"""
Tennis Match-Win Probability Model

A dedicated classifier for "who wins this match" — separate from the 7 stat
models. It is built to weigh the things that actually decide tennis matches,
not ranking alone:

    - Surface mastery   : career + recent win% ON THIS SURFACE (clay/hard/grass)
    - Recent form       : win% over last 10 / 25 matches
    - Momentum           : form trend (recent win% minus career win%), last-5 wins
    - Quality of wins   : career win% vs top-50, avg rank of recent opponents
    - Head-to-head      : prior H2H win% and meeting count (this pairing)
    - Ranking            : rank gap (log-scaled)
    - Context           : best-of-5 (men's Slam), tour

Design notes
------------
* Built from the RAW player-match rows (one row per player per match), so it
  avoids the cartesian row-explosion in the processed stat dataset.
* All form features are computed from PRIOR matches only (shifted) to prevent
  leakage. Scheduling features (days-rest / back-to-back) are deliberately
  EXCLUDED — in historical data they leak the result (you only play
  back-to-back if you keep winning).
* Symmetric by construction: features are player-minus-opponent differences and
  every match contributes both perspectives, so P(A) + P(B) = 1.

Usage:
    python3 -m src.sports.tennis.win_model            # train + scan today
    python3 -m src.sports.tennis.win_model 2026-05-25 # train + scan a date
"""

import os
import sys
import json
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW_DIR = os.path.join(BASE_DIR, "data", "tennis", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "models", "tennis")
MODEL_PATH = os.path.join(MODEL_DIR, "win_model.json")
FEATS_PATH = os.path.join(MODEL_DIR, "win_model_features.json")

# player-level pre-match state columns (computed once, then differenced)
PLAYER_STATE = [
    "win_career", "win_l10", "win_l25", "wins_last5",
    "form_trend", "win_vs_top50", "avg_opp_rank_l10",
    "surf_win", "surf_win_l20", "surf_exp", "log_rank",
]

# final model features: differences (player - opponent) + relative/context terms
FEATURES = (
    [f"d_{c}" for c in PLAYER_STATE]
    + ["h2h_edge", "h2h_n", "is_best_of_5", "is_atp"]
)


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING (from raw player-match rows)
# ---------------------------------------------------------------------------

def _load_raw():
    frames = []
    for tour in ("atp", "wta"):
        p = os.path.join(RAW_DIR, f"{tour}_raw_matches.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p, low_memory=False))
    if not frames:
        raise FileNotFoundError("No raw match data. Run the tennis builder first.")
    df = pd.concat(frames, ignore_index=True)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    df = df.dropna(subset=["tourney_date", "player_id", "opp_id"])
    df = df[df.get("retired", 0) == 0]
    df["won_match"] = df["won_match"].astype(int)
    df["is_atp"] = (df["tour"] == "atp").astype(int)
    df["is_best_of_5"] = (pd.to_numeric(df["best_of"], errors="coerce") == 5).astype(int)
    df["player_rank"] = pd.to_numeric(df["player_rank"], errors="coerce").fillna(250).clip(1, 2000)
    df["opp_rank"] = pd.to_numeric(df["opp_rank"], errors="coerce").fillna(250).clip(1, 2000)
    df["surf"] = df["surface"].where(df["surface"].isin(["Hard", "Clay", "Grass"]), "Hard")
    return df.sort_values(["player_id", "tourney_date"]).reset_index(drop=True)


def _add_cumulative(df):
    """Cumulative (incl. current match) player state. Pre-match = shift within group."""
    g = df.groupby("player_id", sort=False)
    df["win_career"] = g["won_match"].transform(lambda x: x.expanding().mean())
    df["win_l10"] = g["won_match"].transform(lambda x: x.rolling(10, min_periods=1).mean())
    df["win_l25"] = g["won_match"].transform(lambda x: x.rolling(25, min_periods=1).mean())
    df["wins_last5"] = g["won_match"].transform(lambda x: x.rolling(5, min_periods=1).sum())
    df["log_rank"] = np.log1p(df["player_rank"])

    won_vs_top50 = df["won_match"].where(df["opp_rank"] <= 50)
    df["win_vs_top50"] = (
        won_vs_top50.groupby(df["player_id"]).transform(lambda x: x.expanding().mean())
    )
    df["avg_opp_rank_l10"] = g["opp_rank"].transform(lambda x: x.rolling(10, min_periods=1).mean())

    # surface-specific (group by player + surface)
    gs = df.groupby(["player_id", "surf"], sort=False)
    df["surf_win"] = gs["won_match"].transform(lambda x: x.expanding().mean())
    df["surf_win_l20"] = gs["won_match"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df["surf_exp"] = np.log1p(gs.cumcount())

    # head-to-head vs this specific opponent
    gh = df.groupby(["player_id", "opp_id"], sort=False)
    df["h2h_win"] = gh["won_match"].transform(lambda x: x.expanding().mean())
    df["h2h_n"] = gh.cumcount()
    return df


def _to_prematch(df):
    """Shift each cumulative column within its group so a row sees only prior matches."""
    out = df.copy()
    pcols = ["win_career", "win_l10", "win_l25", "wins_last5",
             "win_vs_top50", "avg_opp_rank_l10"]
    for c in pcols:
        out[c] = out.groupby("player_id")[c].shift(1)
    for c in ["surf_win", "surf_win_l20", "surf_exp"]:
        out[c] = out.groupby(["player_id", "surf"])[c].shift(1)
    for c in ["h2h_win", "h2h_n"]:
        out[c] = out.groupby(["player_id", "opp_id"])[c].shift(1)
    # log_rank, is_atp, is_best_of_5 are known pre-match (no shift)

    # sensible defaults for a player's first matches / first time on a surface
    out["win_career"] = out["win_career"].fillna(0.5)
    out["win_l10"] = out["win_l10"].fillna(0.5)
    out["win_l25"] = out["win_l25"].fillna(0.5)
    out["wins_last5"] = out["wins_last5"].fillna(2.0)
    out["win_vs_top50"] = out["win_vs_top50"].fillna(0.30)
    out["avg_opp_rank_l10"] = out["avg_opp_rank_l10"].fillna(150.0)
    out["surf_win"] = out["surf_win"].fillna(out["win_career"])
    out["surf_win_l20"] = out["surf_win_l20"].fillna(out["win_career"])
    out["surf_exp"] = out["surf_exp"].fillna(0.0)
    out["h2h_win"] = out["h2h_win"].fillna(0.5)
    out["h2h_n"] = out["h2h_n"].fillna(0.0)
    out["form_trend"] = out["win_l10"] - out["win_career"]
    return out


def build_training_frame():
    raw = _add_cumulative(_load_raw())
    pre = _to_prematch(raw)

    keep = ["tourney_date", "player_id", "opp_id", "won_match",
            "is_atp", "is_best_of_5"] + PLAYER_STATE + ["h2h_win", "h2h_n"]
    pre = pre[keep]

    # attach opponent's pre-match player state via the unique match triple
    opp = pre[["tourney_date", "player_id", "opp_id"] + PLAYER_STATE].rename(
        columns={"player_id": "_pid", "opp_id": "_oid",
                 **{c: f"opp_{c}" for c in PLAYER_STATE}})
    merged = pre.merge(
        opp,
        left_on=["tourney_date", "opp_id", "player_id"],
        right_on=["tourney_date", "_pid", "_oid"],
        how="inner",
    )

    for c in PLAYER_STATE:
        merged[f"d_{c}"] = merged[c] - merged[f"opp_{c}"]
    merged["h2h_edge"] = merged["h2h_win"] - 0.5

    X = merged[FEATURES].astype(float).fillna(0.0)
    y = merged["won_match"].astype(int)
    dates = merged["tourney_date"]
    return X, y, dates, raw


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------

def train(verbose=True):
    X, y, dates, raw = build_training_frame()
    order = dates.sort_values().index
    X, y = X.loc[order].reset_index(drop=True), y.loc[order].reset_index(drop=True)
    split = int(len(X) * 0.9)

    model = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.04, max_depth=4,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=8,
        reg_alpha=0.2, reg_lambda=1.5, eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    model.fit(X.iloc[:split], y.iloc[:split])

    pv = model.predict_proba(X.iloc[split:])[:, 1]
    yt = y.iloc[split:]
    acc, auc, ll = accuracy_score(yt, pv > 0.5), roc_auc_score(yt, pv), log_loss(yt, pv)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    with open(FEATS_PATH, "w") as f:
        json.dump(FEATURES, f)

    if verbose:
        print(f"   win model: {len(X):,} matches | "
              f"test acc {acc:.3f} | AUC {auc:.3f} | logloss {ll:.3f}")
        imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        print("   top features:")
        for k, v in imp.head(8).items():
            print(f"      {k:<18}{v:.3f}")
    return model, raw


# ---------------------------------------------------------------------------
# PREDICT
# ---------------------------------------------------------------------------

def build_snapshots(raw):
    """Current (incl. latest match) state per player, plus per-surface state + H2H."""
    from src.sports.tennis.scanner import normalize_name
    raw = raw.copy()
    raw["norm"] = raw["player_name"].map(normalize_name)

    last = raw.sort_values("tourney_date").drop_duplicates("player_id", keep="last")
    base = last.set_index("player_id")[
        ["win_career", "win_l10", "win_l25", "wins_last5", "win_vs_top50",
         "avg_opp_rank_l10", "player_rank"]
    ].to_dict("index")

    surf_last = raw.sort_values("tourney_date").drop_duplicates(["player_id", "surf"], keep="last")
    surf = {}
    for _, r in surf_last.iterrows():
        surf[(r["player_id"], r["surf"])] = (r["surf_win"], r["surf_win_l20"], r["surf_exp"])

    name_to_id = (raw.sort_values("tourney_date")
                     .drop_duplicates("norm", keep="last")
                     .set_index("norm")["player_id"].to_dict())

    h2h = defaultdict(lambda: [0, 0])  # (pid, oid) -> [wins, games]
    for pid, oid, w in zip(raw["player_id"], raw["opp_id"], raw["won_match"]):
        h2h[(pid, oid)][0] += int(w)
        h2h[(pid, oid)][1] += 1
    return {"base": base, "surf": surf, "name_to_id": name_to_id, "h2h": h2h}


def _player_vec(snap, name, surface, rank):
    """Return player_state dict + (found_in_history, player_id)."""
    from src.sports.tennis.scanner import normalize_name
    pid = snap["name_to_id"].get(normalize_name(name))
    log_rank = float(np.log1p(rank))
    if pid is None or pid not in snap["base"]:
        return ({"win_career": 0.42, "win_l10": 0.42, "win_l25": 0.42, "wins_last5": 2.0,
                 "form_trend": 0.0, "win_vs_top50": 0.20, "avg_opp_rank_l10": 130.0,
                 "surf_win": 0.42, "surf_win_l20": 0.42, "surf_exp": 0.0,
                 "log_rank": log_rank}, False, pid)
    b = snap["base"][pid]
    sw, sw20, sx = snap["surf"].get((pid, surface), (b["win_career"], b["win_career"], 0.0))
    return ({
        "win_career": b["win_career"], "win_l10": b["win_l10"], "win_l25": b["win_l25"],
        "wins_last5": b["wins_last5"], "form_trend": b["win_l10"] - b["win_career"],
        "win_vs_top50": b["win_vs_top50"], "avg_opp_rank_l10": b["avg_opp_rank_l10"],
        "surf_win": sw, "surf_win_l20": sw20, "surf_exp": sx, "log_rank": log_rank,
    }, True, pid)


def predict_matchup(model, feats, snap, a, b, surface, rank_a, rank_b, is_atp):
    va, ina, pa_id = _player_vec(snap, a, surface, rank_a)
    vb, inb, pb_id = _player_vec(snap, b, surface, rank_b)
    bo5 = 1 if is_atp else 0

    def row(p, o, pid, oid):
        r = {f"d_{c}": p[c] - o[c] for c in PLAYER_STATE}
        w, n = (snap["h2h"].get((pid, oid), [0, 0]) if pid and oid else [0, 0])
        r["h2h_edge"] = (w / n - 0.5) if n else 0.0
        r["h2h_n"] = float(n)
        r["is_best_of_5"], r["is_atp"] = bo5, is_atp
        return pd.DataFrame([r])[feats].astype(float)

    pa = model.predict_proba(row(va, vb, pa_id, pb_id))[0, 1]
    pb = model.predict_proba(row(vb, va, pb_id, pa_id))[0, 1]
    win_a = (pa + (1 - pb)) / 2
    n_h2h = snap["h2h"].get((pa_id, pb_id), [0, 0])[1] if (pa_id and pb_id) else 0
    return win_a, (ina and inb), n_h2h


# ---------------------------------------------------------------------------
# SCAN A DATE (matchups from PrizePicks game relationships)
# ---------------------------------------------------------------------------

def fetch_matchups(date_str):
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                      "Accept": "application/json", "Referer": "https://app.prizepicks.com/",
                      "Origin": "https://app.prizepicks.com"})
    d = s.get("https://partner-api.prizepicks.com/projections?per_page=1000&single_stat=true",
              timeout=25).json()
    pmap, lmap = {}, {}
    for i in d.get("included", []):
        if i.get("type") == "new_player":
            pmap[str(i["id"])] = i["attributes"].get("name")
        elif i.get("type") == "league":
            lmap[str(i["id"])] = i["attributes"].get("name")
    tennis = {k for k, v in lmap.items() if v and "tennis" in v.lower()}
    games = defaultdict(set)
    for p in d.get("data", []):
        a, rels = p["attributes"], p.get("relationships", {})
        ld = rels.get("league", {}).get("data")
        if not ld or str(ld["id"]) not in tennis:
            continue
        if not str(a.get("start_time", "")).startswith(date_str):
            continue
        g, pl = rels.get("game", {}).get("data"), rels.get("new_player", {}).get("data")
        if g and pl and pmap.get(str(pl["id"])):
            games[str(g["id"])].add(pmap[str(pl["id"])])
    return [tuple(sorted(v)) for v in games.values() if len(v) == 2]


def scan_date(date_str, surface="Clay"):
    from src.sports.tennis.rankings import TennisRankings
    from src.sports.tennis.scanner import normalize_name

    print(f"...training win model")
    model, raw = train()
    feats = FEATURES
    snap = build_snapshots(raw)

    rk = TennisRankings()
    rk.load()

    def rank_of(name):
        r = rk.get_rank(name)
        if r == 50.0:  # unknown -> fall back to last seen rank in raw
            pid = snap["name_to_id"].get(normalize_name(name))
            if pid in snap["base"]:
                r = float(snap["base"][pid]["player_rank"])
        return r

    matchups = fetch_matchups(date_str)
    print(f"...{len(matchups)} matchups for {date_str}\n")

    rows = []
    for a, b in matchups:
        is_atp = 1 if rk.get_tour(a) == "atp" else 0
        ra, rb = rank_of(a), rank_of(b)
        wa, ok, nh = predict_matchup(model, feats, snap, a, b, surface, ra, rb, is_atp)
        fav, fr, dog, dr, fp = ((a, ra, b, rb, wa) if wa >= 0.5 else (b, rb, a, ra, 1 - wa))
        rows.append({"TOUR": "ATP" if is_atp else "WTA",
                     "FAV": fav, "rF": ra if wa >= 0.5 else rb,
                     "DOG": dog, "rD": rb if wa >= 0.5 else ra,
                     "FAV%": round(fp * 100, 1), "H2H": nh, "conf": "ok" if ok else "low"})

    res = pd.DataFrame(rows)
    out = os.path.join(BASE_DIR, "output", "tennis", "scans", f"win_prob_{date_str}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res.to_csv(out, index=False)

    for tour in ("WTA", "ATP"):
        sub = res[res.TOUR == tour].sort_values("FAV%", ascending=False)
        if sub.empty:
            continue
        print(f"\n{'='*74}\n  {tour} — {date_str} win probability ({len(sub)} matches)\n{'='*74}")
        print(f"{'FAVORITE':<22}{'Rk':>4}{'WIN%':>7}   {'UNDERDOG':<22}{'Rk':>4}  H2H")
        print("-" * 74)
        for _, r in sub.iterrows():
            star = "" if r.conf == "ok" else " *"
            print(f"{r.FAV[:21]:<22}{int(r.rF):>4}{r['FAV%']:>6.0f}%   "
                  f"{r.DOG[:21]:<22}{int(r.rD):>4}  {r.H2H if r.H2H else '-'}{star}")
    print(f"\n  * = a player not in match history (rank-based only)")
    print(f"\nSaved -> {out}")
    return res


if __name__ == "__main__":
    from datetime import datetime
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    scan_date(date)
