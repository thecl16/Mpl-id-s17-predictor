"""
=============================================================
BAGIAN 2: FEATURE ENGINEERING & PREPROCESSING
=============================================================
Input:  data/mpl_raw_data.csv
Output: data/dataset_siap_ml.csv
        data/team_encoder.json  (mapping nama tim → angka)
"""

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder

RAW_PATH   = "data/mpl_raw_data.csv"
OUTPUT_PATH = "data/dataset_siap_ml.csv"
ENCODER_PATH = "data/team_encoder.json"


# ─────────────────────────────────────────────
# 1. Load & Validate Raw Data
# ─────────────────────────────────────────────
def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"team_a", "score_a", "team_b", "score_b", "result_a"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["date"] = pd.to_datetime(df.get("date", pd.Timestamp.today()), errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    print(f"[LOAD] {len(df)} raw matches loaded.")
    return df


# ─────────────────────────────────────────────
# 2. Expand to per-team rows
# ─────────────────────────────────────────────
def expand_to_team_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each match → 2 rows (one per team perspective).
    This gives us a clean time-ordered history per team.
    """
    rows_a = df.rename(columns={
        "team_a": "team", "score_a": "score_for",
        "team_b": "opponent", "score_b": "score_against",
        "result_a": "result"
    })[["match_id", "date", "stage", "team", "opponent", "score_for", "score_against", "result"]]

    rows_b = df.rename(columns={
        "team_b": "team", "score_b": "score_for",
        "team_a": "opponent", "score_a": "score_against",
        "result_b": "result"
    })[["match_id", "date", "stage", "team", "opponent", "score_for", "score_against", "result"]]

    expanded = pd.concat([rows_a, rows_b], ignore_index=True)
    expanded["win"] = (expanded["result"] == "Win").astype(int)
    expanded = expanded.sort_values(["team", "date"]).reset_index(drop=True)
    return expanded


# ─────────────────────────────────────────────
# 3. Rolling Win Rate (last N games per team)
# ─────────────────────────────────────────────
def add_rolling_win_rate(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.copy()
    df["rolling_win_rate"] = (
        df.groupby("team")["win"]
          .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    df["rolling_win_rate"] = df["rolling_win_rate"].fillna(0.5)
    return df


# ─────────────────────────────────────────────
# 4. Head-to-Head Win Rate
# ─────────────────────────────────────────────
def add_h2h_win_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    h2h_rates = {}

    for (team, opp), group in df.groupby(["team", "opponent"]):
        wins = group["win"].sum()
        total = len(group)
        h2h_rates[(team, opp)] = wins / total if total > 0 else 0.5

    df["h2h_win_rate"] = df.apply(
        lambda r: h2h_rates.get((r["team"], r["opponent"]), 0.5), axis=1
    )
    return df


# ─────────────────────────────────────────────
# 5. Additional Features
# ─────────────────────────────────────────────
def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Average score differential (last 5)
    df["score_diff"] = df["score_for"] - df["score_against"]
    df["avg_score_diff"] = (
        df.groupby("team")["score_diff"]
          .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
          .fillna(0)
    )

    # Win streak (positive = win streak, negative = lose streak)
    def calc_streak(series):
        streaks = []
        s = 0
        for v in series:
            if v == 1:
                s = max(1, s + 1)
            else:
                s = min(-1, s - 1)
            streaks.append(s)
        return streaks

    df["win_streak"] = df.groupby("team")["win"].transform(calc_streak)

    # Stage weight (playoffs matches carry more signal)
    stage_map = {"Regular Season": 1, "Regular": 1,
                 "Upper Bracket": 2, "Lower Bracket": 2,
                 "Semifinal": 3, "Grand Final": 4}
    df["stage_weight"] = df.get("stage", pd.Series("Regular", index=df.index)).map(
        lambda s: stage_map.get(str(s), 1)
    )
    return df


# ─────────────────────────────────────────────
# 6. Encode Team Names
# ─────────────────────────────────────────────
def encode_teams(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    le = LabelEncoder()
    all_teams = pd.concat([df["team"], df["opponent"]]).unique()
    le.fit(all_teams)

    df["team_encoded"]     = le.transform(df["team"])
    df["opponent_encoded"] = le.transform(df["opponent"])

    mapping = {str(name): int(code)
               for name, code in zip(le.classes_, le.transform(le.classes_))}
    return df, mapping


# ─────────────────────────────────────────────
# 7. Build ML-ready match-level dataset
# ─────────────────────────────────────────────
def build_match_features(raw: pd.DataFrame, expanded: pd.DataFrame,
                         encoder: dict) -> pd.DataFrame:
    """
    Merge features back to match-level (one row per match).
    Features: team_a stats vs team_b stats → label: did team_a win?
    """
    feat_cols = ["team", "match_id", "rolling_win_rate", "h2h_win_rate",
                 "avg_score_diff", "win_streak", "team_encoded", "opponent_encoded"]
    feats = expanded[feat_cols].copy()

    # team_a side
    fa = feats.rename(columns={c: f"a_{c}" for c in feat_cols if c not in ("team", "match_id")})
    fa = fa.rename(columns={"team": "team_a"})

    # team_b side — h2h from b's perspective is 1 - a's
    fb = feats.rename(columns={c: f"b_{c}" for c in feat_cols if c not in ("team", "match_id")})
    fb = fb.rename(columns={"team": "team_b"})

    # Merge on match_id
    match_df = raw[["match_id", "team_a", "team_b", "result_a"]].copy()
    match_df["label"] = (match_df["result_a"] == "Win").astype(int)

    merged = match_df.merge(
        fa[["match_id", "team_a", "a_rolling_win_rate", "a_h2h_win_rate",
            "a_avg_score_diff", "a_win_streak", "a_team_encoded"]],
        on=["match_id", "team_a"], how="left"
    ).merge(
        fb[["match_id", "team_b", "b_rolling_win_rate", "b_h2h_win_rate",
            "b_avg_score_diff", "b_win_streak", "b_team_encoded"]],
        on=["match_id", "team_b"], how="left"
    )

    # Differential features
    merged["diff_rolling_wr"]  = merged["a_rolling_win_rate"]  - merged["b_rolling_win_rate"]
    merged["diff_score"]       = merged["a_avg_score_diff"]    - merged["b_avg_score_diff"]
    merged["diff_streak"]      = merged["a_win_streak"]        - merged["b_win_streak"]

    merged = merged.dropna()
    return merged


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    raw = load_raw(RAW_PATH)
    expanded = expand_to_team_rows(raw)
    expanded = add_rolling_win_rate(expanded)
    expanded = add_h2h_win_rate(expanded)
    expanded = add_extra_features(expanded)
    expanded, encoder = encode_teams(expanded)

    ml_df = build_match_features(raw, expanded, encoder)
    ml_df = ml_df.dropna().reset_index(drop=True)

    ml_df.to_csv(OUTPUT_PATH, index=False)
    with open(ENCODER_PATH, "w") as f:
        json.dump(encoder, f, indent=2)

    print(f"\n[SAVED] Dataset ML → {OUTPUT_PATH}  ({len(ml_df)} rows, {ml_df.shape[1]} cols)")
    print(f"[SAVED] Team encoder → {ENCODER_PATH}")
    print(f"\nLabel distribution:\n{ml_df['label'].value_counts().to_string()}")
    print(f"\nFeature columns:\n{[c for c in ml_df.columns if c not in ('match_id','team_a','team_b','result_a')]}")
    print("\nSample:")
    print(ml_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
