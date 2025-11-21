"""
CLEAN MULTI-SEASON RAW QB WAR CALCULATOR (2021–2024)
----------------------------------------------------
- Uses only raw EPA/play (no modeled / enhanced EPA)
- No minimum attempts threshold (only excludes 0-play QBs)
- Replacement level: 25th percentile EPA/play per season, with a small
  fallback of at least 8 QBs in the "replacement pool"
- WAR = (EPA/play - repl_EPA/play) * total_plays / EPA_per_win
Outputs:
- outputs/tables/qb_war_raw_2021_2024_full.csv
- outputs/tables/qb_war_raw_cumulative_2021_2024.csv
- outputs/tables/qb_war_raw_position_summary.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIG / PATHS
# ============================================================

REPL_PERCENTILE = 0.25      # 25th percentile of EPA/play
REPL_MIN_QBS    = 8         # minimum number of QBs in replacement pool
EPA_PER_WIN     = 218.83 * 0.674 * 0.6753   # ≈ 99.5

BASE_DIR   = Path("/Users/anokhpalakurthi/Documents/nfl_offensive_war")
DATA_DIR   = BASE_DIR / "data" / "processed"
OUT_DIR    = BASE_DIR / "outputs"
TABLE_DIR  = OUT_DIR / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

PASSING_PATH = DATA_DIR / "passing_plays.csv"
RUSHING_PATH = DATA_DIR / "rushing_plays.csv"


# ============================================================
# NAME NORMALIZATION
# ============================================================

def normalize_qb_names(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df.loc[df[col] == "Aa.Rodgers", col] = "A.Rodgers"
    return df


# ============================================================
# LOAD DATA (RAW EPA ONLY)
# ============================================================

def load_all_qbs_for_season(season: int) -> pd.DataFrame:
    """
    Load all QB-level aggregates for a given season with:
      - total_epa
      - total_plays
      - epa_per_play

    No minimum attempts filter, except excluding total_plays == 0.
    """

    passing = pd.read_csv(PASSING_PATH)
    rushing = pd.read_csv(RUSHING_PATH)

    p = passing.query("season == @season").copy()
    r = rushing.query("season == @season").copy()

    p = normalize_qb_names(p, "passer_player_name")
    r = normalize_qb_names(r, "rusher_player_name")

    # Passing EPA / plays
    qb_pass = (
        p.groupby("passer_player_name", as_index=False)
        .agg(
            passing_epa=("epa", "sum"),
            passing_plays=("play_id", "count"),
        )
    )

    # Rushing EPA / plays (for QBs)
    qb_rush = (
        r.groupby("rusher_player_name", as_index=False)
        .agg(
            rushing_epa=("epa", "sum"),
            rushing_plays=("play_id", "count"),
        )
    )

    df = qb_pass.merge(
        qb_rush,
        left_on="passer_player_name",
        right_on="rusher_player_name",
        how="left",
    )

    df["rushing_epa"]    = df["rushing_epa"].fillna(0.0)
    df["rushing_plays"]  = df["rushing_plays"].fillna(0).astype(int)
    df["season"]         = season

    df["total_epa"]   = df["passing_epa"] + df["rushing_epa"]
    df["total_plays"] = df["passing_plays"] + df["rushing_plays"]

    # Exclude QBs with zero total plays (degenerate)
    df = df[df["total_plays"] > 0].copy()

    df["epa_per_play"] = df["total_epa"] / df["total_plays"]

    return df[[
        "passer_player_name",
        "season",
        "total_epa",
        "total_plays",
        "epa_per_play",
    ]]


def load_all_qbs_multi(start: int = 2021, end: int = 2024) -> pd.DataFrame:
    frames = [load_all_qbs_for_season(y) for y in range(start, end + 1)]
    out = pd.concat(frames, ignore_index=True)
    print(f"✓ Loaded {len(out)} raw QB-seasons ({start}–{end})")
    return out


# ============================================================
# REPLACEMENT LEVEL & WAR
# ============================================================

def identify_replacement_by_season(
    qb_df: pd.DataFrame,
    percentile: float = REPL_PERCENTILE,
    min_qbs: int = REPL_MIN_QBS,
) -> pd.DataFrame:
    """
    For each season, define a replacement EPA/play as:

      - Take the given percentile (e.g., 25th) of EPA/play
      - Use all QBs at or below that threshold
      - If fewer than min_qbs in that pool, backfill by taking
        the min_qbs lowest EPA/play QBs
    """

    rows = []
    for s, g in qb_df.groupby("season"):
        g_valid = g.dropna(subset=["epa_per_play"]).copy()
        if g_valid.empty:
            raise ValueError(f"No valid EPA/play rows for season {s}")

        thresh = np.percentile(g_valid["epa_per_play"], percentile * 100.0)
        low = g_valid[g_valid["epa_per_play"] <= thresh]

        if len(low) < min_qbs:
            low = g_valid.nsmallest(min_qbs, "epa_per_play")

        replacement = low["epa_per_play"].mean()

        rows.append({
            "season": s,
            "replacement_epa_per_play": replacement,
            "n_replacement_qbs": len(low),
        })

    repl_df = pd.DataFrame(rows)
    print("\n=== Replacement EPA/play by season (raw) ===")
    print(repl_df.to_string(index=False))
    return repl_df


def calculate_war_multi(
    qb_df: pd.DataFrame,
    repl_df: pd.DataFrame,
    epa_per_win: float = EPA_PER_WIN,
) -> pd.DataFrame:
    """
    Compute WAR per season using raw EPA/play vs replacement.
    """

    out_frames = []
    for s, g in qb_df.groupby("season"):
        g = g.copy()
        repl = repl_df.loc[repl_df["season"] == s, "replacement_epa_per_play"].iloc[0]

        g["epa_above_repl"] = (g["epa_per_play"] - repl) * g["total_plays"]
        g["WAR"] = g["epa_above_repl"] / epa_per_win

        out_frames.append(g)

    qb_all = pd.concat(out_frames, ignore_index=True)
    return qb_all


# ============================================================
# RODGERS DIAGNOSTIC (RAW EPA + WAR)
# ============================================================

def rodgers_diagnostic(qb_all: pd.DataFrame, repl_df: pd.DataFrame) -> None:
    """
    Print raw EPA + WAR breakdown for Aaron Rodgers as a sanity check.
    """

    name = "A.Rodgers"
    subset = qb_all[qb_all["passer_player_name"] == name].copy()

    if subset.empty:
        print("\n⚠️ No Rodgers data found — check normalization.")
        return

    rows = []
    for _, r in subset.iterrows():
        season = r["season"]
        repl = repl_df.loc[repl_df["season"] == season, "replacement_epa_per_play"].iloc[0]

        rows.append({
            "season": season,
            "total_plays": int(r["total_plays"]),
            "total_epa": r["total_epa"],
            "epa_per_play": r["epa_per_play"],
            "repl_epa_per_play": repl,
            "epa_above_repl": r["epa_above_repl"],
            "WAR": r["WAR"],
        })

    print("\n=== Rodgers RAW EPA + WAR Diagnostic ===")
    df = pd.DataFrame(rows).sort_values("season")
    numeric_cols = ["total_epa", "epa_per_play", "repl_epa_per_play",
                    "epa_above_repl", "WAR"]
    df[numeric_cols] = df[numeric_cols].round(3)
    print(df.to_string(index=False))


# ============================================================
# CUMULATIVE LEADERBOARD & POSITION SUMMARY
# ============================================================

def make_cumulative_leaderboard(qb_all: pd.DataFrame) -> pd.DataFrame:
    """
    Build cumulative raw WAR leaderboard for 2021–2024.
    """

    cum = (
        qb_all
        .groupby("passer_player_name", as_index=False)
        .agg(
            total_WAR=("WAR", "sum"),
            total_plays=("total_plays", "sum"),
        )
    )

    cum_sorted = cum.sort_values("total_WAR", ascending=False)
    out_path = TABLE_DIR / "qb_war_raw_cumulative_2021_2024.csv"
    cum_sorted.to_csv(out_path, index=False)

    print(f"\n✓ Saved cumulative RAW WAR leaderboard → {out_path}")
    print("\n=== Top 15 QBs by RAW cumulative WAR (2021–2024) ===")
    print(cum_sorted.head(15).round(3).to_string(index=False))

    return cum_sorted


def save_full_qb_table(qb_all: pd.DataFrame) -> None:
    """
    Save full per-season raw WAR table.
    """

    df = qb_all.copy()
    num_cols = ["total_epa", "total_plays", "epa_per_play",
                "epa_above_repl", "WAR"]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].round(4)

    out_path = TABLE_DIR / "qb_war_raw_2021_2024_full.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved full RAW QB WAR table → {out_path}")


def save_qb_position_summary(qb_all: pd.DataFrame) -> None:
    """
    Simple QB-only WAR distribution summary (for comparison with RB/WR/TE).
    """

    df = qb_all.copy()

    summary = {
        "position": ["QB"],
        "count": [len(df)],
        "avg_WAR": [df["WAR"].mean()],
        "median_WAR": [df["WAR"].median()],
        "IQR_WAR": [
            df["WAR"].quantile(0.75) - df["WAR"].quantile(0.25)
        ],
    }

    summary_df = pd.DataFrame(summary)
    out_path = TABLE_DIR / "qb_war_raw_position_summary.csv"
    summary_df.to_csv(out_path, index=False)

    print("\n===== QB RAW WAR Position Summary — 2021–2024 (player-seasons) =====")
    print(summary_df.round(4).to_string(index=False))
    print(f"✓ Saved QB RAW WAR position summary → {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("CLEAN MULTI-SEASON RAW QB WAR CALCULATOR (2021–2024)")
    print("=" * 70)

    # 1) Load raw EPA aggregates
    qb_df = load_all_qbs_multi(2021, 2024)

    # 2) Replacement EPA/play per season
    repl_df = identify_replacement_by_season(qb_df)

    # 3) Compute WAR from raw EPA/play
    qb_all = calculate_war_multi(qb_df, repl_df, EPA_PER_WIN)

    # 4) Save full table
    save_full_qb_table(qb_all)

    # 5) Cumulative leaderboard
    cumulative_df = make_cumulative_leaderboard(qb_all)

    # 6) Rodgers diagnostic (raw)
    rodgers_diagnostic(qb_all, repl_df)

    # 7) Simple QB-only position summary (for RB/WR/TE comparison)
    save_qb_position_summary(qb_all)

if __name__ == "__main__":
    main()