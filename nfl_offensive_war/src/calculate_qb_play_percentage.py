"""
03_calculate_qb_play_percentage.py
----------------------------------
1. Calculates what % of offensive plays involve the QB (pass + QB rush)
2. Runs regression to derive empirical QB share of offensive value
3. Converts TEAM EPA per win → QB-specific EPA per win using both
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

# ============================================================
# PATHS
# ============================================================
DATA_DIR = "/Users/anokhpalakurthi/Documents/nfl_offensive_war/data/processed"
OUTPUT_DIR = "/Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs/tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. QB PLAY PERCENTAGE (pass + QB run / total offense)
# ============================================================

def calculate_qb_play_percentage(season=2024):
    """Compute QB involvement share for a given season."""
    print(f"\n{'='*70}\nCALCULATING QB PLAY PERCENTAGE - {season} SEASON\n{'='*70}")

    passing = pd.read_csv(f"{DATA_DIR}/passing_plays.csv")
    rushing = pd.read_csv(f"{DATA_DIR}/rushing_plays.csv")

    p = passing[passing["season"] == season]
    r = rushing[rushing["season"] == season]

    qb_names = p["passer_player_name"].unique()
    qb_rushing = r[r["rusher_player_name"].isin(qb_names)]
    non_qb_rushing = r[~r["rusher_player_name"].isin(qb_names)]

    total_offensive = len(p) + len(r)
    qb_involved = len(p) + len(qb_rushing)
    qb_play_pct = qb_involved / total_offensive

    print(f"\nPassing plays:          {len(p):>7,}")
    print(f"QB rushing plays:       {len(qb_rushing):>7,}")
    print(f"Non-QB rushing plays:   {len(non_qb_rushing):>7,}")
    print(f"Total offensive plays:  {total_offensive:>7,}")
    print(f"QB-involved plays:      {qb_involved:>7,}")
    print(f"\nQB PLAY PERCENTAGE:     {qb_play_pct:.2%}")
    return qb_play_pct


def calculate_across_all_seasons():
    """Compute average QB play percentage 2021–2024."""
    print(f"\n{'='*70}\nCHECKING CONSISTENCY ACROSS ALL SEASONS\n{'='*70}")
    results = []
    for s in [2021, 2022, 2023, 2024]:
        pct = calculate_qb_play_percentage(s)
        results.append({"season": s, "qb_play_pct": pct})

    df = pd.DataFrame(results)
    avg, std = df["qb_play_pct"].mean(), df["qb_play_pct"].std()

    print(f"\n{'='*70}\nSUMMARY 2021–2024\n{'='*70}")
    print(df.to_string(index=False))
    print(f"\nAverage: {avg:.2%} | Std Dev: {std:.2%}")
    print(f"Using {avg:.2%} as QB play percentage.")
    return avg


# ============================================================
# 2. EMPIRICAL QB SHARE OF OFFENSIVE VALUE
# ============================================================

def derive_empirical_qb_share(season=2024):
    """Run regression (EPA/play ~ CPOE + Success Rate) → derive QB share."""
    print(f"\n{'='*70}\nDERIVING EMPIRICAL QB SHARE FROM REGRESSION ({season})\n{'='*70}")

    passing = pd.read_csv(f"{DATA_DIR}/passing_plays.csv", low_memory=False)
    rushing = pd.read_csv(f"{DATA_DIR}/rushing_plays.csv", low_memory=False)

    p = passing[passing["season"] == season].copy()
    r = rushing[rushing["season"] == season].copy()

    # Aggregate QB passing stats
    qb_pass = (
        p.groupby(["passer_player_id", "passer_player_name"])
        .agg(
            passing_epa=("epa", "sum"),
            passing_attempts=("play_id", "count"),
            avg_cpoe=("cpoe", "mean"),
            passing_success=("success", "mean"),
        )
        .reset_index()
    )

    # Aggregate QB rushing stats
    qb_rush = (
        r[r["rusher_player_name"].isin(qb_pass["passer_player_name"])]
        .groupby(["rusher_player_id", "rusher_player_name"])
        .agg(
            rushing_epa=("epa", "sum"),
            rushing_attempts=("play_id", "count"),
            rushing_success=("success", "mean"),
        )
        .reset_index()
    )

    qb_pass["passer_player_name"] = qb_pass["passer_player_name"].str.title()
    qb_rush["rusher_player_name"] = qb_rush["rusher_player_name"].str.title()

    qb_tot = qb_pass.merge(
        qb_rush[
            ["rusher_player_name", "rushing_epa", "rushing_attempts", "rushing_success"]
        ],
        left_on="passer_player_name",
        right_on="rusher_player_name",
        how="left",
    ).drop(columns=["rusher_player_name"], errors="ignore")

    qb_tot.fillna({"rushing_epa": 0, "rushing_attempts": 0, "rushing_success": 0}, inplace=True)
    qb_tot["total_epa"] = qb_tot["passing_epa"] + qb_tot["rushing_epa"]
    qb_tot["total_plays"] = qb_tot["passing_attempts"] + qb_tot["rushing_attempts"]
    qb_tot = qb_tot[qb_tot["total_plays"] >= 50]
    qb_tot["epa_per_play"] = qb_tot["total_epa"] / qb_tot["total_plays"]
    qb_tot["overall_success"] = (
        qb_tot["passing_success"] * qb_tot["passing_attempts"]
        + qb_tot["rushing_success"] * qb_tot["rushing_attempts"]
    ) / qb_tot["total_plays"]

    # Drop invalid or infinite values
    qb_tot = qb_tot.replace([np.inf, -np.inf], np.nan).dropna(subset=["avg_cpoe", "overall_success", "epa_per_play"])
    print(f"\nValid QBs for regression: {len(qb_tot)} (after dropping NaN/inf values)")

    # Regression
    X = sm.add_constant(qb_tot[["avg_cpoe", "overall_success"]])
    y = qb_tot["epa_per_play"]
    ols = sm.OLS(y, X).fit()

    r2 = ols.rsquared
    qb_share = float(r2)

    print("\nRegression Summary:")
    print(ols.summary())
    print(f"\nR² = {r2:.3f} → QB Share = {qb_share:.3f}")

    out = pd.DataFrame({"metric": ["qb_share_empirical"], "value": [qb_share]})
    out.to_csv(f"{OUTPUT_DIR}/qb_share_empirical.csv", index=False)
    print(f"✓ Saved empirical QB share → {OUTPUT_DIR}/qb_share_empirical.csv")

    return qb_share


# ============================================================
# 3. CALCULATE FINAL QB-SPECIFIC EPA PER WIN
# ============================================================

def calculate_final_qb_epa_per_win(team_epa_per_win=218.83):
    """Combine QB play% and empirical QB share to scale team EPA→QB EPA."""
    print(f"\n{'='*70}\nCALCULATING FINAL QB EPA PER WIN\n{'='*70}")
    qb_play_pct = calculate_across_all_seasons()
    qb_share = derive_empirical_qb_share(2024)

    qb_epa_per_win = team_epa_per_win * qb_play_pct * qb_share

    print(f"\nTeam EPA per Win:            {team_epa_per_win:.2f}")
    print(f"QB Play Percentage:          {qb_play_pct:.2%}")
    print(f"Empirical QB Share:          {qb_share:.2%}")
    print(f"--------------------------------------------")
    print(f"QB-Specific EPA per Win:     {qb_epa_per_win:.2f}")

    pd.DataFrame({
        "metric": ["team_epa_per_win", "qb_play_pct", "qb_share", "qb_epa_per_win"],
        "value": [team_epa_per_win, qb_play_pct, qb_share, qb_epa_per_win],
    }).to_csv(f"{OUTPUT_DIR}/qb_epa_per_win_conversion.csv", index=False)

    print(f"✓ Saved final conversion → {OUTPUT_DIR}/qb_epa_per_win_conversion.csv")
    return qb_epa_per_win


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    calculate_final_qb_epa_per_win()