"""
Validation Script: Compare QB WAR totals to team total wins (regular + postseason),
with robust postseason-wins inference and defensive checks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ------------------------------------------------------------
# 0) Helpers
# ------------------------------------------------------------
def infer_postseason_wins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute postseason_wins and total_wins using whatever columns exist.
    Priority:
      1) total_wins_incl_playoffs - wins
      2) playoff_wins
      3) infer from final_round / playoff_result
      4) else -> zeros
    """
    def col(name):
        for c in df.columns:
            if c.lower() == name.lower():
                return c
        return None

    wins_col = col("wins")
    if wins_col is None:
        raise ValueError("team_results_2024.csv missing a 'wins' column.")

    total_incl_col = col("total_wins_incl_playoffs")
    if total_incl_col is not None:
        df["postseason_wins"] = df[total_incl_col] - df[wins_col]
        df["total_wins"] = df[total_incl_col]
        return df

    playoff_wins_col = col("playoff_wins")
    if playoff_wins_col is not None:
        df["postseason_wins"] = df[playoff_wins_col]
        df["total_wins"] = df[wins_col] + df["postseason_wins"]
        return df

    playoff_result_col = col("playoff_result")
    final_round_col = col("final_round")

    inferred = pd.Series(0, index=df.index, dtype="int64")

    if final_round_col is not None:
        fr_numeric = pd.to_numeric(df[final_round_col], errors="coerce")
        inferred = (
            fr_numeric.fillna(0)
            .clip(lower=0)
            .map({0: 0, 1: 0, 2: 1, 3: 2, 4: 3})
            .fillna(0)
            .astype(int)
        )

    if playoff_result_col is not None:
        pr = df[playoff_result_col].astype(str).str.strip().str.lower()
        map_result_to_wins = {
            "missed": 0, "wc": 0, "wildcard": 0, "wild card": 0,
            "div": 1, "division": 1, "divisional": 1,
            "conf": 2, "conference": 2, "championship": 2,
            "sb": 3, "superbowl": 3, "super bowl": 3,
            "sbchamp": 3, "super bowl champ": 3, "champion": 3,
        }
        inferred_from_text = pr.map(lambda x: map_result_to_wins.get(x, np.nan))
        mask = inferred_from_text.notna()
        inferred.loc[mask] = inferred_from_text.loc[mask].astype(int)

    df["postseason_wins"] = inferred.fillna(0).astype(int)
    df["total_wins"] = df[wins_col] + df["postseason_wins"]
    return df


# ------------------------------------------------------------
# 1) Load data
# ------------------------------------------------------------
season = 2024
print("=" * 70)
print(f"VALIDATING QB WAR VS TEAM TOTAL WINS ({season})")
print("=" * 70)

qb_war = pd.read_csv(
    "/Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs/tables/qb_war_2024_enhanced.csv"
)

team_stats = pd.read_csv(
    "/Users/anokhpalakurthi/Documents/nfl_offensive_war/data/processed/team_results_2024.csv"
)

passing = pd.read_csv(
    "/Users/anokhpalakurthi/Documents/nfl_offensive_war/data/processed/passing_plays.csv"
)

# ------------------------------------------------------------
# 2) Ensure postseason wins exist
# ------------------------------------------------------------
team_stats = infer_postseason_wins(team_stats)

if (team_stats["postseason_wins"] == 0).all():
    print("\n⚠️ Postseason wins still all zero.")
    print("Columns present:", list(team_stats.columns))
    print("\nSample rows:")
    print(team_stats.head(6).to_string(index=False))


# ------------------------------------------------------------
# 3) Map team names to abbreviations
# ------------------------------------------------------------
team_abbrev_map = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

if "team" not in team_stats.columns:
    raise ValueError("team_results_2024.csv missing 'team' column.")

team_stats["team_abbr"] = team_stats["team"].map(team_abbrev_map)


# ------------------------------------------------------------
# 4) CLEANED — Identify primary QB team + filter QB-only + aggregate WAR
# ------------------------------------------------------------

# Count plays by (QB, team)
passing_counts = (
    passing[passing["season"] == season]
    .groupby(["passer_player_name", "posteam"])
    .size()
    .reset_index(name="n_plays")
)

# Pick the team where the QB took the most snaps
passing_season = (
    passing_counts.sort_values("n_plays", ascending=False)
    .drop_duplicates("passer_player_name")[["passer_player_name", "posteam"]]
)

# Filter WAR to actual QBs only
valid_qbs = passing_season["passer_player_name"].unique()
qb_war = qb_war[qb_war["passer_player_name"].isin(valid_qbs)].copy()

# Merge WAR → team and sum
qb_team_war = qb_war.merge(passing_season, on="passer_player_name", how="left")
team_war = qb_team_war.groupby("posteam", as_index=False)["WAR"].sum()
team_war["posteam"] = team_war["posteam"].str.upper()

print("\nTeam WAR sample:")
print(team_war.head())


# ------------------------------------------------------------
# 5) Merge with team stats + compute correlation
# ------------------------------------------------------------
merged = team_stats.merge(
    team_war, left_on="team_abbr", right_on="posteam", how="left"
).rename(columns={"WAR": "total_qb_war"})

merged["total_qb_war"] = merged["total_qb_war"].fillna(0)

print("\nMerged preview:")
print(
    merged[
        ["team", "team_abbr", "wins", "postseason_wins", "total_wins", "total_qb_war"]
    ].head()
)

if merged["total_qb_war"].nunique() <= 1:
    print("\n⚠️ WAR values identical — check team mapping.")
    raise SystemExit()

slope, intercept, r, p, stderr = linregress(merged["total_qb_war"], merged["total_wins"])
merged["predicted_wins"] = slope * merged["total_qb_war"] + intercept
merged["residual"] = merged["total_wins"] - merged["predicted_wins"]

print(f"\nCorrelation (r): {r:.3f}")
print(f"R²: {r**2:.3f}")
print(f"P-value: {p:.5f}")
print(f"Slope: {slope:.3f}   Intercept: {intercept:.3f}")


# ------------------------------------------------------------
# 6) Plot
# ------------------------------------------------------------
plt.figure(figsize=(9, 6))
plt.scatter(merged["total_qb_war"], merged["total_wins"], alpha=0.8)
x = np.linspace(merged["total_qb_war"].min(), merged["total_qb_war"].max(), 100)
plt.plot(x, slope * x + intercept, color="red", linewidth=2, label="Best Fit")
plt.title(f"QB WAR vs Team Total Wins ({season})")
plt.xlabel("Total QB WAR (Team)")
plt.ylabel("Total Wins (Regular + Postseason)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 7) Dump table
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("TEAM-BY-TEAM SUMMARY")
print("=" * 70)

print(
    merged[
        ["team", "wins", "postseason_wins", "total_wins", "total_qb_war", "predicted_wins"]
    ]
    .sort_values(["total_wins", "total_qb_war"], ascending=False)
    .round(2)
    .to_string(index=False)
)

out = "/Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs/tables/team_war_vs_total_wins_2024.csv"
merged.to_csv(out, index=False)
print(f"\n✓ Saved residual table: {out}")