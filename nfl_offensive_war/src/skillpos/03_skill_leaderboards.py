"""
======================================================================
SPWAR 03 — Skill WAR Leaderboards & Positional Analysis (2021–2024)
======================================================================

Inputs:
  - outputs/tables/sp_skill_war_2021_2024.csv

Outputs:
  - outputs/tables/sp_skill_war_season_top10.csv
  - outputs/tables/sp_skill_war_global_top50.csv
  - outputs/tables/spwar_position_summary.csv
  - outputs/plots/spwar_position_boxplot.png
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nfl_data_py as nfl
from pathlib import Path

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE = Path("/Users/anokhpalakurthi/Documents/nfl_offensive_war")
OUT_TBLS = BASE / "outputs/tables"
OUT_PLOTS = BASE / "outputs/plots"

OUT_TBLS.mkdir(parents=True, exist_ok=True)
OUT_PLOTS.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# 1. Load WAR table
# -------------------------------------------------------------------
war = pd.read_csv(OUT_TBLS / "sp_skill_war_2021_2024.csv")
war["season"] = war["season"].astype(int)

print("\nLoaded SPWAR player-season table: ", len(war))

# -------------------------------------------------------------------
# 2. Load player positions from nfl_data_py
# -------------------------------------------------------------------
print("Fetching player position data from nfl_data_py...")

players = nfl.import_players()
players = players.rename(columns={
    "gsis_id": "player_id",
    "display_name": "player_name"
})

players = players[["player_id", "player_name", "position"]].drop_duplicates()

# Merge positions by stable player_id only
war_pos = war.merge(players[["player_id", "position"]], on="player_id", how="left")

# Keep true offensive positions only
war_pos = war_pos[war_pos["position"].isin(["RB", "WR", "TE"])]

print(f"Rows after merge with positions: {len(war_pos)}")
print(war_pos.position.value_counts(dropna=False))
# -------------------------------------------------------------------
# 3. Top-10 per season
# -------------------------------------------------------------------
print("\n======================================================================")
print("Top 10 by Total WAR per Season (Non-QB Skill Players)")
print("======================================================================")

season_top10_rows = []

for yr in sorted(war_pos["season"].unique()):
    top10 = (
        war_pos[war_pos["season"] == yr]
        .sort_values("total_WAR", ascending=False)
        .head(10)
    )
    season_top10_rows.append(top10)
    print(f"\n===== {yr} =====")
    print(top10[["player_name", "position", "total_WAR", "plays_total"]])

season_top10 = pd.concat(season_top10_rows)
season_top10.to_csv(OUT_TBLS / "sp_skill_war_season_top10.csv", index=False)

print(f"\n✓ Saved season Top-10 → {OUT_TBLS / 'sp_skill_war_season_top10.csv'}")

# -------------------------------------------------------------------
# 4. Global Top-50 across 4 seasons
# -------------------------------------------------------------------
global_top50 = (
    war_pos
    .sort_values("total_WAR", ascending=False)
    .head(50)
)

global_top50.to_csv(OUT_TBLS / "sp_skill_war_global_top50.csv", index=False)

print(f"✓ Saved global Top-50 → {OUT_TBLS / 'sp_skill_war_global_top50.csv'}")

# -------------------------------------------------------------------
# 5. Position summary stats
# -------------------------------------------------------------------
summary = (
    war_pos.groupby("position")
    .agg(
        count=("total_WAR", "count"),
        avg_WAR=("total_WAR", "mean"),
        median_WAR=("total_WAR", "median"),
        IQR_WAR=("total_WAR", lambda x: x.quantile(0.75) - x.quantile(0.25)),
    )
)

print("\n======================================================================")
print("Positional WAR Summary — 2021–2024 Cumulative")
print("======================================================================")
print(summary, "\n")

summary.to_csv(OUT_TBLS / "spwar_position_summary.csv")
print(f"✓ Saved position summary → {OUT_TBLS / 'spwar_position_summary.csv'}")

# -------------------------------------------------------------------
# 6. Boxplot visualization
# -------------------------------------------------------------------
sns.set_style("whitegrid")
plt.figure(figsize=(10,6))
sns.boxplot(data=war_pos, x="position", y="total_WAR", palette="Set2")
plt.title("Skill Position WAR Distribution (2021–2024)")
plt.xlabel("Position (RB / WR / TE)")
plt.ylabel("Player-Season WAR")
plt.tight_layout()

plot_path = OUT_PLOTS / "spwar_position_boxplot.png"
plt.savefig(plot_path, dpi=200)
plt.show()

print(f"✓ Saved boxplot → {plot_path}")

# -------------------------------------------------------------------
# 5. Cumulative 2021–2024 Leaderboard (Total WAR)
# -------------------------------------------------------------------
cumulative_skill = (
    war_pos
    .groupby(["player_id", "player_name", "position"], as_index=False)
    .agg(
        total_WAR=("total_WAR", "sum"),
        seasons_count=("season", "nunique"),
        total_plays=("plays_total", "sum")
    )
)

# Sort + take Top 50
top50_cumulative = (
    cumulative_skill
    .sort_values("total_WAR", ascending=False)
    .head(50)
)

print("\n======================================================================")
print("Top 50 Cumulative Skill Players by Total WAR (2021–2024)")
print("======================================================================\n")
print(
    top50_cumulative[
        ["player_name", "position", "total_WAR", "seasons_count", "total_plays"]
    ].to_string(index=False)
)

# Save to CSV
cumulative_path = OUT_TBLS / "sp_skill_cumulative_top10.csv"
top50_cumulative.to_csv(cumulative_path, index=False)
print(f"\n✓ Saved cumulative Top-50 → {cumulative_path}\n")