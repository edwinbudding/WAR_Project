"""
======================================================================
SPWAR 02 — COMPUTING SKILL POSITION WAR (2021–2024)
======================================================================

Inputs:
- data/clean/skill_pbp_2021_2024.csv   (from SPWAR 01)

Method:
- Use *QB EPA-per-win* constant from your Part 4 work:
      QB_EPA_PER_WIN = 99.5
- Exclude any player who ever appears as a QB (passer_player_id)
- Define replacement levels:
      25th percentile EPA/play among players with ≥ 25 plays
      (separately for rush and receiving)
- Compute:
      rush_WAR, rec_WAR, total_WAR, total_WAR_per_300

Output:
- outputs/tables/sp_skill_war_2021_2024.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------------------------------------------------
# Paths / constants
# -------------------------------------------------------------------
BASE = Path("/Users/anokhpalakurthi/Documents/nfl_offensive_war")
CLEAN = BASE / "data" / "clean"
OUT   = BASE / "outputs" / "tables"

PBP_SKILL_PATH = CLEAN / "skill_pbp_2021_2024.csv"
OUT.mkdir(parents=True, exist_ok=True)

QB_EPA_PER_WIN = 99.5       # from your QB WAR work
MIN_PLAYS      = 25         # min total touches for inclusion
REPL_PCT       = 25         # replacement percentile

print("\n======================================================================")
print("SPWAR 03 — COMPUTING SKILL POSITION WAR (2021–2024)")
print("======================================================================\n")

# -------------------------------------------------------------------
# 1. Load clean skill PBP
# -------------------------------------------------------------------
pbp = pd.read_csv(PBP_SKILL_PATH, low_memory=False)

required_cols = [
    "season",
    "player_id",
    "player_name",
    "epa",
    "rush_attempt",
    "pass_attempt",
    "complete_pass",
    "yards_after_catch",
    "passer_player_id",
]
missing = [c for c in required_cols if c not in pbp.columns]
if missing:
    raise ValueError(f"Missing columns in skill PBP: {missing}")

pbp["season"] = pbp["season"].astype(int)
pbp["epa"] = pd.to_numeric(pbp["epa"], errors="coerce").fillna(0.0)

# -------------------------------------------------------------------
# 2. Attach player positions (nfl_data_py) + keep only RB/WR/TE
# -------------------------------------------------------------------
import nfl_data_py as nfl

players = nfl.import_players()
players = players.rename(columns={"gsis_id": "player_id"})

pos_map = players[["player_id", "position"]].drop_duplicates()

pbp = pbp.merge(pos_map, on="player_id", how="left")

before_rows = len(pbp)
skill_pbp = pbp[pbp["position"].isin(["RB", "WR", "TE"])].copy()
after_rows = len(skill_pbp)

print(f"Total plays in clean skill PBP: {before_rows:,}")
print(f"Kept RB/WR/TE plays only:        {after_rows:,}")
print(f"Dropped {before_rows - after_rows:,} non-skill plays\n")

# -------------------------------------------------------------------
# 3. Tag rush vs reception plays
# -------------------------------------------------------------------
skill_pbp["is_rush"] = skill_pbp["rush_attempt"] == 1
skill_pbp["is_rec"]  = (skill_pbp["pass_attempt"] == 1) & (skill_pbp["complete_pass"] == 1)

# -------------------------------------------------------------------
# 4. Aggregate rush & receiving EPA by player-season
# -------------------------------------------------------------------
rush = (
    skill_pbp[skill_pbp["is_rush"]]
    .groupby(["season", "player_id", "player_name"], as_index=False)
    .agg(
        plays_rush=("epa", "count"),
        rush_epa=("epa", "sum"),
        rush_epa_per_play=("epa", "mean"),
    )
)

rec = (
    skill_pbp[skill_pbp["is_rec"]]
    .groupby(["season", "player_id", "player_name"], as_index=False)
    .agg(
        plays_rec=("epa", "count"),
        rec_epa=("epa", "sum"),
        yac_total=("yards_after_catch", "sum"),
        rec_epa_per_play=("epa", "mean"),
    )
)

print("Sample rush aggregates:\n", rush.head(), "\n")
print("Sample rec aggregates:\n", rec.head(), "\n")

# -------------------------------------------------------------------
# 5. Define replacement levels (25th pct EPA/play, ≥ 25 plays)
# -------------------------------------------------------------------
print("-" * 66)
print(f"Defining Replacement Levels ({REPL_PCT}th percentile EPA/play, ≥ {MIN_PLAYS} plays)")
print("-" * 66)

rush_valid = rush[rush["plays_rush"] >= MIN_PLAYS]
rec_valid  = rec[rec["plays_rec"]  >= MIN_PLAYS]

rush_repl = np.percentile(rush_valid["rush_epa_per_play"], REPL_PCT) if len(rush_valid) else 0.0
rec_repl  = np.percentile(rec_valid["rec_epa_per_play"],  REPL_PCT) if len(rec_valid)  else 0.0

print(f"✓ Rush replacement EPA/play: {rush_repl:.4f}")
print(f"✓ Rec  replacement EPA/play: {rec_repl:.4f}\n")

# -------------------------------------------------------------------
# 6. Merge rush + rec and apply volume filter
# -------------------------------------------------------------------
merged = pd.merge(
    rush,
    rec,
    on=["season", "player_id", "player_name"],
    how="outer",
    suffixes=("_rush", "_rec"),
).fillna(
    {
        "plays_rush": 0,
        "rush_epa": 0.0,
        "rush_epa_per_play": 0.0,
        "plays_rec": 0,
        "rec_epa": 0.0,
        "yac_total": 0.0,
        "rec_epa_per_play": 0.0,
    }
)

merged["plays_total"] = merged["plays_rush"] + merged["plays_rec"]
merged = merged[merged["plays_total"] >= MIN_PLAYS].copy()

print(f"Players with ≥ {MIN_PLAYS} total non-QB skill plays: {len(merged)}\n")

# -------------------------------------------------------------------
# 7. Compute WAR using QB_EPA_PER_WIN as common scaler
# -------------------------------------------------------------------
merged["rush_WAR"] = (
    (merged["rush_epa_per_play"] - rush_repl) * merged["plays_rush"]
) / QB_EPA_PER_WIN

merged["rec_WAR"] = (
    (merged["rec_epa_per_play"] - rec_repl) * merged["plays_rec"]
) / QB_EPA_PER_WIN

merged["total_WAR"] = merged["rush_WAR"] + merged["rec_WAR"]
merged["total_WAR_per_300"] = merged["total_WAR"] * (300.0 / merged["plays_total"])

# -------------------------------------------------------------------
# 8. Save WAR table
# -------------------------------------------------------------------
out_path = OUT / "sp_skill_war_2021_2024.csv"
merged.to_csv(out_path, index=False)

print(f"✓ Saved Skill WAR table → {out_path}\n")

# Tiny sanity peek (not a full leaderboard)
print("Sample high-WAR rows:\n")
print(
    merged.sort_values("total_WAR", ascending=False)
    .head(10)[["season", "player_name", "plays_total", "total_WAR", "total_WAR_per_300"]]
)

print("\nDone.\n")

print("\n======================================================================")
print("Justin Jefferson — SPWAR Presence Check")
print("======================================================================")

jj = merged[merged["player_name"].str.contains("Jefferson", case=False, na=False)]

if jj.empty:
    print("⚠️ Still missing — investigate upstream data.")
else:
    print(
        jj[["season", "player_name", "plays_total", "total_WAR"]]
        .sort_values("season")
        .to_string(index=False)
    )