"""
======================================================================
SPWAR 01 — CLEAN SKILL-POSITION PBP (2021–2024)
Removes QB-designed rush, scrambles, and ensures real RB/WR/TE touches
======================================================================
"""

import pandas as pd
from pathlib import Path

BASE = Path("/Users/anokhpalakurthi/Documents/nfl_offensive_war")
RAW = BASE / "data/raw"
CLEAN = BASE / "data/clean"

PBP_PATH = RAW / "pbp_2021_2024.csv"

print("=*70")
print("SPWAR 01 — BUILDING CLEAN SKILL-POSITION PBP (2021–2024)")
print("=*70")

pbp = pd.read_csv(PBP_PATH, low_memory=False)

# Skill touches = rushes OR receptions
skill = pbp[
    (pbp["rush_attempt"] == 1) |
    ((pbp["pass_attempt"] == 1) & (pbp["complete_pass"] == 1))
].copy()

# Remove QB rush touches (scrambles + designed rushes)
skill = skill[
    ~((skill["rush_attempt"] == 1) &
      (skill["passer_player_id"] == skill["rusher_player_id"]))
]

# Assign primary player field
skill["player_id"] = skill.apply(
    lambda r: r["rusher_player_id"]
    if r["rush_attempt"] == 1 else r["receiver_player_id"],
    axis=1
)

skill["player_name"] = skill.apply(
    lambda r: r["rusher_player_name"]
    if r["rush_attempt"] == 1 else r["receiver_player_name"],
    axis=1
)

# Drop plays with no actual player
skill = skill[skill["player_id"].notna()]

skill["season"] = skill["season"].astype(int)
skill["epa"] = skill["epa"].fillna(0)

print(f"Total skill-filtered plays: {len(skill):,}")
print(f"Seasons present: {sorted(skill['season'].unique())}")

# Save
CLEAN.mkdir(exist_ok=True, parents=True)
out_path = CLEAN / "skill_pbp_2021_2024.csv"
skill.to_csv(out_path, index=False)

print(f"\n✓ Saved clean skill PBP → {out_path}")