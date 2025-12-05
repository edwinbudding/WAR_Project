"""
02_team_aggregation_defense.py
===========================================

TEAM DEFENSE AGGREGATION (2021–2024)

Steps:
    1. Load engineered defender-season features (from Script 01)
    2. Aggregate defender stats → TEAM-LEVEL features per season
    3. Import full PBP (2021–2024) and save locally
    4. Compute DEFENSIVE EPA per team
    5. Merge team features + defensive EPA into regression-ready dataset

Output:
    /outputs/team_defense_agg_2021_2024.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import nfl_data_py as nfl

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("/Users/anokhpalakurthi/Documents/nfl_defensive_war")
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFENDER_SEASON_PATH = OUT_DIR / "defense_engineered_2021_2024.csv"
PBP_PATH = DATA_DIR / "pbp_2021_2024_full.csv"
OUT_TEAM_DEF_PATH = OUT_DIR / "team_defense_agg_2021_2024.csv"


# ============================================================
# TEAM CODE STANDARDIZATION
# ============================================================

def standardize_team_code(code: str) -> str:
    if pd.isna(code):
        return np.nan
    code = str(code).upper().strip()

    TEAM_MAP = {
        "ARZ": "ARI",
        "BLT": "BAL",
        "CLV": "CLE",
        "HST": "HOU",

        # CRITICAL FIXES
        "JAX": "JAC",
        "LA":  "LAR",
        "OAK": "LV",
        "SD":  "LAC",
        "STL": "LAR",
        "WSH": "WAS",
    }

    return TEAM_MAP.get(code, code)


# ============================================================
# STEP 0 — IMPORT FULL PBP (2021–2024) IF MISSING
# ============================================================

def import_and_save_pbp(path: Path):
    if path.exists():
        print(f"PBP already exists:\n  {path}\n")
        return

    print("Downloading full PBP data (2021–2024) via nfl_data_py...")
    pbp = nfl.import_pbp_data([2021, 2022, 2023, 2024])
    print(f"Downloaded {len(pbp):,} rows.")

    path.parent.mkdir(parents=True, exist_ok=True)
    pbp.to_csv(path, index=False)
    print(f"✓ Saved PBP to:\n  {path}\n")


# ============================================================
# 1) LOAD DEFENDER-SEASON FEATURES & AGGREGATE TO TEAM LEVEL
# ============================================================

def load_and_aggregate_defender_features(path: Path) -> pd.DataFrame:
    print("============================================")
    print("  TEAM DEFENSE AGGREGATION (2021–2024)")
    print("============================================\n")

    print(f"Loading engineered defender-season file:\n  {path}\n")
    df = pd.read_csv(path)

    # =========================================================
    # FIX: Script 01 already created a standardized 'team' column.
    # Do NOT overwrite it with team_name (which contains full names).
    # Just validate/ensure the team column exists and is clean.
    # =========================================================
    
    if "team" not in df.columns:
        raise ValueError("Expected 'team' column from Script 01 output!")
    
    # Re-apply standardization to be safe (handles any edge cases)
    df["team"] = df["team"].apply(standardize_team_code)
    df = df.dropna(subset=["team"]).copy()
    
    # DEBUG: Show unique teams to verify
    print(f"Unique teams in defender data: {sorted(df['team'].unique())}\n")

    # Determine columns for aggregation
    # CRITICAL: Exclude groupby keys from aggregation to avoid corrupting them
    exclude_from_agg = {"season", "team", "player_id", "player"}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude_from_agg]
    
    grade_cols = [c for c in numeric_cols if c.startswith("grades_")]
    rate_cols = [
        c for c in numeric_cols
        if c.endswith("_rate") or c in ["catch_rate", "yards_per_reception"]
    ]

    agg_dict = {}
    for col in numeric_cols:
        if col in grade_cols or col in rate_cols:
            agg_dict[col] = "mean"
        else:
            agg_dict[col] = "sum"
    
    print(f"Aggregating {len(agg_dict)} numeric columns (excluded: {exclude_from_agg & set(df.columns)})")

    team_agg = (
        df.groupby(["season", "team"], as_index=False)
          .agg(agg_dict)
    )

    print(f"Aggregated to {len(team_agg)} team-seasons.\n")
    print("Season values (should be 2021-2024):", sorted(team_agg["season"].unique()))
    print("\nSample team-level defensive features (pre-EPA merge):")
    print(team_agg[["season", "team"]].head(10), "\n")

    return team_agg


# ============================================================
# 2) COMPUTE TEAM DEFENSIVE EPA FROM FULL PBP
# ============================================================

def compute_team_def_epa_from_pbp(path: Path) -> pd.DataFrame:
    print(f"Loading full PBP from:\n  {path}\n")

    pbp = pd.read_csv(path, low_memory=False)
    pbp = pbp[pbp["season"].between(2021, 2024)]

    if "game_type" in pbp.columns:
        pbp = pbp[pbp["game_type"] == "REG"].copy()

    pbp = pbp.dropna(subset=["defteam", "epa"]).copy()
    pbp["team"] = pbp["defteam"].apply(standardize_team_code)

    team_def = (
        pbp.groupby(["season", "team"], as_index=False)
           .agg(
               def_plays=("epa", "count"),
               def_epa_total=("epa", "sum"),
           )
    )
    team_def["def_epa_per_play"] = team_def["def_epa_total"] / team_def["def_plays"]

    # DEBUG: Show unique teams to verify
    print(f"Unique teams in PBP data: {sorted(team_def['team'].unique())}\n")
    
    print("Sample defensive EPA rows:")
    print(team_def.head(), "\n")

    return team_def


# ============================================================
# 3) MERGE TEAM FEATURES + TEAM DEF EPA
# ============================================================

def build_team_defense_dataset(def_team: pd.DataFrame, team_def_epa: pd.DataFrame) -> pd.DataFrame:
    
    # DEBUG: Check for merge key alignment before merging
    pff_keys = set(zip(def_team["season"], def_team["team"]))
    pbp_keys = set(zip(team_def_epa["season"], team_def_epa["team"]))
    
    matched = pff_keys & pbp_keys
    pff_only = pff_keys - pbp_keys
    pbp_only = pbp_keys - pff_only
    
    print(f"Merge diagnostics:")
    print(f"  PFF team-seasons: {len(pff_keys)}")
    print(f"  PBP team-seasons: {len(pbp_keys)}")
    print(f"  Matched: {len(matched)}")
    
    if pff_only:
        print(f"  PFF-only (won't merge): {sorted(list(pff_only))[:5]}...")
    if pbp_only:
        print(f"  PBP-only (won't merge): {sorted(list(pbp_only))[:5]}...")
    print()
    
    merged = def_team.merge(team_def_epa, on=["season", "team"], how="inner")

    before = len(merged)
    merged = merged.dropna(subset=["def_epa_per_play"]).copy()
    after = len(merged)

    if after != before:
        print(f"Warning: Dropped {before - after} rows missing defensive EPA.\n")

    print(f"Final merged dataset: {len(merged)} team-seasons.\n")
    print("Merged team-defense dataset (head):")
    print(merged[["season", "team", "def_plays", "def_epa_total", "def_epa_per_play"]].head(), "\n")

    return merged


# ============================================================
# MAIN
# ============================================================

def main():
    import_and_save_pbp(PBP_PATH)
    def_team = load_and_aggregate_defender_features(DEFENDER_SEASON_PATH)
    team_def_epa = compute_team_def_epa_from_pbp(PBP_PATH)
    dataset = build_team_defense_dataset(def_team, team_def_epa)

    if len(dataset) == 0:
        print("ERROR: Merge produced 0 rows! Check team code alignment above.")
        return
    
    dataset.to_csv(OUT_TEAM_DEF_PATH, index=False)
    print(f"✓ Saved team defensive dataset → {OUT_TEAM_DEF_PATH}")
    print("Done.\n")


if __name__ == "__main__":
    main()