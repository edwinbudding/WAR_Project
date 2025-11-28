"""
01_load_and_clean.py
Load PFF offensive line blocking data (2021-2024), filter to T/G/C positions,
standardize team names for nflFastR compatibility, and output cleaned data.

Inputs:
    - /Users/anokhpalakurthi/Documents/nfl_offensive_war/data/external/2021_offense_blocking.csv (or similar naming)
    - /Users/anokhpalakurthi/Documents/nfl_offensive_war/data/external/2022_offense_blocking.csv
    - /Users/anokhpalakurthi/Documents/nfl_offensive_war/data/external/2023_offense_blocking.csv
    - /Users/anokhpalakurthi/Documents/nfl_offensive_war/data/external/2024_offense_blocking.csv

Outputs:
    - /Users/anokhpalakurthi/Documents/nfl_offensive_war/data/processed/oline_individual_2021_2024.csv
"""

import pandas as pd
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths - adjust these to match your actual file names
DATA_DIR = Path("/Users/anokhpalakurthi/Documents/nfl_offensive_war/data")
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"

# PFF CSV file names - matched to your actual file names
PFF_FILES = {
    2021: EXTERNAL_DIR / "2021_offense_blocking.csv",
    2022: EXTERNAL_DIR / "2022_offense_blocking.csv",
    2023: EXTERNAL_DIR / "2023_offense_blocking.csv",
    2024: EXTERNAL_DIR / "2024_offense_blocking.csv",
}
# NOTE: If your files are named differently (e.g., with underscores or different prefix),
# update the paths above accordingly

# Positions to keep (Tackles, Guards, Centers)
VALID_POSITIONS = ["T", "G", "C"]

# Team name mapping: PFF abbreviations -> nflFastR abbreviations
# Verified from 2024_offense_blocking.csv
PFF_TO_NFLFASTR_TEAMS = {
    # AFC North
    "BLT": "BAL",  # Baltimore Ravens (PFF uses BLT)
    "CIN": "CIN",  # Cincinnati Bengals
    "CLV": "CLE",  # Cleveland Browns (PFF uses CLV)
    "PIT": "PIT",  # Pittsburgh Steelers
    
    # AFC East
    "BUF": "BUF",  # Buffalo Bills
    "MIA": "MIA",  # Miami Dolphins
    "NE": "NE",    # New England Patriots
    "NYJ": "NYJ",  # New York Jets
    
    # AFC South
    "HST": "HOU",  # Houston Texans (PFF uses HST)
    "IND": "IND",  # Indianapolis Colts
    "JAX": "JAX",  # Jacksonville Jaguars
    "TEN": "TEN",  # Tennessee Titans
    
    # AFC West
    "DEN": "DEN",  # Denver Broncos
    "KC": "KC",    # Kansas City Chiefs
    "LV": "LV",    # Las Vegas Raiders
    "LAC": "LAC",  # Los Angeles Chargers
    
    # NFC North
    "CHI": "CHI",  # Chicago Bears
    "DET": "DET",  # Detroit Lions
    "GB": "GB",    # Green Bay Packers
    "MIN": "MIN",  # Minnesota Vikings
    
    # NFC East
    "DAL": "DAL",  # Dallas Cowboys
    "NYG": "NYG",  # New York Giants
    "PHI": "PHI",  # Philadelphia Eagles
    "WAS": "WAS",  # Washington Commanders
    
    # NFC South
    "ATL": "ATL",  # Atlanta Falcons
    "CAR": "CAR",  # Carolina Panthers
    "NO": "NO",    # New Orleans Saints
    "TB": "TB",    # Tampa Bay Buccaneers
    
    # NFC West
    "ARZ": "ARI",  # Arizona Cardinals (PFF uses ARZ)
    "LA": "LA",    # Los Angeles Rams
    "SF": "SF",    # San Francisco 49ers
    "SEA": "SEA",  # Seattle Seahawks
}

# Columns to keep from PFF data (adjust based on actual column names)
COLUMNS_TO_KEEP = [
    "player",
    "player_id",
    "position",
    "team_name",  # Will be renamed to 'team' after mapping
    "player_game_count",
    "grades_offense",
    "grades_pass_block",
    "grades_run_block",
    "hits_allowed",
    "hurries_allowed",
    "pressures_allowed",
    "sacks_allowed",
    "penalties",
    "declined_penalties",
    "snap_counts_offense",
    "snap_counts_pass_block",
    "snap_counts_run_block",
    "block_percent",
    "pass_block_percent",
    "pbe",  # Pass Block Efficiency
]


# ============================================================================
# FUNCTIONS
# ============================================================================

def load_single_year(filepath: Path, year: int) -> pd.DataFrame:
    """Load a single year's PFF data and add year column."""
    df = pd.read_csv(filepath)
    df["season"] = year
    return df


def filter_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only Tackles, Guards, and Centers."""
    # Check what position values actually exist
    print(f"Unique positions in data: {df['position'].unique()}")
    
    # Filter - adjust this logic if position codes differ
    mask = df["position"].isin(VALID_POSITIONS)
    filtered = df[mask].copy()
    
    print(f"Filtered from {len(df)} to {len(filtered)} rows (T/G/C only)")
    return filtered


def standardize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """Map PFF team abbreviations to nflFastR format."""
    # Check for any unmapped teams
    unique_teams = df["team_name"].unique()
    unmapped = [t for t in unique_teams if t not in PFF_TO_NFLFASTR_TEAMS]
    
    if unmapped:
        print(f"WARNING: Unmapped team abbreviations found: {unmapped}")
        print("Please add these to PFF_TO_NFLFASTR_TEAMS mapping!")
    
    # Apply mapping
    df["team"] = df["team_name"].map(PFF_TO_NFLFASTR_TEAMS)
    
    return df


def categorize_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create position_group column (T, G, C).
    If data has more granular positions (LT, RT, LG, RG), group them.
    """
    # Map granular positions to groups if needed
    position_mapping = {
        "T": "T",
        "LT": "T",
        "RT": "T",
        "G": "G", 
        "LG": "G",
        "RG": "G",
        "C": "C",
    }
    
    df["position_group"] = df["position"].map(position_mapping)
    
    # Check for any unmapped positions
    if df["position_group"].isna().any():
        unmapped = df[df["position_group"].isna()]["position"].unique()
        print(f"WARNING: Unmapped positions: {unmapped}")
    
    return df


def select_and_order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select relevant columns and order them sensibly."""
    # Start with core ID columns
    id_cols = ["season", "player", "player_id", "team", "position", "position_group"]
    
    # Add counting/volume columns
    volume_cols = ["player_game_count", "snap_counts_offense", "snap_counts_pass_block", 
                   "snap_counts_run_block"]
    
    # Add grade columns
    grade_cols = ["grades_offense", "grades_pass_block", "grades_run_block"]
    
    # Add pressure/sack columns
    pressure_cols = ["pressures_allowed", "sacks_allowed", "hits_allowed", "hurries_allowed"]
    
    # Add efficiency columns
    efficiency_cols = ["block_percent", "pass_block_percent", "pbe"]
    
    # Add penalty columns
    penalty_cols = ["penalties", "declined_penalties"]
    
    # Combine all - only keep columns that exist in dataframe
    all_cols = id_cols + volume_cols + grade_cols + pressure_cols + efficiency_cols + penalty_cols
    available_cols = [c for c in all_cols if c in df.columns]
    
    missing = set(all_cols) - set(available_cols)
    if missing:
        print(f"Note: These columns not found in data: {missing}")
    
    return df[available_cols]


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("Loading and cleaning PFF O-Line data (2021-2024)")
    print("=" * 60)
    
    # Load all years
    dfs = []
    for year, filepath in PFF_FILES.items():
        if filepath.exists():
            print(f"\nLoading {year}...")
            df = load_single_year(filepath, year)
            dfs.append(df)
        else:
            print(f"WARNING: File not found: {filepath}")
    
    if not dfs:
        raise FileNotFoundError("No PFF files found! Check file paths in PFF_FILES config.")
    
    # Combine all years
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows loaded: {len(combined)}")
    
    # Filter to T/G/C only
    combined = filter_positions(combined)
    
    # Standardize team names
    combined = standardize_team_names(combined)
    
    # Categorize positions
    combined = categorize_position(combined)
    
    # Select and order columns
    combined = select_and_order_columns(combined)
    
    # Summary stats
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total players: {combined['player'].nunique()}")
    print(f"Total player-seasons: {len(combined)}")
    print(f"\nBy season:")
    print(combined.groupby("season").size())
    print(f"\nBy position group:")
    print(combined.groupby("position_group").size())
    
    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "oline_individual_2021_2024.csv"
    combined.to_csv(output_path, index=False)
    print(f"\nSaved cleaned data to: {output_path}")
    
    return combined


if __name__ == "__main__":
    df = main()