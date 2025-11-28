"""
02_team_aggregation.py
Aggregate individual O-line stats to team-season level, extract team offensive EPA
from play-by-play data, and merge them for regression analysis.

Inputs:
    - /Users/anokhpalakurthi/Documents/nfl_offensive_war/data/processed/oline_individual_2021_2024.csv (from 01_load_and_clean.py)
    - /Users/anokhpalakurthi/Documents/nfl_offensive_war/data/raw/pbp_2021_2024.csv (play-by-play data)

Outputs:
    - /Users/anokhpalakurthi/Documents/nfl_offensive_war/data/processed/oline_team_2021_2024.csv (team-level O-line + offensive EPA)
    - /Users/anokhpalakurthi/Documents/nfl_offensive_war/data/processed/team_offensive_epa_2021_2024.csv (just team EPA for reference)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("/Users/anokhpalakurthi/Documents/nfl_offensive_war/data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Input files
OLINE_INDIVIDUAL_FILE = PROCESSED_DIR / "oline_individual_2021_2024.csv"
PBP_FILE = RAW_DIR / "pbp_2021_2024.csv"

# Output files
TEAM_EPA_FILE = PROCESSED_DIR / "team_offensive_epa_2021_2024.csv"
OLINE_TEAM_FILE = PROCESSED_DIR / "oline_team_2021_2024.csv"


# ============================================================================
# PART 1: EXTRACT TEAM OFFENSIVE EPA FROM PLAY-BY-PLAY
# ============================================================================

def load_pbp_data(filepath: Path) -> pd.DataFrame:
    """Load play-by-play data with only necessary columns for efficiency."""
    print(f"Loading play-by-play data from {filepath}...")
    
    # Only load columns we need
    cols_needed = [
        "season", "game_id", "posteam", "defteam", 
        "epa", "play_type", "down", 
        "rush", "pass", "special"
    ]
    
    # Load with specified columns if they exist
    try:
        df = pd.read_csv(filepath, usecols=cols_needed, low_memory=False)
    except ValueError:
        # If some columns don't exist, load all and filter
        print("Some columns not found, loading full dataset...")
        df = pd.read_csv(filepath, low_memory=False)
        available = [c for c in cols_needed if c in df.columns]
        df = df[available]
    
    print(f"Loaded {len(df):,} plays")
    return df


def filter_offensive_plays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to regular offensive plays (rush/pass), excluding special teams,
    spikes, kneeldowns, etc.
    """
    # Filter criteria matching your QB WAR methodology
    mask = (
        (df["posteam"].notna()) &  # Has a team on offense
        ((df["rush"] == 1) | (df["pass"] == 1)) &  # Rush or pass play
        (df["special"] != 1) &  # Not special teams
        (df["epa"].notna())  # Has EPA value
    )
    
    filtered = df[mask].copy()
    print(f"Filtered to {len(filtered):,} offensive plays")
    return filtered


def calculate_team_offensive_epa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate total offensive EPA per team-season.
    Also calculates rushing EPA and passing EPA separately.
    """
    team_epa = df.groupby(["season", "posteam"]).agg(
        total_offensive_epa=("epa", "sum"),
        total_offensive_plays=("epa", "count"),
        rushing_epa=("epa", lambda x: x[df.loc[x.index, "rush"] == 1].sum()),
        passing_epa=("epa", lambda x: x[df.loc[x.index, "pass"] == 1].sum()),
        rushing_plays=("rush", "sum"),
        passing_plays=("pass", "sum"),
    ).reset_index()
    
    # Rename posteam to team for consistency
    team_epa = team_epa.rename(columns={"posteam": "team"})
    
    # Calculate EPA per play
    team_epa["offensive_epa_per_play"] = (
        team_epa["total_offensive_epa"] / team_epa["total_offensive_plays"]
    )
    team_epa["rushing_epa_per_play"] = (
        team_epa["rushing_epa"] / team_epa["rushing_plays"]
    )
    team_epa["passing_epa_per_play"] = (
        team_epa["passing_epa"] / team_epa["passing_plays"]
    )
    
    print(f"\nTeam offensive EPA calculated for {len(team_epa)} team-seasons")
    return team_epa


# ============================================================================
# PART 2: AGGREGATE O-LINE STATS TO TEAM LEVEL
# ============================================================================

def load_oline_individual(filepath: Path) -> pd.DataFrame:
    """Load cleaned individual O-line data."""
    print(f"\nLoading individual O-line data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} player-seasons")
    return df


def aggregate_oline_to_team(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate individual O-line stats to team-season level.
    Uses snap-weighted averages for grades, sums for counting stats.
    """
    
    # First, calculate snap-weighted grades per team-season
    def snap_weighted_mean(group, value_col, weight_col):
        """Calculate snap-weighted average."""
        weights = group[weight_col]
        values = group[value_col]
        if weights.sum() == 0:
            return np.nan
        return np.average(values, weights=weights)
    
    team_stats = []
    
    for (season, team), group in df.groupby(["season", "team"]):
        stats = {
            "season": season,
            "team": team,
            "n_linemen": len(group),
            
            # Total snaps
            "total_pass_block_snaps": group["snap_counts_pass_block"].sum(),
            "total_run_block_snaps": group["snap_counts_run_block"].sum(),
            "total_offense_snaps": group["snap_counts_offense"].sum(),
            
            # Snap-weighted grades (weighted by relevant snap type)
            "team_pass_block_grade": snap_weighted_mean(
                group, "grades_pass_block", "snap_counts_pass_block"
            ),
            "team_run_block_grade": snap_weighted_mean(
                group, "grades_run_block", "snap_counts_run_block"
            ),
            "team_offense_grade": snap_weighted_mean(
                group, "grades_offense", "snap_counts_offense"
            ),
            
            # Sum of pressure stats
            "total_pressures_allowed": group["pressures_allowed"].sum(),
            "total_sacks_allowed": group["sacks_allowed"].sum(),
            "total_hits_allowed": group["hits_allowed"].sum(),
            "total_hurries_allowed": group["hurries_allowed"].sum(),
            
            # Penalties
            "total_penalties": group["penalties"].sum(),
        }
        
        # Calculate rates
        if stats["total_pass_block_snaps"] > 0:
            stats["pressure_rate"] = (
                stats["total_pressures_allowed"] / stats["total_pass_block_snaps"]
            )
            stats["sack_rate"] = (
                stats["total_sacks_allowed"] / stats["total_pass_block_snaps"]
            )
        else:
            stats["pressure_rate"] = np.nan
            stats["sack_rate"] = np.nan
        
        team_stats.append(stats)
    
    team_df = pd.DataFrame(team_stats)
    print(f"\nAggregated to {len(team_df)} team-seasons")
    
    return team_df


def aggregate_oline_by_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate O-line stats by position group (T, G, C) per team-season.
    Returns a wide dataframe with columns like T_pass_block_grade, G_pass_block_grade, etc.
    """
    
    def snap_weighted_mean(group, value_col, weight_col):
        weights = group[weight_col]
        values = group[value_col]
        if weights.sum() == 0:
            return np.nan
        return np.average(values, weights=weights)
    
    position_stats = []
    
    for (season, team, pos_group), group in df.groupby(["season", "team", "position_group"]):
        stats = {
            "season": season,
            "team": team,
            "position_group": pos_group,
            "n_players": len(group),
            "pass_block_grade": snap_weighted_mean(
                group, "grades_pass_block", "snap_counts_pass_block"
            ),
            "run_block_grade": snap_weighted_mean(
                group, "grades_run_block", "snap_counts_run_block"
            ),
            "pass_block_snaps": group["snap_counts_pass_block"].sum(),
            "run_block_snaps": group["snap_counts_run_block"].sum(),
            "pressures_allowed": group["pressures_allowed"].sum(),
            "sacks_allowed": group["sacks_allowed"].sum(),
        }
        position_stats.append(stats)
    
    pos_df = pd.DataFrame(position_stats)
    
    # Pivot to wide format
    wide_df = pos_df.pivot(
        index=["season", "team"],
        columns="position_group",
        values=["pass_block_grade", "run_block_grade", "pass_block_snaps", 
                "run_block_snaps", "pressures_allowed", "sacks_allowed", "n_players"]
    )
    
    # Flatten column names
    wide_df.columns = [f"{pos}_{stat}" for stat, pos in wide_df.columns]
    wide_df = wide_df.reset_index()
    
    print(f"\nPosition-level aggregation complete: {wide_df.shape}")
    
    return wide_df


# ============================================================================
# PART 3: MERGE O-LINE AND EPA DATA
# ============================================================================

def merge_oline_and_epa(oline_team: pd.DataFrame, team_epa: pd.DataFrame) -> pd.DataFrame:
    """Merge team-level O-line stats with team offensive EPA."""
    
    merged = oline_team.merge(
        team_epa,
        on=["season", "team"],
        how="inner"
    )
    
    print(f"\nMerged dataset: {len(merged)} team-seasons")
    
    # Check for any teams that didn't match
    oline_teams = set(zip(oline_team["season"], oline_team["team"]))
    epa_teams = set(zip(team_epa["season"], team_epa["team"]))
    
    missing_from_epa = oline_teams - epa_teams
    missing_from_oline = epa_teams - oline_teams
    
    if missing_from_epa:
        print(f"WARNING: Teams in O-line data but not EPA: {missing_from_epa}")
    if missing_from_oline:
        print(f"Note: Teams in EPA but not O-line data: {len(missing_from_oline)}")
    
    return merged

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("Aggregating O-Line data and extracting Team Offensive EPA")
    print("=" * 60)
    
    # Part 1: Extract team offensive EPA from play-by-play
    print("\n--- PART 1: Team Offensive EPA ---")
    pbp = load_pbp_data(PBP_FILE)
    pbp = filter_offensive_plays(pbp)
    team_epa = calculate_team_offensive_epa(pbp)
    
    # Save team EPA separately for reference
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    team_epa.to_csv(TEAM_EPA_FILE, index=False)
    print(f"Saved team EPA to: {TEAM_EPA_FILE}")
    
    # Part 2: Aggregate O-line stats to team level
    print("\n--- PART 2: O-Line Team Aggregation ---")
    oline_ind = load_oline_individual(OLINE_INDIVIDUAL_FILE)
    oline_team = aggregate_oline_to_team(oline_ind)
    
    # Also get position-level breakdown
    oline_by_pos = aggregate_oline_by_position(oline_ind)
    
    # Merge position-level stats into team stats
    oline_team = oline_team.merge(oline_by_pos, on=["season", "team"], how="left")
    
    # Part 3: Merge O-line and EPA
    print("\n--- PART 3: Merging Data ---")
    combined = merge_oline_and_epa(oline_team, team_epa)
    
    # Save final merged dataset
    combined.to_csv(OLINE_TEAM_FILE, index=False)
    print(f"Saved merged O-line + EPA data to: {OLINE_TEAM_FILE}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Team-seasons in final dataset: {len(combined)}")
    print(f"\nColumns: {list(combined.columns)}")
    print(f"\nSample correlations with total_offensive_epa:")
    for col in ["team_pass_block_grade", "team_run_block_grade", "pressure_rate", "sack_rate"]:
        if col in combined.columns:
            corr = combined[col].corr(combined["total_offensive_epa"])
            print(f"  {col}: {corr:.3f}")
    
    return combined

if __name__ == "__main__":
    df = main()