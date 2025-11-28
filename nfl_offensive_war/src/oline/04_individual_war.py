"""
04_individual_war.py
Apply regression coefficients to individual O-linemen to estimate their EPA
contribution and convert to WAR.

Inputs:
    - /Users/anokhpalakurthi/Documents/nfl_offensive_war/data/processed/oline_individual_2021_2024.csv (cleaned individual data)
    - /Users/anokhpalakurthi/Documents/nfl_offensive_war/data/processed/oline_team_2021_2024.csv (for replacement level calculation)
    - Regression coefficients from 03_regression_model.py (hardcoded after running)

Outputs:
    - /Users/anokhpalakurthi/Documents/nfl_offensive_war/data/processed/oline_war_2021_2024.csv (individual WAR values)
    - /Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs/tables/oline_war_leaders.csv (top players by WAR)
    - /Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs/figures/oline_war_by_position.png
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("/Users/anokhpalakurthi/Documents/nfl_offensive_war/data")
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = Path("/Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs")
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Input files
OLINE_INDIVIDUAL_FILE = PROCESSED_DIR / "oline_individual_2021_2024.csv"
OLINE_TEAM_FILE = PROCESSED_DIR / "oline_team_2021_2024.csv"

# Output files
OLINE_WAR_FILE = PROCESSED_DIR / "oline_war_2021_2024.csv"

# Ensure output directories exist
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# REGRESSION COEFFICIENTS (UPDATE AFTER RUNNING 03_regression_model.py)
# ============================================================================

# These values should be updated based on your actual regression results!
# Placeholder values below - replace with actual coefficients

PASS_BLOCK_COEF = 5.30   # EPA gained per 1-point increase in team pass block grade
RUN_BLOCK_COEF = 5.09    # EPA gained per 1-point increase in team run block grade

# From your QB WAR methodology: 
# Total offensive EPA per win = ~218.83 (before position adjustments)
# We need to determine what share of this goes to O-line

# Option A: Use remaining share after QB + skill positions
# QB share: 67.4% of plays, so ~67% of offensive value
# Skill positions: receive EPA directly on their touches
# O-line: enables the remaining value

# For now, let's estimate O-line contributes ~15-20% of offensive wins
# This is a parameter you can adjust based on your methodology preferences
OLINE_SHARE_OF_OFFENSE = 0.14  # 14% of offensive value attributed to O-line

# EPA per win for O-line specifically
TOTAL_EPA_PER_WIN = 218.83  # From your QB WAR work
OLINE_EPA_PER_WIN = TOTAL_EPA_PER_WIN * OLINE_SHARE_OF_OFFENSE  # ~30.6 EPA per O-line win
# Replacement level: 25th percentile (matching your QB/skill methodology)
REPLACEMENT_PERCENTILE = 25

# Minimum snaps to qualify (seasonal)
MIN_SNAPS = 50


# ============================================================================
# FUNCTIONS
# ============================================================================

def load_data():
    """Load individual and team-level O-line data."""
    print("Loading data...")
    individual = pd.read_csv(OLINE_INDIVIDUAL_FILE)
    team = pd.read_csv(OLINE_TEAM_FILE)
    print(f"  Individual: {len(individual)} player-seasons")
    print(f"  Team: {len(team)} team-seasons")
    return individual, team


def calculate_replacement_level(df: pd.DataFrame) -> dict:
    """
    Calculate replacement-level grades by position group.
    Uses 25th percentile of qualified players (min snaps threshold).
    """
    print(f"\nCalculating replacement level ({REPLACEMENT_PERCENTILE}th percentile)...")
    print(f"Minimum snaps threshold: {MIN_SNAPS}")
    
    # Filter to qualified players
    qualified = df[df["snap_counts_offense"] >= MIN_SNAPS].copy()
    print(f"Qualified players: {len(qualified)} of {len(df)}")
    
    replacement_levels = {}
    
    for pos in ["T", "G", "C"]:
        pos_data = qualified[qualified["position_group"] == pos]
        
        if len(pos_data) == 0:
            print(f"  WARNING: No qualified {pos} players found!")
            continue
        
        replacement_levels[pos] = {
            "pass_block_grade": np.percentile(pos_data["grades_pass_block"], REPLACEMENT_PERCENTILE),
            "run_block_grade": np.percentile(pos_data["grades_run_block"], REPLACEMENT_PERCENTILE),
            "n_qualified": len(pos_data),
        }
        
        print(f"  {pos}: Pass Block = {replacement_levels[pos]['pass_block_grade']:.1f}, "
              f"Run Block = {replacement_levels[pos]['run_block_grade']:.1f} "
              f"(n={replacement_levels[pos]['n_qualified']})")
    
    return replacement_levels


def calculate_grade_above_replacement(df: pd.DataFrame, replacement_levels: dict) -> pd.DataFrame:
    """
    Calculate each player's grades above replacement level for their position.
    """
    df = df.copy()
    
    df["replacement_pass_block"] = df["position_group"].map(
        lambda x: replacement_levels.get(x, {}).get("pass_block_grade", np.nan)
    )
    df["replacement_run_block"] = df["position_group"].map(
        lambda x: replacement_levels.get(x, {}).get("run_block_grade", np.nan)
    )
    
    # Grade above replacement (can be negative for below-replacement players)
    df["pass_block_above_replacement"] = df["grades_pass_block"] - df["replacement_pass_block"]
    df["run_block_above_replacement"] = df["grades_run_block"] - df["replacement_run_block"]
    
    return df


def calculate_snap_share(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate each player's share of their team's O-line snaps.
    This is used to allocate team-level value to individuals.
    """
    df = df.copy()
    
    # Calculate team totals per season
    team_totals = df.groupby(["season", "team"]).agg({
        "snap_counts_pass_block": "sum",
        "snap_counts_run_block": "sum",
        "snap_counts_offense": "sum",
    }).reset_index()
    
    team_totals = team_totals.rename(columns={
        "snap_counts_pass_block": "team_pass_block_snaps",
        "snap_counts_run_block": "team_run_block_snaps",
        "snap_counts_offense": "team_offense_snaps",
    })
    
    # Merge back
    df = df.merge(team_totals, on=["season", "team"], how="left")
    
    # Calculate shares
    df["pass_block_snap_share"] = df["snap_counts_pass_block"] / df["team_pass_block_snaps"]
    df["run_block_snap_share"] = df["snap_counts_run_block"] / df["team_run_block_snaps"]
    df["offense_snap_share"] = df["snap_counts_offense"] / df["team_offense_snaps"]
    
    return df


def calculate_individual_epa_contribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate each player's EPA contribution using regression coefficients.
    
    Logic:
    - Team pass block grade predicts X EPA (from regression)
    - Player's contribution = their share of that team-level effect
    - Weighted by their snap share and grade above replacement
    """
    df = df.copy()
    
    # Method: Grade Above Replacement × Coefficient × Snap Share
    # This estimates how much EPA this player added vs. a replacement-level player
    
    # Pass blocking contribution
    df["pass_block_epa_contribution"] = (
        df["pass_block_above_replacement"] * 
        PASS_BLOCK_COEF * 
        df["pass_block_snap_share"]
    )
    
    # Run blocking contribution
    df["run_block_epa_contribution"] = (
        df["run_block_above_replacement"] * 
        RUN_BLOCK_COEF * 
        df["run_block_snap_share"]
    )
    
    # Total EPA contribution
    df["total_epa_contribution"] = (
        df["pass_block_epa_contribution"] + 
        df["run_block_epa_contribution"]
    )
    
    return df


def calculate_war(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert EPA contribution to WAR using O-line EPA per win constant.
    """
    df = df.copy()
    
    # WAR = EPA Above Replacement / EPA per Win
    df["war"] = df["total_epa_contribution"] / OLINE_EPA_PER_WIN
    
    # Also calculate component WAR
    df["pass_block_war"] = df["pass_block_epa_contribution"] / OLINE_EPA_PER_WIN
    df["run_block_war"] = df["run_block_epa_contribution"] / OLINE_EPA_PER_WIN
    
    return df


def create_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics by position and season."""
    
    summary = df.groupby(["season", "position_group"]).agg({
        "war": ["mean", "median", "std", "min", "max"],
        "player": "count"
    }).round(2)
    
    summary.columns = ["_".join(col).strip() for col in summary.columns]
    summary = summary.rename(columns={"player_count": "n_players"})
    
    return summary.reset_index()

def create_position_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate overall summary statistics by position group across all seasons."""
    
    summary = df.groupby("position_group").agg({
        "war": ["mean", "median", "std", "min", "max"],
        "player": "nunique"
    }).round(3)
    
    summary.columns = ["avg_war", "median_war", "std_war", "min_war", "max_war", "n_players"]
    summary = summary.reset_index()
    
    # Reorder to T, G, C
    summary["position_group"] = pd.Categorical(summary["position_group"], 
                                                categories=["T", "G", "C"], 
                                                ordered=True)
    summary = summary.sort_values("position_group")
    
    return summary


def plot_war_by_position(df: pd.DataFrame, save_path: Path):
    """Create box plot of WAR distribution by position."""
    
    # Filter to qualified players
    qualified = df[df["snap_counts_offense"] >= MIN_SNAPS].copy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(data=qualified, x="position_group", y="war", 
                order=["T", "G", "C"], ax=ax)
    
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Replacement Level")
    ax.set_xlabel("Position")
    ax.set_ylabel("WAR")
    ax.set_title("O-Line WAR Distribution by Position (2021-2024)")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved WAR by position plot to: {save_path}")


def plot_war_vs_grade(df: pd.DataFrame, save_path: Path):
    """Scatter plot of WAR vs overall offensive grade."""
    
    qualified = df[df["snap_counts_offense"] >= MIN_SNAPS].copy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for pos, color in [("T", "blue"), ("G", "green"), ("C", "orange")]:
        pos_data = qualified[qualified["position_group"] == pos]
        ax.scatter(pos_data["grades_offense"], pos_data["war"], 
                   alpha=0.5, label=pos, c=color)
    
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("PFF Offensive Grade")
    ax.set_ylabel("WAR")
    ax.set_title("O-Line WAR vs PFF Grade (2021-2024)")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved WAR vs grade plot to: {save_path}")


def print_war_leaders(df: pd.DataFrame, n: int = 20):
    """Print top players by WAR."""
    
    # Cumulative WAR across all seasons
    cumulative = df.groupby(["player", "position_group"]).agg({
        "war": "sum",
        "snap_counts_offense": "sum",
        "season": "count"
    }).reset_index()
    
    cumulative = cumulative.rename(columns={"season": "seasons_played"})
    cumulative = cumulative.sort_values("war", ascending=False)
    
    print("\n" + "=" * 60)
    print(f"TOP {n} O-LINEMEN BY CUMULATIVE WAR (2021-2024)")
    print("=" * 60)
    
    for i, row in cumulative.head(n).iterrows():
        print(f"{row['player']:25} ({row['position_group']}) - "
              f"WAR: {row['war']:.2f} ({row['seasons_played']} seasons)")
    
    return cumulative

# ============================================================================
# DIVISOR CALIBRATION (Add this section before main())
# ============================================================================

def load_team_wins() -> pd.DataFrame:
    """
    Load team wins data. 
    You may need to adjust this path to wherever your team results live.
    """
    # Option 1: If you have team_results for multiple years
    team_wins_file = DATA_DIR / "processed" / "team_results_2021_2024.csv"
    
    # Option 2: If you only have 2024, you can build from pbp
    if not team_wins_file.exists():
        print("Building team wins from play-by-play data...")
        pbp = pd.read_csv(DATA_DIR / "raw" / "pbp_2021_2024.csv", 
                          usecols=["season", "game_id", "home_team", "away_team", 
                                   "home_score", "away_score"])
        
        # Get unique games
        games = pbp.groupby("game_id").first().reset_index()
        
        # Calculate wins for each team
        home_wins = games[games["home_score"] > games["away_score"]].groupby(
            ["season", "home_team"]).size().reset_index(name="wins")
        home_wins = home_wins.rename(columns={"home_team": "team"})
        
        away_wins = games[games["away_score"] > games["home_score"]].groupby(
            ["season", "away_team"]).size().reset_index(name="wins")
        away_wins = away_wins.rename(columns={"away_team": "team"})
        
        # Combine
        team_wins = pd.concat([home_wins, away_wins]).groupby(
            ["season", "team"])["wins"].sum().reset_index()
        
        return team_wins
    
    return pd.read_csv(team_wins_file)


def calculate_war_with_divisor(df: pd.DataFrame, replacement_levels: dict, 
                                divisor: float) -> pd.DataFrame:
    """Calculate WAR using a specific divisor for team-level adjustment."""
    df = df.copy()
    
    # Grade above replacement (already calculated, but recalc for safety)
    df["replacement_pass_block"] = df["position_group"].map(
        lambda x: replacement_levels.get(x, {}).get("pass_block_grade", np.nan)
    )
    df["replacement_run_block"] = df["position_group"].map(
        lambda x: replacement_levels.get(x, {}).get("run_block_grade", np.nan)
    )
    df["pass_block_above_replacement"] = df["grades_pass_block"] - df["replacement_pass_block"]
    df["run_block_above_replacement"] = df["grades_run_block"] - df["replacement_run_block"]
    
    # EPA contribution WITH DIVISOR
    df["pass_block_epa_contribution"] = (
        (df["pass_block_above_replacement"] / divisor) * 
        PASS_BLOCK_COEF * 
        df["pass_block_snap_share"]
    )
    df["run_block_epa_contribution"] = (
        (df["run_block_above_replacement"] / divisor) * 
        RUN_BLOCK_COEF * 
        df["run_block_snap_share"]
    )
    df["total_epa_contribution"] = (
        df["pass_block_epa_contribution"] + df["run_block_epa_contribution"]
    )
    
    # WAR
    df["war"] = df["total_epa_contribution"] / OLINE_EPA_PER_WIN
    
    return df


def find_optimal_divisor(individual: pd.DataFrame, replacement_levels: dict,
                         divisor_range: list = None) -> dict:
    """
    Test different divisors and find the one that produces the best
    correlation between team O-line WAR and team wins.
    """
    if divisor_range is None:
        divisor_range = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    # Load team wins
    team_wins = load_team_wins()
    
    results = []
    
    for divisor in divisor_range:
        # Calculate WAR with this divisor
        df_war = calculate_war_with_divisor(individual.copy(), replacement_levels, divisor)
        
        # Aggregate to team-season level (sum of all linemen)
        team_war = df_war.groupby(["season", "team"]).agg({
            "war": "sum"
        }).reset_index()
        team_war = team_war.rename(columns={"war": "team_oline_war"})
        
        # Merge with wins
        merged = team_war.merge(team_wins, on=["season", "team"], how="inner")
        
        if len(merged) == 0:
            print(f"  Divisor {divisor}: No matching team-seasons found")
            continue
        
        # Calculate correlation
        corr = merged["team_oline_war"].corr(merged["wins"])
        
        # Calculate mean and max individual WAR for reference
        mean_war = df_war["war"].mean()
        max_war = df_war["war"].max()
        
        # Position-specific peaks
        t_max = df_war[df_war["position_group"] == "T"]["war"].max()
        g_max = df_war[df_war["position_group"] == "G"]["war"].max()
        c_max = df_war[df_war["position_group"] == "C"]["war"].max()
        
        results.append({
            "divisor": divisor,
            "correlation": corr,
            "r_squared": corr ** 2,
            "mean_war": mean_war,
            "max_war": max_war,
            "T_peak": t_max,
            "G_peak": g_max,
            "C_peak": c_max,
            "n_teams": len(merged),
        })
        
        print(f"  Divisor {divisor:.1f}: r={corr:.3f}, R²={corr**2:.3f}, "
              f"mean={mean_war:.3f}, T_peak={t_max:.2f}, G_peak={g_max:.2f}, C_peak={c_max:.2f}")
    
    results_df = pd.DataFrame(results)
    
    # Find optimal (highest correlation)
    best = results_df.loc[results_df["correlation"].idxmax()]
    
    print(f"\n  OPTIMAL DIVISOR: {best['divisor']:.1f} (r={best['correlation']:.3f})")
    
    return {
        "optimal_divisor": best["divisor"],
        "results_df": results_df,
        "best_result": best.to_dict()
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("Calculating Individual O-Line WAR")
    print("=" * 60)
    
    # Load data
    individual, team = load_data()
    
    # Calculate replacement levels by position
    replacement_levels = calculate_replacement_level(individual)
    
    # Calculate grades above replacement
    individual = calculate_grade_above_replacement(individual, replacement_levels)
    
    # Calculate snap shares
    individual = calculate_snap_share(individual)
    
    # === MOVE CALIBRATION HERE (before calculating WAR) ===
    print("\n" + "=" * 60)
    print("CALIBRATING DIVISOR AGAINST TEAM WINS")
    print("=" * 60)
    
    calibration = find_optimal_divisor(individual, replacement_levels)
    OPTIMAL_DIVISOR = calibration["optimal_divisor"]
    
    # Use optimal divisor in EPA/WAR calculation
    individual = calculate_war_with_divisor(individual, replacement_levels, OPTIMAL_DIVISOR)
    # === END CALIBRATION BLOCK ===
    
    # Summary stats
    summary = create_summary_stats(individual)
    print("\n" + "=" * 60)
    print("WAR SUMMARY BY POSITION AND SEASON")
    print("=" * 60)
    print(summary.to_string(index=False))
    
    # Plots
    plot_war_by_position(individual, FIGURES_DIR / "oline_war_by_position.png")
    plot_war_vs_grade(individual, FIGURES_DIR / "oline_war_vs_grade.png")
    
    # Leaders
    cumulative = print_war_leaders(individual)
    
    # Save outputs
    individual.to_csv(OLINE_WAR_FILE, index=False)
    print(f"\nSaved individual WAR data to: {OLINE_WAR_FILE}")
    
    cumulative.to_csv(TABLES_DIR / "oline_war_leaders.csv", index=False)
    print(f"Saved WAR leaders to: {TABLES_DIR / 'oline_war_leaders.csv'}")
    
    # Print methodology summary
    print("\n" + "=" * 60)
    print("METHODOLOGY SUMMARY")
    print("=" * 60)
    print(f"Pass Block Coefficient: {PASS_BLOCK_COEF} EPA per grade point")
    print(f"Run Block Coefficient: {RUN_BLOCK_COEF} EPA per grade point")
    print(f"O-Line Share of Offense: {OLINE_SHARE_OF_OFFENSE:.1%}")
    print(f"O-Line EPA per Win: {OLINE_EPA_PER_WIN:.1f}")
    print(f"Replacement Level: {REPLACEMENT_PERCENTILE}th percentile")
    print(f"Minimum Snaps: {MIN_SNAPS}")
    print(f"Optimal Divisor: {OPTIMAL_DIVISOR}")  # Add this line

    position_summary = create_position_summary(individual)
    print("\n" + "=" * 60)
    print("OVERALL WAR BY POSITION (2021-2024)")
    print("=" * 60)
    print(position_summary.to_string(index=False))
    
    return individual

if __name__ == "__main__":
    df = main()