"""
Validate O-line WAR against Approximate Value (AV) and team wins.
Uses cumulative data (2021-2024).

Inputs:
    - data/processed/oline_war_2021_2024.csv (from 04_individual_war.py)
    - data/external/2021_2024_OL_AV.csv (Stathead cumulative AV)

Outputs:
    - outputs/figures/oline_war_vs_av.png
    - outputs/figures/oline_war_av_vs_wins.png
    - outputs/tables/oline_validation_summary.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("/Users/anokhpalakurthi/Documents/nfl_offensive_war/data")
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
OUTPUTS_DIR = Path("/Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs")
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Input files
OLINE_WAR_FILE = PROCESSED_DIR / "oline_war_2021_2024.csv"
OLINE_AV_FILE = EXTERNAL_DIR / "2021_2024_OL_AV.csv"

# Ensure output directories exist
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# FUNCTIONS
# ============================================================================

def load_war_data() -> pd.DataFrame:
    """Load O-line WAR data and aggregate to cumulative totals."""
    print(f"Loading WAR data from {OLINE_WAR_FILE}...")
    df = pd.read_csv(OLINE_WAR_FILE)
    print(f"  Loaded {len(df)} player-seasons")
    
    # Aggregate to cumulative WAR per player
    cumulative = df.groupby(["player", "position_group"]).agg({
        "war": "sum",
        "snap_counts_offense": "sum",
        "season": "count"
    }).reset_index()
    
    cumulative = cumulative.rename(columns={"season": "seasons_played"})
    
    # Clean player names for matching
    cumulative["player_clean"] = (cumulative["player"]
        .str.replace(".", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.lower()
        .str.strip())
    
    print(f"  Aggregated to {len(cumulative)} unique players")
    
    return cumulative


def load_av_data() -> pd.DataFrame:
    """Load cumulative AV data from Stathead."""
    print(f"\nLoading AV data from {OLINE_AV_FILE}...")
    df = pd.read_csv(OLINE_AV_FILE)
    print(f"  Loaded {len(df)} players")
    
    # Flag multi-team players
    df["multi_team"] = df["Team"].str.contains(",", na=False)
    print(f"  Multi-team players: {df['multi_team'].sum()}")
    
    # Clean player names for matching
    df["player_clean"] = (df["Player"]
        .str.replace(".", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.lower()
        .str.strip())
    
    # Map position to position group
    def map_position(pos):
        pos = str(pos).upper()
        if "C" in pos and "G" not in pos and "T" not in pos:
            return "C"
        elif "G" in pos:
            return "G"
        elif "T" in pos:
            return "T"
        elif pos == "OL":
            return "unknown"
        else:
            return "unknown"
    
    df["position_group"] = df["Pos"].apply(map_position)
    
    return df


def match_war_and_av(war_df: pd.DataFrame, av_df: pd.DataFrame) -> pd.DataFrame:
    """Match WAR and AV data by player name."""
    print("\nMatching WAR and AV data...")
    
    # Merge on cleaned player name
    merged = war_df.merge(
        av_df[["player_clean", "Player", "AV", "G", "GS", "multi_team", "position_group"]],
        on="player_clean",
        how="inner",
        suffixes=("_war", "_av")
    )
    
    # Use WAR position group (more reliable since it comes from our data)
    merged["position"] = merged["position_group_war"]
    
    print(f"  Matched {len(merged)} players")
    print(f"  WAR players not matched: {len(war_df) - len(merged)}")
    
    # Show some unmatched players for debugging
    matched_names = set(merged["player_clean"])
    unmatched_war = war_df[~war_df["player_clean"].isin(matched_names)]
    if len(unmatched_war) > 0:
        print(f"  Top unmatched WAR players by WAR:")
        top_unmatched = unmatched_war.nlargest(5, "war")[["player", "war"]]
        for _, row in top_unmatched.iterrows():
            print(f"    {row['player']}: {row['war']:.2f}")
    
    return merged


def calculate_correlations(merged_df: pd.DataFrame) -> dict:
    """Calculate player-level correlations between WAR and AV."""
    print("\nCalculating player-level correlations...")
    
    # Exclude multi-team players for cleaner analysis (optional)
    single_team = merged_df[~merged_df["multi_team"]]
    print(f"  Single-team players: {len(single_team)}")
    
    # Overall correlation (all players)
    pearson_r, pearson_p = stats.pearsonr(merged_df["war"], merged_df["AV"])
    spearman_r, spearman_p = stats.spearmanr(merged_df["war"], merged_df["AV"])
    
    print(f"\n  All players (n={len(merged_df)}):")
    print(f"    Pearson r:  {pearson_r:.3f} (p={pearson_p:.4f})")
    print(f"    Spearman r: {spearman_r:.3f} (p={spearman_p:.4f})")
    print(f"    R²: {pearson_r**2:.3f}")
    
    # Single-team only
    if len(single_team) > 10:
        pearson_single, _ = stats.pearsonr(single_team["war"], single_team["AV"])
        print(f"\n  Single-team only (n={len(single_team)}):")
        print(f"    Pearson r: {pearson_single:.3f}")
    
    # By position
    print("\n  By position:")
    position_corrs = {}
    for pos in ["T", "G", "C"]:
        pos_data = merged_df[merged_df["position"] == pos]
        if len(pos_data) >= 10:
            r, p = stats.pearsonr(pos_data["war"], pos_data["AV"])
            position_corrs[pos] = {"pearson": r, "n": len(pos_data)}
            print(f"    {pos}: r={r:.3f} (n={len(pos_data)})")
    
    return {
        "pearson": pearson_r,
        "spearman": spearman_r,
        "r_squared": pearson_r ** 2,
        "n": len(merged_df),
        "by_position": position_corrs
    }


def plot_war_vs_av(merged_df: pd.DataFrame, save_path: Path):
    """Scatter plot of cumulative WAR vs cumulative AV."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {"T": "blue", "G": "green", "C": "orange"}
    
    for pos in ["T", "G", "C"]:
        pos_data = merged_df[merged_df["position"] == pos]
        ax.scatter(pos_data["AV"], pos_data["war"],
                   alpha=0.6, label=f"{pos} (n={len(pos_data)})", 
                   c=colors.get(pos, "gray"), s=60, edgecolor="white")
    
    # Add regression line
    z = np.polyfit(merged_df["AV"], merged_df["war"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged_df["AV"].min(), merged_df["AV"].max(), 100)
    
    r_squared = merged_df["AV"].corr(merged_df["war"]) ** 2
    ax.plot(x_line, p(x_line), color="darkred", linewidth=2, linestyle="--",
            label=f"R² = {r_squared:.3f}")
    
    # Label some notable players
    top_players = merged_df.nlargest(5, "war")
    for _, row in top_players.iterrows():
        ax.annotate(row["player"], (row["AV"], row["war"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    
    ax.set_xlabel("Approximate Value (AV) - Cumulative 2021-2024", fontsize=12)
    ax.set_ylabel("WAR - Cumulative 2021-2024", fontsize=12)
    ax.set_title("O-Line WAR vs Approximate Value (2021-2024)", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved WAR vs AV plot to: {save_path}")


def print_ranking_comparison(merged_df: pd.DataFrame, n: int = 15):
    """Print side-by-side ranking comparison of WAR vs AV."""
    print("\n" + "=" * 70)
    print(f"TOP {n} PLAYERS: WAR vs AV RANKING COMPARISON")
    print("=" * 70)
    
    # Get top N by each metric
    top_war = merged_df.nlargest(n, "war")[["player", "position", "war", "AV"]].reset_index(drop=True)
    top_av = merged_df.nlargest(n, "AV")[["player", "position", "AV", "war"]].reset_index(drop=True)
    
    print(f"\n{'TOP BY WAR':<35} | {'TOP BY AV':<35}")
    print("-" * 70)
    
    for i in range(n):
        war_row = top_war.iloc[i]
        av_row = top_av.iloc[i]
        
        war_str = f"{i+1}. {war_row['player']} ({war_row['position']}): {war_row['war']:.2f}"
        av_str = f"{i+1}. {av_row['player']} ({av_row['position']}): {av_row['AV']}"
        
        print(f"{war_str:<35} | {av_str:<35}")


def save_validation_summary(player_corrs: dict, save_path: Path):
    """Save validation results to CSV."""
    summary = pd.DataFrame([
        {"metric": "WAR vs AV (player-level)", 
         "pearson_r": player_corrs["pearson"],
         "spearman_r": player_corrs["spearman"],
         "r_squared": player_corrs["r_squared"], 
         "n": player_corrs["n"]},
    ])
    
    # Add position-specific correlations
    for pos, vals in player_corrs.get("by_position", {}).items():
        summary = pd.concat([summary, pd.DataFrame([{
            "metric": f"WAR vs AV ({pos} only)",
            "pearson_r": vals["pearson"],
            "spearman_r": np.nan,
            "r_squared": vals["pearson"] ** 2,
            "n": vals["n"]
        }])], ignore_index=True)
    
    summary.to_csv(save_path, index=False)
    print(f"\nSaved validation summary to: {save_path}")

def load_team_wins() -> pd.DataFrame:
    """Load team wins data from play-by-play."""
    print("\nLoading team wins...")
    
    pbp_file = DATA_DIR / "raw" / "pbp_2021_2024.csv"
    
    pbp = pd.read_csv(pbp_file, 
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
    
    # Combine and sum across all seasons for cumulative wins
    team_wins = pd.concat([home_wins, away_wins]).groupby("team")["wins"].sum().reset_index()
    team_wins = team_wins.rename(columns={"wins": "total_wins"})
    
    print(f"  Loaded cumulative wins for {len(team_wins)} teams")
    
    return team_wins


def aggregate_to_team_level(war_df: pd.DataFrame, av_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate WAR and AV to team level for comparison with wins."""
    print("\nAggregating to team level...")
    
    # Load season-level WAR data for team aggregation
    season_war = pd.read_csv(OLINE_WAR_FILE)
    
    # Aggregate WAR by team (summing across all seasons)
    team_war = season_war.groupby("team").agg({
        "war": "sum"
    }).reset_index()
    team_war = team_war.rename(columns={"war": "team_oline_war"})
    
    # For AV, we need to handle multi-team players
    # Only use single-team players for clean team attribution
    single_team_av = av_df[~av_df["multi_team"]].copy()
    
    # Map Stathead team abbreviations to nflFastR
    STATHEAD_TO_NFLFASTR = {
        "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BUF": "BUF",
        "CAR": "CAR", "CHI": "CHI", "CIN": "CIN", "CLE": "CLE",
        "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GNB": "GB",
        "HOU": "HOU", "IND": "IND", "JAX": "JAX", "KAN": "KC",
        "LAC": "LAC", "LAR": "LA", "LVR": "LV", "MIA": "MIA",
        "MIN": "MIN", "NOR": "NO", "NWE": "NE", "NYG": "NYG",
        "NYJ": "NYJ", "PHI": "PHI", "PIT": "PIT", "SEA": "SEA",
        "SFO": "SF", "TAM": "TB", "TEN": "TEN", "WAS": "WAS",
    }
    
    single_team_av["team"] = single_team_av["Team"].map(STATHEAD_TO_NFLFASTR)
    
    team_av = single_team_av.groupby("team").agg({
        "AV": "sum"
    }).reset_index()
    team_av = team_av.rename(columns={"AV": "team_oline_av"})
    
    print(f"  Teams with WAR data: {len(team_war)}")
    print(f"  Teams with AV data: {len(team_av)}")
    
    return team_war, team_av


def compare_war_vs_av_to_wins(team_war: pd.DataFrame, team_av: pd.DataFrame, 
                               team_wins: pd.DataFrame) -> dict:
    """Compare how well WAR and AV each correlate with team wins."""
    print("\nComparing WAR vs AV correlation with wins...")
    
    # Merge WAR with wins
    war_wins = team_war.merge(team_wins, on="team", how="inner")
    
    # Merge AV with wins
    av_wins = team_av.merge(team_wins, on="team", how="inner")
    
    # WAR vs wins
    war_r, war_p = stats.pearsonr(war_wins["team_oline_war"], war_wins["total_wins"])
    
    # AV vs wins
    av_r, av_p = stats.pearsonr(av_wins["team_oline_av"], av_wins["total_wins"])
    
    print(f"\n  O-Line WAR vs Wins: r={war_r:.3f} (R²={war_r**2:.3f}, n={len(war_wins)})")
    print(f"  O-Line AV vs Wins:  r={av_r:.3f} (R²={av_r**2:.3f}, n={len(av_wins)})")
    
    if war_r > av_r:
        print(f"\n  --> WAR outperforms AV by {war_r - av_r:.3f}")
    else:
        print(f"\n  --> AV outperforms WAR by {av_r - war_r:.3f}")
    
    return {
        "war_r": war_r,
        "war_r_squared": war_r ** 2,
        "war_n": len(war_wins),
        "av_r": av_r,
        "av_r_squared": av_r ** 2,
        "av_n": len(av_wins),
        "war_wins_df": war_wins,
        "av_wins_df": av_wins
    }


def plot_team_metrics_vs_wins(war_wins: pd.DataFrame, av_wins: pd.DataFrame, save_path: Path):
    """Side-by-side scatter plots of team WAR and AV vs wins."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # WAR vs Wins
    ax1 = axes[0]
    ax1.scatter(war_wins["team_oline_war"], war_wins["total_wins"],
                alpha=0.7, color="steelblue", s=80, edgecolor="white")
    
    z1 = np.polyfit(war_wins["team_oline_war"], war_wins["total_wins"], 1)
    p1 = np.poly1d(z1)
    x1_line = np.linspace(war_wins["team_oline_war"].min(), war_wins["team_oline_war"].max(), 100)
    ax1.plot(x1_line, p1(x1_line), color="darkred", linewidth=2)
    
    r1 = war_wins["team_oline_war"].corr(war_wins["total_wins"])
    ax1.set_xlabel("Team O-Line WAR (Cumulative 2021-2024)", fontsize=12)
    ax1.set_ylabel("Team Wins (Cumulative 2021-2024)", fontsize=12)
    ax1.set_title(f"O-Line WAR vs Wins (R² = {r1**2:.3f})", fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    # AV vs Wins
    ax2 = axes[1]
    ax2.scatter(av_wins["team_oline_av"], av_wins["total_wins"],
                alpha=0.7, color="forestgreen", s=80, edgecolor="white")
    
    z2 = np.polyfit(av_wins["team_oline_av"], av_wins["total_wins"], 1)
    p2 = np.poly1d(z2)
    x2_line = np.linspace(av_wins["team_oline_av"].min(), av_wins["team_oline_av"].max(), 100)
    ax2.plot(x2_line, p2(x2_line), color="darkred", linewidth=2)
    
    r2 = av_wins["team_oline_av"].corr(av_wins["total_wins"])
    ax2.set_xlabel("Team O-Line AV (Cumulative 2021-2024)", fontsize=12)
    ax2.set_ylabel("Team Wins (Cumulative 2021-2024)", fontsize=12)
    ax2.set_title(f"O-Line AV vs Wins (R² = {r2**2:.3f})", fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved team metrics vs wins plot to: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("O-Line WAR Validation: Cumulative WAR vs AV (2021-2024)")
    print("=" * 60)
    
    # Load data
    war_df = load_war_data()
    av_df = load_av_data()
    
    # Match WAR and AV at player level
    merged = match_war_and_av(war_df, av_df)
    
    # Player-level correlations
    player_corrs = calculate_correlations(merged)
    
    # Ranking comparison
    print_ranking_comparison(merged)
    
    # Plot
    plot_war_vs_av(merged, FIGURES_DIR / "oline_war_vs_av.png")
    
    # Save summary
    save_validation_summary(player_corrs, TABLES_DIR / "oline_validation_summary.csv")

    # --- Team-level comparison with wins ---
    team_wins = load_team_wins()
    team_war, team_av = aggregate_to_team_level(war_df, av_df)
    wins_comparison = compare_war_vs_av_to_wins(team_war, team_av, team_wins)
    
    # Plot team metrics vs wins
    plot_team_metrics_vs_wins(
        wins_comparison["war_wins_df"], 
        wins_comparison["av_wins_df"],
        FIGURES_DIR / "oline_war_av_vs_wins.png"
    )
    
    # Print final summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Players matched: {len(merged)}")
    print(f"WAR vs AV correlation: r={player_corrs['pearson']:.3f} (R²={player_corrs['r_squared']:.3f})")
    print(f"\nTeam-level correlation with wins:")
    print(f"  O-Line WAR vs Wins: r={wins_comparison['war_r']:.3f} (R²={wins_comparison['war_r_squared']:.3f})")
    print(f"  O-Line AV vs Wins:  r={wins_comparison['av_r']:.3f} (R²={wins_comparison['av_r_squared']:.3f})")
    
    return merged

if __name__ == "__main__":
    merged = main()