"""
analyze_def_epa_per_win.py
Empirically derive defensive EPA per win for WAR calculation.

This ensures our defensive WAR is on the same scale as offensive WAR.
"""
import pandas as pd
import numpy as np
from scipy import stats

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================

TEAM_DEF_PATH = "/Users/anokhpalakurthi/Documents/nfl_defensive_war/outputs/team_defense_agg_2021_2024.csv"
PBP_PATH = "/Users/anokhpalakurthi/Documents/nfl_defensive_war/data/pbp_2021_2024_full.csv"

# ============================================================================
# FUNCTIONS
# ============================================================================

def build_team_wins_from_pbp(pbp_path: str) -> pd.DataFrame:
    """Build team wins per season from play-by-play data."""
    print("Building team wins from play-by-play data...")
    
    # Load only needed columns
    pbp = pd.read_csv(pbp_path, usecols=[
        "season", "game_id", "home_team", "away_team", 
        "home_score", "away_score"
    ])
    
    # Get unique games (one row per game)
    games = pbp.groupby("game_id").first().reset_index()
    print(f"  Total games: {len(games)}")
    
    # Home wins
    home_wins = games[games["home_score"] > games["away_score"]].copy()
    home_wins = home_wins.groupby(["season", "home_team"]).size().reset_index(name="wins")
    home_wins = home_wins.rename(columns={"home_team": "team"})
    
    # Away wins
    away_wins = games[games["away_score"] > games["home_score"]].copy()
    away_wins = away_wins.groupby(["season", "away_team"]).size().reset_index(name="wins")
    away_wins = away_wins.rename(columns={"away_team": "team"})
    
    # Home ties (0.5 wins each)
    ties = games[games["home_score"] == games["away_score"]].copy()
    if len(ties) > 0:
        home_ties = ties.groupby(["season", "home_team"]).size().reset_index(name="ties")
        home_ties = home_ties.rename(columns={"home_team": "team"})
        home_ties["wins"] = home_ties["ties"] * 0.5
        
        away_ties = ties.groupby(["season", "away_team"]).size().reset_index(name="ties")
        away_ties = away_ties.rename(columns={"away_team": "team"})
        away_ties["wins"] = away_ties["ties"] * 0.5
        
        # Combine all
        all_wins = pd.concat([
            home_wins[["season", "team", "wins"]],
            away_wins[["season", "team", "wins"]],
            home_ties[["season", "team", "wins"]],
            away_ties[["season", "team", "wins"]]
        ])
    else:
        all_wins = pd.concat([
            home_wins[["season", "team", "wins"]],
            away_wins[["season", "team", "wins"]]
        ])
    
    # Sum wins per team per season
    team_wins = all_wins.groupby(["season", "team"])["wins"].sum().reset_index()
    
    print(f"  Team-seasons: {len(team_wins)}")
    print(f"  Avg wins: {team_wins['wins'].mean():.1f}")
    
    return team_wins


def main():
    print("=" * 60)
    print("DERIVING DEFENSIVE EPA PER WIN")
    print("=" * 60)
    
    # Load team defensive data
    print("\n[1] Loading team defensive data...")
    team_df = pd.read_csv(TEAM_DEF_PATH)
    print(f"  Team-seasons: {len(team_df)}")
    
    # Check if def_epa_total exists, otherwise calculate it
    if "def_epa_total" not in team_df.columns:
        print("  Calculating total defensive EPA...")
        team_df["def_epa_total"] = team_df["def_epa_per_play"] * team_df["def_plays"]
    
    print(f"  Def EPA range: [{team_df['def_epa_total'].min():.1f}, {team_df['def_epa_total'].max():.1f}]")
    
    # Build team wins
    print("\n[2] Building team wins...")
    team_wins = build_team_wins_from_pbp(PBP_PATH)
    
    # Merge
    print("\n[3] Merging data...")
    merged = team_df.merge(team_wins, on=["season", "team"], how="inner")
    print(f"  Matched team-seasons: {len(merged)}")
    
    # Regression: wins ~ def_epa_total
    # Note: Lower (more negative) defensive EPA = better defense = more wins
    # So we expect NEGATIVE slope
    print("\n[4] Running regression: Wins ~ Defensive EPA...")
    
    slope, intercept, r, p, se = stats.linregress(
        merged["def_epa_total"], 
        merged["wins"]
    )
    
    print(f"\n  Results:")
    print(f"    Slope:     {slope:.6f} wins per EPA point")
    print(f"    Intercept: {intercept:.2f}")
    print(f"    R:         {r:.3f}")
    print(f"    R²:        {r**2:.3f}")
    print(f"    p-value:   {p:.6f}")
    
    # EPA per win = 1 / |slope|
    epa_per_win = 1 / abs(slope)
    print(f"\n  DEFENSIVE EPA PER WIN: {epa_per_win:.2f}")
    
    # Sanity check with offensive comparison
    print("\n[5] Context...")
    print(f"    Your offensive EPA per win: ~218.83")
    print(f"    Defensive EPA per win:      {epa_per_win:.2f}")
    print(f"    Ratio (def/off):            {epa_per_win / 218.83:.2f}")
    
    # Additional analysis: defensive EPA correlation with wins
    print("\n[6] Validation...")
    print(f"    Better defense (lower EPA) → More wins")
    print(f"    Correlation: {r:.3f} (expect negative)")
    
    if r < 0:
        print("    ✓ Confirmed: negative correlation as expected")
    else:
        print("    ⚠️ Warning: positive correlation unexpected!")
    
    # Summary stats
    print("\n[7] Summary Statistics...")
    print(f"    Avg team defensive EPA: {merged['def_epa_total'].mean():.1f}")
    print(f"    Std team defensive EPA: {merged['def_epa_total'].std():.1f}")
    print(f"    Avg team wins:          {merged['wins'].mean():.1f}")
    
    # What this means for our model
    print("\n" + "=" * 60)
    print("RECOMMENDATION FOR WAR MODEL")
    print("=" * 60)
    print(f"""
    Add to 03_war_model.py:
    
    DEFENSE_EPA_PER_WIN = {epa_per_win:.2f}
    
    Then in compute_team_war():
        team_df["def_war_pool"] = (
            (replacement_epa - team_df["def_epa_per_play"]) * team_df["def_plays"]
        ) / DEFENSE_EPA_PER_WIN
    """)
    
    return {
        "epa_per_win": epa_per_win,
        "r_squared": r**2,
        "slope": slope,
        "merged_df": merged
    }


if __name__ == "__main__":
    results = main()