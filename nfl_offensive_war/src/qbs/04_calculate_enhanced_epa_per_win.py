"""
Calculate EPA per Win - Empirical Approach

This script analyzes game-level EPA differentials to determine
how much offensive EPA translates to one win.

Methodology:
1. For each game, calculate home and away offensive EPA
2. Determine winner's EPA - loser's EPA
3. Average EPA differential across all games
4. Scale to full season: differential × 17 games = EPA per win

This approach:
- Uses our actual data (2021-2024)
- Works for all positions (not just QBs)
- Transparent and defensible
- Avoids arbitrary constants
"""

import pandas as pd
import numpy as np
import os

def load_play_data():
    """Load the cleaned play-by-play data"""
    print("Loading play-by-play data...")
    
    # Use relative path from src folder
    pbp = pd.read_csv('/Users/anokhpalakurthi/Documents/nfl_offensive_war/data/raw/pbp_2021_2024.csv')
    print(f"Loaded {len(pbp):,} plays from 2021-2024")
    return pbp

def filter_to_offensive_plays(pbp):
    """
    Filter to plays that count as offensive production
    (passing and rushing plays, no penalties/special teams)
    """
    print("\nFiltering to offensive plays...")
    
    # Keep only pass and rush plays (play_type already filtered in data)
    offensive = pbp[pbp['play_type'].isin(['pass', 'run'])].copy()
    
    # Remove plays with missing EPA (rare but possible)
    offensive = offensive[offensive['epa'].notna()]
    
    # Remove plays with missing game results
    offensive = offensive[offensive['home_score'].notna() & 
                          offensive['away_score'].notna()]
    
    print(f"Kept {len(offensive):,} offensive plays")
    return offensive

def calculate_game_epa(pbp):
    """
    Calculate total offensive EPA for each team in each game
    """
    print("\nCalculating game-level offensive EPA...")
    
    # Home team offensive EPA
    home_epa = pbp[pbp['posteam'] == pbp['home_team']].groupby('game_id').agg({
        'epa': 'sum',
        'home_team': 'first',
        'away_team': 'first',
        'home_score': 'first',
        'away_score': 'first',
        'season': 'first',
        'week': 'first'
    }).rename(columns={'epa': 'home_offensive_epa'})
    
    # Away team offensive EPA
    away_epa = pbp[pbp['posteam'] == pbp['away_team']].groupby('game_id').agg({
        'epa': 'sum'
    }).rename(columns={'epa': 'away_offensive_epa'})
    
    # Merge together
    games = home_epa.join(away_epa)
    
    print(f"Analyzed {len(games):,} games")
    return games

def analyze_epa_differentials(games):
    """
    Calculate EPA differential between winner and loser
    """
    print("\nAnalyzing EPA differentials (Winner - Loser)...")
    
    # Determine winner and calculate EPA differential
    games['winner'] = np.where(
        games['home_score'] > games['away_score'],
        'home',
        np.where(games['home_score'] < games['away_score'], 'away', 'tie')
    )
    
    # Calculate EPA differential (Winner EPA - Loser EPA)
    games['epa_differential'] = np.where(
        games['winner'] == 'home',
        games['home_offensive_epa'] - games['away_offensive_epa'],
        np.where(
            games['winner'] == 'away',
            games['away_offensive_epa'] - games['home_offensive_epa'],
            0  # ties get 0 differential
        )
    )
    
    # Remove ties (very rare, but they exist)
    games_no_ties = games[games['winner'] != 'tie']
    
    print(f"Removed {len(games) - len(games_no_ties)} tied games")
    print(f"Analyzing {len(games_no_ties):,} decided games")
    
    return games_no_ties

def calculate_epa_per_win(games):
    """
    Calculate EPA per win conversion factor
    """
    print("\n" + "="*60)
    print("EPA PER WIN CALCULATION")
    print("="*60)
    
    # Average EPA differential per game
    avg_differential = games['epa_differential'].mean()
    std_differential = games['epa_differential'].std()
    median_differential = games['epa_differential'].median()
    
    print(f"\nEPA Differential (Winner - Loser) per game:")
    print(f"  Mean:   {avg_differential:>8.3f} EPA")
    print(f"  Median: {median_differential:>8.3f} EPA")
    print(f"  Std:    {std_differential:>8.3f} EPA")
    
    # Scale to full season (17 games in modern NFL)
    epa_per_win = avg_differential * 17
    
    print(f"\nEPA per Win (differential × 17 games):")
    print(f"  {epa_per_win:.2f} EPA = 1 Win")
    
    # Compare to historical ~27 points per win
    # (note: this is OFFENSIVE EPA only, not points)
    print(f"\nFor reference:")
    print(f"  Traditional: ~27 points = 1 win")
    print(f"  Our metric:  {epa_per_win:.1f} offensive EPA = 1 win")
    
    return epa_per_win, avg_differential

def season_breakdown(games):
    """
    Show EPA per win by season (sanity check)
    """
    print("\n" + "="*60)
    print("SEASON-BY-SEASON BREAKDOWN")
    print("="*60)
    
    season_stats = games.groupby('season').agg({
        'epa_differential': ['count', 'mean', 'std']
    }).round(3)
    
    season_stats.columns = ['Games', 'Avg_Differential', 'Std_Differential']
    season_stats['EPA_per_Win'] = season_stats['Avg_Differential'] * 17
    
    print("\n", season_stats)
    
    # Check if seasons are relatively consistent
    consistency = season_stats['EPA_per_Win'].std()
    print(f"\nStd dev across seasons: {consistency:.2f} EPA")
    print("(Lower is better - means metric is stable)")

def main():
    """
    Main execution flow
    """
    print("="*60)
    print("CALCULATING EPA PER WIN FROM 2021-2024 DATA")
    print("="*60)
    
    # Load and filter data
    pbp = load_play_data()
    offensive = filter_to_offensive_plays(pbp)
    
    # Calculate game-level EPA
    games = calculate_game_epa(offensive)
    
    # Analyze differentials
    games = analyze_epa_differentials(games)
    
    # Calculate EPA per win
    epa_per_win, avg_differential = calculate_epa_per_win(games)
    
    # Season breakdown
    season_breakdown(games)
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Create outputs directory if it doesn't exist
    os.makedirs('/Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs/tables', exist_ok=True)
    
    # Save game-level data for reference
    games.to_csv('/Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs/tables/game_epa_differentials.csv', index=True)
    print("✓ Saved: /Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs/tables/game_epa_differentials.csv")
    
    # Save conversion factor
    conversion = pd.DataFrame({
        'metric': ['epa_per_win', 'avg_differential_per_game'],
        'value': [epa_per_win, avg_differential]
    })
    conversion.to_csv('/Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs/tables/epa_per_win_conversion.csv', index=False)
    print("✓ Saved: /Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs/tables/epa_per_win_conversion.csv")
    
    print(f"\n✓ EPA per Win: {epa_per_win:.2f}")
    print("\nThis conversion factor will be used for all WAR calculations")
    print("(QB, RB, WR, TE - and potentially OL/Defense in future)")

if __name__ == "__main__":
    main()