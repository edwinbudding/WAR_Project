"""
Defensive WAR Calibration Analysis
===================================

Exploring solutions for:
1. Martin Emerson / Paulson Adebo problem (elite team inflation)
2. Flattened WAR values (divisor = 11 too aggressive)

"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path("/Users/anokhpalakurthi/Documents/nfl_defensive_war")
PLAYER_PATH = BASE_DIR / "outputs" / "defense_engineered_2021_2024.csv"
WAR_PATH = BASE_DIR / "outputs" / "defensive_war_2021_2024.csv"

def main():
    print("=" * 70)
    print(" DEFENSIVE WAR CALIBRATION ANALYSIS")
    print("=" * 70)
    
    # Load data
    players = pd.read_csv(PLAYER_PATH)
    war_df = pd.read_csv(WAR_PATH)
    
    qualified = war_df[war_df['snap_counts_defense'] >= 200].copy()
    print(f"\n    Qualified players: {len(qualified)}")
    
    # ============================================================
    # PART 1: SNAP CONCENTRATION ANALYSIS
    # ============================================================
    print("\n" + "=" * 70)
    print(" PART 1: SNAP CONCENTRATION BY TEAM")
    print("=" * 70)
    
    print("""
    Question: How many players take the bulk of defensive snaps?
    
    If top 7 players take 85% of snaps, maybe divisor should be ~7-8
    instead of 11.
    """)
    
    # Calculate snap concentration by team-season
    team_seasons = qualified.groupby(['team', 'season']).agg({
        'snap_counts_defense': ['sum', 'count', 'max', 'mean']
    }).reset_index()
    team_seasons.columns = ['team', 'season', 'total_snaps', 'n_players', 'max_snaps', 'mean_snaps']
    
    print(f"\n    Team-seasons: {len(team_seasons)}")
    print(f"    Avg qualified players per team: {team_seasons['n_players'].mean():.1f}")
    print(f"    Range: {team_seasons['n_players'].min()} - {team_seasons['n_players'].max()}")
    
    # For each team, calculate what % of snaps top N players take
    print("\n    SNAP CONCENTRATION (avg across team-seasons):")
    
    for top_n in [5, 7, 9, 11]:
        concentrations = []
        for (team, season), group in qualified.groupby(['team', 'season']):
            sorted_group = group.sort_values('snap_counts_defense', ascending=False)
            total = sorted_group['snap_counts_defense'].sum()
            top_n_snaps = sorted_group.head(top_n)['snap_counts_defense'].sum()
            concentrations.append(top_n_snaps / total if total > 0 else 0)
        
        avg_conc = np.mean(concentrations)
        print(f"      Top {top_n} players: {avg_conc:.1%} of team snaps")
    
    # ============================================================
    # PART 2: EFFECTIVE STARTERS CALCULATION
    # ============================================================
    print("\n" + "=" * 70)
    print(" PART 2: EFFECTIVE STARTERS (SNAP-WEIGHTED DIVISOR)")
    print("=" * 70)
    
    print("""
    Idea: Instead of dividing by 11, calculate "effective starters"
    based on snap distribution.
    
    Method: HHI-style calculation
    effective_starters = 1 / sum(snap_share^2)
    
    If 11 players split snaps equally: effective = 11
    If 7 players dominate: effective ≈ 7
    """)
    
    effective_starters = []
    for (team, season), group in qualified.groupby(['team', 'season']):
        total = group['snap_counts_defense'].sum()
        if total > 0:
            shares = group['snap_counts_defense'] / total
            hhi = (shares ** 2).sum()
            eff = 1 / hhi if hhi > 0 else 11
            effective_starters.append({
                'team': team,
                'season': season,
                'n_players': len(group),
                'effective_starters': eff
            })
    
    eff_df = pd.DataFrame(effective_starters)
    
    print(f"\n    Effective starters distribution:")
    print(f"      Mean: {eff_df['effective_starters'].mean():.2f}")
    print(f"      Median: {eff_df['effective_starters'].median():.2f}")
    print(f"      Min: {eff_df['effective_starters'].min():.2f}")
    print(f"      Max: {eff_df['effective_starters'].max():.2f}")
    print(f"      Std: {eff_df['effective_starters'].std():.2f}")
    
    # Show examples
    print("\n    Teams with LOWEST effective starters (concentrated snaps):")
    for _, row in eff_df.nsmallest(5, 'effective_starters').iterrows():
        print(f"      {row['season']} {row['team']}: {row['effective_starters']:.2f} effective ({row['n_players']} qualified)")
    
    print("\n    Teams with HIGHEST effective starters (distributed snaps):")
    for _, row in eff_df.nlargest(5, 'effective_starters').iterrows():
        print(f"      {row['season']} {row['team']}: {row['effective_starters']:.2f} effective ({row['n_players']} qualified)")
    
    # ============================================================
    # PART 3: DIVISOR COMPARISON
    # ============================================================
    print("\n" + "=" * 70)
    print(" PART 3: DIVISOR IMPACT ON WAR SCALE")
    print("=" * 70)
    
    print("""
    Current: Divisor = 11, then adjusted to 5.5 for final output
    
    Comparing different divisor approaches:
    """)
    
    # Current top WAR values
    current_max = qualified['def_war'].max()
    current_mean_cb = qualified[qualified['role'] == 'CB']['def_war'].mean()
    
    print(f"\n    Current WAR scale (divisor effects already applied):")
    print(f"      Max WAR: {current_max:.3f}")
    print(f"      Mean CB WAR: {current_mean_cb:.3f}")
    
    # Simulate different divisors
    print("\n    Simulated WAR with different divisors:")
    print("    (scaling current values proportionally)")
    
    current_divisor = 5.5  # What we're using now based on transcript
    
    for new_divisor in [4, 5, 6, 7, 8]:
        scale_factor = current_divisor / new_divisor
        new_max = current_max * scale_factor
        new_mean_cb = current_mean_cb * scale_factor
        print(f"      Divisor {new_divisor}: Max WAR = {new_max:.2f}, Mean CB = {new_mean_cb:.3f}")
    
    # ============================================================
    # PART 4: TEAM CONTEXT REGRESSION OPTIONS
    # ============================================================
    print("\n" + "=" * 70)
    print(" PART 4: TEAM CONTEXT REGRESSION OPTIONS")
    print("=" * 70)
    
    print("""
    Current: 50% regression to league mean for all positions
    
    Problem: Emerson/Adebo still inflated on elite defenses
    
    Options:
    A) Increase regression to 60-70%
    B) Position-specific regression (CBs get more regression)
    C) Cap individual WAR as % of team WAR
    D) Peer comparison within team
    """)
    
    # Analyze team WAR concentration by position
    print("\n    WAR CONCENTRATION BY POSITION:")
    print("    (What % of team's pass defense WAR goes to top CB?)")
    
    for role in ['CB', 'S', 'EDGE', 'IDL', 'LB']:
        role_shares = []
        for (team, season), group in qualified.groupby(['team', 'season']):
            role_group = group[group['role'] == role]
            if len(role_group) > 0:
                team_role_war = role_group['def_war'].sum()
                max_player_war = role_group['def_war'].max()
                if team_role_war > 0:
                    share = max_player_war / team_role_war
                    role_shares.append(share)
        
        if role_shares:
            print(f"      {role}: Top player gets {np.mean(role_shares):.1%} of position group WAR")
    
    # Look at Emerson and Adebo specifically
    print("\n    CASE STUDY: EMERSON & ADEBO TEAM CONTEXT")
    
    # Emerson 2023 CLE
    cle_2023 = qualified[(qualified['team'] == 'CLE') & (qualified['season'] == 2023)]
    print(f"\n    2023 CLE Defense ({len(cle_2023)} qualified players):")
    print(f"      Team total WAR: {cle_2023['def_war'].sum():.2f}")
    emerson = cle_2023[cle_2023['player'].str.contains('Emerson', case=False)]
    if len(emerson) > 0:
        emerson_war = emerson['def_war'].values[0]
        emerson_share = emerson_war / cle_2023['def_war'].sum()
        print(f"      Emerson WAR: {emerson_war:.2f} ({emerson_share:.1%} of team)")
    
    cle_cbs = cle_2023[cle_2023['role'] == 'CB']
    print(f"      Other CBs: {cle_cbs[~cle_cbs['player'].str.contains('Emerson', case=False)][['player', 'def_war']].to_string(index=False)}")
    
    # Adebo 2023 NO
    no_2023 = qualified[(qualified['team'] == 'NO') & (qualified['season'] == 2023)]
    print(f"\n    2023 NO Defense ({len(no_2023)} qualified players):")
    print(f"      Team total WAR: {no_2023['def_war'].sum():.2f}")
    adebo = no_2023[no_2023['player'].str.contains('Adebo', case=False)]
    if len(adebo) > 0:
        adebo_war = adebo['def_war'].values[0]
        adebo_share = adebo_war / no_2023['def_war'].sum()
        print(f"      Adebo WAR: {adebo_war:.2f} ({adebo_share:.1%} of team)")
    
    no_cbs = no_2023[no_2023['role'] == 'CB']
    print(f"      Other CBs: {no_cbs[~no_cbs['player'].str.contains('Adebo', case=False)][['player', 'def_war']].to_string(index=False)}")
    
    # ============================================================
    # PART 5: POSITION-SPECIFIC REGRESSION
    # ============================================================
    print("\n" + "=" * 70)
    print(" PART 5: POSITION-SPECIFIC REGRESSION PROPOSAL")
    print("=" * 70)
    
    print("""
    Idea: CBs and Safeties should get MORE regression because:
    - Coverage success is harder to isolate individually
    - They benefit more from good pass rush / scheme
    
    EDGE/IDL should get LESS regression because:
    - Sacks/pressures are more individually attributable
    - Less dependent on secondary play
    
    Proposed regression factors:
      CB:   65% toward mean (more regression)
      S:    60% toward mean
      LB:   55% toward mean  
      IDL:  50% toward mean (current)
      EDGE: 45% toward mean (less regression)
    """)
    
    # Simulate impact
    print("\n    Simulated impact on top players:")
    
    top_players = qualified.nlargest(10, 'def_war')[['player', 'season', 'team', 'role', 'def_war']].copy()
    
    # Current is 50% regression. Simulate position-specific.
    position_regression = {'CB': 0.65, 'S': 0.60, 'LB': 0.55, 'IDL': 0.50, 'EDGE': 0.45}
    league_mean_war = qualified['def_war'].mean()
    
    for idx, row in top_players.iterrows():
        current_war = row['def_war']
        role = row['role']
        
        # Reverse current 50% regression to get "raw" team-based WAR
        # current = 0.5 * league_mean + 0.5 * raw
        # raw = (current - 0.5 * league_mean) / 0.5
        raw_war = (current_war - 0.5 * league_mean_war) / 0.5
        
        # Apply new position-specific regression
        new_regression = position_regression.get(role, 0.50)
        new_war = new_regression * league_mean_war + (1 - new_regression) * raw_war
        
        top_players.loc[idx, 'new_war'] = new_war
        top_players.loc[idx, 'change'] = new_war - current_war
    
    print(top_players[['player', 'role', 'def_war', 'new_war', 'change']].to_string(index=False))
    
    # ============================================================
    # PART 6: RECOMMENDATIONS
    # ============================================================
    print("\n" + "=" * 70)
    print(" RECOMMENDATIONS")
    print("=" * 70)
    
    print("""
    FOR EMERSON/ADEBO PROBLEM:
    
    Option 1: Position-specific regression (recommended)
      - CB/S get 60-65% regression (vs current 50%)
      - EDGE gets 45% regression
      - Empirically justified: coverage harder to isolate
    
    Option 2: Cap individual WAR at 15-20% of team WAR
      - Prevents any one player from dominating
      - Simpler but less nuanced
    
    Option 3: Peer comparison
      - Compare to teammates at same position
      - If all CBs on team look elite, regress more
      - Complex to implement
    
    FOR FLATTENED WAR VALUES:
    
    Option 1: Use effective starters as divisor (~7-8)
      - Empirically derived from snap concentration
      - Would increase WAR values by ~40-50%
    
    Option 2: Position-group divisors
      - Pass defense ÷ 6, Run defense ÷ 5
      - More granular allocation
    
    Option 3: Simple divisor reduction
      - Change from 5.5 to 4.5 or 4.0
      - Quick fix, less principled
    
    COMBINED RECOMMENDATION:
    
    1. Reduce divisor to ~4.5 (based on effective starters analysis)
    2. Increase CB/S regression to 60%
    3. Keep EDGE/IDL at 50%
    
    This should:
    - Bump up overall WAR scale
    - Reduce Emerson/Adebo inflation
    - Maintain EDGE/IDL values (more individually attributable)
    """)
    
    print("\n" + "=" * 70)
    print(" DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()