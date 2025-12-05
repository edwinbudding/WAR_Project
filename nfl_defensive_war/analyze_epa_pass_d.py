"""
Analyze EPA on Pass Defense by Position
========================================

Similar to our run tackle analysis, let's see if we can empirically ground
coverage value by position.

Questions:
1. What's the EPA on plays where each position gets a pass defense?
2. Can we use this to adjust coverage value by position?
3. What other coverage metrics might help value shutdown corners?
4. What are empirical coverage multipliers for each position? (PART 6 - v19)
5. Should yards/coverage snap have position-specific adjustments? (PART 7 - v21)

Key finding for v19:
- LBs allow significantly higher QB rating than CBs/Ss
- This should translate to a coverage multiplier penalty for LBs
- Derived from actual PFF data, not arbitrary assumptions

Key finding for v21:
- Yards per coverage snap varies by position
- But existing coverage multipliers already account for this
- No need for additional position-specific yards adjustments

Limitations:
- pass_defense_1_player_id only captures plays where someone broke up the pass
- It doesn't capture "avoided entirely" or "completion allowed"
- True shutdown corner value requires tracking data or film grading
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/Users/anokhpalakurthi/Documents/nfl_defensive_war/data")
PBP_PATH = DATA_DIR / "pbp_2021_2024_full.csv"
PLAYER_PATH = Path("/Users/anokhpalakurthi/Documents/nfl_defensive_war/outputs/defense_engineered_2021_2024.csv")
CROSSWALK_PATH = DATA_DIR / "gsis_to_position_crosswalk.csv"


def main():
    print("=" * 70)
    print(" PASS DEFENSE EPA ANALYSIS BY POSITION")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading data...")
    pbp = pd.read_csv(PBP_PATH, low_memory=False)
    print(f"    Total plays: {len(pbp):,}")
    
    # Load crosswalk
    crosswalk = pd.read_csv(CROSSWALK_PATH)
    gsis_to_role = dict(zip(crosswalk['gsis_id'], crosswalk['role']))
    print(f"    Position crosswalk: {len(gsis_to_role):,} players")
    
    # Filter to pass plays
    if 'play_type' in pbp.columns:
        pass_plays = pbp[pbp['play_type'] == 'pass'].copy()
    else:
        pass_plays = pbp[pbp['pass'] == 1].copy()
    
    pass_plays = pass_plays.dropna(subset=['epa'])
    print(f"    Pass plays with EPA: {len(pass_plays):,}")
    
    # ==================== PART 1: PASS DEFENSE PLAYS ====================
    print("\n" + "=" * 70)
    print(" PART 1: EPA ON PASS BREAKUPS BY POSITION")
    print("=" * 70)
    
    pbu_position_epa = {}  # Store for later multiplier calculation
    
    if 'pass_defense_1_player_id' in pbp.columns:
        # Filter to plays with a pass defense
        pbu_plays = pass_plays[pass_plays['pass_defense_1_player_id'].notna()].copy()
        print(f"\n    Plays with pass defense: {len(pbu_plays):,}")
        
        # Map defender to position
        pbu_plays['defender_role'] = pbu_plays['pass_defense_1_player_id'].map(gsis_to_role)
        known_pbu = pbu_plays[pbu_plays['defender_role'].notna()]
        print(f"    With known defender position: {len(known_pbu):,}")
        
        if len(known_pbu) > 0:
            print("\n    EPA on Pass Breakups by Defender Position:")
            print("    (Negative offense EPA = good for defense)")
            
            for pos in ['CB', 'S', 'LB', 'EDGE', 'IDL']:
                subset = known_pbu[known_pbu['defender_role'] == pos]
                if len(subset) > 50:
                    off_epa_mean = subset['epa'].mean()
                    def_epa_mean = -off_epa_mean  # Flip for defense perspective
                    pbu_position_epa[pos] = def_epa_mean
                    print(f"      {pos}: n={len(subset):,}, Off EPA={off_epa_mean:.3f}, Def EPA={def_epa_mean:.3f}")
            
            # Derive PBU multipliers by position
            if len(pbu_position_epa) >= 3:
                print("\n    PBU VALUE BY POSITION:")
                cb_pbu_epa = pbu_position_epa.get('CB', 1.5)
                
                print(f"\n    Using CB as baseline (1.00x):")
                for pos in ['CB', 'S', 'LB', 'EDGE', 'IDL']:
                    if pos in pbu_position_epa:
                        mult = pbu_position_epa[pos] / cb_pbu_epa
                        print(f"      {pos}: {pbu_position_epa[pos]:.3f} EPA → {mult:.2f}x")
    
    # ==================== PART 2: INTERCEPTION VALUE ====================
    print("\n" + "=" * 70)
    print(" PART 2: INTERCEPTION EPA BY POSITION")
    print("=" * 70)
    
    if 'interception_player_id' in pbp.columns:
        int_plays = pass_plays[pass_plays['interception'] == 1].copy()
        int_plays['int_role'] = int_plays['interception_player_id'].map(gsis_to_role)
        known_int = int_plays[int_plays['int_role'].notna()]
        
        print(f"\n    Interceptions with known position: {len(known_int):,}")
        print("\n    EPA on Interceptions by Position:")
        
        for pos in ['CB', 'S', 'LB', 'EDGE', 'IDL']:
            subset = known_int[known_int['int_role'] == pos]
            if len(subset) > 20:
                epa_mean = subset['epa'].mean()
                # Defense EPA is negative of offense EPA
                print(f"      {pos}: n={len(subset):,}, Def EPA={-epa_mean:.3f}")
    
    # ==================== PART 3: COMPLETION ANALYSIS ====================
    print("\n" + "=" * 70)
    print(" PART 3: COMPLETION VS INCOMPLETION EPA")
    print("=" * 70)
    
    # Basic completion EPA analysis
    if 'complete_pass' in pass_plays.columns:
        completions = pass_plays[pass_plays['complete_pass'] == 1]
        incompletions = pass_plays[(pass_plays['complete_pass'] == 0) & 
                                   (pass_plays['interception'] == 0) &
                                   (pass_plays['sack'] == 0)]
        
        print(f"\n    Completions: n={len(completions):,}, Offense EPA={completions['epa'].mean():.3f}")
        print(f"    Incompletions: n={len(incompletions):,}, Offense EPA={incompletions['epa'].mean():.3f}")
        print(f"    Difference: {completions['epa'].mean() - incompletions['epa'].mean():.3f}")
        
        # By air yards if available
        if 'air_yards' in pass_plays.columns:
            print("\n    Completion EPA by Air Yards (depth of target):")
            pass_plays['air_yards_bin'] = pd.cut(pass_plays['air_yards'], 
                                                  bins=[-10, 0, 5, 10, 15, 20, 50],
                                                  labels=['Behind', '0-5', '5-10', '10-15', '15-20', '20+'])
            
            for bin_label in ['Behind', '0-5', '5-10', '10-15', '15-20', '20+']:
                bin_data = pass_plays[pass_plays['air_yards_bin'] == bin_label]
                if len(bin_data) > 100:
                    comp_rate = bin_data['complete_pass'].mean()
                    epa_mean = bin_data['epa'].mean()
                    print(f"      {bin_label} yards: n={len(bin_data):,}, Comp%={comp_rate:.1%}, EPA={epa_mean:.3f}")
    
    # ==================== PART 4: PFF COVERAGE DATA ====================
    print("\n" + "=" * 70)
    print(" PART 4: PFF COVERAGE METRICS BY POSITION")
    print("=" * 70)
    
    players = pd.read_csv(PLAYER_PATH)
    qualified = players[players['snap_counts_defense'] >= 200].copy()
    
    # Check what coverage columns we have
    cov_cols = [col for col in qualified.columns if any(x in col.lower() for x in 
                ['target', 'catch', 'yard', 'coverage', 'qb_rating'])]
    print(f"\n    Coverage-related columns: {cov_cols}")
    
    print("\n    Coverage Metrics by Position (>= 200 snaps):")
    
    for pos in ['CB', 'S', 'LB']:
        pos_df = qualified[qualified['role'] == pos]
        if len(pos_df) > 50:
            print(f"\n    {pos} (n={len(pos_df)}):")
            
            if 'targets' in pos_df.columns:
                print(f"      Avg targets: {pos_df['targets'].mean():.1f}")
            if 'catch_rate' in pos_df.columns:
                print(f"      Avg catch rate: {pos_df['catch_rate'].mean():.1%}")
            if 'yards' in pos_df.columns:
                print(f"      Avg yards allowed: {pos_df['yards'].mean():.1f}")
            if 'qb_rating_against' in pos_df.columns:
                print(f"      Avg QB rating against: {pos_df['qb_rating_against'].mean():.1f}")
            if 'snap_counts_coverage' in pos_df.columns and 'targets' in pos_df.columns:
                # Snaps per target (higher = avoided more)
                spt = pos_df['snap_counts_coverage'] / pos_df['targets'].replace(0, np.nan)
                print(f"      Avg snaps/target: {spt.mean():.1f}")
            if 'targets' in pos_df.columns and 'yards' in pos_df.columns:
                # Yards per target
                ypt = pos_df['yards'] / pos_df['targets'].replace(0, np.nan)
                print(f"      Avg yards/target: {ypt.mean():.1f}")
    
    # ==================== PART 5: SHUTDOWN CORNER ANALYSIS ====================
    print("\n" + "=" * 70)
    print(" PART 5: IDENTIFYING SHUTDOWN CORNERS")
    print("=" * 70)
    
    cbs = qualified[qualified['role'] == 'CB'].copy()
    
    if 'snap_counts_coverage' in cbs.columns and 'targets' in cbs.columns:
        cbs['snaps_per_target'] = cbs['snap_counts_coverage'] / cbs['targets'].replace(0, np.nan)
        cbs['yards_per_target'] = cbs['yards'] / cbs['targets'].replace(0, np.nan) if 'yards' in cbs.columns else np.nan
        
        # High snaps/target = being avoided (shutdown indicator)
        print("\n    Top 10 CBs by Snaps/Target (most avoided):")
        top_avoided = cbs.nlargest(10, 'snaps_per_target')[['player', 'season', 'team', 
            'snap_counts_coverage', 'targets', 'snaps_per_target', 'catch_rate', 'qb_rating_against']]
        print(top_avoided.to_string(index=False))
        
        # Composite shutdown score: high snaps/target + low catch rate + low QB rating
        cbs['shutdown_score'] = (
            (cbs['snaps_per_target'] - cbs['snaps_per_target'].mean()) / cbs['snaps_per_target'].std() +
            -(cbs['catch_rate'] - cbs['catch_rate'].mean()) / cbs['catch_rate'].std() +
            -(cbs['qb_rating_against'] - cbs['qb_rating_against'].mean()) / cbs['qb_rating_against'].std()
        ) / 3
        
        print("\n    Top 10 CBs by Composite Shutdown Score:")
        print("    (combines: avoided + low catch rate + low QB rating)")
        top_shutdown = cbs.nlargest(10, 'shutdown_score')[['player', 'season', 'team',
            'snaps_per_target', 'catch_rate', 'qb_rating_against', 'shutdown_score']]
        print(top_shutdown.to_string(index=False))
        
        # Compare to current WAR rankings
        print("\n    Checking if shutdown corners are properly valued in current model...")
        # This would need def_war column to compare
    
    # ==================== PART 6: POSITION-SPECIFIC COVERAGE EFFECTIVENESS ====================
    print("\n" + "=" * 70)
    print(" PART 6: COVERAGE EFFECTIVENESS BY POSITION (FOR v19 MULTIPLIERS)")
    print("=" * 70)
    
    print("""
    Goal: Derive empirical coverage multipliers by position.
    
    Problem: LBs get same credit for coverage snaps as CBs, but LBs are 
    objectively worse at coverage (higher QB rating against, higher catch rate).
    
    Approach: Use QB rating against to derive position multipliers.
    - League average QB rating is ~95
    - Below 95 = good coverage
    - Above 95 = bad coverage
    """)
    
    players = pd.read_csv(PLAYER_PATH)
    qualified = players[players['snap_counts_defense'] >= 200].copy()
    
    # Only players with significant coverage snaps
    cov_qualified = qualified[qualified['snap_counts_coverage'] >= 100].copy()
    
    print(f"\n    Players with >= 200 def snaps and >= 100 cov snaps: {len(cov_qualified)}")
    
    if 'qb_rating_against' in cov_qualified.columns:
        print("\n    QB RATING AGAINST BY POSITION:")
        print("    (Lower = better coverage)")
        
        position_stats = []
        for pos in ['CB', 'S', 'LB', 'EDGE', 'IDL']:
            pos_df = cov_qualified[cov_qualified['role'] == pos]
            if len(pos_df) >= 20:
                qbr_mean = pos_df['qb_rating_against'].mean()
                qbr_median = pos_df['qb_rating_against'].median()
                qbr_std = pos_df['qb_rating_against'].std()
                qbr_25 = pos_df['qb_rating_against'].quantile(0.25)
                qbr_75 = pos_df['qb_rating_against'].quantile(0.75)
                
                position_stats.append({
                    'position': pos,
                    'n': len(pos_df),
                    'mean_qbr': qbr_mean,
                    'median_qbr': qbr_median,
                    'std_qbr': qbr_std,
                    '25th': qbr_25,
                    '75th': qbr_75,
                })
                
                print(f"\n      {pos} (n={len(pos_df)}):")
                print(f"        Mean QBR:   {qbr_mean:.1f}")
                print(f"        Median QBR: {qbr_median:.1f}")
                print(f"        25th-75th:  {qbr_25:.1f} - {qbr_75:.1f}")
                print(f"        Std Dev:    {qbr_std:.1f}")
        
        # Calculate multipliers based on QB rating
        print("\n    DERIVING COVERAGE MULTIPLIERS:")
        print("    Method: Use inverse of QB rating relative to CB baseline")
        
        pos_stats_df = pd.DataFrame(position_stats)
        
        # CB is baseline (best coverage position)
        cb_qbr = pos_stats_df[pos_stats_df['position'] == 'CB']['mean_qbr'].values[0]
        
        print(f"\n    CB baseline QBR: {cb_qbr:.1f}")
        print("\n    Position multipliers (CB = 1.00):")
        
        for _, row in pos_stats_df.iterrows():
            pos = row['position']
            qbr = row['mean_qbr']
            # Multiplier = CB_QBR / position_QBR (lower QBR = better = higher multiplier)
            # But we want to penalize worse coverage, so:
            # multiplier = CB_QBR / position_QBR
            multiplier = cb_qbr / qbr
            print(f"      {pos}: {qbr:.1f} QBR → {multiplier:.2f}x multiplier")
        
        # Alternative: EPA-equivalent approach
        print("\n    ALTERNATIVE: EPA-equivalent multipliers")
        print("    Method: Convert QBR difference to EPA impact")
        print("""
    QB Rating formula estimates expected points per pass attempt.
    Every 10 points of QB rating ≈ 0.1 EPA difference per target.
    
    If CB allows 95 QBR and LB allows 101 QBR:
      Difference = 6 points → ~0.06 EPA/target worse
      Over 40 targets: 2.4 EPA difference per season
        """)
        
        # Calculate EPA-based multipliers
        print("\n    EPA-ADJUSTED MULTIPLIERS:")
        baseline_qbr = 95.0  # League average
        
        for _, row in pos_stats_df.iterrows():
            pos = row['position']
            qbr = row['mean_qbr']
            qbr_diff = qbr - baseline_qbr
            # Every 10 QBR above baseline = 0.10 reduction in value
            epa_multiplier = 1.0 - (qbr_diff / 10.0) * 0.10
            epa_multiplier = max(0.5, min(1.2, epa_multiplier))  # Clip to reasonable range
            print(f"      {pos}: {qbr:.1f} QBR (diff: {qbr_diff:+.1f}) → {epa_multiplier:.2f}x")
        
        # Final recommended multipliers
        print("\n    RECOMMENDED COVERAGE MULTIPLIERS FOR v19:")
        print("    (Based on QB rating relative to league average)")
        
        for _, row in pos_stats_df.iterrows():
            pos = row['position']
            qbr = row['mean_qbr']
            
            # Empirical multiplier: penalize positions with worse coverage
            if pos == 'CB':
                mult = 1.00
            elif pos == 'S':
                mult = cb_qbr / qbr  # Slight penalty if S is worse than CB
            elif pos == 'LB':
                mult = cb_qbr / qbr  # Bigger penalty for LBs
            else:
                mult = 0.50  # EDGE/IDL shouldn't get much coverage credit
            
            mult = round(mult, 2)
            print(f"      {pos}: {mult:.2f}x")
    
    # Also look at catch rate by position
    if 'catch_rate' in cov_qualified.columns:
        print("\n    CATCH RATE BY POSITION:")
        print("    (Lower = better coverage)")
        
        for pos in ['CB', 'S', 'LB']:
            pos_df = cov_qualified[cov_qualified['role'] == pos]
            if len(pos_df) >= 20:
                cr_mean = pos_df['catch_rate'].mean()
                print(f"      {pos}: {cr_mean:.1%} catch rate allowed")
    
    # Yards per target
    if 'yards' in cov_qualified.columns and 'targets' in cov_qualified.columns:
        print("\n    YARDS PER TARGET BY POSITION:")
        print("    (Lower = better coverage)")
        
        for pos in ['CB', 'S', 'LB']:
            pos_df = cov_qualified[cov_qualified['role'] == pos]
            if len(pos_df) >= 20:
                pos_df = pos_df[pos_df['targets'] > 10]  # Minimum targets
                ypt = (pos_df['yards'] / pos_df['targets']).mean()
                print(f"      {pos}: {ypt:.1f} yards/target allowed")

    print("\n" + "=" * 70)
    print(" SUMMARY: OPTIONS FOR BETTER COVERAGE VALUATION")
    print("=" * 70)
    
    print("""
    Current approach:
      - Coverage snaps (55%) + PBUs (45%)
      - Lockdown adjustment: snaps/target × catch rate inverse
      - LB liability: QB rating against penalty
    
    Potential improvements:
    
    1. YARDS PER TARGET as efficiency metric
       - Directly measures "when targeted, how much damage?"
       - Problem: doesn't capture "avoided entirely"
    
    2. QB RATING AGAINST (already have for LBs)
       - Could apply to CBs/Ss too, not just LBs
       - Comprehensive: comp%, yards, TDs, INTs
    
    3. MORE AGGRESSIVE LOCKDOWN MULTIPLIER
       - Current range: 0.6x to 1.4x
       - Could widen to 0.4x to 1.8x for bigger separation
    
    4. COMPOSITE SHUTDOWN SCORE
       - Combine snaps/target + catch_rate + qb_rating
       - Use as coverage multiplier
    
    5. EPA-BASED (limited by data)
       - Only have EPA on pass breakups, not all targets
       - Would need tracking data for true coverage EPA
    
    The fundamental limitation: Without tracking data, we can't know
    who was "supposed to" cover a receiver on completions. We only
    know who broke up incomplete passes.
    """)

    # ==================== PART 7: YARDS PER COVERAGE SNAP BY POSITION ====================
    print("\n" + "=" * 70)
    print(" PART 7: YARDS PER COVERAGE SNAP BY POSITION (FOR v21 EFFICIENCY)")
    print("=" * 70)
    
    print("""
    Goal: Determine if yards/coverage snap should have position-specific
    adjustments, similar to how we derived coverage multipliers from catch rate.
    
    Question: Is 0.8 yards/snap equally "good" for a CB vs a LB vs a Safety?
    
    Hypothesis: LBs may allow more yards/snap because they cover TEs/RBs
    in shorter routes, while CBs face WRs on deeper routes.
    """)
    
    players = pd.read_csv(PLAYER_PATH)
    qualified = players[players['snap_counts_defense'] >= 200].copy()
    
    # Only players with significant coverage snaps
    cov_qualified = qualified[qualified['snap_counts_coverage'] >= 100].copy()
    
    # Calculate yards per coverage snap
    cov_qualified['yards_per_cov_snap'] = np.where(
        cov_qualified['snap_counts_coverage'] > 0,
        cov_qualified['yards'] / cov_qualified['snap_counts_coverage'],
        np.nan
    )
    
    print(f"\n    Players with >= 200 def snaps and >= 100 cov snaps: {len(cov_qualified)}")
    
    print("\n    YARDS PER COVERAGE SNAP BY POSITION:")
    print("    (Lower = better coverage efficiency)")
    
    position_yards_stats = []
    for pos in ['CB', 'S', 'LB', 'EDGE', 'IDL']:
        pos_df = cov_qualified[cov_qualified['role'] == pos]
        pos_df = pos_df[pos_df['yards_per_cov_snap'].notna()]
        
        if len(pos_df) >= 20:
            yps_mean = pos_df['yards_per_cov_snap'].mean()
            yps_median = pos_df['yards_per_cov_snap'].median()
            yps_std = pos_df['yards_per_cov_snap'].std()
            yps_25 = pos_df['yards_per_cov_snap'].quantile(0.25)
            yps_75 = pos_df['yards_per_cov_snap'].quantile(0.75)
            yps_10 = pos_df['yards_per_cov_snap'].quantile(0.10)
            yps_90 = pos_df['yards_per_cov_snap'].quantile(0.90)
            
            position_yards_stats.append({
                'position': pos,
                'n': len(pos_df),
                'mean_yps': yps_mean,
                'median_yps': yps_median,
                'std_yps': yps_std,
                '10th': yps_10,
                '25th': yps_25,
                '75th': yps_75,
                '90th': yps_90,
            })
            
            print(f"\n      {pos} (n={len(pos_df)}):")
            print(f"        Mean:       {yps_mean:.3f} yards/snap")
            print(f"        Median:     {yps_median:.3f} yards/snap")
            print(f"        10th-90th:  {yps_10:.3f} - {yps_90:.3f}")
            print(f"        25th-75th:  {yps_25:.3f} - {yps_75:.3f}")
            print(f"        Std Dev:    {yps_std:.3f}")
    
    # Create DataFrame for analysis
    pos_yards_df = pd.DataFrame(position_yards_stats)
    
    # Calculate position multipliers based on yards/snap
    print("\n    DERIVING YARDS EFFICIENCY MULTIPLIERS:")
    print("    Method: Use CB as baseline, adjust other positions")
    
    cb_yps = pos_yards_df[pos_yards_df['position'] == 'CB']['mean_yps'].values[0]
    
    print(f"\n    CB baseline yards/snap: {cb_yps:.3f}")
    print("\n    Option 1: Inverse ratio multipliers (penalize worse efficiency)")
    print("    Logic: Higher yards/snap = worse coverage = lower multiplier")
    
    for _, row in pos_yards_df.iterrows():
        pos = row['position']
        yps = row['mean_yps']
        # Inverse ratio: lower yards/snap = higher multiplier
        # multiplier = CB_yps / position_yps
        # But we want: LOWER yards = BETTER, so actually:
        # If LB allows MORE yards, they should get LESS credit
        # multiplier = CB_yps / position_yps (if pos_yps > cb_yps, mult < 1)
        if yps > 0:
            mult_inverse = cb_yps / yps
            print(f"      {pos}: {yps:.3f} yds/snap → {mult_inverse:.3f}x multiplier")
    
    print("\n    Option 2: Standardized difference multipliers")
    print("    Logic: Convert yards/snap difference to standard deviations")
    
    overall_mean = cov_qualified['yards_per_cov_snap'].mean()
    overall_std = cov_qualified['yards_per_cov_snap'].std()
    
    print(f"\n    Overall mean: {overall_mean:.3f}, std: {overall_std:.3f}")
    
    for _, row in pos_yards_df.iterrows():
        pos = row['position']
        yps = row['mean_yps']
        z_score = (yps - overall_mean) / overall_std
        # Convert z-score to multiplier: 
        # z = 0 → 1.0x, z = +1 (worse) → 0.9x, z = -1 (better) → 1.1x
        mult_zscore = 1.0 - (z_score * 0.10)
        mult_zscore = max(0.5, min(1.5, mult_zscore))
        print(f"      {pos}: {yps:.3f} yds/snap (z={z_score:+.2f}) → {mult_zscore:.2f}x")
    
    # Compare with existing catch rate multipliers
    print("\n    COMPARISON WITH EXISTING CATCH RATE MULTIPLIERS:")
    print("    Current v21 coverage multipliers from catch rate analysis:")
    print("      CB: 1.00x")
    print("      S:  0.94x")
    print("      LB: 0.81x")
    
    print("\n    Yards/snap derived multipliers:")
    for _, row in pos_yards_df.iterrows():
        pos = row['position']
        yps = row['mean_yps']
        if pos in ['CB', 'S', 'LB']:
            mult = cb_yps / yps if yps > 0 else 1.0
            print(f"      {pos}: {mult:.2f}x (from yards/snap)")
    
    # Correlation between yards/snap and other metrics
    print("\n    CORRELATION: YARDS/SNAP vs OTHER COVERAGE METRICS:")
    
    cov_positions = cov_qualified[cov_qualified['role'].isin(['CB', 'S', 'LB'])]
    
    if 'catch_rate' in cov_positions.columns:
        corr_cr = cov_positions['yards_per_cov_snap'].corr(cov_positions['catch_rate'])
        print(f"      Yards/snap vs Catch Rate: r = {corr_cr:.3f}")
    
    if 'qb_rating_against' in cov_positions.columns:
        corr_qbr = cov_positions['yards_per_cov_snap'].corr(cov_positions['qb_rating_against'])
        print(f"      Yards/snap vs QB Rating:  r = {corr_qbr:.3f}")
    
    if 'targets' in cov_positions.columns:
        cov_positions = cov_positions.copy()
        cov_positions['yards_per_target'] = cov_positions['yards'] / cov_positions['targets'].replace(0, np.nan)
        corr_ypt = cov_positions['yards_per_cov_snap'].corr(cov_positions['yards_per_target'])
        print(f"      Yards/snap vs Yards/target: r = {corr_ypt:.3f}")
    
    # Best and worst by position for yards/snap
    print("\n    TOP 5 BEST YARDS/SNAP BY POSITION (lowest = best):")
    for pos in ['CB', 'S', 'LB']:
        pos_df = cov_qualified[cov_qualified['role'] == pos].nsmallest(5, 'yards_per_cov_snap')
        print(f"\n      {pos}:")
        for _, row in pos_df.iterrows():
            print(f"        {row['player']} ({row['season']} {row['team']}): {row['yards_per_cov_snap']:.3f}")
    
    print("\n    TOP 5 WORST YARDS/SNAP BY POSITION (highest = worst):")
    for pos in ['CB', 'S', 'LB']:
        pos_df = cov_qualified[cov_qualified['role'] == pos].nlargest(5, 'yards_per_cov_snap')
        print(f"\n      {pos}:")
        for _, row in pos_df.iterrows():
            print(f"        {row['player']} ({row['season']} {row['team']}): {row['yards_per_cov_snap']:.3f}")
    
    # Final recommendation
    print("\n    " + "=" * 60)
    print("    RECOMMENDATION FOR v21 YARDS EFFICIENCY MULTIPLIERS")
    print("    " + "=" * 60)
    
    print("""
    Finding: Yards per coverage snap DOES vary significantly by position.
    
    However, our existing coverage multipliers (derived from catch rate)
    already capture most of this positional difference:
    
    The yards/snap metric in v21 already gets the coverage multipliers
    applied to it (CB 1.00x, S 0.94x, LB 0.81x).
    
    Options:
    
    A) KEEP CURRENT APPROACH (recommended):
       - Apply existing coverage multipliers to yards_efficiency metric
       - This already penalizes LBs for worse coverage ability
       - Simpler, empirically grounded
    
    B) ADD POSITION-SPECIFIC YARDS BASELINE:
       - Instead of raw yards/snap, use yards vs position average
       - CB with 0.9 yds/snap vs CB avg of 0.95 = good
       - LB with 0.9 yds/snap vs LB avg of 0.75 = bad
       - More complex, might double-penalize LBs
    
    C) COMPOSITE MULTIPLIER:
       - Average the catch rate multiplier and yards/snap multiplier
       - CB: (1.00 + 1.00) / 2 = 1.00x
       - S:  (0.94 + S_yps_mult) / 2
       - LB: (0.81 + LB_yps_mult) / 2
    
    Current implementation already handles this reasonably well.
    The key insight is that the efficiency metric (yards/snap) WITHIN
    each position separates good from bad players, while the coverage
    multipliers adjust for baseline positional differences.
    """)

    # ==================== PART 8: TACKLE VALUE IN PASS DEFENSE ====================
    print("\n" + "=" * 70)
    print(" PART 8: TACKLE VALUE IN PASS DEFENSE (YAC LIMITATION)")
    print("=" * 70)
    
    print("""
    Goal: Determine if tackles after completions should be valued in pass defense.
    
    Current model: We credit coverage snaps, PBUs, INTs, yards efficiency.
    Missing: Credit for limiting YAC (yards after catch) via tackling.
    
    A corner who allows a catch but tackles immediately is more valuable
    than one who allows the same catch plus 15 YAC.
    """)
    
    # Load play-by-play for YAC analysis
    pbp = pd.read_csv(PBP_PATH, low_memory=False)
    
    # Filter to completed passes
    if 'complete_pass' in pbp.columns:
        completions = pbp[(pbp['complete_pass'] == 1) & (pbp['epa'].notna())].copy()
    else:
        completions = pbp[(pbp['pass'] == 1) & (pbp['epa'].notna())].copy()
    
    print(f"\n    Completed passes with EPA: {len(completions):,}")
    
    # Check for YAC columns
    yac_cols = [col for col in pbp.columns if 'yac' in col.lower() or 'after_catch' in col.lower()]
    print(f"    YAC-related columns: {yac_cols}")
    
    if 'yards_after_catch' in completions.columns:
        completions['yac'] = completions['yards_after_catch']
    elif 'yac' in completions.columns:
        completions['yac'] = completions['yac']
    else:
        print("    WARNING: No YAC column found, skipping YAC analysis")
        completions['yac'] = np.nan
    
    if completions['yac'].notna().sum() > 1000:
        print("\n    EPA BY YAC BUCKET (on completions):")
        print("    Logic: Lower YAC = better tackling = more defensive value")
        
        # Create YAC buckets
        completions['yac_bucket'] = pd.cut(
            completions['yac'],
            bins=[-np.inf, 0, 3, 6, 10, 15, np.inf],
            labels=['0 or less', '1-3', '4-6', '7-10', '11-15', '16+']
        )
        
        yac_results = []
        for bucket in ['0 or less', '1-3', '4-6', '7-10', '11-15', '16+']:
            subset = completions[completions['yac_bucket'] == bucket]
            if len(subset) > 100:
                epa_mean = subset['epa'].mean()
                yac_mean = subset['yac'].mean()
                count = len(subset)
                yac_results.append({
                    'bucket': bucket,
                    'n': count,
                    'off_epa': epa_mean,
                    'def_epa': -epa_mean,
                    'avg_yac': yac_mean
                })
                print(f"      {bucket} YAC: n={count:,}, Off EPA={epa_mean:.3f}, Def EPA={-epa_mean:.3f}")
        
        if len(yac_results) >= 2:
            yac_df = pd.DataFrame(yac_results)
            
            # Calculate value of limiting YAC
            print("\n    VALUE OF LIMITING YAC:")
            
            # Compare 0-3 YAC (good tackle) vs 7+ YAC (poor tackle)
            good_tackle = yac_df[yac_df['bucket'].isin(['0 or less', '1-3'])]['def_epa'].mean()
            poor_tackle = yac_df[yac_df['bucket'].isin(['7-10', '11-15', '16+'])]['def_epa'].mean()
            
            print(f"      Good tackle (0-3 YAC): {good_tackle:.3f} def EPA")
            print(f"      Poor tackle (7+ YAC):  {poor_tackle:.3f} def EPA")
            print(f"      Difference: {good_tackle - poor_tackle:.3f} def EPA per play")
            
            # EPA per YAC yard
            # Regression: EPA vs YAC
            valid_yac = completions[completions['yac'].notna() & completions['epa'].notna()]
            if len(valid_yac) > 1000:
                corr = valid_yac['yac'].corr(valid_yac['epa'])
                print(f"\n      Correlation (YAC vs Off EPA): r = {corr:.3f}")
                
                # Simple slope calculation
                yac_std = valid_yac['yac'].std()
                epa_std = valid_yac['epa'].std()
                slope = corr * (epa_std / yac_std)
                print(f"      Approx EPA per YAC yard: {slope:.3f}")
                print(f"      → Each YAC yard prevented saves ~{-slope:.3f} EPA")
        
        # Tackle after catch by position
        print("\n    TACKLING AFTER CATCH BY POSITION:")
        
        # Load crosswalk
        crosswalk = pd.read_csv(CROSSWALK_PATH)
        gsis_to_role = dict(zip(crosswalk['gsis_id'], crosswalk['role']))
        
        # Find tackle column
        tackle_col = None
        for col in ['solo_tackle_1_player_id', 'tackle_for_loss_1_player_id']:
            if col in completions.columns:
                tackle_col = col
                break
        
        if tackle_col:
            completions['tackler_role'] = completions[tackle_col].map(gsis_to_role)
            known_tackles = completions[completions['tackler_role'].notna()]
            
            print(f"      Completions with known tackler: {len(known_tackles):,}")
            
            for pos in ['CB', 'S', 'LB', 'EDGE', 'IDL']:
                subset = known_tackles[known_tackles['tackler_role'] == pos]
                if len(subset) > 50:
                    yac_mean = subset['yac'].mean()
                    epa_mean = subset['epa'].mean()
                    print(f"        {pos}: n={len(subset):,}, Avg YAC={yac_mean:.1f}, Off EPA={epa_mean:.3f}")
        
        # Derive tackle weight for pass defense
        print("\n    " + "=" * 60)
        print("    DERIVING TACKLE WEIGHT FOR PASS DEFENSE")
        print("    " + "=" * 60)
        
        # Load player data for frequency
        players = pd.read_csv(PLAYER_PATH)
        qualified = players[players['snap_counts_defense'] >= 200].copy()
        coverage_players = qualified[qualified['role'].isin(['CB', 'S', 'LB'])]
        
        # Get average tackles per player
        if 'tackles' in coverage_players.columns:
            avg_tackles = coverage_players['tackles'].mean()
        elif 'solo_tackles' in coverage_players.columns:
            avg_tackles = coverage_players['solo_tackles'].mean()
        else:
            avg_tackles = 40  # Estimate
        
        # Estimate tackles on pass plays (roughly 55-60% of plays are passes)
        avg_pass_tackles = avg_tackles * 0.55
        
        print(f"\n      Avg tackles per player-season: {avg_tackles:.1f}")
        print(f"      Estimated pass play tackles: {avg_pass_tackles:.1f}")
        
        # Value per tackle (YAC limitation)
        # Average completion allows X YAC, good tackle limits to Y YAC
        avg_yac = completions['yac'].mean()
        good_yac = completions[completions['yac'] <= 3]['yac'].mean()
        yac_saved = avg_yac - good_yac
        
        # EPA value of YAC saved
        if 'slope' in dir():
            tackle_value = yac_saved * (-slope)
        else:
            tackle_value = yac_saved * 0.05  # Estimate ~0.05 EPA per yard
        
        print(f"\n      Avg YAC on completions: {avg_yac:.1f}")
        print(f"      Good tackle YAC (<=3): {good_yac:.1f}")
        print(f"      YAC saved per good tackle: {yac_saved:.1f}")
        print(f"      EPA value per tackle: {tackle_value:.3f}")
        
        # Compare to other pass defense activities
        print("\n    TACKLE VALUE VS OTHER PASS DEFENSE ACTIVITIES:")
        
        # PBU value (from Part 1)
        pbu_epa = 1.5  # Approximate from earlier analysis
        int_epa = 4.5  # Approximate from earlier analysis
        
        # Coverage snap value (harder to quantify, but small per snap)
        cov_snap_value = 0.01  # Rough estimate
        
        print(f"      PBU: ~{pbu_epa:.2f} def EPA")
        print(f"      INT: ~{int_epa:.2f} def EPA")
        print(f"      Tackle (YAC limit): ~{tackle_value:.3f} def EPA")
        print(f"      Coverage snap: ~{cov_snap_value:.3f} def EPA")
        
        # Season contribution
        avg_pbus = coverage_players['pbus'].mean() if 'pbus' in coverage_players.columns else 5
        avg_ints = coverage_players['interceptions'].mean() if 'interceptions' in coverage_players.columns else 1
        avg_cov_snaps = coverage_players['snap_counts_coverage'].mean() if 'snap_counts_coverage' in coverage_players.columns else 400
        
        pbu_total = avg_pbus * pbu_epa
        int_total = avg_ints * int_epa
        tackle_total = avg_pass_tackles * tackle_value
        cov_total = avg_cov_snaps * cov_snap_value
        
        total_pass_def = pbu_total + int_total + tackle_total + cov_total
        
        print(f"\n    SEASON VALUE CONTRIBUTION (pass defense):")
        print(f"      PBUs: {avg_pbus:.1f} × {pbu_epa:.2f} = {pbu_total:.1f} EPA ({pbu_total/total_pass_def:.1%})")
        print(f"      INTs: {avg_ints:.1f} × {int_epa:.2f} = {int_total:.1f} EPA ({int_total/total_pass_def:.1%})")
        print(f"      Tackles: {avg_pass_tackles:.1f} × {tackle_value:.3f} = {tackle_total:.1f} EPA ({tackle_total/total_pass_def:.1%})")
        print(f"      Coverage: {avg_cov_snaps:.0f} × {cov_snap_value:.3f} = {cov_total:.1f} EPA ({cov_total/total_pass_def:.1%})")
        
        print(f"\n    RECOMMENDED TACKLE WEIGHT FOR PASS DEFENSE:")
        tackle_pct = tackle_total / total_pass_def * 100
        print(f"      Empirically derived: {tackle_pct:.1f}%")
        print(f"      (This would come from the 69% pass defense allocation)")
    
    else:
        print("    Insufficient YAC data for analysis")

    # ==================== PART 9: PASS RUSH SNAPS VS PRODUCTION ====================
    print("\n" + "=" * 70)
    print(" PART 9: PASS RUSH SNAPS VS PRODUCTION")
    print("=" * 70)
    
    print("""
    Question: Should we credit pass rush snaps like we credit coverage snaps?
    
    Current model:
      - Coverage snaps get 35% weight (presence = deterrence value)
      - Pass rush snaps get 0% weight (only sacks/hits/hurries count)
    
    Concern: Adding snap credit might overrate players on good defenses
    who accumulate snaps without production (the "Emerson problem").
    
    Analysis: Look at relationship between snaps and production by position.
    """)
    
    # Load player data
    players = pd.read_csv(PLAYER_PATH)
    qualified = players[players['snap_counts_defense'] >= 200].copy()
    
    print(f"\n    Players with >= 200 defensive snaps: {len(qualified)}")
    
    # Check for pass rush snap column
    snap_cols = [c for c in qualified.columns if 'snap' in c.lower()]
    print(f"    Snap columns available: {snap_cols}")
    
    if 'snap_counts_pass_rush' not in qualified.columns:
        print("    ERROR: snap_counts_pass_rush column not found")
        return
    
    print(f"\n    Pass rush snap column found: snap_counts_pass_rush")
    
    # Calculate production rates
    # Check what pressure columns we have
    pressure_cols = [c for c in qualified.columns if any(x in c.lower() for x in ['sack', 'hit', 'hurr', 'pressure'])]
    print(f"    Pressure columns available: {pressure_cols}")
    
    # Use available columns
    if 'sacks' in qualified.columns:
        qualified['sacks_col'] = qualified['sacks']
    elif 'sack' in qualified.columns:
        qualified['sacks_col'] = qualified['sack']
    else:
        qualified['sacks_col'] = 0
    
    if 'hits' in qualified.columns:
        qualified['hits_col'] = qualified['hits']
    elif 'qb_hits' in qualified.columns:
        qualified['hits_col'] = qualified['qb_hits']
    else:
        qualified['hits_col'] = 0
    
    if 'hurries' in qualified.columns:
        qualified['hurries_col'] = qualified['hurries']
    else:
        qualified['hurries_col'] = 0
    
    # Total pressures
    qualified['total_pressures'] = qualified['sacks_col'] + qualified['hits_col'] + qualified['hurries_col']
    qualified['pressures_per_snap'] = qualified['total_pressures'] / qualified['snap_counts_pass_rush'].replace(0, np.nan)
    qualified['sacks_per_snap'] = qualified['sacks_col'] / qualified['snap_counts_pass_rush'].replace(0, np.nan)
    
    print("\n    PASS RUSH SNAPS BY POSITION:")
    
    position_snap_stats = []
    for pos in ['EDGE', 'IDL', 'LB', 'CB', 'S']:
        pos_df = qualified[qualified['role'] == pos]
        if len(pos_df) >= 20:
            snap_mean = pos_df['snap_counts_pass_rush'].mean()
            sacks_mean = pos_df['sacks_col'].mean()
            hits_mean = pos_df['hits_col'].mean()
            hurries_mean = pos_df['hurries_col'].mean()
            pressures_mean = pos_df['total_pressures'].mean()
            pressure_rate = pos_df['pressures_per_snap'].mean()
            sack_rate = pos_df['sacks_per_snap'].mean()
            
            position_snap_stats.append({
                'pos': pos,
                'n': len(pos_df),
                'avg_snaps': snap_mean,
                'avg_sacks': sacks_mean,
                'avg_pressures': pressures_mean,
                'pressure_rate': pressure_rate,
                'sack_rate': sack_rate
            })
            
            print(f"\n      {pos} (n={len(pos_df)}):")
            print(f"        Avg pass rush snaps: {snap_mean:.1f}")
            print(f"        Avg sacks: {sacks_mean:.1f}, hits: {hits_mean:.1f}, hurries: {hurries_mean:.1f}")
            print(f"        Pressure rate: {pressure_rate:.4f} per snap")
            print(f"        Sack rate: {sack_rate:.4f} per snap")
    
    pos_snap_df = pd.DataFrame(position_snap_stats)
    
    # Correlation: Do more snaps = more production?
    print("\n    CORRELATION: SNAPS VS PRODUCTION")
    print("    (Does playing more = producing more, or just being on field?)")
    
    for pos in ['EDGE', 'IDL', 'LB']:
        pos_df = qualified[qualified['role'] == pos]
        if len(pos_df) >= 30:
            corr_sacks = pos_df['snap_counts_pass_rush'].corr(pos_df['sacks_col'])
            corr_pressures = pos_df['snap_counts_pass_rush'].corr(pos_df['total_pressures'])
            
            print(f"\n      {pos}:")
            print(f"        Snaps vs Sacks: r = {corr_sacks:.3f}")
            print(f"        Snaps vs Total Pressures: r = {corr_pressures:.3f}")
    
    # Key question: Do high-snap, low-production players exist?
    print("\n    HIGH-SNAP, LOW-PRODUCTION PLAYERS:")
    print("    (Players who might get inflated value from snap-based credit)")
    
    for pos in ['EDGE', 'IDL']:
        pos_df = qualified[qualified['role'] == pos].copy()
        if len(pos_df) >= 30:
            # High snaps = top 25%, low production = bottom 25%
            snap_75 = pos_df['snap_counts_pass_rush'].quantile(0.75)
            prod_25 = pos_df['pressures_per_snap'].quantile(0.25)
            
            high_snap_low_prod = pos_df[
                (pos_df['snap_counts_pass_rush'] >= snap_75) & 
                (pos_df['pressures_per_snap'] <= prod_25)
            ]
            
            print(f"\n      {pos}: {len(high_snap_low_prod)} players with high snaps, low production")
            if len(high_snap_low_prod) > 0 and len(high_snap_low_prod) <= 5:
                for _, row in high_snap_low_prod.iterrows():
                    print(f"        - {row['player']} ({row['season']} {row['team']}): "
                          f"{row['snap_counts_pass_rush']:.0f} snaps, "
                          f"{row['sacks_col']:.1f} sacks, {row['total_pressures']:.0f} pressures")
    
    # Compare: Production rate by team defensive quality
    print("\n    PRODUCTION RATE BY TEAM CONTEXT:")
    print("    (Do players on good defenses have inflated or deflated production rates?)")
    
    if 'team' in qualified.columns and 'season' in qualified.columns:
        team_seasons = qualified.groupby(['team', 'season']).agg({
            'snap_counts_pass_rush': 'sum',
            'sacks_col': 'sum',
            'total_pressures': 'sum'
        }).reset_index()
        team_seasons.columns = ['team', 'season', 'team_pr_snaps', 'team_sacks', 'team_pressures']
        
        qualified = qualified.merge(team_seasons, on=['team', 'season'], how='left', suffixes=('', '_team'))
        
        # Team pressure rate
        qualified['team_pressure_rate'] = qualified['team_pressures'] / qualified['team_pr_snaps']
        
        # Split into good vs bad pass rushes
        team_pr_median = qualified['team_pressure_rate'].median()
        good_pr = qualified[qualified['team_pressure_rate'] >= team_pr_median]
        bad_pr = qualified[qualified['team_pressure_rate'] < team_pr_median]
        
        print(f"\n      Good pass rush teams (above median team pressure rate):")
        print(f"        Avg individual pressure rate: {good_pr['pressures_per_snap'].mean():.4f}")
        print(f"      Bad pass rush teams (below median team pressure rate):")
        print(f"        Avg individual pressure rate: {bad_pr['pressures_per_snap'].mean():.4f}")
        
        ratio = good_pr['pressures_per_snap'].mean() / bad_pr['pressures_per_snap'].mean()
        print(f"      Ratio (good/bad): {ratio:.2f}x")
        
        if ratio > 1.1:
            print(f"\n      → Players on good pass rush teams have HIGHER pressure rates")
            print(f"      → Adding snap credit would compound the 'good defense' advantage")
            print(f"      → Recommend AGAINST adding pass rush snaps, or use lower weight")
        elif ratio < 0.9:
            print(f"\n      → Players on good pass rush teams have LOWER pressure rates")
            print(f"      → Snap credit might help balance opportunity differences")
            print(f"      → Could consider adding pass rush snaps")
        else:
            print(f"\n      → Pressure rates similar regardless of team quality")
            print(f"      → Snap credit would be neutral re: team context")
    
    # Compare to coverage snaps logic
    print("\n    COMPARISON: PASS RUSH VS COVERAGE SNAP LOGIC")
    
    print("""
    Coverage snaps rationale:
      - A CB who isn't targeted is providing value (deterrence)
      - "Not being thrown at" is a measurable positive signal
      - Snaps without targets = doing your job
    
    Pass rush snaps question:
      - Is a pass rusher who isn't getting pressures providing value?
      - "Occupying blockers" is harder to measure
      - Snaps without pressures = ??? (value or failure?)
    
    Key difference:
      - Coverage: Can measure "avoided" (low targets per snap)
      - Pass rush: No clear "avoided" equivalent
      - A double-teamed DT is valuable but we can't easily measure it
    """)
    
    # Recommendation
    print("\n    " + "=" * 60)
    print("    RECOMMENDATION: PASS RUSH SNAPS")
    print("    " + "=" * 60)
    
    print("""
    Key findings:
    
    1. SNAPS VS PRODUCTION CORRELATION
       - If high (r > 0.7): Snaps ≈ production, snap credit is redundant
       - If moderate (0.4-0.7): Snaps partially capture value
       - If low (r < 0.4): Snaps and production measure different things
    
    2. TEAM CONTEXT EFFECT
       - If good team players have higher rates: Don't add snaps (Emerson problem)
       - If rates are equal: Snaps would be neutral
       - If good team players have lower rates: Snaps could help
    
    3. POSITION DIFFERENCES
       - IDL: Most likely to benefit from snap credit (occupying blockers, two-gapping)
       - EDGE: Less clear (expected to produce pressures)
    
    Options:
    
    A) NO SNAP CREDIT (current approach):
       - Keeps it simple
       - Avoids Emerson problem
       - But misses "occupying blockers" value for IDL
    
    B) IDL-ONLY SNAP CREDIT:
       - Add snap_counts_pass_rush at modest weight for IDL only
       - Recognizes their unique role as space-eaters
       - EDGE still judged on production
    
    C) SMALL SNAP CREDIT FOR ALL (10-15%):
       - Add snap_counts_pass_rush at modest weight
       - Apply position multipliers (IDL 1.0x, EDGE 0.5x)
       - Still primarily rewards production
    
    D) EFFICIENCY-BASED:
       - Don't credit snaps directly
       - Use pressure_rate as efficiency metric
       - Rewards productive players regardless of opportunity
    """)

    print("\n" + "=" * 70)
    print(" DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()