"""
Combined EPA Analysis - Run Defense
====================================

Analyses in this script:
1. EPA by Play Type (pass vs run)
2. EPA by Tackler Position (to ground position-specific run defense value)
3. TFL vs Regular Stop EPA (for allocation weights)
4. Run defense allocation weight derivation
5. NEW: Tackle Value by Depth (should we credit non-stop tackles?)

Run this on your machine:
    python analyze_epa_run_d.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Adjust paths as needed
DATA_DIR = Path("/Users/anokhpalakurthi/Documents/nfl_defensive_war/data")
PBP_PATH = DATA_DIR / "pbp_2021_2024_full.csv"
PLAYER_PATH = Path("/Users/anokhpalakurthi/Documents/nfl_defensive_war/outputs/defense_engineered_2021_2024.csv")
CROSSWALK_PATH = DATA_DIR / "gsis_to_position_crosswalk.csv"


def analyze_play_type(pbp):
    """Analyze EPA by play type (pass vs run)."""
    print("\n" + "=" * 70)
    print(" PART 1: EPA ANALYSIS BY PLAY TYPE")
    print("=" * 70)
    
    # Filter to pass/run plays
    if 'play_type' in pbp.columns:
        offensive_plays = pbp[pbp['play_type'].isin(['pass', 'run'])].copy()
    elif 'pass' in pbp.columns and 'rush' in pbp.columns:
        offensive_plays = pbp[(pbp['pass'] == 1) | (pbp['rush'] == 1)].copy()
        offensive_plays['play_type'] = np.where(offensive_plays['pass'] == 1, 'pass', 'run')
    else:
        print("    ERROR: Cannot identify play types.")
        return
    
    epa_col = 'epa' if 'epa' in offensive_plays.columns else 'EPA'
    offensive_plays = offensive_plays.dropna(subset=[epa_col])
    print(f"\n    Offensive plays with EPA: {len(offensive_plays):,}")
    
    # Play distribution
    print("\n[A] PLAY TYPE DISTRIBUTION:")
    play_counts = offensive_plays['play_type'].value_counts()
    play_pcts = offensive_plays['play_type'].value_counts(normalize=True) * 100
    for play_type in ['pass', 'run']:
        if play_type in play_counts.index:
            print(f"    {play_type.upper()}: {play_counts[play_type]:,} plays ({play_pcts[play_type]:.1f}%)")
    
    # EPA magnitude comparison
    pass_plays = offensive_plays[offensive_plays['play_type'] == 'pass']
    run_plays = offensive_plays[offensive_plays['play_type'] == 'run']
    
    print("\n[B] EPA MAGNITUDE COMPARISON:")
    pass_epa_abs_mean = pass_plays[epa_col].abs().mean()
    run_epa_abs_mean = run_plays[epa_col].abs().mean()
    print(f"    Pass plays avg |EPA|: {pass_epa_abs_mean:.4f}")
    print(f"    Run plays avg |EPA|:  {run_epa_abs_mean:.4f}")
    print(f"    Ratio (pass/run): {pass_epa_abs_mean / run_epa_abs_mean:.2f}x")
    
    # Suggested weights by EPA magnitude
    pass_epa_total_abs = pass_plays[epa_col].abs().sum()
    run_epa_total_abs = run_plays[epa_col].abs().sum()
    total_epa_abs = pass_epa_total_abs + run_epa_total_abs
    
    print("\n[C] SUGGESTED WEIGHTS BY EPA MAGNITUDE:")
    print(f"    Pass-related: {pass_epa_total_abs / total_epa_abs:.1%}")
    print(f"    Run-related: {run_epa_total_abs / total_epa_abs:.1%}")
    
    # Turnovers
    print("\n[D] TURNOVER EPA VALUES:")
    if 'interception' in offensive_plays.columns:
        ints = offensive_plays[offensive_plays['interception'] == 1]
        print(f"    Interceptions: n={len(ints):,}, Def EPA={-ints[epa_col].mean():.3f}")
    if 'fumble_lost' in offensive_plays.columns:
        fumbles = offensive_plays[offensive_plays['fumble_lost'] == 1]
        print(f"    Fumbles lost: n={len(fumbles):,}, Def EPA={-fumbles[epa_col].mean():.3f}")
    if 'sack' in offensive_plays.columns:
        sacks = offensive_plays[offensive_plays['sack'] == 1]
        print(f"    Sacks: n={len(sacks):,}, Def EPA={-sacks[epa_col].mean():.3f}")


def analyze_tackler_position(pbp, crosswalk_path):
    """Analyze EPA by tackler position on run plays."""
    print("\n" + "=" * 70)
    print(" PART 2: EPA ANALYSIS BY TACKLER POSITION")
    print("=" * 70)
    
    # Load crosswalk
    crosswalk = pd.read_csv(crosswalk_path)
    gsis_to_role = dict(zip(crosswalk['gsis_id'], crosswalk['role']))
    print(f"\n    Position crosswalk: {len(gsis_to_role):,} players")
    
    # Filter to run plays
    if 'play_type' in pbp.columns:
        run_plays = pbp[pbp['play_type'] == 'run'].copy()
    elif 'rush' in pbp.columns:
        run_plays = pbp[pbp['rush'] == 1].copy()
    else:
        print("    ERROR: Cannot identify run plays")
        return None
    
    epa_col = 'epa' if 'epa' in run_plays.columns else 'EPA'
    yards_col = 'yards_gained' if 'yards_gained' in run_plays.columns else None
    run_plays = run_plays.dropna(subset=[epa_col])
    print(f"    Run plays with EPA: {len(run_plays):,}")
    
    # Find tackler column
    tackler_col = 'solo_tackle_1_player_id' if 'solo_tackle_1_player_id' in pbp.columns else None
    if tackler_col is None:
        print("    WARNING: No tackler column found")
        return None
    
    print(f"    Using tackler column: {tackler_col}")
    
    # Map tackler to position using crosswalk
    run_plays['tackler_position'] = run_plays[tackler_col].map(gsis_to_role)
    known_tackles = run_plays[run_plays['tackler_position'].notna()]
    print(f"    Plays with known tackler position: {len(known_tackles):,}")
    
    if len(known_tackles) == 0:
        print("    No tackles matched to positions.")
        return None
    
    # Results
    print("\n    EPA BY TACKLER POSITION (Run Plays):")
    print("    (Higher EPA = more yards allowed = worse for defense)")
    
    results = []
    for pos in ['IDL', 'EDGE', 'LB', 'S', 'CB']:
        subset = known_tackles[known_tackles['tackler_position'] == pos]
        if len(subset) > 100:
            epa_mean = subset[epa_col].mean()
            count = len(subset)
            yds_mean = subset[yards_col].mean() if yards_col else np.nan
            results.append({'pos': pos, 'n': count, 'epa': epa_mean, 'yards': yds_mean, 'def_epa': -epa_mean})
            
            if yards_col:
                print(f"      {pos}: n={count:,}, Off EPA={epa_mean:.3f}, Def EPA={-epa_mean:.3f}, Yards={yds_mean:.1f}")
            else:
                print(f"      {pos}: n={count:,}, Off EPA={epa_mean:.3f}, Def EPA={-epa_mean:.3f}")
    
    # Position comparison
    if len(results) >= 2:
        print("\n    POSITION MULTIPLIERS (IDL = baseline):")
        results_df = pd.DataFrame(results)
        idl_epa = results_df[results_df['pos'] == 'IDL']['def_epa'].values[0]
        
        for _, row in results_df.sort_values('def_epa', ascending=False).iterrows():
            multiplier = row['def_epa'] / idl_epa if idl_epa != 0 else np.nan
            print(f"      {row['pos']}: {row['def_epa']:.3f} EPA → {multiplier:.2f}x")
    
    return known_tackles


def analyze_tfl_vs_stop(pbp, crosswalk_path):
    """Analyze EPA on TFLs vs regular stops."""
    print("\n" + "=" * 70)
    print(" PART 3: TFL VS REGULAR STOP EPA")
    print("=" * 70)
    
    # Filter to run plays
    if 'play_type' in pbp.columns:
        run_plays = pbp[pbp['play_type'] == 'run'].copy()
    elif 'rush' in pbp.columns:
        run_plays = pbp[pbp['rush'] == 1].copy()
    else:
        print("    ERROR: Cannot identify run plays")
        return
    
    epa_col = 'epa'
    yards_col = 'yards_gained'
    run_plays = run_plays.dropna(subset=[epa_col])
    
    # Check for TFL column
    tfl_cols = [col for col in pbp.columns if 'tackle_for_loss' in col.lower() or 'tfl' in col.lower()]
    print(f"\n    TFL-related columns: {tfl_cols}")
    
    if 'tackle_for_loss_1_player_id' in run_plays.columns:
        # TFL plays
        tfl_plays = run_plays[run_plays['tackle_for_loss_1_player_id'].notna()]
        
        # Non-TFL plays (regular stops)
        non_tfl_plays = run_plays[run_plays['tackle_for_loss_1_player_id'].isna()]
        
        # For "stops", filter to plays with negative or low EPA (defensive success)
        # A "stop" is typically a play held to < 45% of yards needed for 1st down
        # For simplicity, use yards < 4 as proxy
        if yards_col in run_plays.columns:
            stops = non_tfl_plays[non_tfl_plays[yards_col] <= 3]  # Short gains
            good_runs = non_tfl_plays[non_tfl_plays[yards_col] > 3]  # Decent gains
        else:
            stops = non_tfl_plays[non_tfl_plays[epa_col] < 0]  # Negative EPA
            good_runs = non_tfl_plays[non_tfl_plays[epa_col] >= 0]
        
        print(f"\n    PLAY COUNTS:")
        print(f"      TFLs: {len(tfl_plays):,}")
        print(f"      Stops (<=3 yards): {len(stops):,}")
        print(f"      Good runs (>3 yards): {len(good_runs):,}")
        
        print(f"\n    EPA BY OUTCOME:")
        tfl_epa = tfl_plays[epa_col].mean()
        stop_epa = stops[epa_col].mean()
        good_run_epa = good_runs[epa_col].mean()
        all_run_epa = run_plays[epa_col].mean()
        
        print(f"      TFL:            Off EPA={tfl_epa:.3f}, Def EPA={-tfl_epa:.3f}")
        print(f"      Stop (<=3 yds): Off EPA={stop_epa:.3f}, Def EPA={-stop_epa:.3f}")
        print(f"      Good run:       Off EPA={good_run_epa:.3f}, Def EPA={-good_run_epa:.3f}")
        print(f"      All runs:       Off EPA={all_run_epa:.3f}")
        
        print(f"\n    TFL VALUE COMPARISON:")
        print(f"      TFL vs Stop: {-tfl_epa - (-stop_epa):.3f} additional def EPA")
        print(f"      TFL vs All:  {-tfl_epa - (-all_run_epa):.3f} additional def EPA")
        
        # TFL multiplier
        if stop_epa != 0:
            tfl_multiplier = (-tfl_epa) / (-stop_epa)
            print(f"      TFL is {tfl_multiplier:.2f}x more valuable than a stop")
        
        if yards_col in run_plays.columns:
            print(f"\n    AVERAGE YARDS:")
            print(f"      TFL: {tfl_plays[yards_col].mean():.1f} yards")
            print(f"      Stop: {stops[yards_col].mean():.1f} yards")
            print(f"      Good run: {good_runs[yards_col].mean():.1f} yards")
        
        # TFL by position
        print("\n    TFL EPA BY POSITION:")
        crosswalk = pd.read_csv(crosswalk_path)
        gsis_to_role = dict(zip(crosswalk['gsis_id'], crosswalk['role']))
        
        tfl_plays = tfl_plays.copy()
        tfl_plays['tfl_role'] = tfl_plays['tackle_for_loss_1_player_id'].map(gsis_to_role)
        known_tfl = tfl_plays[tfl_plays['tfl_role'].notna()]
        
        for pos in ['IDL', 'EDGE', 'LB', 'S', 'CB']:
            subset = known_tfl[known_tfl['tfl_role'] == pos]
            if len(subset) > 50:
                epa_mean = subset[epa_col].mean()
                print(f"      {pos}: n={len(subset):,}, Off EPA={epa_mean:.3f}, Def EPA={-epa_mean:.3f}")
        
        return tfl_plays, stops
    
    return None, None


def derive_run_defense_weights(pbp):
    """Derive run defense allocation weights from EPA."""
    print("\n" + "=" * 70)
    print(" PART 4: RUN DEFENSE ALLOCATION WEIGHT DERIVATION")
    print("=" * 70)
    
    print("\n    Current model weights: Stops 64%, TFLs 27%, FF 9%")
    
    # Load player stats to get average counts
    players = pd.read_csv(PLAYER_PATH)
    qualified = players[players['snap_counts_defense'] >= 200]
    run_defenders = qualified[qualified['role'].isin(['IDL', 'EDGE', 'LB'])]
    
    avg_stops = run_defenders['stops'].mean() if 'stops' in run_defenders.columns else 20
    avg_tfls = run_defenders['tackles_for_loss'].mean() if 'tackles_for_loss' in run_defenders.columns else 5
    
    print(f"\n    FREQUENCY (avg per player-season):")
    print(f"      Stops: {avg_stops:.1f}")
    print(f"      TFLs: {avg_tfls:.1f}")
    print(f"      Ratio (Stops/TFLs): {avg_stops/avg_tfls:.1f}x")
    
    # Get EPA values from run plays
    if 'play_type' in pbp.columns:
        run_plays = pbp[pbp['play_type'] == 'run'].copy()
    else:
        run_plays = pbp[pbp['rush'] == 1].copy()
    run_plays = run_plays.dropna(subset=['epa'])
    
    # TFL EPA
    tfl_plays = run_plays[run_plays['tackle_for_loss_1_player_id'].notna()]
    tfl_epa = -tfl_plays['epa'].mean() if len(tfl_plays) > 0 else 1.0
    
    # Stop EPA (non-TFL, short gain)
    if 'yards_gained' in run_plays.columns:
        non_tfl = run_plays[run_plays['tackle_for_loss_1_player_id'].isna()]
        stops = non_tfl[non_tfl['yards_gained'] <= 3]
        stop_epa = -stops['epa'].mean() if len(stops) > 0 else 0.3
    else:
        stop_epa = 0.3  # Estimate
    
    print(f"\n    EPA VALUES:")
    print(f"      TFL: {tfl_epa:.3f} def EPA")
    print(f"      Stop: {stop_epa:.3f} def EPA")
    
    # Total contribution
    stop_total = avg_stops * stop_epa
    tfl_total = avg_tfls * tfl_epa
    total_run = stop_total + tfl_total
    
    print(f"\n    SEASON VALUE CONTRIBUTION:")
    print(f"      Stops: {avg_stops:.0f} × {stop_epa:.3f} = {stop_total:.1f} EPA ({stop_total/total_run:.1%})")
    print(f"      TFLs: {avg_tfls:.1f} × {tfl_epa:.3f} = {tfl_total:.1f} EPA ({tfl_total/total_run:.1%})")
    
    print(f"\n    EMPIRICALLY-DERIVED WEIGHTS:")
    print(f"      Stops: {stop_total/total_run:.0%}")
    print(f"      TFLs: {tfl_total/total_run:.0%}")


def analyze_tackle_value_by_depth(pbp, crosswalk_path):
    """
    NEW PART 5: Analyze tackle value by yards gained (depth).
    
    Goal: Determine if non-stop tackles (4-7 yards) should be credited
    and at what weight relative to stops/TFLs.
    """
    print("\n" + "=" * 70)
    print(" PART 5: TACKLE VALUE BY DEPTH (FOR v22 CONSIDERATION)")
    print("=" * 70)
    
    print("""
    Goal: Should we credit tackles that aren't "stops" (<=3 yards)?
    
    Current model only values:
      - TFLs (negative yards)
      - Stops (0-3 yards, or <45% of yards to first down)
    
    Missing: Tackles at 4-6 yards that limit damage vs 8+ yard runs.
    
    Question: Is a tackle at 5 yards worth something, or only stops matter?
    """)
    
    # Load crosswalk
    crosswalk = pd.read_csv(crosswalk_path)
    gsis_to_role = dict(zip(crosswalk['gsis_id'], crosswalk['role']))
    
    # Filter to run plays
    if 'play_type' in pbp.columns:
        run_plays = pbp[pbp['play_type'] == 'run'].copy()
    else:
        run_plays = pbp[pbp['rush'] == 1].copy()
    
    run_plays = run_plays.dropna(subset=['epa'])
    yards_col = 'yards_gained'
    
    if yards_col not in run_plays.columns:
        print("    ERROR: No yards_gained column found")
        return
    
    print(f"\n    Run plays with EPA and yards: {len(run_plays):,}")
    
    # Create depth buckets
    print("\n    EPA BY YARDS GAINED BUCKET:")
    print("    (Lower yards = better for defense)")
    
    run_plays['yards_bucket'] = pd.cut(
        run_plays[yards_col],
        bins=[-np.inf, -1, 0, 3, 6, 10, 15, np.inf],
        labels=['TFL (<0)', '0 yards', '1-3 yards', '4-6 yards', '7-10 yards', '11-15 yards', '16+ yards']
    )
    
    depth_results = []
    for bucket in ['TFL (<0)', '0 yards', '1-3 yards', '4-6 yards', '7-10 yards', '11-15 yards', '16+ yards']:
        subset = run_plays[run_plays['yards_bucket'] == bucket]
        if len(subset) > 100:
            epa_mean = subset['epa'].mean()
            yards_mean = subset[yards_col].mean()
            count = len(subset)
            depth_results.append({
                'bucket': bucket,
                'n': count,
                'pct': count / len(run_plays) * 100,
                'off_epa': epa_mean,
                'def_epa': -epa_mean,
                'avg_yards': yards_mean
            })
            print(f"      {bucket:15s}: n={count:,} ({count/len(run_plays)*100:5.1f}%), "
                  f"Off EPA={epa_mean:+.3f}, Def EPA={-epa_mean:+.3f}")
    
    depth_df = pd.DataFrame(depth_results)
    
    # Calculate marginal value of each tackle type
    print("\n    MARGINAL VALUE ANALYSIS:")
    print("    Comparing each outcome to the average run play")
    
    avg_run_epa = run_plays['epa'].mean()
    print(f"\n      Average run play EPA: {avg_run_epa:.3f}")
    print(f"      (Offense perspective - positive means good for offense)")
    
    print("\n      Value vs Average Run:")
    for _, row in depth_df.iterrows():
        marginal = avg_run_epa - row['off_epa']  # How much better than average (for defense)
        print(f"        {row['bucket']:15s}: {marginal:+.3f} EPA saved vs avg run")
    
    # Key comparison: 4-6 yard tackles vs letting play continue
    print("\n    KEY QUESTION: VALUE OF 4-6 YARD TACKLES")
    
    limiting_tackles = depth_df[depth_df['bucket'] == '4-6 yards']
    big_runs = depth_df[depth_df['bucket'].isin(['7-10 yards', '11-15 yards', '16+ yards'])]
    
    if len(limiting_tackles) > 0 and len(big_runs) > 0:
        limit_epa = limiting_tackles['def_epa'].values[0]
        big_run_epa = big_runs['def_epa'].mean()
        
        print(f"\n      4-6 yard tackle def EPA: {limit_epa:.3f}")
        print(f"      7+ yard run def EPA:     {big_run_epa:.3f}")
        print(f"      Difference:              {limit_epa - big_run_epa:.3f}")
        
        if limit_epa > big_run_epa:
            print(f"\n      → 4-6 yard tackles ARE valuable vs bigger runs!")
        else:
            print(f"\n      → 4-6 yard tackles still represent negative plays")
    
    # Compare to stops and TFLs
    print("\n    RELATIVE VALUE COMPARISON:")
    
    tfl_row = depth_df[depth_df['bucket'] == 'TFL (<0)']
    stop_rows = depth_df[depth_df['bucket'].isin(['0 yards', '1-3 yards'])]
    limit_row = depth_df[depth_df['bucket'] == '4-6 yards']
    
    if len(tfl_row) > 0 and len(stop_rows) > 0 and len(limit_row) > 0:
        tfl_epa = tfl_row['def_epa'].values[0]
        stop_epa = stop_rows['def_epa'].mean()
        limit_epa = limit_row['def_epa'].values[0]
        
        print(f"\n      TFL def EPA:              {tfl_epa:+.3f} (baseline = 1.00x)")
        print(f"      Stop (0-3 yds) def EPA:   {stop_epa:+.3f} ({stop_epa/tfl_epa:.2f}x)")
        print(f"      Limiting (4-6 yds) def EPA: {limit_epa:+.3f} ({limit_epa/tfl_epa:.2f}x)")
        
        # Only credit if positive defensive EPA
        if limit_epa > 0:
            print(f"\n      → 4-6 yard tackles have POSITIVE defensive value")
            print(f"      → Should be credited at {limit_epa/tfl_epa:.2f}x of TFL value")
        else:
            print(f"\n      → 4-6 yard tackles have NEGATIVE defensive value")
            print(f"      → Should NOT be credited (they represent offensive success)")
    
    # Tackle depth by position
    print("\n    TACKLE DEPTH BY POSITION:")
    
    tackler_col = 'solo_tackle_1_player_id' if 'solo_tackle_1_player_id' in run_plays.columns else None
    
    if tackler_col:
        run_plays['tackler_role'] = run_plays[tackler_col].map(gsis_to_role)
        known_tackles = run_plays[run_plays['tackler_role'].notna()]
        
        print(f"      Tackles with known position: {len(known_tackles):,}")
        
        pos_depth_results = []
        for pos in ['IDL', 'EDGE', 'LB', 'S', 'CB']:
            subset = known_tackles[known_tackles['tackler_role'] == pos]
            if len(subset) > 200:
                yards_mean = subset[yards_col].mean()
                epa_mean = subset['epa'].mean()
                
                # Breakdown by depth
                tfl_pct = (subset[yards_col] < 0).mean() * 100
                stop_pct = ((subset[yards_col] >= 0) & (subset[yards_col] <= 3)).mean() * 100
                limit_pct = ((subset[yards_col] > 3) & (subset[yards_col] <= 6)).mean() * 100
                big_pct = (subset[yards_col] > 6).mean() * 100
                
                pos_depth_results.append({
                    'pos': pos,
                    'n': len(subset),
                    'avg_yards': yards_mean,
                    'def_epa': -epa_mean,
                    'tfl_pct': tfl_pct,
                    'stop_pct': stop_pct,
                    'limit_pct': limit_pct,
                    'big_pct': big_pct
                })
                
                print(f"\n        {pos} (n={len(subset):,}):")
                print(f"          Avg yards: {yards_mean:.1f}")
                print(f"          TFL%: {tfl_pct:.1f}%, Stop%: {stop_pct:.1f}%, "
                      f"Limit%: {limit_pct:.1f}%, Big%: {big_pct:.1f}%")
        
        if pos_depth_results:
            pos_df = pd.DataFrame(pos_depth_results)
            
            print("\n    POSITION TACKLE QUALITY SUMMARY:")
            print("    (Higher TFL% + Stop% = better)")
            
            pos_df['quality_pct'] = pos_df['tfl_pct'] + pos_df['stop_pct']
            for _, row in pos_df.sort_values('quality_pct', ascending=False).iterrows():
                print(f"      {row['pos']}: {row['quality_pct']:.1f}% quality tackles "
                      f"(TFL+Stop), avg {row['avg_yards']:.1f} yards")
    
    # Derive tackle weights for run defense
    print("\n    " + "=" * 60)
    print("    DERIVING TACKLE WEIGHT FOR RUN DEFENSE")
    print("    " + "=" * 60)
    
    # Load player data for frequency
    players = pd.read_csv(PLAYER_PATH)
    qualified = players[players['snap_counts_defense'] >= 200]
    run_defenders = qualified[qualified['role'].isin(['IDL', 'EDGE', 'LB'])]
    
    # Get tackle counts
    if 'tackles' in run_defenders.columns:
        avg_tackles = run_defenders['tackles'].mean()
    elif 'solo_tackles' in run_defenders.columns:
        avg_tackles = run_defenders['solo_tackles'].mean()
    else:
        avg_tackles = 50  # Estimate
    
    avg_stops = run_defenders['stops'].mean() if 'stops' in run_defenders.columns else 20
    avg_tfls = run_defenders['tackles_for_loss'].mean() if 'tackles_for_loss' in run_defenders.columns else 5
    
    # Estimate "limiting tackles" (4-6 yards) - tackles that aren't stops or TFLs
    # Rough estimate: ~20% of non-stop tackles are in 4-6 yard range
    avg_non_stop_tackles = avg_tackles - avg_stops - avg_tfls
    avg_limiting_tackles = avg_non_stop_tackles * 0.35  # Estimate based on distribution
    
    print(f"\n      Avg tackles per player-season: {avg_tackles:.1f}")
    print(f"      Avg TFLs: {avg_tfls:.1f}")
    print(f"      Avg stops: {avg_stops:.1f}")
    print(f"      Avg other tackles: {avg_non_stop_tackles:.1f}")
    print(f"      Estimated 4-6 yard tackles: {avg_limiting_tackles:.1f}")
    
    # EPA values from analysis above
    if len(tfl_row) > 0 and len(stop_rows) > 0 and len(limit_row) > 0:
        tfl_epa_val = tfl_row['def_epa'].values[0]
        stop_epa_val = stop_rows['def_epa'].mean()
        limit_epa_val = limit_row['def_epa'].values[0]
        
        # Only include limiting tackles if they have positive defensive value
        if limit_epa_val > 0:
            # Total contribution
            tfl_total = avg_tfls * tfl_epa_val
            stop_total = avg_stops * stop_epa_val
            limit_total = avg_limiting_tackles * limit_epa_val
            
            total_run = tfl_total + stop_total + limit_total
            
            print(f"\n    SEASON VALUE CONTRIBUTION (with limiting tackles):")
            print(f"      TFLs: {avg_tfls:.1f} × {tfl_epa_val:.3f} = {tfl_total:.1f} EPA ({tfl_total/total_run:.1%})")
            print(f"      Stops: {avg_stops:.1f} × {stop_epa_val:.3f} = {stop_total:.1f} EPA ({stop_total/total_run:.1%})")
            print(f"      Limiting: {avg_limiting_tackles:.1f} × {limit_epa_val:.3f} = {limit_total:.1f} EPA ({limit_total/total_run:.1%})")
            
            print(f"\n    RECOMMENDED WEIGHTS FOR v22:")
            print(f"      TFLs: {tfl_total/total_run:.0%}")
            print(f"      Stops: {stop_total/total_run:.0%}")
            print(f"      Limiting Tackles (4-6 yds): {limit_total/total_run:.0%}")
            
            print(f"\n    COMPARISON TO CURRENT v21:")
            print(f"      Current: TFLs 27%, Stops 64%, FF 9%")
            print(f"      Proposed: TFLs {tfl_total/total_run:.0%}, Stops {stop_total/total_run:.0%}, "
                  f"Limiting {limit_total/total_run:.0%}")
        else:
            print(f"\n    CONCLUSION: Limiting tackles (4-6 yards) have negative def EPA")
            print(f"    → Should NOT be credited in the model")
            print(f"    → Current approach of only valuing TFLs/Stops is correct")
    
    # Final recommendation
    print("\n    " + "=" * 60)
    print("    FINAL RECOMMENDATION")
    print("    " + "=" * 60)
    
    print("""
    The key question is whether 4-6 yard tackles represent defensive VALUE
    or just defensive INVOLVEMENT.
    
    If 4-6 yard tackles have POSITIVE defensive EPA:
      → They prevented a bigger play
      → Should be credited at reduced weight
      → Add to model with empirical weight
    
    If 4-6 yard tackles have NEGATIVE defensive EPA:
      → They represent offensive success (just not a big play)
      → Should NOT be credited
      → Current TFL/Stop-only approach is correct
    
    The data above should inform this decision.
    """)


def analyze_snap_vs_production_run(pbp):
    """
    PART 6: Analyze relationship between run defense snaps and production.
    
    Goal: Determine if run defense snaps should be credited like coverage snaps,
    or if snaps without production represent empty playing time.
    """
    print("\n" + "=" * 70)
    print(" PART 6: RUN DEFENSE SNAPS VS PRODUCTION")
    print("=" * 70)
    
    print("""
    Question: Should we credit run defense snaps like we credit coverage snaps?
    
    Current model:
      - Coverage snaps get 35% weight (presence = deterrence value)
      - Run defense snaps get 0% weight (only stops/TFLs count)
    
    Concern: Adding snap credit might overrate players on good defenses
    who accumulate snaps without production (the "Emerson problem").
    
    Analysis: Look at relationship between snaps and production by position.
    """)
    
    # Load player data
    players = pd.read_csv(PLAYER_PATH)
    qualified = players[players['snap_counts_defense'] >= 200].copy()
    
    print(f"\n    Players with >= 200 defensive snaps: {len(qualified)}")
    
    # Check for run defense snap column
    if 'snap_counts_run_defense' not in qualified.columns:
        print("    ERROR: snap_counts_run_defense column not found")
        print(f"    Available columns: {[c for c in qualified.columns if 'snap' in c.lower()]}")
        return
    
    print(f"\n    Run defense snap column found: snap_counts_run_defense")
    
    # Calculate production rates
    qualified['stops_per_snap'] = qualified['stops'] / qualified['snap_counts_run_defense'].replace(0, np.nan)
    qualified['tfls_per_snap'] = qualified['tackles_for_loss'] / qualified['snap_counts_run_defense'].replace(0, np.nan)
    qualified['run_production'] = qualified['stops'] + qualified['tackles_for_loss'] * 2  # Weight TFLs 2x
    qualified['run_production_per_snap'] = qualified['run_production'] / qualified['snap_counts_run_defense'].replace(0, np.nan)
    
    print("\n    RUN DEFENSE SNAPS BY POSITION:")
    
    position_snap_stats = []
    for pos in ['IDL', 'EDGE', 'LB', 'S', 'CB']:
        pos_df = qualified[qualified['role'] == pos]
        if len(pos_df) >= 20:
            snap_mean = pos_df['snap_counts_run_defense'].mean()
            snap_median = pos_df['snap_counts_run_defense'].median()
            stops_mean = pos_df['stops'].mean()
            tfls_mean = pos_df['tackles_for_loss'].mean()
            stops_per_snap = pos_df['stops_per_snap'].mean()
            tfls_per_snap = pos_df['tfls_per_snap'].mean()
            prod_per_snap = pos_df['run_production_per_snap'].mean()
            
            position_snap_stats.append({
                'pos': pos,
                'n': len(pos_df),
                'avg_snaps': snap_mean,
                'avg_stops': stops_mean,
                'avg_tfls': tfls_mean,
                'stops_per_snap': stops_per_snap,
                'tfls_per_snap': tfls_per_snap,
                'prod_per_snap': prod_per_snap
            })
            
            print(f"\n      {pos} (n={len(pos_df)}):")
            print(f"        Avg run def snaps: {snap_mean:.1f}")
            print(f"        Avg stops: {stops_mean:.1f}, Avg TFLs: {tfls_mean:.1f}")
            print(f"        Stops/snap: {stops_per_snap:.4f}")
            print(f"        TFLs/snap: {tfls_per_snap:.4f}")
            print(f"        Production/snap: {prod_per_snap:.4f}")
    
    pos_snap_df = pd.DataFrame(position_snap_stats)
    
    # Correlation: Do more snaps = more production?
    print("\n    CORRELATION: SNAPS VS PRODUCTION")
    print("    (Does playing more = producing more, or just being on field?)")
    
    for pos in ['IDL', 'EDGE', 'LB']:
        pos_df = qualified[qualified['role'] == pos]
        if len(pos_df) >= 30:
            corr_stops = pos_df['snap_counts_run_defense'].corr(pos_df['stops'])
            corr_tfls = pos_df['snap_counts_run_defense'].corr(pos_df['tackles_for_loss'])
            corr_prod = pos_df['snap_counts_run_defense'].corr(pos_df['run_production'])
            
            print(f"\n      {pos}:")
            print(f"        Snaps vs Stops: r = {corr_stops:.3f}")
            print(f"        Snaps vs TFLs:  r = {corr_tfls:.3f}")
            print(f"        Snaps vs Total Production: r = {corr_prod:.3f}")
    
    # Key question: Do high-snap, low-production players exist?
    print("\n    HIGH-SNAP, LOW-PRODUCTION PLAYERS:")
    print("    (Players who might get inflated value from snap-based credit)")
    
    for pos in ['IDL', 'EDGE', 'LB']:
        pos_df = qualified[qualified['role'] == pos].copy()
        if len(pos_df) >= 30:
            # High snaps = top 25%, low production = bottom 25%
            snap_75 = pos_df['snap_counts_run_defense'].quantile(0.75)
            prod_25 = pos_df['run_production_per_snap'].quantile(0.25)
            
            high_snap_low_prod = pos_df[
                (pos_df['snap_counts_run_defense'] >= snap_75) & 
                (pos_df['run_production_per_snap'] <= prod_25)
            ]
            
            print(f"\n      {pos}: {len(high_snap_low_prod)} players with high snaps, low production")
            if len(high_snap_low_prod) > 0 and len(high_snap_low_prod) <= 5:
                for _, row in high_snap_low_prod.iterrows():
                    print(f"        - {row['player']} ({row['season']} {row['team']}): "
                          f"{row['snap_counts_run_defense']:.0f} snaps, "
                          f"{row['stops']:.0f} stops, {row['tackles_for_loss']:.0f} TFLs")
    
    # Compare: Production rate by team defensive quality
    print("\n    PRODUCTION RATE BY TEAM CONTEXT:")
    print("    (Do players on good defenses have inflated or deflated production rates?)")
    
    # Use team's total defensive snaps as proxy for opportunities
    if 'team' in qualified.columns and 'season' in qualified.columns:
        team_seasons = qualified.groupby(['team', 'season']).agg({
            'snap_counts_run_defense': 'sum',
            'stops': 'sum',
            'tackles_for_loss': 'sum'
        }).reset_index()
        team_seasons.columns = ['team', 'season', 'team_run_snaps', 'team_stops', 'team_tfls']
        
        qualified = qualified.merge(team_seasons, on=['team', 'season'], how='left')
        
        # Team production rate
        qualified['team_prod_rate'] = (qualified['team_stops'] + qualified['team_tfls'] * 2) / qualified['team_run_snaps']
        
        # Split into good vs bad defenses
        team_prod_median = qualified['team_prod_rate'].median()
        good_def = qualified[qualified['team_prod_rate'] >= team_prod_median]
        bad_def = qualified[qualified['team_prod_rate'] < team_prod_median]
        
        print(f"\n      Good defenses (above median team production rate):")
        print(f"        Avg individual production/snap: {good_def['run_production_per_snap'].mean():.4f}")
        print(f"      Bad defenses (below median team production rate):")
        print(f"        Avg individual production/snap: {bad_def['run_production_per_snap'].mean():.4f}")
        
        ratio = good_def['run_production_per_snap'].mean() / bad_def['run_production_per_snap'].mean()
        print(f"      Ratio (good/bad): {ratio:.2f}x")
        
        if ratio > 1.1:
            print(f"\n      → Players on good defenses have HIGHER production rates")
            print(f"      → Adding snap credit would compound the 'good defense' advantage")
            print(f"      → Recommend AGAINST adding run defense snaps, or use lower weight")
        elif ratio < 0.9:
            print(f"\n      → Players on good defenses have LOWER production rates")
            print(f"      → Snap credit might help balance opportunity differences")
            print(f"      → Could consider adding run defense snaps")
        else:
            print(f"\n      → Production rates similar regardless of team quality")
            print(f"      → Snap credit would be neutral re: team context")
    
    # Recommendation
    print("\n    " + "=" * 60)
    print("    RECOMMENDATION: RUN DEFENSE SNAPS")
    print("    " + "=" * 60)
    
    print("""
    Key findings:
    
    1. SNAPS VS PRODUCTION CORRELATION
       - If high (r > 0.7): Snaps ≈ production, snap credit is redundant
       - If moderate (0.4-0.7): Snaps partially capture value
       - If low (r < 0.4): Snaps and production measure different things
    
    2. TEAM CONTEXT EFFECT
       - If good defense players have higher rates: Don't add snaps (Emerson problem)
       - If rates are equal: Snaps would be neutral
       - If good defense players have lower rates: Snaps could help
    
    3. POSITION DIFFERENCES
       - IDL: Most likely to benefit from snap credit (two-gapping, occupying blockers)
       - EDGE: Less clear (expected to produce)
       - LB: Least likely (high opportunity, should produce)
    
    Options:
    
    A) NO SNAP CREDIT (current approach):
       - Keeps it simple
       - Avoids Emerson problem
       - But misses "occupying blockers" value
    
    B) SMALL SNAP CREDIT (10-15%):
       - Add snap_counts_run_defense at modest weight
       - Apply position multipliers (IDL 1.0x, EDGE 0.5x, LB 0.2x)
       - Still primarily rewards production
    
    C) EFFICIENCY-BASED:
       - Don't credit snaps directly
       - Use production_per_snap as efficiency metric
       - Rewards productive players regardless of opportunity
    """)


def main():
    print("=" * 70)
    print(" COMBINED EPA ANALYSIS - RUN DEFENSE (2021-2024)")
    print("=" * 70)
    
    # Load data once
    print("\nLoading play-by-play data...")
    pbp = pd.read_csv(PBP_PATH, low_memory=False)
    print(f"Total plays: {len(pbp):,}")
    
    # Run all analyses
    analyze_play_type(pbp)
    analyze_tackler_position(pbp, CROSSWALK_PATH)
    analyze_tfl_vs_stop(pbp, CROSSWALK_PATH)
    derive_run_defense_weights(pbp)
    analyze_tackle_value_by_depth(pbp, CROSSWALK_PATH)  # NEW PART 5
    
    # Part 6: Snap counts vs production
    analyze_snap_vs_production_run(pbp)
    
    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY: RUN DEFENSE FINDINGS")
    print("=" * 70)
    
    print("""
    1. POSITION MULTIPLIERS (from tackle EPA):
       IDL: 1.00x (baseline - tackles at LOS)
       EDGE: 1.07x (slightly better than IDL)
       LB: 0.21x (tackles happen 1.7 yards further downfield)
       CB: 0.00x (tackles too late, positive EPA)
       S: 0.00x (last line of defense)
       
       → Already implemented in v21
       
    2. TFL VS STOP VALUE:
       TFLs are worth ~3x more than regular stops
       Current weights (27% TFL, 64% Stop) reflect this
       
    3. TACKLE VALUE BY DEPTH (Part 5):
       See analysis above for whether 4-6 yard tackles
       should be credited in future versions.
       
    4. POSITION-SPECIFIC TACKLE QUALITY:
       IDL/EDGE: Higher % of TFLs and stops
       LB: More tackles but at greater depth
       CB/S: Last line - tackles represent failures
    """)
    
    print("=" * 70)
    print(" DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()