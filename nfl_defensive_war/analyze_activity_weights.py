"""
analysis_05_activity_weights.py
===============================

EMPIRICALLY DERIVE DEFENSIVE ACTIVITY WEIGHTS

This script calculates the proper split of defensive value across activities
based on actual EPA data from play-by-play.

Previous analysis established:
    - Pass plays: 70% of total EPA magnitude
    - Run plays: 30% of total EPA magnitude

This script determines:
    - How to split the 70% pass-related into coverage vs pass rush
    - How to handle turnovers (separate category or included in coverage?)
    - Final activity weights for the WAR model

Methodology:
    1. Calculate total defensive EPA from each outcome type
    2. Group outcomes into activities (pass rush, coverage, run defense, turnovers)
    3. Derive weights from EPA contribution

Run this on your machine:
    python analysis_05_activity_weights.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/Users/anokhpalakurthi/Documents/nfl_defensive_war/data")
PBP_PATH = DATA_DIR / "pbp_2021_2024_full.csv"


def main():
    print("=" * 70)
    print(" EMPIRICAL DERIVATION OF ACTIVITY WEIGHTS")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading play-by-play data...")
    pbp = pd.read_csv(PBP_PATH, low_memory=False)
    print(f"    Total plays: {len(pbp):,}")
    
    # Filter to offensive plays with EPA
    if 'play_type' in pbp.columns:
        plays = pbp[pbp['play_type'].isin(['pass', 'run'])].copy()
    else:
        plays = pbp[(pbp['pass'] == 1) | (pbp['rush'] == 1)].copy()
        plays['play_type'] = np.where(plays['pass'] == 1, 'pass', 'run')
    
    plays = plays.dropna(subset=['epa'])
    print(f"    Offensive plays with EPA: {len(plays):,}")
    
    # ==================== PART 1: CATEGORIZE PLAY OUTCOMES ====================
    print("\n" + "=" * 70)
    print(" PART 1: PLAY OUTCOME CATEGORIZATION")
    print("=" * 70)
    
    # Separate pass and run plays
    pass_plays = plays[plays['play_type'] == 'pass'].copy()
    run_plays = plays[plays['play_type'] == 'run'].copy()
    
    print(f"\n    Pass plays: {len(pass_plays):,}")
    print(f"    Run plays: {len(run_plays):,}")
    
    # Categorize pass play outcomes
    print("\n    PASS PLAY OUTCOMES:")
    
    # Sacks (pass rush success)
    sacks = pass_plays[pass_plays['sack'] == 1]
    n_sacks = len(sacks)
    sack_epa = -sacks['epa'].sum()  # Defense perspective
    sack_epa_avg = -sacks['epa'].mean()
    print(f"      Sacks: n={n_sacks:,}, Total Def EPA={sack_epa:,.1f}, Avg={sack_epa_avg:.3f}")
    
    # Interceptions (coverage + turnover)
    ints = pass_plays[pass_plays['interception'] == 1]
    n_ints = len(ints)
    int_epa = -ints['epa'].sum()
    int_epa_avg = -ints['epa'].mean()
    print(f"      Interceptions: n={n_ints:,}, Total Def EPA={int_epa:,.1f}, Avg={int_epa_avg:.3f}")
    
    # Incompletions (coverage success, non-INT, non-sack)
    incompletions = pass_plays[
        (pass_plays['complete_pass'] == 0) & 
        (pass_plays['interception'] == 0) & 
        (pass_plays['sack'] == 0)
    ]
    n_incomp = len(incompletions)
    incomp_epa = -incompletions['epa'].sum()
    incomp_epa_avg = -incompletions['epa'].mean()
    print(f"      Incompletions: n={n_incomp:,}, Total Def EPA={incomp_epa:,.1f}, Avg={incomp_epa_avg:.3f}")
    
    # Completions (coverage failure)
    completions = pass_plays[pass_plays['complete_pass'] == 1]
    n_comp = len(completions)
    comp_epa = -completions['epa'].sum()  # Negative for defense = bad
    comp_epa_avg = -completions['epa'].mean()
    print(f"      Completions: n={n_comp:,}, Total Def EPA={comp_epa:,.1f}, Avg={comp_epa_avg:.3f}")
    
    # Run play outcomes
    print("\n    RUN PLAY OUTCOMES:")
    
    # Fumbles lost on run plays
    if 'fumble_lost' in run_plays.columns:
        run_fumbles = run_plays[run_plays['fumble_lost'] == 1]
        n_run_fumbles = len(run_fumbles)
        run_fumble_epa = -run_fumbles['epa'].sum()
        print(f"      Fumbles lost: n={n_run_fumbles:,}, Total Def EPA={run_fumble_epa:,.1f}")
    else:
        n_run_fumbles = 0
        run_fumble_epa = 0
    
    # Regular run plays (non-fumble)
    regular_runs = run_plays[run_plays.get('fumble_lost', 0) != 1]
    n_regular_runs = len(regular_runs)
    regular_run_epa = -regular_runs['epa'].sum()
    regular_run_epa_avg = -regular_runs['epa'].mean()
    print(f"      Regular runs: n={n_regular_runs:,}, Total Def EPA={regular_run_epa:,.1f}, Avg={regular_run_epa_avg:.3f}")
    
    # Pass play fumbles
    if 'fumble_lost' in pass_plays.columns:
        pass_fumbles = pass_plays[(pass_plays['fumble_lost'] == 1) & (pass_plays['interception'] == 0)]
        n_pass_fumbles = len(pass_fumbles)
        pass_fumble_epa = -pass_fumbles['epa'].sum()
        print(f"      Pass fumbles: n={n_pass_fumbles:,}, Total Def EPA={pass_fumble_epa:,.1f}")
    else:
        n_pass_fumbles = 0
        pass_fumble_epa = 0
    
    # ==================== PART 2: GROUP INTO ACTIVITIES ====================
    print("\n" + "=" * 70)
    print(" PART 2: GROUP OUTCOMES INTO ACTIVITIES")
    print("=" * 70)
    
    # Activity definitions:
    # - PASS RUSH: Sacks (direct pass rush outcome)
    # - COVERAGE: Incompletions + Completions (coverage outcomes, net effect)
    # - TURNOVERS: INTs + Fumbles (big plays, separate category)
    # - RUN DEFENSE: Regular run plays
    
    # Calculate EPA by activity
    pass_rush_epa = sack_epa
    coverage_epa = incomp_epa + comp_epa  # Net coverage (incomp good, comp bad)
    turnover_epa = int_epa + run_fumble_epa + pass_fumble_epa
    run_defense_epa = regular_run_epa
    
    print("\n    ACTIVITY EPA TOTALS (Defense perspective, positive = good):")
    print(f"      Pass Rush (sacks):        {pass_rush_epa:>10,.1f} EPA")
    print(f"      Coverage (incomp + comp): {coverage_epa:>10,.1f} EPA")
    print(f"      Turnovers (INT + fumbles):{turnover_epa:>10,.1f} EPA")
    print(f"      Run Defense:              {run_defense_epa:>10,.1f} EPA")
    
    # Note: Coverage can be negative if completions outweigh incompletions
    # This is expected - offenses complete more passes than they don't
    
    # ==================== PART 3: CALCULATE USING |EPA| MAGNITUDE ====================
    print("\n" + "=" * 70)
    print(" PART 3: WEIGHTS BY EPA MAGNITUDE (absolute value)")
    print("=" * 70)
    
    # For weights, we care about HOW MUCH each activity matters (magnitude)
    # not whether it nets positive or negative
    
    # Pass rush magnitude (sacks)
    pass_rush_mag = sacks['epa'].abs().sum()
    
    # Coverage magnitude (incompletions + completions)
    coverage_mag = incompletions['epa'].abs().sum() + completions['epa'].abs().sum()
    
    # Turnover magnitude (INTs + fumbles)
    turnover_mag = ints['epa'].abs().sum() + run_fumbles['epa'].abs().sum() + pass_fumbles['epa'].abs().sum()
    
    # Run defense magnitude
    run_defense_mag = regular_runs['epa'].abs().sum()
    
    total_mag = pass_rush_mag + coverage_mag + turnover_mag + run_defense_mag
    
    print("\n    ACTIVITY |EPA| MAGNITUDE:")
    print(f"      Pass Rush:    {pass_rush_mag:>12,.1f} ({pass_rush_mag/total_mag*100:>5.1f}%)")
    print(f"      Coverage:     {coverage_mag:>12,.1f} ({coverage_mag/total_mag*100:>5.1f}%)")
    print(f"      Turnovers:    {turnover_mag:>12,.1f} ({turnover_mag/total_mag*100:>5.1f}%)")
    print(f"      Run Defense:  {run_defense_mag:>12,.1f} ({run_defense_mag/total_mag*100:>5.1f}%)")
    print(f"      TOTAL:        {total_mag:>12,.1f}")
    
    # Raw weights
    w_pass_rush = pass_rush_mag / total_mag
    w_coverage = coverage_mag / total_mag
    w_turnovers = turnover_mag / total_mag
    w_run_defense = run_defense_mag / total_mag
    
    print("\n    RAW EMPIRICAL WEIGHTS:")
    print(f"      Pass Rush:    {w_pass_rush:.1%}")
    print(f"      Coverage:     {w_coverage:.1%}")
    print(f"      Turnovers:    {w_turnovers:.1%}")
    print(f"      Run Defense:  {w_run_defense:.1%}")
    
    # ==================== PART 4: ALTERNATIVE - POSITIVE EPA ONLY ====================
    print("\n" + "=" * 70)
    print(" PART 4: WEIGHTS BY POSITIVE DEFENSIVE EPA ONLY")
    print("=" * 70)
    
    # Only count plays where defense gained EPA (successful plays)
    # This answers: "What activities generate defensive value?"
    
    # Pass rush - successful sacks
    pr_positive = sack_epa  # Already positive (sacks always good for D)
    
    # Coverage - only incompletions (completions are bad for D)
    cov_positive = incomp_epa
    
    # Turnovers - always positive for D
    to_positive = turnover_epa
    
    # Run defense - only negative EPA runs (good for D)
    run_positive = -run_plays[run_plays['epa'] < 0]['epa'].sum()
    
    total_positive = pr_positive + cov_positive + to_positive + run_positive
    
    print("\n    POSITIVE DEFENSIVE EPA BY ACTIVITY:")
    print(f"      Pass Rush:    {pr_positive:>10,.1f} ({pr_positive/total_positive*100:>5.1f}%)")
    print(f"      Coverage:     {cov_positive:>10,.1f} ({cov_positive/total_positive*100:>5.1f}%)")
    print(f"      Turnovers:    {to_positive:>10,.1f} ({to_positive/total_positive*100:>5.1f}%)")
    print(f"      Run Defense:  {run_positive:>10,.1f} ({run_positive/total_positive*100:>5.1f}%)")
    print(f"      TOTAL:        {total_positive:>10,.1f}")
    
    w2_pass_rush = pr_positive / total_positive
    w2_coverage = cov_positive / total_positive
    w2_turnovers = to_positive / total_positive
    w2_run_defense = run_positive / total_positive
    
    print("\n    POSITIVE-EPA WEIGHTS:")
    print(f"      Pass Rush:    {w2_pass_rush:.1%}")
    print(f"      Coverage:     {w2_coverage:.1%}")
    print(f"      Turnovers:    {w2_turnovers:.1%}")
    print(f"      Run Defense:  {w2_run_defense:.1%}")
    
    # ==================== PART 5: COMPARISON AND RECOMMENDATION ====================
    print("\n" + "=" * 70)
    print(" PART 5: COMPARISON AND RECOMMENDATION")
    print("=" * 70)
    
    print("\n    WEIGHT COMPARISON:")
    print(f"    {'Activity':<15} {'Magnitude':<12} {'Positive EPA':<12} {'Current v13':<12}")
    print(f"    {'-'*51}")
    print(f"    {'Pass Rush':<15} {w_pass_rush*100:>8.1f}%    {w2_pass_rush*100:>8.1f}%    {'25.0%':>8}")
    print(f"    {'Coverage':<15} {w_coverage*100:>8.1f}%    {w2_coverage*100:>8.1f}%    {'35.0%':>8}")
    print(f"    {'Turnovers':<15} {w_turnovers*100:>8.1f}%    {w2_turnovers*100:>8.1f}%    {'20.0%':>8}")
    print(f"    {'Run Defense':<15} {w_run_defense*100:>8.1f}%    {w2_run_defense*100:>8.1f}%    {'20.0%':>8}")
    
    # Check pass-related total
    print(f"\n    PASS VS RUN SPLIT:")
    print(f"      Magnitude method - Pass: {(w_pass_rush + w_coverage + w_turnovers)*100:.1f}%, Run: {w_run_defense*100:.1f}%")
    print(f"      Positive EPA method - Pass: {(w2_pass_rush + w2_coverage + w2_turnovers)*100:.1f}%, Run: {w2_run_defense*100:.1f}%")
    print(f"      Current v13 - Pass: 80.0%, Run: 20.0%")
    print(f"      Original EPA magnitude (Part 1 analysis) - Pass: 69.5%, Run: 30.5%")
    
    # ==================== PART 6: RECOMMENDED WEIGHTS ====================
    print("\n" + "=" * 70)
    print(" PART 6: RECOMMENDED WEIGHTS")
    print("=" * 70)
    
    # Average the two methods and round to nice numbers
    avg_pr = (w_pass_rush + w2_pass_rush) / 2
    avg_cov = (w_coverage + w2_coverage) / 2
    avg_to = (w_turnovers + w2_turnovers) / 2
    avg_run = (w_run_defense + w2_run_defense) / 2
    
    print("\n    AVERAGED EMPIRICAL WEIGHTS:")
    print(f"      Pass Rush:    {avg_pr:.1%}")
    print(f"      Coverage:     {avg_cov:.1%}")
    print(f"      Turnovers:    {avg_to:.1%}")
    print(f"      Run Defense:  {avg_run:.1%}")
    
    # Round to nearest 5%
    def round_to_5(x):
        return round(x * 20) / 20
    
    rec_pr = round_to_5(avg_pr)
    rec_cov = round_to_5(avg_cov)
    rec_to = round_to_5(avg_to)
    rec_run = round_to_5(avg_run)
    
    # Normalize to sum to 1.0
    total_rec = rec_pr + rec_cov + rec_to + rec_run
    rec_pr /= total_rec
    rec_cov /= total_rec
    rec_to /= total_rec
    rec_run /= total_rec
    
    print("\n    RECOMMENDED WEIGHTS (rounded, normalized):")
    print(f"      Pass Rush:    {rec_pr:.0%}")
    print(f"      Coverage:     {rec_cov:.0%}")
    print(f"      Turnovers:    {rec_to:.0%}")
    print(f"      Run Defense:  {rec_run:.0%}")
    print(f"      Total:        {rec_pr + rec_cov + rec_to + rec_run:.0%}")
    
    # Save results
    results = {
        'method': ['Magnitude', 'Positive EPA', 'Average', 'Recommended'],
        'pass_rush': [w_pass_rush, w2_pass_rush, avg_pr, rec_pr],
        'coverage': [w_coverage, w2_coverage, avg_cov, rec_cov],
        'turnovers': [w_turnovers, w2_turnovers, avg_to, rec_to],
        'run_defense': [w_run_defense, w2_run_defense, avg_run, rec_run],
    }
    results_df = pd.DataFrame(results)
    
    output_path = DATA_DIR.parent / "outputs" / "empirical_activity_weights.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n    Saved results to: {output_path}")
    
    print("\n" + "=" * 70)
    print(" DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()