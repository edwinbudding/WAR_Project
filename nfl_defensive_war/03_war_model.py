"""
03_defense_activity_war.py
===========================================

DEFENSIVE WAR MODEL (2021–2024) — v23 WITH CB ROLE ADJUSTMENT

Key changes from v22:
    - Added CB ROLE ADJUSTMENT based on snap share
    - CB1s (>40% of team CB snaps) get 1.15x boost for tougher assignments
    - CB3s (<25% of team CB snaps) get 0.90x for easier matchups
    - Fixes "Lattimore ranked below teammates" problem
    
Previous changes (v22):
    - Position-specific regression (CB 65%, EDGE 45%)
    - Reduced final WAR divisor to 4.5

Philosophy: Two-bucket structure with efficiency adjustment + role adjustment
    - Pass Defense (coverage + pass rush + INTs)
    - Run Defense (stops + TFLs + forced fumbles)
    - Coverage rewards EFFICIENCY, not just volume
    - CB1s get credit for facing tougher assignments
    - Position-specific regression accounts for attribution difficulty

Within Pass Defense (69%):
    Coverage snaps:  35%
    Yards/snap eff:  25%  ← Lower yards = higher share
    PBUs:            15%
    Sacks:           11%
    Hits:             2%
    Hurries:          2%
    INTs:            10%

CB Role Adjustment:
    snap_share = player_cb_cov_snaps / team_total_cb_cov_snaps
    CB1 (>40% share): 1.15x multiplier (tougher assignments)
    CB2 (25-40%):     1.00x (neutral)
    CB3 (<25%):       0.90x (easier matchups)

Efficiency metric explanation:
    yards_per_snap = yards_allowed / coverage_snaps
    efficiency = 1 / (yards_per_snap + 0.1)  # Invert so lower yards = higher
    share = player_efficiency / team_total_efficiency
    
    Example:
    - CB A: 1000 snaps, 800 yards → 0.80 yds/snap → efficiency = 1.11
    - CB B: 1000 snaps, 1400 yards → 1.40 yds/snap → efficiency = 0.67
    - CB A gets ~1.7x more credit for efficiency component

Position multipliers (unchanged):
    Run Defense: EDGE 1.07x, IDL 1.00x, LB 0.21x, CB/S 0.00x
    Coverage:    CB 1.00x, S 0.94x, LB 0.81x, EDGE/IDL 0.50x

Team context:
    Using 50% regression to league mean to balance team effects.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("/Users/anokhpalakurthi/Documents/nfl_defensive_war")
OUT_DIR = BASE_DIR / "outputs"

TEAM_AGG_PATH = OUT_DIR / "team_defense_agg_2021_2024.csv"
PLAYER_PATH = OUT_DIR / "defense_engineered_2021_2024.csv"

WAR_OUT = OUT_DIR / "defensive_war_2021_2024.csv"
WEIGHTS_OUT = OUT_DIR / "defense_activity_weights.csv"
SUMMARY_OUT = OUT_DIR / "defense_model_summary.txt"


# ============================================================
# CONFIGURATION
# ============================================================

MIN_SNAPS_FOR_WAR = 50
MIN_SNAPS_FOR_REPLACEMENT = 50
DEFENSIVE_UNIT_DIVISOR = 11

# Base regression factor for team WAR (applied at team level)
TEAM_WAR_REGRESSION_FACTOR = 0.50

# Position-specific regression factors (applied at player level)
# Higher = more regression toward mean = less extreme values
# CBs/Ss get more regression (coverage harder to isolate)
# EDGE gets less regression (sacks more individually attributable)
POSITION_REGRESSION_FACTORS = {
    "CB": 0.65,
    "S": 0.60,
    "LB": 0.55,
    "IDL": 0.50,
    "EDGE": 0.45,
    "UNKNOWN": 0.55,
    "HYBRID": 0.55,
}

# Final WAR scaling divisor (to get reasonable WAR magnitudes)
# Based on effective starters analysis suggesting ~3-5
FINAL_WAR_DIVISOR = 3

# Small constant to avoid division by zero in efficiency calc
EFFICIENCY_SMOOTHING = 0.1


# ============================================================
# TWO-BUCKET ACTIVITY WEIGHTS
# ============================================================

# Top-level bucket weights (from EPA magnitude analysis)
BUCKET_WEIGHTS = {
    "pass_defense": 0.69,
    "run_defense": 0.31,
}

# Within-bucket allocations
# v21: Added yards efficiency, reduced coverage snaps
PASS_DEFENSE_ALLOCATION = {
    # Coverage component (now split between volume and efficiency)
    "coverage_snaps": 0.35,        # Reduced from 0.63
    "yards_efficiency": 0.25,      # NEW: Lower yards/snap = higher share
    "pass_break_ups": 0.15,        # Increased from 0.12
    # Pass rush component (unchanged)
    "sacks": 0.11,
    "hits": 0.02,
    "hurries": 0.02,
    # Turnover component (unchanged)
    "interceptions": 0.10,
}

RUN_DEFENSE_ALLOCATION = {
    "stops": 0.64,
    "tackles_for_loss": 0.27,
    "forced_fumbles": 0.09,
}

# Metrics that get coverage position multipliers applied
COVERAGE_METRICS = {"coverage_snaps", "yards_efficiency", "pass_break_ups", "interceptions"}

# Metrics that DON'T get position multipliers (pass rush)
PASS_RUSH_METRICS = {"sacks", "hits", "hurries"}


# ============================================================
# POSITION MULTIPLIERS
# ============================================================

# Run defense multipliers (from tackle EPA by field position)
RUN_DEFENSE_POSITION_MULTIPLIERS = {
    "EDGE": 1.07,
    "IDL": 1.00,
    "LB": 0.21,
    "CB": 0.00,
    "S": 0.00,
    "UNKNOWN": 0.50,
    "HYBRID": 0.50,
}

# Box safety adjustment for run defense
SAFETY_BOX_RATE_THRESHOLD = 0.10
SAFETY_MAX_RUN_MULTIPLIER = 0.21
SAFETY_BOX_RATE_FOR_MAX = 0.50

# Coverage multipliers (from catch rate by position)
COVERAGE_POSITION_MULTIPLIERS = {
    "CB": 1.00,
    "S": 0.94,
    "LB": 0.81,
    "EDGE": 0.50,
    "IDL": 0.50,
    "UNKNOWN": 0.75,
    "HYBRID": 0.75,
}

# CB Role Adjustment (CB1 vs CB2/CB3)
# CB1s face tougher assignments (opponent WR1) so their yards efficiency
# should be adjusted upward. We use snap share as proxy for role.
# CB with highest share of team's CB coverage snaps = CB1
CB_ROLE_ADJUSTMENT = {
    "CB1": 1.15,    # Boost for taking tougher assignments
    "CB2": 1.00,    # Neutral
    "CB3": 0.90,    # Slight penalty for easier matchups
}
CB1_SNAP_SHARE_THRESHOLD = 0.40  # >40% of team CB snaps = CB1
CB2_SNAP_SHARE_THRESHOLD = 0.25  # 25-40% = CB2, <25% = CB3


# ============================================================
# STEP 1: LOAD DATA
# ============================================================

def load_data():
    print("=" * 70)
    print(" DEFENSIVE WAR MODEL (2021–2024) — v23 WITH CB ROLE ADJUSTMENT")
    print("=" * 70)
    
    print("\n[1] Loading data...")
    
    team_df = pd.read_csv(TEAM_AGG_PATH)
    player_df = pd.read_csv(PLAYER_PATH)
    
    print(f"    Team-seasons: {len(team_df)}")
    print(f"    Player-seasons: {len(player_df)}")
    
    # Check for yards column
    if "yards" in player_df.columns:
        print(f"    Yards allowed data: Available ✓")
    else:
        print(f"    WARNING: 'yards' column not found!")
    
    print(f"\n    Role distribution:")
    for role, count in player_df["role"].value_counts().items():
        print(f"      {role}: {count}")
    
    return team_df, player_df


# ============================================================
# STEP 2: BUILD TEAM ACTIVITY METRICS
# ============================================================

def build_team_activity_metrics(player_df):
    """Aggregate player stats to team-level activity metrics."""
    print("\n[2] Building team activity metrics...")
    
    team_metrics = []
    
    for (season, team), group in player_df.groupby(["season", "team"]):
        row = {"season": season, "team": team}
        
        # Pass defense metrics
        row["team_cov_snaps"] = group["snap_counts_coverage"].sum()
        row["team_pbus"] = group["pass_break_ups"].sum()
        row["team_sacks"] = group["sacks"].sum()
        row["team_hits"] = group["hits"].sum()
        row["team_hurries"] = group["hurries"].sum()
        row["team_ints"] = group["interceptions"].sum()
        row["team_yards"] = group["yards"].sum() if "yards" in group.columns else 0
        
        # Run defense metrics
        row["team_stops"] = group["stops"].sum()
        row["team_tfls"] = group["tackles_for_loss"].sum()
        row["team_forced_fumbles"] = group["forced_fumbles"].sum() if "forced_fumbles" in group.columns else 0
        
        # General
        row["team_def_snaps"] = group["snap_counts_defense"].sum()
        
        team_metrics.append(row)
    
    team_activity_df = pd.DataFrame(team_metrics)
    
    print(f"    Built metrics for {len(team_activity_df)} team-seasons")
    
    return team_activity_df


# ============================================================
# STEP 3: DISPLAY WEIGHTS
# ============================================================

def display_weights():
    """Display the two-bucket weight structure."""
    print("\n[3] Using TWO-BUCKET structure with CB ROLE ADJUSTMENT (v23)...")
    print("    NEW: Yards per coverage snap efficiency metric added")
    print("    Coverage snaps reduced from 63% to 35%")
    print("    Yards efficiency: 25% (lower yards = higher share)")
    print(f"    Team WAR regressed {TEAM_WAR_REGRESSION_FACTOR:.0%} toward league mean")
    
    print("\n    Bucket weights:")
    print(f"      Pass Defense: {BUCKET_WEIGHTS['pass_defense']:.0%}")
    print(f"      Run Defense:  {BUCKET_WEIGHTS['run_defense']:.0%}")
    
    print("\n    Pass Defense allocation (69% of total):")
    for metric, weight in PASS_DEFENSE_ALLOCATION.items():
        mult_note = "(× cov mult)" if metric in COVERAGE_METRICS else ""
        new_note = " ← NEW" if metric == "yards_efficiency" else ""
        print(f"      {metric}: {weight:.0%} {mult_note}{new_note}")
    
    print("\n    Run Defense allocation (31% of total):")
    for metric, weight in RUN_DEFENSE_ALLOCATION.items():
        print(f"      {metric}: {weight:.0%}")
    
    print("\n    Position multipliers:")
    print("      Run Defense: EDGE 1.07x, IDL 1.00x, LB 0.21x, CB/S 0.00x")
    print("      Coverage:    CB 1.00x, S 0.94x, LB 0.81x, EDGE/IDL 0.50x")
    
    print("\n    CB Role Adjustment (snap share as proxy for assignment difficulty):")
    print(f"      CB1 (>{CB1_SNAP_SHARE_THRESHOLD:.0%} of team CB snaps): {CB_ROLE_ADJUSTMENT['CB1']:.2f}x")
    print(f"      CB2 ({CB2_SNAP_SHARE_THRESHOLD:.0%}-{CB1_SNAP_SHARE_THRESHOLD:.0%}): {CB_ROLE_ADJUSTMENT['CB2']:.2f}x")
    print(f"      CB3 (<{CB2_SNAP_SHARE_THRESHOLD:.0%}): {CB_ROLE_ADJUSTMENT['CB3']:.2f}x")
    
    # Save weights
    weights_data = []
    for metric, weight in PASS_DEFENSE_ALLOCATION.items():
        weights_data.append({
            "bucket": "pass_defense",
            "metric": metric,
            "weight_in_bucket": weight,
            "weight_total": weight * BUCKET_WEIGHTS["pass_defense"]
        })
    for metric, weight in RUN_DEFENSE_ALLOCATION.items():
        weights_data.append({
            "bucket": "run_defense", 
            "metric": metric,
            "weight_in_bucket": weight,
            "weight_total": weight * BUCKET_WEIGHTS["run_defense"]
        })
    
    weights_df = pd.DataFrame(weights_data)
    weights_df.to_csv(WEIGHTS_OUT, index=False)
    print(f"\n    Saved weights → {WEIGHTS_OUT}")


# ============================================================
# STEP 4: COMPUTE TEAM DEFENSIVE WAR
# ============================================================

def compute_team_war(team_df):
    """Compute team-level defensive WAR with regression to mean."""
    print("\n[4] Computing team defensive WAR...")
    
    replacement_epa = team_df["def_epa_per_play"].quantile(0.75)
    
    team_df["team_def_war_raw"] = (
        (replacement_epa - team_df["def_epa_per_play"]) * team_df["def_plays"]
    )
    team_df["team_def_war_actual"] = team_df["team_def_war_raw"] / DEFENSIVE_UNIT_DIVISOR
    
    # Calculate league average
    league_avg_war = team_df["team_def_war_actual"].mean()
    
    # Apply 50% regression to mean
    team_df["team_def_war"] = (
        TEAM_WAR_REGRESSION_FACTOR * league_avg_war + 
        (1 - TEAM_WAR_REGRESSION_FACTOR) * team_df["team_def_war_actual"]
    )
    
    print(f"    Replacement EPA/play: {replacement_epa:.4f}")
    print(f"    Actual team WAR range: [{team_df['team_def_war_actual'].min():.1f}, {team_df['team_def_war_actual'].max():.1f}]")
    print(f"    League average WAR: {league_avg_war:.2f}")
    print(f"    Regression factor: {TEAM_WAR_REGRESSION_FACTOR:.0%} toward mean")
    print(f"    Adjusted team WAR range: [{team_df['team_def_war'].min():.1f}, {team_df['team_def_war'].max():.1f}]")
    
    print("\n    Top 5 team defenses (actual → adjusted):")
    for _, row in team_df.nlargest(5, "team_def_war_actual").iterrows():
        print(f"      {row['season']} {row['team']}: {row['team_def_war_actual']:.1f} → {row['team_def_war']:.1f}")
    
    print("\n    Bottom 5 team defenses (actual → adjusted):")
    for _, row in team_df.nsmallest(5, "team_def_war_actual").iterrows():
        print(f"      {row['season']} {row['team']}: {row['team_def_war_actual']:.1f} → {row['team_def_war']:.1f}")
    
    return team_df


# ============================================================
# STEP 5: ALLOCATE WAR TO PLAYERS
# ============================================================

def allocate_war_to_players(player_df, team_df):
    """Split team WAR into two buckets, allocate to players with efficiency metrics."""
    print("\n[5] Allocating WAR to players (with yards efficiency)...")
    print(f"    Using {TEAM_WAR_REGRESSION_FACTOR:.0%} regression to league mean")
    
    # Merge team WAR
    player_df = player_df.merge(
        team_df[["season", "team", "team_def_war"]],
        on=["season", "team"],
        how="left"
    )
    
    # Initialize WAR columns
    player_df["pass_defense_war"] = 0.0
    player_df["run_defense_war"] = 0.0
    
    # Track efficiency stats for display
    efficiency_examples = []
    
    # Process each team-season
    for (season, team), group in player_df.groupby(["season", "team"]):
        team_war = group["team_def_war"].iloc[0]
        if pd.isna(team_war):
            continue
        
        indices = group.index
        roles = group["role"].values
        
        # ========== PASS DEFENSE ALLOCATION ==========
        pass_def_pool = team_war * BUCKET_WEIGHTS["pass_defense"]
        
        # Get coverage position multipliers
        coverage_multipliers = np.array([
            COVERAGE_POSITION_MULTIPLIERS.get(role, 0.75) for role in roles
        ])
        
        pass_def_shares = np.zeros(len(group))
        
        # Calculate yards per coverage snap efficiency
        # Lower yards/snap = higher efficiency = higher share
        cov_snaps = group["snap_counts_coverage"].values.astype(float)
        yards = group["yards"].values.astype(float) if "yards" in group.columns else np.zeros(len(group))
        
        # Yards per snap (safe division - avoids warning)
        yards_per_snap = np.divide(
            yards,
            cov_snaps,
            out=np.zeros_like(yards, dtype=float),
            where=cov_snaps > 0
        )
        
        # Invert: lower yards/snap = higher efficiency
        # Add smoothing to avoid extreme values
        efficiency = 1 / (yards_per_snap + EFFICIENCY_SMOOTHING)
        
        # Normalize efficiency to sum to 1 for team
        team_efficiency_total = efficiency.sum()
        if team_efficiency_total > 0:
            efficiency_share = efficiency / team_efficiency_total
        else:
            efficiency_share = np.zeros(len(group))
        
        # Column name mapping
        col_mapping = {
            "coverage_snaps": "snap_counts_coverage",
            "pass_break_ups": "pass_break_ups",
            "sacks": "sacks",
            "hits": "hits",
            "hurries": "hurries",
            "interceptions": "interceptions",
        }
        
        for metric, weight in PASS_DEFENSE_ALLOCATION.items():
            if metric == "yards_efficiency":
                # Use pre-calculated efficiency share
                metric_share = efficiency_share * weight
                # Apply coverage multipliers
                metric_share = metric_share * coverage_multipliers
                pass_def_shares += metric_share
                continue
            
            col = col_mapping.get(metric, metric)
            if col not in group.columns:
                continue
                
            team_total = group[col].sum()
            if team_total <= 0:
                continue
            
            metric_share = (group[col].values / team_total) * weight
            
            # Apply coverage multipliers to coverage metrics only
            if metric in COVERAGE_METRICS:
                metric_share = metric_share * coverage_multipliers
            
            pass_def_shares += metric_share
        
        # ========== CB ROLE ADJUSTMENT ==========
        # CB1s face tougher assignments, so boost their share
        # Use snap share within team's CBs as proxy for role
        cb_mask = (roles == "CB")
        if cb_mask.sum() > 1:  # Only apply if multiple CBs
            cb_cov_snaps = np.where(cb_mask, cov_snaps, 0)
            team_cb_total = cb_cov_snaps.sum()
            
            if team_cb_total > 0:
                cb_snap_shares = cb_cov_snaps / team_cb_total
                
                # Assign role multipliers based on snap share
                cb_role_multipliers = np.ones(len(group))
                for i, (is_cb, snap_share) in enumerate(zip(cb_mask, cb_snap_shares)):
                    if is_cb:
                        if snap_share >= CB1_SNAP_SHARE_THRESHOLD:
                            cb_role_multipliers[i] = CB_ROLE_ADJUSTMENT["CB1"]
                        elif snap_share >= CB2_SNAP_SHARE_THRESHOLD:
                            cb_role_multipliers[i] = CB_ROLE_ADJUSTMENT["CB2"]
                        else:
                            cb_role_multipliers[i] = CB_ROLE_ADJUSTMENT["CB3"]
                
                # Apply CB role adjustment to pass defense shares
                pass_def_shares = pass_def_shares * cb_role_multipliers
        
        player_df.loc[indices, "pass_defense_war"] = pass_def_shares * pass_def_pool
        
        # ========== RUN DEFENSE ALLOCATION ==========
        run_def_pool = team_war * BUCKET_WEIGHTS["run_defense"]
        
        # Get run defense position multipliers
        position_multipliers = np.array([
            RUN_DEFENSE_POSITION_MULTIPLIERS.get(role, 0.5) for role in roles
        ])
        
        # Box safety adjustment
        if "snap_counts_box" in group.columns and "snap_counts_defense" in group.columns:
            box_snaps = group["snap_counts_box"].values
            total_snaps = group["snap_counts_defense"].values
            box_rate = np.where(total_snaps > 0, box_snaps / total_snaps, 0)
            
            safety_mask = (roles == "S")
            if safety_mask.sum() > 0:
                safety_multipliers = np.clip(
                    (box_rate - SAFETY_BOX_RATE_THRESHOLD) / 
                    (SAFETY_BOX_RATE_FOR_MAX - SAFETY_BOX_RATE_THRESHOLD),
                    0, 1
                ) * SAFETY_MAX_RUN_MULTIPLIER
                
                position_multipliers = np.where(
                    safety_mask, 
                    safety_multipliers, 
                    position_multipliers
                )
        
        run_def_shares = np.zeros(len(group))
        
        for metric, weight in RUN_DEFENSE_ALLOCATION.items():
            col = metric
            if col not in group.columns:
                continue
                
            team_total = group[col].sum()
            if team_total <= 0:
                continue
            
            metric_share = (group[col].values / team_total) * weight
            run_def_shares += metric_share
        
        # Apply position multipliers
        run_def_shares = run_def_shares * position_multipliers
        
        player_df.loc[indices, "run_defense_war"] = run_def_shares * run_def_pool
    
    # Sum to get total WAR
    player_df["raw_def_war"] = player_df["pass_defense_war"] + player_df["run_defense_war"]
    
    # Calculate yards per snap for output
    player_df["yards_per_cov_snap"] = np.where(
        player_df["snap_counts_coverage"] > 0,
        player_df["yards"] / player_df["snap_counts_coverage"],
        0
    )
    
    # Show efficiency examples
    print("\n    Yards efficiency examples (qualified CBs):")
    qualified_cbs = player_df[
        (player_df["role"] == "CB") & 
        (player_df["snap_counts_coverage"] >= 400)
    ].copy()
    
    if len(qualified_cbs) > 0:
        # Best efficiency
        best = qualified_cbs.nsmallest(3, "yards_per_cov_snap")
        print("      Best (lowest yards/snap):")
        for _, row in best.iterrows():
            print(f"        {row['player']} ({row['season']} {row['team']}): {row['yards_per_cov_snap']:.2f} yds/snap")
        
        # Worst efficiency
        worst = qualified_cbs.nlargest(3, "yards_per_cov_snap")
        print("      Worst (highest yards/snap):")
        for _, row in worst.iterrows():
            print(f"        {row['player']} ({row['season']} {row['team']}): {row['yards_per_cov_snap']:.2f} yds/snap")
    
    # Show summary
    print(f"\n    Bucket WAR ranges:")
    print(f"      Pass Defense: [{player_df['pass_defense_war'].min():.3f}, {player_df['pass_defense_war'].max():.3f}]")
    print(f"      Run Defense:  [{player_df['run_defense_war'].min():.3f}, {player_df['run_defense_war'].max():.3f}]")
    print(f"\n    Raw WAR range: [{player_df['raw_def_war'].min():.3f}, {player_df['raw_def_war'].max():.3f}]")
    
    return player_df


# ============================================================
# STEP 5B: APPLY POSITION-SPECIFIC REGRESSION
# ============================================================

def apply_position_regression(player_df):
    """
    Apply position-specific regression toward the mean.
    
    CBs/Ss get more regression (coverage harder to isolate individually)
    EDGE gets less regression (sacks more individually attributable)
    """
    print("\n[5b] Applying position-specific regression...")
    
    # Calculate league average WAR
    qualified = player_df[player_df["snap_counts_defense"] >= MIN_SNAPS_FOR_WAR]
    league_avg_war = qualified["raw_def_war"].mean()
    
    print(f"    League average raw WAR: {league_avg_war:.4f}")
    print(f"\n    Position regression factors:")
    for role, factor in sorted(POSITION_REGRESSION_FACTORS.items(), key=lambda x: -x[1]):
        print(f"      {role}: {factor:.0%} toward mean")
    
    # Apply position-specific regression
    player_df["position_regression_factor"] = player_df["role"].map(POSITION_REGRESSION_FACTORS)
    player_df["position_regression_factor"] = player_df["position_regression_factor"].fillna(0.55)
    
    # Store pre-regression values
    player_df["raw_def_war_pre_regression"] = player_df["raw_def_war"]
    player_df["pass_defense_war_pre_regression"] = player_df["pass_defense_war"]
    player_df["run_defense_war_pre_regression"] = player_df["run_defense_war"]
    
    # Apply regression: regressed_war = factor * league_avg + (1 - factor) * actual
    player_df["raw_def_war"] = (
        player_df["position_regression_factor"] * league_avg_war +
        (1 - player_df["position_regression_factor"]) * player_df["raw_def_war_pre_regression"]
    )
    
    # Also regress the bucket WARs proportionally
    pass_avg = qualified["pass_defense_war"].mean()
    run_avg = qualified["run_defense_war"].mean()
    
    player_df["pass_defense_war"] = (
        player_df["position_regression_factor"] * pass_avg +
        (1 - player_df["position_regression_factor"]) * player_df["pass_defense_war_pre_regression"]
    )
    
    player_df["run_defense_war"] = (
        player_df["position_regression_factor"] * run_avg +
        (1 - player_df["position_regression_factor"]) * player_df["run_defense_war_pre_regression"]
    )
    
    # Apply final divisor scaling
    player_df["raw_def_war"] = player_df["raw_def_war"] * (11 / FINAL_WAR_DIVISOR)
    player_df["pass_defense_war"] = player_df["pass_defense_war"] * (11 / FINAL_WAR_DIVISOR)
    player_df["run_defense_war"] = player_df["run_defense_war"] * (11 / FINAL_WAR_DIVISOR)
    
    print(f"\n    Final WAR divisor: {FINAL_WAR_DIVISOR} (scaling by {11/FINAL_WAR_DIVISOR:.2f}x)")
    
    # Show impact on top players
    print(f"\n    Impact on top players (before → after regression):")
    qualified_after = player_df[player_df["snap_counts_defense"] >= MIN_SNAPS_FOR_WAR].copy()
    top_before = qualified_after.nlargest(10, "raw_def_war_pre_regression")
    
    for _, row in top_before.iterrows():
        before = row["raw_def_war_pre_regression"] * (11 / FINAL_WAR_DIVISOR)  # Scale for comparison
        after = row["raw_def_war"]
        change = after - before
        print(f"      {row['player']:20s} ({row['role']:4s}): {before:.3f} → {after:.3f} ({change:+.3f})")
    
    print(f"\n    Regressed WAR range: [{player_df['raw_def_war'].min():.3f}, {player_df['raw_def_war'].max():.3f}]")
    
    return player_df


# ============================================================
# STEP 6: APPLY REPLACEMENT LEVEL
# ============================================================

def apply_replacement_level(player_df):
    """Apply position-specific replacement levels."""
    print("\n[6] Applying role-specific replacement levels...")
    
    qualified = player_df[player_df["snap_counts_defense"] >= MIN_SNAPS_FOR_REPLACEMENT].copy()
    
    replacement_levels = (
        qualified.groupby("role")["raw_def_war"]
        .quantile(0.25)
        .reset_index()
        .rename(columns={"raw_def_war": "replacement_war"})
    )
    
    print("\n    Replacement levels (25th percentile):")
    for _, row in replacement_levels.sort_values("replacement_war").iterrows():
        print(f"      {row['role']}: {row['replacement_war']:.3f}")
    
    overall_replacement = qualified["raw_def_war"].quantile(0.25)
    
    player_df = player_df.merge(replacement_levels, on="role", how="left")
    player_df["replacement_war"] = player_df["replacement_war"].fillna(overall_replacement)
    
    player_df["def_war"] = player_df["raw_def_war"] - player_df["replacement_war"]
    
    # Summary
    qualified_war = player_df[player_df["snap_counts_defense"] >= MIN_SNAPS_FOR_WAR]
    
    print(f"\n    Final WAR distribution (>= {MIN_SNAPS_FOR_WAR} snaps, n={len(qualified_war)}):")
    print(f"      Mean:   {qualified_war['def_war'].mean():.3f}")
    print(f"      Std:    {qualified_war['def_war'].std():.3f}")
    print(f"      Median: {qualified_war['def_war'].median():.3f}")
    print(f"      Min:    {qualified_war['def_war'].min():.3f}")
    print(f"      Max:    {qualified_war['def_war'].max():.3f}")
    
    return player_df


# ============================================================
# STEP 7: OUTPUT RESULTS
# ============================================================

def output_results(player_df):
    """Save results and display leaderboards."""
    print("\n[7] Saving results...")
    
    # Calculate WAR/700 (per 700 snaps, ~full season starter)
    player_df["war_per_700"] = (player_df["def_war"] / player_df["snap_counts_defense"]) * 700
    player_df["war_per_700"] = player_df["war_per_700"].replace([np.inf, -np.inf], np.nan)
    
    # Output columns
    id_cols = ["season", "team", "player", "player_id", "position", "role"]
    stat_cols = ["snap_counts_defense", "snap_counts_coverage", "yards", "yards_per_cov_snap",
                 "total_pressures", "sacks", "stops", "interceptions", "forced_fumbles", "pass_break_ups"]
    war_cols = ["pass_defense_war", "run_defense_war", "raw_def_war", "replacement_war", "def_war", "war_per_700"]
    
    out_cols = [c for c in id_cols + stat_cols + war_cols if c in player_df.columns]
    out_df = player_df[out_cols].copy()
    out_df = out_df.sort_values("def_war", ascending=False)
    
    out_df.to_csv(WAR_OUT, index=False)
    print(f"    Saved → {WAR_OUT}")
    
    # Display results
    qualified = out_df[out_df["snap_counts_defense"] >= MIN_SNAPS_FOR_WAR].copy()
    
    # ==================== POSITIONAL SUMMARY TABLE ====================
    print(f"\n    " + "=" * 66)
    print(f"    POSITIONAL SUMMARY (>= {MIN_SNAPS_FOR_WAR} snaps)")
    print(f"    " + "=" * 66)
    
    position_summary = []
    for role in ["EDGE", "IDL", "LB", "CB", "S"]:
        role_df = qualified[qualified["role"] == role]
        if len(role_df) == 0:
            continue
        
        summary = {
            "Position": role,
            "Count": len(role_df),
            "Mean WAR": role_df["def_war"].mean(),
            "Median WAR": role_df["def_war"].median(),
            "Std WAR": role_df["def_war"].std(),
            "Min WAR": role_df["def_war"].min(),
            "Max WAR": role_df["def_war"].max(),
            "Avg WAR/700": role_df["war_per_700"].mean(),
            "Avg Pass Def": role_df["pass_defense_war"].mean(),
            "Avg Run Def": role_df["run_defense_war"].mean(),
        }
        
        # Add yards efficiency for coverage positions
        if role in ["CB", "S", "LB"]:
            summary["Avg Yds/Snap"] = role_df["yards_per_cov_snap"].mean()
        
        position_summary.append(summary)
    
    pos_summary_df = pd.DataFrame(position_summary)
    
    war_cols_display = ["Position", "Count", "Mean WAR", "Median WAR", "Std WAR", "Min WAR", "Max WAR", "Avg WAR/700"]
    print(f"\n    WAR by Position:")
    print(pos_summary_df[war_cols_display].to_string(index=False))
    
    bucket_cols_display = ["Position", "Avg Pass Def", "Avg Run Def"]
    print(f"\n    Average Bucket WAR by Position:")
    print(pos_summary_df[bucket_cols_display].to_string(index=False))
    
    # ==================== LEADERBOARDS ====================
    display_cols = ["player", "season", "team", "role", "snap_counts_defense", "yards_per_cov_snap",
                    "pass_defense_war", "run_defense_war", "def_war", "war_per_700"]
    display_cols = [c for c in display_cols if c in qualified.columns]
    
    print(f"\n    Top 25 Defensive WAR (>= {MIN_SNAPS_FOR_WAR} snaps):")
    print(qualified[display_cols].head(25).to_string(index=False))
    
    print(f"\n    Bottom 10 Defensive WAR (>= {MIN_SNAPS_FOR_WAR} snaps):")
    print(qualified[display_cols].tail(10).to_string(index=False))
    
    print(f"\n    Top 5 by Role:")
    for role in ["EDGE", "IDL", "LB", "CB", "S"]:
        role_df = qualified[qualified["role"] == role].head(5)
        if len(role_df) > 0:
            print(f"\n    {role}:")
            if role in ["CB", "S", "LB"]:
                mini_cols = ["player", "season", "team", "snap_counts_defense", "yards_per_cov_snap", "def_war", "war_per_700"]
            else:
                mini_cols = ["player", "season", "team", "snap_counts_defense", "def_war", "war_per_700"]
            mini_cols = [c for c in mini_cols if c in role_df.columns]
            print(role_df[mini_cols].to_string(index=False))
    
    # ==================== CUMULATIVE WAR (2021-2024) ====================
    print(f"\n    " + "=" * 66)
    print(f"    CUMULATIVE WAR (2021-2024)")
    print(f"    " + "=" * 66)
    
    cumulative = qualified.groupby("player").agg({
        "def_war": "sum",
        "season": "count",
        "snap_counts_defense": "sum",
        "snap_counts_coverage": "sum",
        "yards": "sum",
        "pass_defense_war": "sum",
        "run_defense_war": "sum",
        "role": "first",
        "team": "last",
    }).rename(columns={"season": "seasons"})
    
    cumulative = cumulative.reset_index()
    
    # Calculate cumulative yards per snap and WAR/700
    cumulative["yards_per_cov_snap"] = np.where(
        cumulative["snap_counts_coverage"] > 0,
        cumulative["yards"] / cumulative["snap_counts_coverage"],
        0
    )
    cumulative["war_per_700"] = (cumulative["def_war"] / cumulative["snap_counts_defense"]) * 700
    
    cumulative = cumulative.sort_values("def_war", ascending=False)
    
    print(f"\n    Top 20 Cumulative Defensive WAR (2021-2024):")
    cum_cols = ["player", "role", "team", "seasons", "snap_counts_defense",
                "yards_per_cov_snap", "pass_defense_war", "run_defense_war", "def_war", "war_per_700"]
    cum_cols = [c for c in cum_cols if c in cumulative.columns]
    print(cumulative[cum_cols].head(20).to_string(index=False))
    
    print(f"\n    Top 5 Cumulative by Role:")
    for role in ["EDGE", "IDL", "LB", "CB", "S"]:
        role_df = cumulative[cumulative["role"] == role].head(5)
        if len(role_df) > 0:
            print(f"\n    {role}:")
            if role in ["CB", "S", "LB"]:
                mini_cols = ["player", "team", "seasons", "snap_counts_defense", "yards_per_cov_snap", "def_war", "war_per_700"]
            else:
                mini_cols = ["player", "team", "seasons", "snap_counts_defense", "def_war", "war_per_700"]
            mini_cols = [c for c in mini_cols if c in role_df.columns]
            print(role_df[mini_cols].to_string(index=False))
    
    # ============================================================
    # DIAGNOSTIC: PAULSON ADEBO SEASON-BY-SEASON BREAKDOWN
    # ============================================================
    print("\n" + "=" * 70)
    print(" DIAGNOSTIC: PAULSON ADEBO SEASON-BY-SEASON")
    print("=" * 70)
    
    adebo_seasons = qualified[qualified["player"].str.contains("Adebo", case=False, na=False)]
    
    if len(adebo_seasons) > 0:
        print(f"\n    Found {len(adebo_seasons)} season(s) for Adebo:")
        
        diag_cols = ["player", "season", "team", "snap_counts_defense", "snap_counts_coverage",
                     "yards", "yards_per_cov_snap", "pbus", "interceptions", "targets", "catch_rate",
                     "pass_defense_war", "run_defense_war", "def_war", "war_per_700"]
        diag_cols = [c for c in diag_cols if c in adebo_seasons.columns]
        
        print("\n    Season-by-Season Stats:")
        print(adebo_seasons[diag_cols].to_string(index=False))
        
        # Show his rank each season
        print("\n    Rank Among CBs Each Season:")
        for _, row in adebo_seasons.iterrows():
            season = row["season"]
            war = row["def_war"]
            season_cbs = qualified[(qualified["season"] == season) & (qualified["role"] == "CB")]
            rank = (season_cbs["def_war"] > war).sum() + 1
            total = len(season_cbs)
            print(f"      {season}: {war:.2f} WAR (#{rank} of {total} CBs)")
        
        # Cumulative breakdown
        print(f"\n    Cumulative Totals:")
        print(f"      Total Snaps: {adebo_seasons['snap_counts_defense'].sum():.0f}")
        print(f"      Total WAR: {adebo_seasons['def_war'].sum():.2f}")
        print(f"      Seasons Qualified: {len(adebo_seasons)}")
        
        # Compare to other top CBs
        print("\n    Comparison to Other Top Cumulative CBs:")
        top_cbs = cumulative[cumulative["role"] == "CB"].head(5)
        compare_cols = ["player", "seasons", "snap_counts_defense", "def_war", "war_per_700"]
        compare_cols = [c for c in compare_cols if c in top_cbs.columns]
        print(top_cbs[compare_cols].to_string(index=False))
    else:
        print("\n    Adebo not found in qualified players (may have <200 snaps)")
        
        # Check unqualified data
        all_adebo = player_df[player_df["player"].str.contains("Adebo", case=False, na=False)]
        if len(all_adebo) > 0:
            print(f"\n    Found in raw data (before filtering):")
            raw_cols = ["player", "season", "team", "snap_counts_defense", "snap_counts_coverage"]
            raw_cols = [c for c in raw_cols if c in all_adebo.columns]
            print(all_adebo[raw_cols].to_string(index=False))
    
    print("=" * 70)
    
    # Save summary
    with open(SUMMARY_OUT, "w") as f:
        f.write("DEFENSIVE WAR MODEL SUMMARY (v23 — WITH CB ROLE ADJUSTMENT)\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("KEY CHANGES IN v23:\n")
        f.write("  - CB Role Adjustment based on snap share:\n")
        f.write(f"      CB1 (>{CB1_SNAP_SHARE_THRESHOLD:.0%} of team CB snaps): {CB_ROLE_ADJUSTMENT['CB1']:.2f}x\n")
        f.write(f"      CB2 ({CB2_SNAP_SHARE_THRESHOLD:.0%}-{CB1_SNAP_SHARE_THRESHOLD:.0%}): {CB_ROLE_ADJUSTMENT['CB2']:.2f}x\n")
        f.write(f"      CB3 (<{CB2_SNAP_SHARE_THRESHOLD:.0%}): {CB_ROLE_ADJUSTMENT['CB3']:.2f}x\n")
        f.write("  - Fixes 'CB1 penalized for tough assignments' problem\n\n")
        
        f.write("PREVIOUS CHANGES (v22):\n")
        f.write("  - Position-specific regression toward mean:\n")
        f.write("      CB: 65%, S: 60%, LB: 55%, IDL: 50%, EDGE: 45%\n")
        f.write("  - Reduced final WAR divisor from 5.5 to 4.5\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("  1. Two-bucket structure: Pass Defense (69%) + Run Defense (31%)\n")
        f.write("  2. Coverage efficiency: Lower yards/snap = higher WAR share\n")
        f.write("  3. CB Role Adjustment: CB1s boosted, CB3s reduced\n")
        f.write("  4. Team WAR regressed 50% toward league mean\n")
        f.write("  5. Position multipliers applied within buckets\n")
        f.write("  6. Position-specific regression (CBs more, EDGE less)\n")
        f.write("  7. Role-specific replacement levels\n\n")
        
        f.write("PASS DEFENSE ALLOCATION (69% of total):\n")
        for metric, weight in PASS_DEFENSE_ALLOCATION.items():
            f.write(f"  {metric}: {weight:.0%}\n")
        
        f.write("\nRUN DEFENSE ALLOCATION (31% of total):\n")
        for metric, weight in RUN_DEFENSE_ALLOCATION.items():
            f.write(f"  {metric}: {weight:.0%}\n")
        
        f.write("\nPOSITION REGRESSION FACTORS:\n")
        for role, factor in sorted(POSITION_REGRESSION_FACTORS.items(), key=lambda x: -x[1]):
            f.write(f"  {role}: {factor:.0%} toward mean\n")
        
        f.write("\nPOSITION MULTIPLIERS:\n")
        f.write("  Run Defense: EDGE 1.07x, IDL 1.00x, LB 0.21x, CB/S 0.00x\n")
        f.write("  Coverage:    CB 1.00x, S 0.94x, LB 0.81x, EDGE/IDL 0.50x\n")
        
        f.write("\n\nPOSITIONAL SUMMARY:\n")
        f.write(pos_summary_df[war_cols_display].to_string(index=False))
        
        f.write("\n\nAVERAGE BUCKET WAR BY POSITION:\n")
        f.write(pos_summary_df[bucket_cols_display].to_string(index=False))
        
        f.write(f"\n\nTOP 25 DEFENSIVE WAR:\n")
        f.write(qualified[display_cols].head(25).to_string(index=False))
        
        f.write(f"\n\nTOP 5 BY ROLE:\n")
        for role in ["EDGE", "IDL", "LB", "CB", "S"]:
            role_df = qualified[qualified["role"] == role].head(5)
            if len(role_df) > 0:
                f.write(f"\n{role}:\n")
                f.write(role_df[["player", "season", "team", "def_war"]].to_string(index=False))
                f.write("\n")
    
    print(f"    Saved summary → {SUMMARY_OUT}")
    
    return out_df


# ============================================================
# MAIN
# ============================================================

def main():
    team_df, player_df = load_data()
    team_activity_df = build_team_activity_metrics(player_df)
    display_weights()
    team_df = compute_team_war(team_df)
    player_df = allocate_war_to_players(player_df, team_df)
    player_df = apply_position_regression(player_df)  # NEW: Position-specific regression
    player_df = apply_replacement_level(player_df)
    output_results(player_df)
    
    print("\n" + "=" * 70)
    print(" DONE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
