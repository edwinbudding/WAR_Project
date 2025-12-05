"""
Build Player ID Crosswalk
=========================

nflfastR roster data contains both:
- gsis_id (what PBP uses: "00-0029585")
- full player names

We can match full names to your PFF player data to create a mapping.

This script:
1. Loads nflfastR roster data (you may need to download it)
2. Creates gsis_id -> full_name mapping
3. Matches to your PFF player data by name
4. Creates gsis_id -> position mapping for the EPA analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path("/Users/anokhpalakurthi/Documents/nfl_defensive_war/data")
PBP_PATH = DATA_DIR / "pbp_2021_2024_full.csv"
PLAYER_PATH = Path("/Users/anokhpalakurthi/Documents/nfl_defensive_war/outputs/defense_engineered_2021_2024.csv")

# nflfastR roster data - you may need to download this
# Option 1: If you have it locally
ROSTER_PATH = DATA_DIR / "roster_2021_2024.csv"  # Adjust filename

# Option 2: Download from nflfastR GitHub
ROSTER_URL = "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{year}.csv"


def download_rosters(years=[2021, 2022, 2023, 2024]):
    """Download roster data from nflfastR."""
    print("Downloading roster data from nflfastR...")
    dfs = []
    for year in years:
        url = ROSTER_URL.format(year=year)
        try:
            df = pd.read_csv(url)
            df['season'] = year
            dfs.append(df)
            print(f"  {year}: {len(df)} players")
        except Exception as e:
            print(f"  {year}: FAILED - {e}")
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None


def main():
    print("=" * 70)
    print(" BUILD GSIS_ID TO POSITION CROSSWALK")
    print("=" * 70)
    
    # Try to load local roster data first
    roster = None
    if ROSTER_PATH.exists():
        print(f"\n[1] Loading local roster data: {ROSTER_PATH}")
        roster = pd.read_csv(ROSTER_PATH)
    else:
        print(f"\n[1] No local roster found at {ROSTER_PATH}")
        print("    Attempting to download from nflfastR...")
        roster = download_rosters()
        if roster is not None:
            # Save for future use
            roster.to_csv(DATA_DIR / "roster_2021_2024.csv", index=False)
            print(f"    Saved roster data to {DATA_DIR / 'roster_2021_2024.csv'}")
    
    if roster is None:
        print("    ERROR: Could not load roster data")
        return
    
    print(f"    Roster records: {len(roster):,}")
    print(f"    Columns: {list(roster.columns)[:15]}")
    
    # Check what ID columns exist
    print("\n[2] ROSTER ID COLUMNS:")
    id_cols = [col for col in roster.columns if 'id' in col.lower()]
    print(f"    {id_cols}")
    
    # Sample the data
    print("\n    Sample roster data:")
    sample_cols = ['full_name', 'gsis_id', 'position', 'team'] 
    sample_cols = [c for c in sample_cols if c in roster.columns]
    if sample_cols:
        print(roster[sample_cols].head(10).to_string())
    
    # Load PFF player data
    print("\n[3] LOADING PFF PLAYER DATA:")
    players = pd.read_csv(PLAYER_PATH)
    print(f"    Players: {len(players):,}")
    
    # Create name -> role mapping from PFF data
    # Use most common role per player name
    pff_name_to_role = players.groupby('player')['role'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]).to_dict()
    print(f"    Unique player names in PFF: {len(pff_name_to_role)}")
    
    # Create gsis_id -> full_name mapping from roster
    print("\n[4] BUILDING CROSSWALK:")
    
    name_col = 'full_name' if 'full_name' in roster.columns else 'player_name'
    gsis_col = 'gsis_id'
    
    if gsis_col not in roster.columns:
        print(f"    ERROR: No gsis_id column in roster data")
        print(f"    Available: {roster.columns.tolist()}")
        return
    
    # Get unique gsis_id -> name mappings
    roster_clean = roster[[gsis_col, name_col]].dropna().drop_duplicates()
    gsis_to_name = dict(zip(roster_clean[gsis_col], roster_clean[name_col]))
    print(f"    GSIS ID -> Name mappings: {len(gsis_to_name)}")
    
    # Now match names to PFF roles
    gsis_to_role = {}
    matched = 0
    unmatched_names = []
    
    for gsis_id, full_name in gsis_to_name.items():
        if full_name in pff_name_to_role:
            gsis_to_role[gsis_id] = pff_name_to_role[full_name]
            matched += 1
        else:
            unmatched_names.append(full_name)
    
    print(f"    Matched GSIS IDs to PFF roles: {matched}")
    print(f"    Unmatched names: {len(unmatched_names)}")
    
    if unmatched_names:
        print(f"\n    Sample unmatched names (in roster but not PFF):")
        for name in unmatched_names[:10]:
            print(f"      {name}")
    
    # Save the crosswalk
    crosswalk_path = DATA_DIR / "gsis_to_position_crosswalk.csv"
    crosswalk_df = pd.DataFrame([
        {'gsis_id': k, 'role': v} for k, v in gsis_to_role.items()
    ])
    crosswalk_df.to_csv(crosswalk_path, index=False)
    print(f"\n    Saved crosswalk to {crosswalk_path}")
    
    # Test the crosswalk on PBP data
    print("\n[5] TESTING CROSSWALK ON PBP DATA:")
    pbp = pd.read_csv(PBP_PATH, low_memory=False)
    
    if 'play_type' in pbp.columns:
        run_plays = pbp[pbp['play_type'] == 'run'].copy()
    else:
        run_plays = pbp[pbp['rush'] == 1].copy()
    
    run_plays = run_plays.dropna(subset=['epa', 'solo_tackle_1_player_id'])
    print(f"    Run plays with tackler: {len(run_plays):,}")
    
    run_plays['tackler_role'] = run_plays['solo_tackle_1_player_id'].map(gsis_to_role)
    matched_plays = run_plays['tackler_role'].notna().sum()
    print(f"    Plays with matched tackler position: {matched_plays:,} ({matched_plays/len(run_plays)*100:.1f}%)")
    
    if matched_plays > 0:
        print("\n" + "=" * 70)
        print(" EPA BY TACKLER POSITION (Run Plays)")
        print("=" * 70)
        
        known = run_plays[run_plays['tackler_role'].notna()]
        
        print("\n    BY POSITION:")
        for pos in ['IDL', 'EDGE', 'LB', 'S', 'CB']:
            subset = known[known['tackler_role'] == pos]
            if len(subset) > 100:
                epa_mean = subset['epa'].mean()
                yards_mean = subset['yards_gained'].mean() if 'yards_gained' in subset.columns else np.nan
                print(f"      {pos}: n={len(subset):,}, EPA={epa_mean:.3f}, Yards={yards_mean:.1f}")
        
        # Key comparison
        idl = known[known['tackler_role'] == 'IDL']['epa']
        lb = known[known['tackler_role'] == 'LB']['epa']
        
        if len(idl) > 100 and len(lb) > 100:
            print(f"\n    IDL avg EPA on tackles: {idl.mean():.3f}")
            print(f"    LB avg EPA on tackles:  {lb.mean():.3f}")
            print(f"    Difference: {lb.mean() - idl.mean():.3f}")
    
    print("\n" + "=" * 70)
    print(" DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()