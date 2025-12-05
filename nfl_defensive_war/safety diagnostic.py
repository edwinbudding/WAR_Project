"""
Quick diagnostic: Safety box rate distribution
Add to existing analysis or run standalone to see FS vs SS split
"""

import pandas as pd
from pathlib import Path

PLAYER_PATH = Path("/Users/anokhpalakurthi/Documents/nfl_defensive_war/outputs/defense_engineered_2021_2024.csv")

def main():
    print("=" * 70)
    print(" SAFETY BOX RATE ANALYSIS")
    print("=" * 70)
    
    players = pd.read_csv(PLAYER_PATH)
    
    # Filter to safeties with minimum snaps
    safeties = players[(players['role'] == 'S') & (players['snap_counts_defense'] >= 200)].copy()
    print(f"\n    Safeties with >= 200 snaps: {len(safeties)}")
    
    # Calculate box rate and deep rate
    safeties['box_rate'] = safeties['snap_counts_box'] / safeties['snap_counts_defense']
    safeties['deep_rate'] = safeties['snap_counts_fs'] / safeties['snap_counts_defense']
    
    print("\n    BOX RATE DISTRIBUTION:")
    print(f"      Min:    {safeties['box_rate'].min():.1%}")
    print(f"      25th:   {safeties['box_rate'].quantile(0.25):.1%}")
    print(f"      Median: {safeties['box_rate'].median():.1%}")
    print(f"      75th:   {safeties['box_rate'].quantile(0.75):.1%}")
    print(f"      Max:    {safeties['box_rate'].max():.1%}")
    
    print("\n    DEEP RATE DISTRIBUTION:")
    print(f"      Min:    {safeties['deep_rate'].min():.1%}")
    print(f"      25th:   {safeties['deep_rate'].quantile(0.25):.1%}")
    print(f"      Median: {safeties['deep_rate'].median():.1%}")
    print(f"      75th:   {safeties['deep_rate'].quantile(0.75):.1%}")
    print(f"      Max:    {safeties['deep_rate'].max():.1%}")
    
    # Define box-heavy vs deep safeties
    # Box-heavy: > 25% box snaps
    # Deep-heavy: > 30% deep snaps
    box_heavy = safeties[safeties['box_rate'] > 0.25]
    deep_heavy = safeties[safeties['deep_rate'] > 0.30]
    
    print(f"\n    Box-heavy safeties (>25% box): {len(box_heavy)} ({len(box_heavy)/len(safeties):.1%})")
    print(f"    Deep-heavy safeties (>30% deep): {len(deep_heavy)} ({len(deep_heavy)/len(safeties):.1%})")
    
    # Top box-heavy safeties
    print("\n    TOP 10 BOX-HEAVY SAFETIES:")
    top_box = safeties.nlargest(10, 'box_rate')[['player', 'season', 'team', 'box_rate', 'deep_rate', 'snap_counts_defense']]
    for _, row in top_box.iterrows():
        print(f"      {row['player']:<20} {row['season']} {row['team']}: {row['box_rate']:.1%} box, {row['deep_rate']:.1%} deep")
    
    # Top deep safeties
    print("\n    TOP 10 DEEP SAFETIES:")
    top_deep = safeties.nlargest(10, 'deep_rate')[['player', 'season', 'team', 'box_rate', 'deep_rate', 'snap_counts_defense']]
    for _, row in top_deep.iterrows():
        print(f"      {row['player']:<20} {row['season']} {row['team']}: {row['box_rate']:.1%} box, {row['deep_rate']:.1%} deep")
    
    # Check Kyle Hamilton specifically
    print("\n    KYLE HAMILTON:")
    hamilton = safeties[safeties['player'].str.contains('Hamilton', case=False)]
    if len(hamilton) > 0:
        for _, row in hamilton.iterrows():
            print(f"      {row['season']} {row['team']}: {row['box_rate']:.1%} box, {row['deep_rate']:.1%} deep")
    
    # Check Marcus Williams
    print("\n    MARCUS WILLIAMS:")
    williams = safeties[safeties['player'].str.contains('Marcus Williams', case=False)]
    if len(williams) > 0:
        for _, row in williams.iterrows():
            print(f"      {row['season']} {row['team']}: {row['box_rate']:.1%} box, {row['deep_rate']:.1%} deep")
    
    print("\n" + "=" * 70)
    print(" DONE")
    print("=" * 70)

if __name__ == "__main__":
    main()