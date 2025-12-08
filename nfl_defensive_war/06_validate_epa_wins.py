"""
06_validate_epa_wins.py
========================
Validate the relationship between defensive EPA and wins.

Key questions:
1. What's the correlation between team defensive EPA and wins?
2. How many defensive EPA saved = 1 win?
3. Does our WAR scaling make sense given this relationship?

"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("/Users/anokhpalakurthi/Documents/nfl_defensive_war")
OUTPUTS_DIR = BASE_DIR / "outputs"
TEAM_AGG_PATH = OUTPUTS_DIR / "team_defense_agg_2021_2024.csv"
WAR_PATH = OUTPUTS_DIR / "defensive_war_2021_2024.csv"

# ============================================================
# 2024 TEAM RECORDS (INCLUDING POSTSEASON)
# ============================================================

# Format: team abbreviation -> (wins, losses, ties)
TEAM_RECORDS_2024 = {
    # AFC East
    "BUF": (15, 5, 0),   # Lost Divisional
    "MIA": (8, 9, 0),
    "NE": (4, 13, 0),
    "NYJ": (5, 12, 0),
    # AFC North
    "BAL": (13, 5, 0),   # Lost Divisional
    "PIT": (10, 8, 0),   # Lost Wild Card
    "CLE": (3, 14, 0),
    "CIN": (9, 8, 0),
    # AFC South
    "HOU": (11, 7, 0),   # Lost Divisional
    "IND": (8, 9, 0),
    "TEN": (3, 14, 0),
    "JAC": (4, 13, 0),
    # AFC West
    "KC": (19, 2, 0),    # Super Bowl Champions
    "LAC": (11, 7, 0),   # Lost Wild Card
    "DEN": (10, 7, 0),   # Lost Wild Card
    "LV": (4, 13, 0),
    # NFC East
    "PHI": (16, 4, 0),   # Lost Super Bowl
    "WAS": (14, 6, 0),   # Lost Divisional
    "DAL": (7, 10, 0),
    "NYG": (3, 14, 0),
    # NFC North
    "DET": (16, 3, 0),   # Lost Championship
    "MIN": (14, 5, 0),   # Lost Wild Card
    "GB": (12, 6, 0),    # Lost Wild Card
    "CHI": (5, 12, 0),
    # NFC South
    "TB": (10, 8, 0),    # Lost Wild Card
    "ATL": (8, 9, 0),
    "NO": (5, 12, 0),
    "CAR": (5, 12, 0),
    # NFC West
    "LAR": (11, 7, 0),   # Lost Divisional
    "SEA": (10, 7, 0),
    "ARI": (8, 9, 0),
    "SF": (6, 11, 0),
}

def get_2024_wins_df():
    """Convert hardcoded records to DataFrame."""
    records = []
    for team, (w, l, t) in TEAM_RECORDS_2024.items():
        records.append({
            "season": 2024,
            "team": team,
            "wins": w,
            "losses": l,
            "ties": t,
            "win_pct": (w + 0.5 * t) / (w + l + t)
        })
    return pd.DataFrame(records)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load team and player WAR data."""
    print("Loading data...")
    
    team_df = pd.read_csv(TEAM_AGG_PATH)
    player_df = pd.read_csv(WAR_PATH)
    
    # Add wins data for 2024
    wins_df = get_2024_wins_df()
    team_df = team_df.merge(wins_df, on=["season", "team"], how="left")
    
    print(f"  Team-seasons: {len(team_df)}")
    print(f"  Team-seasons with wins data (2024): {team_df['wins'].notna().sum()}")
    print(f"  Player-seasons: {len(player_df)}")
    
    return team_df, player_df


# ============================================================
# ANALYSIS 1: DEFENSIVE EPA vs WINS (2024 ONLY)
# ============================================================

def analyze_epa_wins(team_df):
    """Check correlation between defensive EPA and wins."""
    print("\n" + "=" * 60)
    print(" DEFENSIVE EPA vs WINS RELATIONSHIP (2024)")
    print("=" * 60)
    
    # Filter to 2024 only (where we have wins data)
    df_2024 = team_df[team_df["season"] == 2024].copy()
    
    if len(df_2024) == 0:
        print("\n  ERROR: No 2024 data found.")
        return None
    
    print(f"\n  Teams in analysis: {len(df_2024)}")
    
    # Calculate total defensive EPA
    df_2024["total_def_epa"] = df_2024["def_epa_per_play"] * df_2024["def_plays"]
    
    # Correlation
    corr = df_2024["total_def_epa"].corr(df_2024["wins"])
    print(f"\n  Correlation (Total Def EPA vs Wins): {corr:.3f}")
    
    # Note: Lower (more negative) defensive EPA = better defense = more wins
    # So we expect NEGATIVE correlation
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df_2024["total_def_epa"], df_2024["wins"]
    )
    
    print(f"\n  Linear Regression:")
    print(f"    Slope: {slope:.4f} wins per EPA")
    print(f"    Intercept: {intercept:.2f} wins")
    print(f"    R-squared: {r_value**2:.3f}")
    print(f"    P-value: {p_value:.6f}")
    
    # Interpretation
    epa_per_win = 1 / slope if slope != 0 else np.nan
    print(f"\n  Interpretation:")
    print(f"    1 Win = {epa_per_win:.1f} Defensive EPA saved")
    print(f"    (Negative because lower EPA = better defense)")
    
    # Show best/worst defenses
    print(f"\n  Top 5 Defenses by EPA (2024):")
    for _, row in df_2024.nsmallest(5, "total_def_epa").iterrows():
        print(f"    {row['team']}: {row['total_def_epa']:.1f} EPA, {row['wins']:.0f} wins")
    
    print(f"\n  Bottom 5 Defenses by EPA (2024):")
    for _, row in df_2024.nlargest(5, "total_def_epa").iterrows():
        print(f"    {row['team']}: {row['total_def_epa']:.1f} EPA, {row['wins']:.0f} wins")
    
    return {
        "correlation": corr,
        "slope": slope,
        "r_squared": r_value**2,
        "epa_per_win": epa_per_win,
        "df_2024": df_2024
    }


# ============================================================
# ANALYSIS 2: SUMMED PLAYER WAR vs TEAM WINS (2024 ONLY)
# ============================================================

def analyze_war_wins(player_df, team_df):
    """Check if summed player WAR correlates with team wins."""
    print("\n" + "=" * 60)
    print(" SUMMED PLAYER WAR vs WINS (2024)")
    print("=" * 60)
    
    # Filter to 2024 only
    player_2024 = player_df[player_df["season"] == 2024].copy()
    
    # Sum player WAR by team
    team_war = player_2024.groupby("team").agg({
        "def_war": "sum",
        "pass_defense_war": "sum",
        "run_defense_war": "sum",
    }).reset_index()
    
    team_war.columns = ["team", "total_player_war", "total_pass_war", "total_run_war"]
    
    print(f"\n  Team WAR distribution (2024):")
    print(f"    Mean: {team_war['total_player_war'].mean():.2f}")
    print(f"    Std:  {team_war['total_player_war'].std():.2f}")
    print(f"    Min:  {team_war['total_player_war'].min():.2f}")
    print(f"    Max:  {team_war['total_player_war'].max():.2f}")
    
    # Merge with wins data
    wins_df = get_2024_wins_df()
    merged = team_war.merge(wins_df[["team", "wins"]], on="team")
    
    print(f"\n  Teams matched: {len(merged)}")
    
    # Correlation
    corr = merged["total_player_war"].corr(merged["wins"])
    print(f"\n  Correlation (Summed Player WAR vs Wins): {corr:.3f}")
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        merged["total_player_war"], merged["wins"]
    )
    
    print(f"\n  Linear Regression:")
    print(f"    Slope: {slope:.2f} wins per WAR")
    print(f"    Intercept: {intercept:.2f} wins (baseline)")
    print(f"    R-squared: {r_value**2:.3f}")
    print(f"    P-value: {p_value:.6f}")
    
    print(f"\n  Interpretation:")
    print(f"    1 Player WAR ≈ {slope:.2f} Wins")
    print(f"    Baseline (0 WAR team) ≈ {intercept:.1f} wins")
    
    # Ideally: slope ≈ 1.0 and intercept ≈ 4 (replacement-level team)
    if slope > 0:
        print(f"\n  Scaling assessment:")
        if 0.8 <= slope <= 1.2:
            print(f"    ✓ Slope near 1.0 - WAR scaling looks good!")
        elif slope < 0.8:
            print(f"    ⚠ Slope < 1.0 - WAR may be OVERSCALED (values too high)")
            print(f"      Consider increasing divisor by {1/slope:.1f}x")
        else:
            print(f"    ⚠ Slope > 1.0 - WAR may be UNDERSCALED (values too low)")
            print(f"      Consider decreasing divisor by {slope:.1f}x")
    
    # Show best/worst teams
    print(f"\n  Top 5 Teams by Summed Player WAR (2024):")
    for _, row in merged.nlargest(5, "total_player_war").iterrows():
        print(f"    {row['team']}: {row['total_player_war']:.2f} WAR, {row['wins']:.0f} wins")
    
    print(f"\n  Bottom 5 Teams by Summed Player WAR (2024):")
    for _, row in merged.nsmallest(5, "total_player_war").iterrows():
        print(f"    {row['team']}: {row['total_player_war']:.2f} WAR, {row['wins']:.0f} wins")
    
    return merged


# ============================================================
# ANALYSIS 3: SANITY CHECK CURRENT SCALING (2024)
# ============================================================

def sanity_check_scaling(player_df):
    """Check if current WAR values make sense."""
    print("\n" + "=" * 60)
    print(" SANITY CHECK: CURRENT WAR SCALING (2024)")
    print("=" * 60)
    
    # Filter to 2024 and qualified players
    df_2024 = player_df[(player_df["season"] == 2024) & (player_df["snap_counts_defense"] >= 50)].copy()
    
    # Average WAR by position
    pos_avg = df_2024.groupby("role")["def_war"].mean()
    print("\n  Average WAR by Position (2024):")
    for pos in ["EDGE", "S", "CB", "LB", "IDL"]:
        if pos in pos_avg.index:
            print(f"    {pos}: {pos_avg[pos]:.3f}")
    
    # Expected team total (rough estimate)
    # 2 EDGE + 2 IDL + 3 LB + 2 CB + 2 S = 11 starters
    expected_team = (
        2 * pos_avg.get("EDGE", 0) +
        2 * pos_avg.get("IDL", 0) +
        3 * pos_avg.get("LB", 0) +
        2 * pos_avg.get("CB", 0) +
        2 * pos_avg.get("S", 0)
    )
    
    print(f"\n  Expected average team WAR (11 starters): {expected_team:.2f}")
    
    # Actual team totals for 2024
    team_totals = df_2024.groupby("team")["def_war"].sum()
    print(f"\n  Actual team WAR totals (2024):")
    print(f"    Mean: {team_totals.mean():.2f}")
    print(f"    Std:  {team_totals.std():.2f}")
    print(f"    Min:  {team_totals.min():.2f}")
    print(f"    Max:  {team_totals.max():.2f}")
    
    # If 1 WAR = 1 Win, what does this imply?
    print(f"\n  If 1 WAR = 1 Win:")
    print(f"    Average team defense contributes: {team_totals.mean():.1f} wins")
    print(f"    Best defense contributes: {team_totals.max():.1f} wins")
    print(f"    Worst defense contributes: {team_totals.min():.1f} wins")


# ============================================================
# VISUALIZATION
# ============================================================

def plot_war_vs_wins(merged, save_path=None):
    """Scatter plot of team WAR vs wins."""
    if merged is None or "wins" not in merged.columns:
        print("\n  Cannot create plot (missing wins data)")
        return
    
    print("\n  Creating WAR vs Wins scatter plot...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(merged["total_player_war"], merged["wins"], alpha=0.7, s=100, edgecolor="black")
    
    # Label each point with team name
    for _, row in merged.iterrows():
        ax.annotate(row["team"], (row["total_player_war"], row["wins"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    
    # Add regression line
    slope, intercept, _, _, _ = stats.linregress(
        merged["total_player_war"], merged["wins"]
    )
    x_line = np.linspace(merged["total_player_war"].min() - 0.5, merged["total_player_war"].max() + 0.5, 100)
    ax.plot(x_line, slope * x_line + intercept, "r--", linewidth=2,
            label=f"Fit: {slope:.2f}x + {intercept:.1f}")
    
    # Add reference line (1 WAR = 1 Win, baseline at 8.5 for average)
    ax.plot(x_line, x_line + 8.5 - merged["total_player_war"].mean(), "g--", alpha=0.5, 
            linewidth=2, label="Reference: 1 WAR = 1 Win")
    
    ax.set_xlabel("Summed Player Defensive WAR", fontsize=12)
    ax.set_ylabel("Team Wins (incl. playoffs)", fontsize=12)
    ax.set_title("Defensive WAR vs Team Wins (2024 Season)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    
    plt.show()
    return fig


# ============================================================
# MAIN
# ============================================================

def main():
    team_df, player_df = load_data()
    
    # Run analyses (2024 only)
    epa_results = analyze_epa_wins(team_df)
    war_results = analyze_war_wins(player_df, team_df)
    sanity_check_scaling(player_df)
    
    # Plot
    if war_results is not None:
        FIG_DIR = OUTPUTS_DIR / "figures"
        FIG_DIR.mkdir(exist_ok=True)
        plot_war_vs_wins(war_results, save_path=FIG_DIR / "05_war_vs_wins_2024.png")
    
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    
if __name__ == "__main__":
    main()