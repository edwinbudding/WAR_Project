"""
06_war_vs_market_showdown.py
Compare WAR methodologies (mine vs PFF) against market valuations.
The ultimate question: Who's right about positional value - the nerds or the money?

Inputs:
    - Hardcoded WAR averages from my analysis and PFF's 2018 study
    - OTC market salary data (2021)

Outputs:
    - outputs/figures/war_vs_market_comparison.png
    - outputs/tables/war_market_correlation_summary.csv
    - Console output with correlation analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUTS_DIR = Path("/Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs")
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Ensure output directories exist
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA
# ============================================================================

# My WAR averages (2021-2024)
MY_WAR = {
    "QB": 1.05,
    "WR": 0.30,
    "T": 0.16,
    "G": 0.15,
    "C": 0.15,
    "TE": 0.15,
    "RB": 0.06,
}

# PFF WAR averages (2006-2018)
PFF_WAR = {
    "QB": 1.63,
    "WR": 0.28,
    "T": 0.09,
    "G": 0.10,
    "C": 0.10,
    "TE": 0.18,
    "RB": 0.10,
}

# OTC Market Data (2021) - Avg/Player salary
# Combining LT + RT for T, LG + RG for G
OTC_SALARY = {
    "QB": 8443048,
    "LT": 4889341,
    "RT": 4271117,
    "WR": 3443415,
    "C": 3217725,
    "G": 2872489,
    "TE": 2578763,
    "RB": 2327709,
}


# ============================================================================
# FUNCTIONS
# ============================================================================

def build_comparison_df() -> pd.DataFrame:
    """Build DataFrame comparing WAR and market values."""
    
    # Combine tackle salaries (average of LT and RT)
    tackle_salary = (OTC_SALARY["LT"] + OTC_SALARY["RT"]) / 2
    
    # Build comparison data for offensive positions only
    data = []
    
    positions = ["QB", "WR", "T", "G", "C", "TE", "RB"]
    
    for pos in positions:
        row = {
            "position": pos,
            "my_war": MY_WAR[pos],
            "pff_war": PFF_WAR[pos],
        }
        
        # Map salary
        if pos == "T":
            row["salary"] = tackle_salary
        elif pos in OTC_SALARY:
            row["salary"] = OTC_SALARY[pos]
        else:
            row["salary"] = np.nan
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Calculate ranks (higher WAR/salary = higher rank, so rank 1 is best)
    df["my_war_rank"] = df["my_war"].rank(ascending=False)
    df["pff_war_rank"] = df["pff_war"].rank(ascending=False)
    df["salary_rank"] = df["salary"].rank(ascending=False)
    
    # Calculate proportions of total
    df["my_war_prop"] = df["my_war"] / df["my_war"].sum()
    df["pff_war_prop"] = df["pff_war"] / df["pff_war"].sum()
    df["salary_prop"] = df["salary"] / df["salary"].sum()
    
    return df


def calculate_correlations(df: pd.DataFrame) -> dict:
    """Calculate all pairwise correlations."""
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    
    results = {}
    
    # --- My WAR vs PFF WAR ---
    print("\n1. MY WAR vs PFF WAR (Do we agree on positional value?)")
    
    # Spearman (rank-based)
    r_spearman, p_spearman = stats.spearmanr(df["my_war"], df["pff_war"])
    print(f"   Spearman (rank): r = {r_spearman:.3f} (p = {p_spearman:.3f})")
    
    # Pearson (raw values)
    r_pearson, p_pearson = stats.pearsonr(df["my_war"], df["pff_war"])
    print(f"   Pearson (raw):   r = {r_pearson:.3f} (p = {p_pearson:.3f})")
    
    results["my_vs_pff"] = {
        "spearman": r_spearman,
        "pearson": r_pearson,
        "description": "My WAR vs PFF WAR"
    }
    
    # --- My WAR vs Market ---
    print("\n2. MY WAR vs MARKET (Does my WAR match what teams pay?)")
    
    r_spearman, p_spearman = stats.spearmanr(df["my_war"], df["salary"])
    print(f"   Spearman (rank): r = {r_spearman:.3f} (p = {p_spearman:.3f})")
    
    r_pearson, p_pearson = stats.pearsonr(df["my_war"], df["salary"])
    print(f"   Pearson (raw):   r = {r_pearson:.3f} (p = {p_pearson:.3f})")
    
    results["my_vs_market"] = {
        "spearman": r_spearman,
        "pearson": r_pearson,
        "description": "My WAR vs Market Salary"
    }
    
    # --- PFF WAR vs Market ---
    print("\n3. PFF WAR vs MARKET (Does PFF WAR match what teams pay?)")
    
    r_spearman, p_spearman = stats.spearmanr(df["pff_war"], df["salary"])
    print(f"   Spearman (rank): r = {r_spearman:.3f} (p = {p_spearman:.3f})")
    
    r_pearson, p_pearson = stats.pearsonr(df["pff_war"], df["salary"])
    print(f"   Pearson (raw):   r = {r_pearson:.3f} (p = {p_pearson:.3f})")
    
    results["pff_vs_market"] = {
        "spearman": r_spearman,
        "pearson": r_pearson,
        "description": "PFF WAR vs Market Salary"
    }
    
    return results


def print_ranking_comparison(df: pd.DataFrame):
    """Print side-by-side ranking comparison."""
    print("\n" + "=" * 60)
    print("POSITIONAL VALUE RANKINGS")
    print("=" * 60)
    
    # Sort by my WAR rank
    df_sorted = df.sort_values("my_war_rank")
    
    print(f"\n{'Position':<10} {'My WAR':<12} {'PFF WAR':<12} {'Market $':<15} {'My Rank':<10} {'PFF Rank':<10} {'$ Rank':<10}")
    print("-" * 80)
    
    for _, row in df_sorted.iterrows():
        print(f"{row['position']:<10} {row['my_war']:<12.2f} {row['pff_war']:<12.2f} ${row['salary']/1e6:<14.2f}M {int(row['my_war_rank']):<10} {int(row['pff_war_rank']):<10} {int(row['salary_rank']):<10}")


def identify_discrepancies(df: pd.DataFrame):
    """Identify where WAR and market disagree most."""
    print("\n" + "=" * 60)
    print("KEY DISCREPANCIES: WHERE WAR AND MARKET DISAGREE")
    print("=" * 60)
    
    # Calculate rank differences
    df["my_vs_market_diff"] = abs(df["my_war_rank"] - df["salary_rank"])
    df["pff_vs_market_diff"] = abs(df["pff_war_rank"] - df["salary_rank"])
    
    print("\nMy WAR vs Market (largest rank differences):")
    for _, row in df.nlargest(3, "my_vs_market_diff").iterrows():
        direction = "OVERVALUED by market" if row["salary_rank"] < row["my_war_rank"] else "UNDERVALUED by market"
        print(f"  {row['position']}: WAR rank {int(row['my_war_rank'])}, Salary rank {int(row['salary_rank'])} → {direction}")
    
    print("\nPFF WAR vs Market (largest rank differences):")
    for _, row in df.nlargest(3, "pff_vs_market_diff").iterrows():
        direction = "OVERVALUED by market" if row["salary_rank"] < row["pff_war_rank"] else "UNDERVALUED by market"
        print(f"  {row['position']}: WAR rank {int(row['pff_war_rank'])}, Salary rank {int(row['salary_rank'])} → {direction}")


def plot_comparison(df: pd.DataFrame, save_path: Path):
    """Create visualization comparing WAR methodologies and market."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    positions = df["position"].tolist()
    x = np.arange(len(positions))
    
    # --- Plot 1: My WAR vs PFF WAR ---
    ax1 = axes[0]
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df["my_war"], width, label="My WAR", color="steelblue", alpha=0.8)
    bars2 = ax1.bar(x + width/2, df["pff_war"], width, label="PFF WAR", color="coral", alpha=0.8)
    
    ax1.set_xlabel("Position", fontsize=11)
    ax1.set_ylabel("Average WAR", fontsize=11)
    ax1.set_title("My WAR vs PFF WAR by Position", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(positions)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    
    # --- Plot 2: WAR Rank vs Salary Rank ---
    ax2 = axes[1]
    
    ax2.scatter(df["salary_rank"], df["my_war_rank"], s=100, c="steelblue", 
                label="My WAR", alpha=0.8, edgecolor="white", linewidth=1.5)
    ax2.scatter(df["salary_rank"], df["pff_war_rank"], s=100, c="coral",
                label="PFF WAR", alpha=0.8, edgecolor="white", linewidth=1.5)
    
    # Add diagonal reference line (perfect agreement)
    ax2.plot([0, 8], [0, 8], "k--", alpha=0.5, label="Perfect Agreement")
    
    # Label points
    for _, row in df.iterrows():
        ax2.annotate(row["position"], (row["salary_rank"], row["my_war_rank"]),
                     xytext=(5, 5), textcoords="offset points", fontsize=9, color="steelblue")
    
    ax2.set_xlabel("Market Salary Rank (1 = highest paid)", fontsize=11)
    ax2.set_ylabel("WAR Rank (1 = highest WAR)", fontsize=11)
    ax2.set_title("WAR Rank vs Market Rank", fontsize=12)
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 8)
    
    # --- Plot 3: Normalized comparison (proportions) ---
    ax3 = axes[2]
    
    x3 = np.arange(len(positions))
    width3 = 0.25
    
    bars1 = ax3.bar(x3 - width3, df["my_war_prop"] * 100, width3, label="My WAR %", color="steelblue", alpha=0.8)
    bars2 = ax3.bar(x3, df["pff_war_prop"] * 100, width3, label="PFF WAR %", color="coral", alpha=0.8)
    bars3 = ax3.bar(x3 + width3, df["salary_prop"] * 100, width3, label="Salary %", color="forestgreen", alpha=0.8)
    
    ax3.set_xlabel("Position", fontsize=11)
    ax3.set_ylabel("% of Total", fontsize=11)
    ax3.set_title("Share of Total Value: WAR vs Market", fontsize=12)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(positions)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved comparison plot to: {save_path}")


def save_summary(df: pd.DataFrame, correlations: dict, save_path: Path):
    """Save summary to CSV."""
    # Correlation summary
    corr_rows = []
    for key, vals in correlations.items():
        corr_rows.append({
            "comparison": vals["description"],
            "spearman_r": vals["spearman"],
            "pearson_r": vals["pearson"]
        })
    
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(save_path, index=False)
    print(f"Saved correlation summary to: {save_path}")
    
    # Also save full comparison table
    df.to_csv(save_path.parent / "war_market_full_comparison.csv", index=False)
    print(f"Saved full comparison to: {save_path.parent / 'war_market_full_comparison.csv'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("WAR VS MARKET SHOWDOWN")
    print("Who's Right About Positional Value?")
    print("=" * 60)
    
    # Build comparison DataFrame
    df = build_comparison_df()
    
    # Print ranking comparison
    print_ranking_comparison(df)
    
    # Calculate correlations
    correlations = calculate_correlations(df)
    
    # Identify discrepancies
    identify_discrepancies(df)
    
    # Plot
    plot_comparison(df, FIGURES_DIR / "war_vs_market_showdown.png")
    
    # Save
    save_summary(df, correlations, TABLES_DIR / "war_market_correlation_summary.csv")
    
    # Final verdict
    print("\n" + "=" * 60)
    print("THE VERDICT")
    print("=" * 60)
    
    my_market_r = correlations["my_vs_market"]["spearman"]
    pff_market_r = correlations["pff_vs_market"]["spearman"]
    
    print(f"\nMy WAR correlation with market:  r = {my_market_r:.3f}")
    print(f"PFF WAR correlation with market: r = {pff_market_r:.3f}")
    
    if my_market_r > pff_market_r:
        print(f"\n→ My WAR aligns better with market valuations by {my_market_r - pff_market_r:.3f}")
    else:
        print(f"\n→ PFF WAR aligns better with market valuations by {pff_market_r - my_market_r:.3f}")
    
    my_pff_r = correlations["my_vs_pff"]["spearman"]
    print(f"\nMy WAR vs PFF WAR agreement: r = {my_pff_r:.3f}")
    
    if my_pff_r > 0.7:
        print("→ Strong agreement on positional value hierarchy despite different methodologies")
    elif my_pff_r > 0.5:
        print("→ Moderate agreement on positional value hierarchy")
    else:
        print("→ Notable disagreements on positional value hierarchy")
    
    return df, correlations


if __name__ == "__main__":
    df, correlations = main()