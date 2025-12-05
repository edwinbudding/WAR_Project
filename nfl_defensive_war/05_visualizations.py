import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("/Users/anokhpalakurthi/Documents/nfl_defensive_war")
OUTPUTS_DIR = BASE_DIR / "outputs"
WAR_PATH = OUTPUTS_DIR / "defensive_war_2021_2024.csv"

# Output paths for figures
FIG_DIR = OUTPUTS_DIR / "figures"

# Color palette by position
POSITION_COLORS = {
    "EDGE": "#1f77b4",  # Blue
    "IDL": "#ff7f0e",   # Orange
    "LB": "#2ca02c",    # Green
    "CB": "#d62728",    # Red
    "S": "#9467bd",     # Purple
}

# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load WAR data."""
    print("Loading data...")
    df = pd.read_csv(WAR_PATH)
    
    # Filter to qualified players (50+ snaps to match blog)
    df = df[df["snap_counts_defense"] >= 50].copy()
    print(f"  Loaded {len(df)} player-seasons")
    
    return df


# ============================================================
# VISUALIZATION 1: AVERAGE WAR BY POSITION
# ============================================================

def plot_positional_war(df, save_path=None):
    """Bar chart showing average WAR by position."""
    print("\nCreating: Average WAR by Position...")
    
    # Calculate positional averages
    pos_stats = df.groupby("role").agg({
        "def_war": ["mean", "std", "count"]
    }).round(3)
    pos_stats.columns = ["mean", "std", "n"]
    pos_stats = pos_stats.loc[["EDGE", "S", "CB", "LB", "IDL"]]  # Order by mean
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    positions = pos_stats.index.tolist()
    means = pos_stats["mean"].values
    stds = pos_stats["std"].values
    colors = [POSITION_COLORS[pos] for pos in positions]
    
    # Bar chart
    bars = ax.bar(positions, means, color=colors, edgecolor="black", linewidth=1.2)
    
    # Add error bars (std dev)
    ax.errorbar(positions, means, yerr=stds, fmt="none", color="black", capsize=5, capthick=1.5)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f"{mean:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    
    # Formatting
    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("Average WAR", fontsize=12)
    ax.set_title("Average Defensive WAR by Position (2021-2024)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(means) + max(stds) + 0.05)
    ax.axhline(y=0, color="black", linewidth=0.5)
    
    # Add subtitle
    ax.text(0.5, -0.12, "Error bars represent standard deviation | n = 3,293 player-seasons (50+ snaps)",
            ha="center", transform=ax.transAxes, fontsize=10, style="italic", color="gray")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    
    plt.show()
    return fig


# ============================================================
# VISUALIZATION 2: TOP 10 CUMULATIVE WAR LEADERS
# ============================================================

def plot_top_players(df, save_path=None):
    """Horizontal bar chart of top 10 cumulative WAR leaders."""
    print("\nCreating: Top 10 Cumulative WAR Leaders...")
    
    # Calculate cumulative WAR
    cumulative = df.groupby("player").agg({
        "def_war": "sum",
        "role": "first",
        "season": "count"
    }).reset_index()
    cumulative.columns = ["player", "total_war", "role", "seasons"]
    cumulative = cumulative.sort_values("total_war", ascending=True).tail(10)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    players = cumulative["player"].values
    wars = cumulative["total_war"].values
    roles = cumulative["role"].values
    colors = [POSITION_COLORS.get(role, "gray") for role in roles]
    
    # Horizontal bar chart
    bars = ax.barh(players, wars, color=colors, edgecolor="black", linewidth=1.2)
    
    # Add value labels
    for bar, war in zip(bars, wars):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f"{war:.2f}", ha="left", va="center", fontsize=11, fontweight="bold")
    
    # Formatting
    ax.set_xlabel("Cumulative WAR (2021-2024)", fontsize=12)
    ax.set_ylabel("")
    ax.set_title("Top 10 Defensive WAR Leaders (2021-2024)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(wars) + 0.5)
    
    # Legend
    legend_patches = [mpatches.Patch(color=POSITION_COLORS[pos], label=pos) 
                      for pos in ["EDGE", "LB", "IDL", "CB", "S"]]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    
    plt.show()
    return fig


# ============================================================
# VISUALIZATION 3: MY WAR vs MARKET/PFF COMPARISON
# ============================================================

def plot_validation_scatter(save_path=None):
    """Scatter plot comparing My WAR vs Market Implied vs PFF."""
    print("\nCreating: Validation Scatter Plot...")
    
    # Data from model output
    data = {
        "Position": ["EDGE", "S", "CB", "LB", "IDL"],
        "My WAR": [0.18, 0.15, 0.13, 0.12, 0.10],
        "Market Implied": [0.18, 0.11, 0.12, 0.10, 0.14],
        "PFF WAR": [0.06, 0.23, 0.23, 0.11, 0.06],
    }
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot diagonal line (perfect agreement)
    ax.plot([0, 0.25], [0, 0.25], "k--", alpha=0.5, label="Perfect Agreement")
    
    # Scatter: My WAR vs Market
    for _, row in df.iterrows():
        ax.scatter(row["My WAR"], row["Market Implied"], 
                   color=POSITION_COLORS[row["Position"]], s=150, 
                   edgecolor="black", linewidth=1.5, zorder=3)
        ax.annotate(row["Position"], (row["My WAR"], row["Market Implied"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=10, fontweight="bold")
    
    # Scatter: My WAR vs PFF (different marker)
    for _, row in df.iterrows():
        ax.scatter(row["My WAR"], row["PFF WAR"], 
                   color=POSITION_COLORS[row["Position"]], s=150, 
                   marker="^", edgecolor="black", linewidth=1.5, alpha=0.6, zorder=2)
    
    # Formatting
    ax.set_xlabel("My WAR Estimate", fontsize=12)
    ax.set_ylabel("Comparison WAR", fontsize=12)
    ax.set_title("My WAR vs Market & PFF Estimates", fontsize=14, fontweight="bold")
    ax.set_xlim(0.04, 0.22)
    ax.set_ylim(0.04, 0.26)
    ax.set_aspect("equal")
    
    # Custom legend
    circle = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", 
                        markersize=10, markeredgecolor="black", label="vs Market Implied")
    triangle = plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", 
                          markersize=10, markeredgecolor="black", alpha=0.6, label="vs PFF WAR")
    ax.legend(handles=[circle, triangle], loc="upper left", fontsize=10)
    
    # Add annotation
    ax.text(0.17, 0.07, "EDGE aligns\nwith market", fontsize=9, style="italic", 
            ha="center", color="gray")
    ax.text(0.08, 0.22, "PFF values\ncoverage higher", fontsize=9, style="italic",
            ha="center", color="gray")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    
    plt.show()
    return fig


# ============================================================
# VISUALIZATION 4: PASS vs RUN DEFENSE WAR BY POSITION
# ============================================================

def plot_war_breakdown(df, save_path=None):
    """Stacked bar chart showing pass vs run defense WAR by position."""
    print("\nCreating: Pass vs Run Defense WAR Breakdown...")
    
    # Calculate averages by position
    pos_stats = df.groupby("role").agg({
        "pass_defense_war": "mean",
        "run_defense_war": "mean",
    }).round(3)
    pos_stats = pos_stats.loc[["EDGE", "S", "CB", "LB", "IDL"]]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    positions = pos_stats.index.tolist()
    pass_war = pos_stats["pass_defense_war"].values
    run_war = pos_stats["run_defense_war"].values
    
    x = np.arange(len(positions))
    width = 0.6
    
    # Stacked bars
    bars1 = ax.bar(x, pass_war, width, label="Pass Defense WAR", color="#3498db", edgecolor="black")
    bars2 = ax.bar(x, run_war, width, bottom=pass_war, label="Run Defense WAR", color="#e74c3c", edgecolor="black")
    
    # Add total labels on top
    totals = pass_war + run_war
    for i, (p, r, t) in enumerate(zip(pass_war, run_war, totals)):
        ax.text(i, t + 0.01, f"{t:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        # Add percentage labels inside bars
        if p > 0.05:
            ax.text(i, p/2, f"{p/(p+r)*100:.0f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        if r > 0.05:
            ax.text(i, p + r/2, f"{r/(p+r)*100:.0f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    
    # Formatting
    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("Average WAR", fontsize=12)
    ax.set_title("Pass Defense vs Run Defense WAR by Position", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, max(totals) + 0.08)
    
    # Add subtitle
    ax.text(0.5, -0.1, "CB/S value comes almost entirely from pass defense | EDGE contributes to both",
            ha="center", transform=ax.transAxes, fontsize=10, style="italic", color="gray")
    
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
    # Create figures directory
    FIG_DIR.mkdir(exist_ok=True)
    
    # Load data
    df = load_data()
    
    # Create visualizations
    plot_positional_war(df, save_path=FIG_DIR / "01_positional_war.png")
    plot_top_players(df, save_path=FIG_DIR / "02_top_players.png")
    plot_validation_scatter(save_path=FIG_DIR / "03_validation_scatter.png")
    plot_war_breakdown(df, save_path=FIG_DIR / "04_war_breakdown.png")
    
    print("\n" + "=" * 50)
    print(" ALL VISUALIZATIONS COMPLETE")
    print("=" * 50)
    print(f"\nFigures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()