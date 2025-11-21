# ============================================================
# BENCHMARK: QB WAR vs APPROXIMATE VALUE (AV) — 2024
# ============================================================

import pandas as pd
from scipy.stats import linregress
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

print("=" * 70)
print("BENCHMARKING QB WAR vs APPROXIMATE VALUE (AV) — 2024")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load and Clean Data
# ------------------------------------------------------------
war_path = "/Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs/tables/qb_war_2024_enhanced.csv"
av_path = "/Users/anokhpalakurthi/Documents/nfl_offensive_war/data/external/qb_av_2024.csv"

war = pd.read_csv(war_path)
av = pd.read_csv(av_path)

# Standardize columns (do not change your mappings)
war = war.rename(columns={"passer_player_name": "Player", "WAR": "QB_WAR"})
av = av.rename(columns={"Team": "team_abbr"})
av = av.drop(columns=["AV.1"], errors="ignore")

# Clean player names
war["Player_clean"] = war["Player"].astype(str).str.lower().str.strip()
av["Player_clean"] = av["Player"].astype(str).str.lower().str.strip()

# ------------------------------------------------------------
# 2. Merge AV + WAR
# ------------------------------------------------------------
merged = pd.merge(av, war, on="Player_clean", how="inner", suffixes=("_AV", "_WAR"))

# Regression (QB-level)
slope, intercept, r_value, p_value, std_err = linregress(merged["QB_WAR"], merged["AV"])
print(f"Correlation (r): {r_value:.3f}")
print(f"R²: {r_value**2:.3f}")
print(f"P-value: {p_value:.5f}")
print(f"Slope: {slope:.3f}   Intercept: {intercept:.3f}")

# Save merged QB-level table
output_path = "/Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs/tables/qb_war_vs_av_2024.csv"
merged.to_csv(output_path, index=False)
print(f"\n✓ Saved merged dataset: {output_path}")
print(f"Matched players: {len(merged)} / {len(av)}")

# ============================================================
# 3. TEAM-LEVEL BENCHMARK: AV vs WINS
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING QB AV VS TEAM TOTAL WINS (2024)")
print("=" * 70)

team_results_path = "/Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs/tables/team_war_vs_total_wins_2024.csv"
team = pd.read_csv(team_results_path)

# Standardize casing for merges
team.columns = team.columns.str.strip().str.lower()
av.columns = av.columns.str.strip().str.lower()

# Team abbreviation normalization (critical — unchanged)
team["team"] = team["team"].astype(str).str.upper().str[:3]
av["team_abbr"] = av["team_abbr"].astype(str).str.upper().str[:3]
av["team"] = av["team_abbr"]

# Select top QB per team
team_top_qb_av = (
    av.sort_values("av", ascending=False)
      .groupby("team", as_index=False)
      .first()[["team", "av"]]
      .rename(columns={"av": "total_qb_av"})
)

# Merge with wins
merged_av = pd.merge(team, team_top_qb_av, on="team", how="left")
merged_av = merged_av.dropna(subset=["total_qb_av", "total_wins"])

# Regression: AV vs Wins
slope, intercept, r_val, p_val, stderr = stats.linregress(
    merged_av["total_qb_av"], merged_av["total_wins"]
)
print(f"Correlation (r): {r_val:.3f}")
print(f"R²: {r_val**2:.3f}")
print(f"P-value: {p_val:.5f}")
print(f"Slope: {slope:.3f}   Intercept: {intercept:.3f}\n")

# Save merged team-level table
out_path = "/Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs/tables/team_av_vs_total_wins_2024.csv"
merged_av.to_csv(out_path, index=False)
print(f"✓ Saved AV vs Wins dataset: {out_path}")

# Leaderboard
print("\nTop 10 teams by *Top QB AV* and Wins:")
print(
    merged_av[["team", "total_qb_av", "total_wins"]]
    .sort_values("total_qb_av", ascending=False)
    .head(10)
    .to_string(index=False)
)

# ============================================================
# 4. VISUALIZATION
# ============================================================

fig_dir = "/Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs/figures"
os.makedirs(fig_dir, exist_ok=True)

# --- Plot 1: QB WAR vs Wins ---
plt.figure(figsize=(8, 6))
sns.regplot(
    x="total_qb_war",
    y="total_wins",
    data=team,
    scatter_kws={"s": 60, "alpha": 0.8},
    line_kws={"color": "red"},
)
plt.title("QB WAR vs Team Wins (2024)", fontsize=14, weight="bold")
plt.xlabel("Total QB WAR (Team-Level)")
plt.ylabel("Total Wins (Including Playoffs)")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "team_war_vs_wins_2024.png"), dpi=300)
plt.close()

# --- Plot 2: QB AV vs Wins ---
plt.figure(figsize=(8, 6))
sns.regplot(
    x="total_qb_av",
    y="total_wins",
    data=merged_av,
    scatter_kws={"s": 60, "alpha": 0.8},
    line_kws={"color": "red"},
)
plt.title("QB AV (Top QB Only) vs Team Wins (2024)", fontsize=14, weight="bold")
plt.xlabel("Top QB Approximate Value (AV)")
plt.ylabel("Total Wins (Including Playoffs)")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "team_av_vs_wins_2024.png"), dpi=300)
plt.close()

print("\nVisualization complete!")

# ============================================================
# 5. DUAL REGRESSION COMPARISON: WAR→AV vs AV Identity
# ============================================================

plt.figure(figsize=(10, 7))

# Scatter
plt.scatter(merged["QB_WAR"], merged["AV"], s=60, alpha=0.75, label="Players")

# Fit 1: WAR → AV
slope1, intercept1, _, _, _ = linregress(merged["QB_WAR"], merged["AV"])
x_vals = np.linspace(merged["QB_WAR"].min(), merged["QB_WAR"].max(), 200)
plt.plot(x_vals, slope1*x_vals + intercept1, color="red", linewidth=2.2,
         label=f"WAR → AV Fit (R² = {r_value**2:.3f})")

# Fit 2: AV identity line (scaled)
x2 = np.linspace(merged["QB_WAR"].min(), merged["QB_WAR"].max(), 200)
scaled_identity = np.interp(x2,
                            (merged["QB_WAR"].min(), merged["QB_WAR"].max()),
                            (merged["AV"].min(), merged["AV"].max()))
plt.plot(x2, scaled_identity, color="blue", linewidth=2.2, linestyle="--",
         label="AV Scale Reference Line")

# Formatting
plt.title("Comparison: WAR→AV Fit vs AV Baseline (2024)", fontsize=16, weight="bold")
plt.xlabel("QB WAR", fontsize=13)
plt.ylabel("AV", fontsize=13)
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

out_path = "/Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs/figures/dual_regression_war_vs_av.png"
plt.savefig(out_path, dpi=300)
plt.show()

print(f"✓ Saved dual regression comparison: {out_path}")