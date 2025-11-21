"""
======================================================================
SPWAR 04 - SKILL WAR VALIDATION — Player-Level + Team-Level (2024)
======================================================================

Validates Skill Position WAR with two checks:

1) Player-level: Skill WAR vs PFR Approximate Value (AV)
2) Team-level: Sum of Skill WAR vs Wins (incl. postseason)

Outputs:
 - Printed correlations + regression summaries
 - Player-level scatter plot
 - Team-level scatter plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from matplotlib.lines import Line2D

# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------
BASE = Path("/Users/anokhpalakurthi/Documents/nfl_offensive_war")
OUT_TBLS = BASE / "outputs" / "tables"
OUT_FIGS = BASE / "outputs" / "figures"
DATA_EXTERNAL = BASE / "data" / "external"
DATA_PROCESSED = BASE / "data" / "processed"

OUT_FIGS.mkdir(parents=True, exist_ok=True)

SKILL_WAR_PATH = OUT_TBLS / "sp_skill_war_2021_2024.csv"
PFR_AV_PATH = DATA_EXTERNAL / "skill_AV_2024.csv"
TEAM_RESULTS_PATH = DATA_PROCESSED / "team_results_2024.csv"

# ===================================================================
# PLAYER-LEVEL VALIDATION
# ===================================================================
print("\n========================= PLAYER-LEVEL =========================\n")

skill = pd.read_csv(SKILL_WAR_PATH)
skill_2024 = (
    skill[skill["season"] == 2024]
    [["player_id", "player_name", "total_WAR", "plays_total"]]
    .groupby(["player_id", "player_name"], as_index=False)
    .agg(total_WAR=("total_WAR", "sum"),
         total_plays=("plays_total", "sum"))
)

av = pd.read_csv(PFR_AV_PATH)[["Player", "AV", "Pos", "Team"]]
av = av[av["Pos"].isin(["RB", "WR", "TE"])]
av["AV"] = pd.to_numeric(av["AV"], errors="coerce")

merged = skill_2024.merge(
    av, left_on="player_name", right_on="Player", how="inner"
).dropna(subset=["total_WAR", "AV"])

print(f"Merged players: {len(merged)}")

# Correlations
pearson_r = merged["total_WAR"].corr(merged["AV"])
spearman_r = merged["total_WAR"].rank().corr(merged["AV"].rank())
print(f"Pearson r  : {pearson_r:.3f}")
print(f"Spearman ρ : {spearman_r:.3f}")

# Correlation by position
print("\nBy Position:")
for pos in ["RB", "WR", "TE"]:
    sub = merged[merged["Pos"] == pos]
    if len(sub) > 0:
        r = sub["total_WAR"].corr(sub["AV"])
        rho = sub["total_WAR"].rank().corr(sub["AV"].rank())
        print(f"  {pos}: n={len(sub):3d}  r={r:.3f}  ρ={rho:.3f}")

# Regression
X = sm.add_constant(merged["total_WAR"])
model = sm.OLS(merged["AV"], X).fit()
print("\nPlayer-Level Regression Summary:\n")
print(model.summary())

# Scatter Plot
pos_colors = {"RB": "tab:blue", "WR": "tab:orange", "TE": "tab:green"}

plt.figure(figsize=(8, 6))
plt.scatter(
    merged["total_WAR"], merged["AV"],
    c=merged["Pos"].map(pos_colors), edgecolor="k", alpha=0.85
)

legend_elems = [
    Line2D([0], [0], marker='o', color='w', label=p,
           markerfacecolor=c, markeredgecolor="k", markersize=8)
    for p, c in pos_colors.items()
]
plt.legend(handles=legend_elems, title="Position")
plt.xlabel("Skill WAR (2024)")
plt.ylabel("Approximate Value (PFR AV)")
plt.title("Skill Position WAR vs AV — 2024")
plt.tight_layout()

scatter_path = OUT_FIGS / "skill_war_vs_av_2024.png"
plt.savefig(scatter_path, dpi=200)
plt.close()
print(f"✓ Saved → {scatter_path}")

# ===================================================================
# TEAM-LEVEL VALIDATION
# ===================================================================
print("\n========================= TEAM-LEVEL =========================\n")

# Aggregate Skill WAR by team (using AV file's team)
team_skill = (
    merged.groupby("Team", as_index=False)
    .agg(team_skill_WAR=("total_WAR", "sum"))
    .rename(columns={"Team": "team"})
)

# Load wins
team_results = pd.read_csv(TEAM_RESULTS_PATH)

# CRITICAL FIX: remove duplicate team column before renaming
team_results = team_results.loc[:, ~team_results.columns.duplicated()]

# Normalize column names
team_results.columns = team_results.columns.str.lower().str.strip()
team_results = team_results.rename(columns={"tm": "team", "w": "wins"})
team_results = team_results[["team", "wins"]].drop_duplicates()

# Map name → abbr
TEAM_MAP = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
    "Detroit Lions": "DET", "Green Bay Packers": "GNB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KAN",
    "Las Vegas Raiders": "LVR", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NWE",
    "New Orleans Saints": "NOR", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SFO",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TAM",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

team_results["team"] = team_results["team"].map(TEAM_MAP)
team_results = team_results.dropna(subset=["team"])

# Merge wins + team WAR
team_merged = team_results.merge(team_skill, on="team", how="left")
team_merged["team_skill_WAR"] = team_merged["team_skill_WAR"].fillna(0.0)

print(f"Merged teams: {len(team_merged)}")

# Correlations
r, p_r = pearsonr(team_merged["team_skill_WAR"], team_merged["wins"])
rho, p_rho = spearmanr(team_merged["team_skill_WAR"], team_merged["wins"])

X = sm.add_constant(team_merged["team_skill_WAR"])
win_model = sm.OLS(team_merged["wins"], X).fit()

beta0 = win_model.params["const"]
beta1 = win_model.params["team_skill_WAR"]
r2 = win_model.rsquared

print(f"Pearson r   : {r:.3f}  (p={p_r:.3f})")
print(f"Spearman ρ  : {rho:.3f}  (p={p_rho:.3f})")
print("\nTeam-Level Regression Summary:\n")
print(win_model.summary())

# Scatter plot for teams
plt.figure(figsize=(9, 6))
plt.scatter(team_merged["team_skill_WAR"], team_merged["wins"], alpha=0.85)

xg = np.linspace(team_merged["team_skill_WAR"].min(),
                 team_merged["team_skill_WAR"].max(), 100)
plt.plot(xg, beta0 + beta1 * xg, color="black")

for _, row in team_merged.iterrows():
    plt.text(row["team_skill_WAR"] + 0.03,
             row["wins"] + 0.03,
             row["team"], fontsize=7)

plt.xlabel("Team Skill WAR — 2024")
plt.ylabel("Team Wins (incl. postseason)")
plt.title(f"Wins vs Team Skill Position WAR — 2024\n"
          f"R²={r2:.3f}, r={r:.3f}")
plt.grid(alpha=0.3)
plt.tight_layout()

plot_path = OUT_FIGS / "team_skill_war_vs_wins_2024.png"
plt.savefig(plot_path, dpi=200)
plt.close()
print(f"✓ Saved → {plot_path}")


# ======================= Skill AV vs Wins (TEAM LEVEL) =======================

team_skill_av = (
    merged.groupby("Team", as_index=False)
    .agg(team_skill_AV=("AV", "sum"))
    .rename(columns={"Team": "team"})
)

team_av_merged = team_results.merge(team_skill_av, on="team", how="left")
team_av_merged["team_skill_AV"] = team_av_merged["team_skill_AV"].fillna(0)

# Correlations
r_av, p_av = pearsonr(team_av_merged["team_skill_AV"], team_av_merged["wins"])
rho_av, p_rho_av = spearmanr(team_av_merged["team_skill_AV"], team_av_merged["wins"])

# Regression
Xa = sm.add_constant(team_av_merged["team_skill_AV"])
win_model_av = sm.OLS(team_av_merged["wins"], Xa).fit()
beta1_av = win_model_av.params["team_skill_AV"]
r2_av = win_model_av.rsquared

print("\n===== TEAM-LEVEL: Skill AV vs Wins (2024) =====")
print(f"Pearson r   : {r_av:.3f} (p={p_av:.3f})")
print(f"Spearman ρ  : {rho_av:.3f} (p={p_rho_av:.3f})")
print(f"R² (variance explained): {r2_av:.3f}")
print(win_model_av.summary())

# ===================================================================
# FIGURE: Team Skill Position AV vs Wins (2024)
# ===================================================================
print("\nCreating complementary AV vs Wins plot…")

plt.figure(figsize=(9, 6))
plt.scatter(team_av_merged["team_skill_AV"], team_av_merged["wins"], alpha=0.8)
    
# Regression line
xg_av = np.linspace(team_av_merged["team_skill_AV"].min(),
                    team_av_merged["team_skill_AV"].max(), 100)
av_beta0 = win_model_av.params["const"]
av_beta1 = win_model_av.params["team_skill_AV"]
plt.plot(xg_av, av_beta0 + av_beta1 * xg_av, color="black", linewidth=2)

# Label each point
for _, row in team_av_merged.iterrows():
    plt.text(row["team_skill_AV"] + 0.03,
             row["wins"] + 0.03,
             row["team"], fontsize=7)

plt.xlabel("Team Skill Position AV — 2024")
plt.ylabel("Team Wins (incl. postseason)")
plt.title(f"Wins vs Team Skill Position AV — 2024\nR²={r2_av:.3f}, r={r_av:.3f}")
plt.grid(alpha=0.3)

plot_av_path = OUT_FIGS / "team_skill_av_vs_wins_2024.png"
plt.tight_layout()
plt.savefig(plot_av_path, dpi=200)
plt.close()

print(f"Saved → {plot_av_path}")

print("\nDone.\n")