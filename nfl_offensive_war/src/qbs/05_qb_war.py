"""
CLEAN MULTI-SEASON ENHANCED QB WAR CALCULATOR (2021–2024)
Now with Weighted_WAR (seasonal z-blend) + cumulative Weighted_WAR_total.
"""

# ============================================================
# CONFIG
# ============================================================

BLEND_ALPHA = 0.50         
REPL_PERCENTILE = 0.25
REPL_MIN_OBS = 8
ATTEMPT_MIN = 100

PLAYS_PER_FULL_SEASON = 700

# Weighted_WAR blend: volume vs efficiency
WEIGHTED_WAR_VOLUME_WEIGHT = 0.70
WEIGHTED_WAR_RATE_WEIGHT = 0.30

import os, pandas as pd, numpy as np, statsmodels.api as sm
import matplotlib.pyplot as plt, seaborn as sns

DATA_DIR = "/Users/anokhpalakurthi/Documents/nfl_offensive_war/data/processed"
OUT_DIR = "/Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs"
TABLE_DIR = f"{OUT_DIR}/tables"
FIG_DIR = f"{OUT_DIR}/figures"
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# NAME NORMALIZATION
# ============================================================

def normalize_qb_names(df, col):
    df = df.copy()
    df.loc[df[col] == "Aa.Rodgers", col] = "A.Rodgers"
    return df

# ============================================================
# LOAD DATA
# ============================================================

def load_all_qbs(season, min_attempts=ATTEMPT_MIN):
    passing = pd.read_csv(f"{DATA_DIR}/passing_plays.csv")
    rushing = pd.read_csv(f"{DATA_DIR}/rushing_plays.csv")

    p = passing.query("season == @season").copy()
    r = rushing.query("season == @season").copy()
    p = normalize_qb_names(p, "passer_player_name")
    r = normalize_qb_names(r, "rusher_player_name")

    qb_pass = (
        p.groupby(["passer_player_name"], as_index=False)
        .agg(
            passing_epa=("epa", "sum"),
            passing_plays=("play_id", "count"),
            avg_cpoe=("cpoe", "mean"),
            passing_success=("success", "mean")
        )
    )

    qb_rush = (
        r.groupby(["rusher_player_name"], as_index=False)
        .agg(
            rushing_epa=("epa", "sum"),
            rushing_plays=("play_id", "count"),
            rushing_success=("success", "mean")
        )
    )

    df = qb_pass.merge(
        qb_rush, left_on="passer_player_name", right_on="rusher_player_name", how="left"
    )
    df.fillna({"rushing_epa": 0, "rushing_plays": 0, "rushing_success": 0}, inplace=True)
    df["season"] = season

    df["total_epa"] = df["passing_epa"] + df["rushing_epa"]
    df["total_plays"] = df["passing_plays"] + df["rushing_plays"]
    df["epa_per_play"] = df["total_epa"] / df["total_plays"].replace(0, np.nan)

    df["overall_success"] = (
        (df["passing_success"] * df["passing_plays"] +
         df["rushing_success"] * df["rushing_plays"])
        / df["total_plays"].replace(0, np.nan)
    )

    df = df[df["total_plays"] >= min_attempts].copy()

    return df[[
        "passer_player_name", "season", "total_epa", "total_plays",
        "epa_per_play", "avg_cpoe", "overall_success"
    ]]


def load_all_qbs_multi(start=2021, end=2024, min_attempts=ATTEMPT_MIN):
    dfs = [load_all_qbs(y, min_attempts) for y in range(start, end + 1)]
    out = pd.concat(dfs, ignore_index=True)
    print(f"✓ Loaded {len(out)} QB-seasons ({start}–{end})")
    return out

# ============================================================
# MODEL + ENHANCEMENT
# ============================================================

def estimate_empirical_coefficients(qb_df):
    rows = []
    for season, g in qb_df.groupby("season"):
        g = g.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["avg_cpoe", "overall_success", "epa_per_play"]
        )

        if len(g) < 6:
            print(f"⚠️  Season {season}: too few valid QBs ({len(g)}). Skipping regression.")
            continue

        X = sm.add_constant(g[["avg_cpoe", "overall_success"]])
        y = g["epa_per_play"]
        model = sm.OLS(y, X).fit()

        rows.append({
            "season": season,
            "intercept": model.params.get("const", np.nan),
            "beta_cpoe": model.params.get("avg_cpoe", np.nan),
            "beta_success": model.params.get("overall_success", np.nan),
            "r2": model.rsquared,
            "n": len(g)
        })

    coeffs = pd.DataFrame(rows)
    print("\n=== Regression Diagnostics ===")
    print(coeffs.to_string(index=False))
    return coeffs


def apply_seasonal_enhancement(qb_df, coeff_df, alpha=BLEND_ALPHA):
    qb_df = qb_df.copy()
    modeled_list = []

    for _, row in qb_df.iterrows():
        c = coeff_df[coeff_df["season"] == row["season"]].iloc[0]
        modeled_list.append(
            c["intercept"] +
            c["beta_cpoe"] * row["avg_cpoe"] +
            c["beta_success"] * row["overall_success"]
        )

    qb_df["modeled_epa_per_play"] = modeled_list

    qb_df["enhanced_epa_per_play"] = (
        alpha * qb_df["epa_per_play"] +
        (1 - alpha) * qb_df["modeled_epa_per_play"]
    )

    qb_df["enhanced_epa"] = qb_df["enhanced_epa_per_play"] * qb_df["total_plays"]
    return qb_df

# ============================================================
# WAR CALCULATION
# ============================================================

def calculate_epa_per_win_fixed():
    return 218.83 * 0.674 * 0.6753 


def identify_replacement_by_season(qb_df, percentile=REPL_PERCENTILE, min_obs=REPL_MIN_OBS):
    levels = []
    for s, g in qb_df.groupby("season"):
        thresh = np.percentile(g["enhanced_epa_per_play"], percentile * 100)
        low = g[g["enhanced_epa_per_play"] <= thresh]
        if len(low) < min_obs:
            low = g.nsmallest(min_obs, "enhanced_epa_per_play")
        levels.append({"season": s, "replacement": low["enhanced_epa_per_play"].mean()})
    return pd.DataFrame(levels)


def calculate_war_multi(qb_df, repl_df, epa_per_win):
    dfs = []
    for s, g in qb_df.groupby("season"):
        g = g.copy()
        repl = repl_df.loc[repl_df["season"] == s, "replacement"].iloc[0]

        g["epa_above_repl"] = (g["enhanced_epa_per_play"] - repl) * g["total_plays"]
        g["WAR"] = g["epa_above_repl"] / epa_per_win
        g["WAR_per_700"] = g["WAR"] * (PLAYS_PER_FULL_SEASON / g["total_plays"])

        dfs.append(g)

    return pd.concat(dfs, ignore_index=True)

# ============================================================
# NEW: WEIGHTED_WAR (seasonal Z-blend)
# ============================================================

def add_weighted_war_per_season(qb_all):
    qb_all = qb_all.copy()

    # Initialize columns
    qb_all["WAR_z"] = np.nan
    qb_all["WAR700_z"] = np.nan
    qb_all["Weighted_WAR"] = np.nan

    for season, g in qb_all.groupby("season"):
        idx = g.index

        war = g["WAR"]
        war700 = g["WAR_per_700"]

        war_std = war.std(ddof=0)
        war700_std = war700.std(ddof=0)

        if war_std == 0 or np.isnan(war_std):
            war_z = pd.Series(0.0, index=idx)
        else:
            war_z = (war - war.mean()) / war_std

        if war700_std == 0 or np.isnan(war700_std):
            war700_z = pd.Series(0.0, index=idx)
        else:
            war700_z = (war700 - war700.mean()) / war700_std

        qb_all.loc[idx, "WAR_z"] = war_z
        qb_all.loc[idx, "WAR700_z"] = war700_z
        qb_all.loc[idx, "Weighted_WAR"] = (
            WEIGHTED_WAR_VOLUME_WEIGHT * war_z +
            WEIGHTED_WAR_RATE_WEIGHT * war700_z
        )

    return qb_all

# ============================================================
# DIAGNOSTIC (Rodgers)
# ============================================================

def rodgers_diagnostic(qb_df, coeffs):
    name = "A.Rodgers"
    subset = qb_df[qb_df["passer_player_name"] == name]

    if subset.empty:
        print("\n⚠️ No Rodgers data found — check normalization.")
        return

    rows = []
    for _, r in subset.iterrows():
        coef = coeffs[coeffs["season"] == r["season"]].iloc[0]
        modeled = (
            coef["intercept"] +
            coef["beta_cpoe"] * r["avg_cpoe"] +
            coef["beta_success"] * r["overall_success"]
        )
        enhanced = BLEND_ALPHA * r["epa_per_play"] + (1 - BLEND_ALPHA) * modeled

        rows.append({
            "season": r["season"],
            "actual": r["epa_per_play"],
            "modeled": modeled,
            "enhanced": enhanced
        })

    print("\n=== Rodgers EPA Diagnostic ===")
    print(pd.DataFrame(rows).to_string(index=False))

# ============================================================
# YEAR-SPECIFIC SCATTERPLOT GRID (WAR vs WAR_per_700)
# ============================================================

def visualize_war_scatter(qb_all, fig_dir=FIG_DIR):

    years = sorted(qb_all["season"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, year in enumerate(years):
        ax = axes[i]
        df = qb_all[qb_all["season"] == year]

        ax.scatter(df["WAR"], df["WAR_per_700"], alpha=0.6, s=40)
        ax.axhline(0, color='black', linewidth=0.6)
        ax.axvline(0, color='black', linewidth=0.6)

        top = df.sort_values("WAR_per_700", ascending=False).head(5)
        for _, row in top.iterrows():
            ax.text(row["WAR"], row["WAR_per_700"], row["passer_player_name"],
                    fontsize=8, weight="bold")

        ax.set_title(f"{year} WAR vs WAR_per_700")
        ax.set_xlabel("WAR (raw)")
        ax.set_ylabel("WAR per 700 plays")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{fig_dir}/war_vs_war700_grid.png"
    plt.savefig(out_path, dpi=200)
    print(f"✓ Saved WAR grid to: {out_path}")

# ============================================================
# NEW: YEAR-SPECIFIC SCATTER (WAR vs Weighted_WAR)
# ============================================================

def visualize_weighted_war_scatter(qb_all, fig_dir=FIG_DIR):

    years = sorted(qb_all["season"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, year in enumerate(years):
        ax = axes[i]
        df = qb_all[qb_all["season"] == year]

        ax.scatter(df["WAR"], df["Weighted_WAR"], alpha=0.6, s=40)
        ax.axhline(0, color='black', linewidth=0.6)
        ax.axvline(0, color='black', linewidth=0.6)

        top = df.sort_values("Weighted_WAR", ascending=False).head(5)
        for _, row in top.iterrows():
            ax.text(row["WAR"], row["Weighted_WAR"], row["passer_player_name"],
                    fontsize=8, weight="bold")

        ax.set_title(f"{year} WAR vs Weighted_WAR")
        ax.set_xlabel("WAR (raw)")
        ax.set_ylabel("Weighted_WAR (z-blend)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{fig_dir}/war_vs_weighted_war_grid.png"
    plt.savefig(out_path, dpi=200)
    print(f"✓ Saved WAR vs Weighted_WAR grid to: {out_path}")

# ============================================================
# CUMULATIVE LEADERBOARDS
# ============================================================

def make_cumulative_leaderboards(qb_all, table_dir=TABLE_DIR):

    # Base cumulative WAR + volume
    cum_war = (
        qb_all.groupby("passer_player_name", as_index=False)
        .agg(total_WAR=("WAR", "sum"),
             total_plays=("total_plays", "sum"))
    )

    # Plays-weighted average WAR_per_700
    cum_war700 = (
        qb_all.groupby("passer_player_name")
        .apply(lambda g: np.average(g["WAR_per_700"], weights=g["total_plays"]))
        .reset_index(name="weighted_WAR_per_700")
    )

    cum_df = cum_war.merge(cum_war700, on="passer_player_name", how="left")

    # Z-scores across the cumulative leaderboard
    def zscore(series):
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=series.index)
        return (series - series.mean()) / std

    cum_df["total_WAR_z"] = zscore(cum_df["total_WAR"])
    cum_df["WAR700_total_z"] = zscore(cum_df["weighted_WAR_per_700"])

    cum_df["Weighted_WAR_total"] = (
        WEIGHTED_WAR_VOLUME_WEIGHT * cum_df["total_WAR_z"] +
        WEIGHTED_WAR_RATE_WEIGHT * cum_df["WAR700_total_z"]
    )

    # Round for output
    num_cols = [
        "total_WAR", "total_plays", "weighted_WAR_per_700",
        "total_WAR_z", "WAR700_total_z", "Weighted_WAR_total"
    ]
    cum_df_rounded = cum_df.copy()
    cum_df_rounded[num_cols] = cum_df_rounded[num_cols].round(2)

    out_path = f"{table_dir}/cumulative_war_2021_2024.csv"
    cum_df_rounded.to_csv(out_path, index=False)
    print(f"✓ Saved cumulative WAR leaderboard to: {out_path}")

    return cum_df

# ============================================================
# COMBINED SCATTER: MULTI-YEAR (WAR vs Weighted_WAR)
# ============================================================

def visualize_combined_scatter(qb_all, fig_dir=FIG_DIR):

    plt.figure(figsize=(10, 7))
    cmap = {2021: "tab:blue", 2022: "tab:green",
            2023: "tab:orange", 2024: "tab:red"}

    for year in sorted(qb_all["season"].unique()):
        df = qb_all[qb_all["season"] == year]
        plt.scatter(df["WAR"], df["Weighted_WAR"],
                    color=cmap[year], alpha=0.6, s=45, label=str(year))

    plt.axhline(0, color='black', linewidth=0.6)
    plt.axvline(0, color='black', linewidth=0.6)

    plt.title("Combined WAR vs Weighted_WAR (2021–2024)")
    plt.xlabel("WAR (raw)")
    plt.ylabel("Weighted_WAR (seasonal z-blend)")
    plt.grid(alpha=0.3)
    plt.legend(title="Season")

    path = f"{fig_dir}/combined_war_vs_weighted.png"
    plt.savefig(path, dpi=200)
    print(f"✓ Saved combined WAR vs Weighted_WAR scatter to: {path}")


# ============================================================
# PRINT TABLES (ORDER BY Weighted_WAR / Weighted_WAR_total)
# ============================================================

def print_all_tables(qb_all, cumulative_df):

    print("\n================== FULL TABLES (2021–2024) ==================\n")

    for year in sorted(qb_all["season"].unique()):
        df = qb_all[qb_all["season"] == year].copy()

        df_print = df[[
            "passer_player_name",
            "total_plays",
            "WAR",
            "WAR_per_700",
            "Weighted_WAR"
        ]]

        df_print[["WAR", "WAR_per_700", "Weighted_WAR"]] = df_print[
            ["WAR", "WAR_per_700", "Weighted_WAR"]
        ].round(2)

        df_print = df_print.sort_values("Weighted_WAR", ascending=False)

        print(f"\n=== {year} FULL QB TABLE (sorted by Weighted_WAR) ===")
        print(df_print.to_string(index=False))

    print("\n=== CUMULATIVE 2021–2024 TABLE (sorted by Weighted_WAR_total) ===")
    cum_print = cumulative_df.copy()
    cols = [
        "passer_player_name",
        "total_plays",
        "total_WAR",
        "weighted_WAR_per_700",
        "total_WAR_z",
        "WAR700_total_z",
        "Weighted_WAR_total"
    ]
    cum_print = cum_print[cols]
    cum_print[[
        "total_WAR",
        "weighted_WAR_per_700",
        "total_WAR_z",
        "WAR700_total_z",
        "Weighted_WAR_total"
    ]] = cum_print[[
        "total_WAR",
        "weighted_WAR_per_700",
        "total_WAR_z",
        "WAR700_total_z",
        "Weighted_WAR_total"
    ]].round(2)

    cum_sorted = cum_print.sort_values("Weighted_WAR_total", ascending=False)
    print(cum_sorted.to_string(index=False))

# ============================================================
# OPTIONAL: SAVE FULL PER-SEASON DATASET
# ============================================================

def save_full_qb_table(qb_all, table_dir=TABLE_DIR):
    df = qb_all.copy()
    num_cols = [
        "total_epa", "total_plays", "epa_per_play",
        "enhanced_epa_per_play", "enhanced_epa",
        "epa_above_repl", "WAR", "WAR_per_700",
        "WAR_z", "WAR700_z", "Weighted_WAR"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].round(2)

    out_path = f"{table_dir}/qb_war_enhanced_2021_2024_full.csv"
    df.to_csv(out_path, index=False)
    print(f"✓ Saved full QB WAR table to: {out_path}")

# ============================================================
# QB POSITIONAL SUMMARY (FOR COMPARISON WITH SKILL POSITIONS)
# ============================================================

def save_qb_position_summary(qb_all, table_dir=TABLE_DIR):
    """
    Summarize QB WAR distribution across 2021–2024 player-seasons.
    Mirrors what you did for RB/WR/TE in SPWAR 04, but QB-only.
    """

    df = qb_all.copy()

    # Enforce the same min plays rule (defensive coding)
    df = df[df["total_plays"] >= ATTEMPT_MIN].copy()

    summary = {
        "position": ["QB"],
        "count": [len(df)],
        "avg_WAR": [df["WAR"].mean()],
        "median_WAR": [df["WAR"].median()],
        "IQR_WAR": [
            df["WAR"].quantile(0.75) - df["WAR"].quantile(0.25)
        ],
        "avg_WAR_700": [df["WAR_per_700"].mean()],
        "median_WAR_700": [df["WAR_per_700"].median()],
        "IQR_WAR_700": [
            df["WAR_per_700"].quantile(0.75) - df["WAR_per_700"].quantile(0.25)
        ],
    }

    summary_df = pd.DataFrame(summary)

    out_path = f"{table_dir}/qb_war_position_summary.csv"
    summary_df.to_csv(out_path, index=False)

    print("\n===== QB WAR Position Summary — 2021–2024 (player-seasons) =====")
    print(summary_df.to_string(index=False))
    print(f"✓ Saved QB WAR position summary → {out_path}")


def visualize_qb_war_boxplot(qb_all, fig_dir=FIG_DIR):
    """
    Simple one-box QB WAR distribution plot (2021–2024).
    """

    df = qb_all.copy()
    df = df[df["total_plays"] >= ATTEMPT_MIN].copy()

    plt.figure(figsize=(5, 6))
    sns.boxplot(y=df["WAR"])
    plt.title("QB WAR Distribution per Season (2021–2024)")
    plt.ylabel("WAR")
    plt.tight_layout()

    out_path = f"{fig_dir}/qb_war_boxplot.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"✓ Saved QB WAR boxplot → {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("CLEAN MULTI-SEASON ENHANCED QB WAR CALCULATOR (2021–2024)")
    print("=" * 70)

    qb_df = load_all_qbs_multi(2021, 2024)
    coeff_df = estimate_empirical_coefficients(qb_df)
    qb_df = apply_seasonal_enhancement(qb_df, coeff_df)

    epa_per_win = calculate_epa_per_win_fixed()
    print(f"\nEPA per win constant: {epa_per_win:.2f}")

    repl_df = identify_replacement_by_season(qb_df)
    qb_all = calculate_war_multi(qb_df, repl_df, epa_per_win)

    # Add Weighted_WAR (seasonal z-blend)
    qb_all = add_weighted_war_per_season(qb_all)

    visualize_war_scatter(qb_all)
    visualize_weighted_war_scatter(qb_all)
    cumulative_df = make_cumulative_leaderboards(qb_all)
    visualize_combined_scatter(qb_all)

    save_full_qb_table(qb_all)
    print_all_tables(qb_all, cumulative_df)

    # NEW: QB-only positional summary + boxplot
    save_qb_position_summary(qb_all)
    visualize_qb_war_boxplot(qb_all)

if __name__ == "__main__":
    main()
