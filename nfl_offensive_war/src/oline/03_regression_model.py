"""
03_regression_model.py
Run regression analysis to estimate the relationship between team O-line metrics
and team offensive EPA. This gives us coefficients to allocate value to individuals.

Inputs:
    - /Users/anokhpalakurthi/Documents/nfl_offensive_war/data/processed/oline_team_2021_2024.csv (from 02_team_aggregation.py)

Outputs:
    - /Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs/tables/oline_regression_results.csv
    - /Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs/figures/oline_epa_correlations.png
    - Prints regression coefficients for use in 04_individual_war.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("/Users/anokhpalakurthi/Documents/nfl_offensive_war/data")
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = Path("/Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs")
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Input
OLINE_TEAM_FILE = PROCESSED_DIR / "oline_team_2021_2024.csv"

# Ensure output directories exist
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def load_team_data(filepath: Path) -> pd.DataFrame:
    """Load team-level O-line + EPA data."""
    print(f"Loading team data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} team-seasons")
    return df


def exploratory_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlations between O-line metrics and offensive EPA."""
    
    # Define O-line predictor columns
    oline_cols = [
        "team_pass_block_grade", "team_run_block_grade", "team_offense_grade",
        "pressure_rate", "sack_rate", 
        "total_pressures_allowed", "total_sacks_allowed",
    ]
    
    # Add position-specific columns if they exist
    for pos in ["T", "G", "C"]:
        for stat in ["pass_block_grade", "run_block_grade"]:
            col = f"{pos}_{stat}"
            if col in df.columns:
                oline_cols.append(col)
    
    # Keep only columns that exist
    oline_cols = [c for c in oline_cols if c in df.columns]
    
    # Calculate correlations with EPA measures
    epa_cols = ["total_offensive_epa", "rushing_epa", "passing_epa",
                "offensive_epa_per_play", "rushing_epa_per_play", "passing_epa_per_play"]
    epa_cols = [c for c in epa_cols if c in df.columns]
    
    correlations = []
    for oline_col in oline_cols:
        for epa_col in epa_cols:
            corr = df[oline_col].corr(df[epa_col])
            correlations.append({
                "oline_metric": oline_col,
                "epa_metric": epa_col,
                "correlation": corr
            })
    
    corr_df = pd.DataFrame(correlations)
    
    # Print key findings
    print("\n" + "=" * 60)
    print("KEY CORRELATIONS WITH TOTAL OFFENSIVE EPA")
    print("=" * 60)
    
    key_corrs = corr_df[corr_df["epa_metric"] == "total_offensive_epa"].sort_values(
        "correlation", ascending=False
    )
    for _, row in key_corrs.iterrows():
        print(f"  {row['oline_metric']}: {row['correlation']:.3f}")
    
    return corr_df


def plot_correlations(df: pd.DataFrame, save_path: Path):
    """Create correlation heatmap for O-line metrics vs EPA."""
    
    # Select relevant columns
    oline_cols = ["team_pass_block_grade", "team_run_block_grade", 
                  "pressure_rate", "sack_rate"]
    epa_cols = ["total_offensive_epa", "rushing_epa", "passing_epa"]
    
    # Keep only columns that exist
    oline_cols = [c for c in oline_cols if c in df.columns]
    epa_cols = [c for c in epa_cols if c in df.columns]
    
    # Create correlation matrix
    corr_matrix = df[oline_cols + epa_cols].corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", 
                center=0, ax=ax, square=True)
    ax.set_title("O-Line Metrics vs Offensive EPA Correlations (2021-2024)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved correlation plot to: {save_path}")

def plot_grade_vs_epa_scatter(df: pd.DataFrame, save_path: Path):
    """
    Create partial regression plots showing team O-line grades vs offensive EPA,
    controlling for the other variable. This matches the multivariate regression coefficients.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get clean data
    mask = (df["team_pass_block_grade"].notna() & 
            df["team_run_block_grade"].notna() & 
            df["total_offensive_epa"].notna())
    df_clean = df[mask].copy()
    
    x_pass = df_clean["team_pass_block_grade"]
    x_run = df_clean["team_run_block_grade"]
    y = df_clean["total_offensive_epa"]
    
    # --- Plot 1: Pass Block Grade (controlling for Run Block) ---
    ax1 = axes[0]
    
    # Residualize: regress EPA on run block, get residuals
    z_y_on_run = np.polyfit(x_run, y, 1)
    y_resid_run = y - np.poly1d(z_y_on_run)(x_run)
    
    # Residualize: regress pass block on run block, get residuals
    z_pass_on_run = np.polyfit(x_run, x_pass, 1)
    x_pass_resid = x_pass - np.poly1d(z_pass_on_run)(x_run)
    
    ax1.scatter(x_pass_resid, y_resid_run, alpha=0.6, color="steelblue", 
                edgecolor="white", s=80)
    
    # Regression line on residuals
    z1 = np.polyfit(x_pass_resid, y_resid_run, 1)
    p1 = np.poly1d(z1)
    x1_line = np.linspace(x_pass_resid.min(), x_pass_resid.max(), 100)
    ax1.plot(x1_line, p1(x1_line), color="darkred", linewidth=2,
             label=f"Coefficient: {z1[0]:.2f} EPA/grade point")
    
    ax1.set_xlabel("Team Pass Block Grade (residualized)", fontsize=12)
    ax1.set_ylabel("Total Offensive EPA (residualized)", fontsize=12)
    ax1.set_title("Pass Block Grade vs Offensive EPA\n(controlling for Run Block)", fontsize=13)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    
    # --- Plot 2: Run Block Grade (controlling for Pass Block) ---
    ax2 = axes[1]
    
    # Residualize: regress EPA on pass block, get residuals
    z_y_on_pass = np.polyfit(x_pass, y, 1)
    y_resid_pass = y - np.poly1d(z_y_on_pass)(x_pass)
    
    # Residualize: regress run block on pass block, get residuals
    z_run_on_pass = np.polyfit(x_pass, x_run, 1)
    x_run_resid = x_run - np.poly1d(z_run_on_pass)(x_pass)
    
    ax2.scatter(x_run_resid, y_resid_pass, alpha=0.6, color="forestgreen",
                edgecolor="white", s=80)
    
    # Regression line on residuals
    z2 = np.polyfit(x_run_resid, y_resid_pass, 1)
    p2 = np.poly1d(z2)
    x2_line = np.linspace(x_run_resid.min(), x_run_resid.max(), 100)
    ax2.plot(x2_line, p2(x2_line), color="darkred", linewidth=2,
             label=f"Coefficient: {z2[0]:.2f} EPA/grade point")
    
    ax2.set_xlabel("Team Run Block Grade (residualized)", fontsize=12)
    ax2.set_ylabel("Total Offensive EPA (residualized)", fontsize=12)
    ax2.set_title("Run Block Grade vs Offensive EPA\n(controlling for Pass Block)", fontsize=13)
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved partial regression plot to: {save_path}")

def run_regression_total_epa(df: pd.DataFrame) -> dict:
    """
    Regression Model 1: Predict total offensive EPA from O-line grades.
    This is the primary model for allocating value.
    """
    print("\n" + "=" * 60)
    print("REGRESSION 1: O-Line Grades -> Total Offensive EPA")
    print("=" * 60)
    
    # Predictors: pass block and run block grades
    X_cols = ["team_pass_block_grade", "team_run_block_grade"]
    X = df[X_cols].copy()
    y = df["total_offensive_epa"].copy()
    
    # Drop any rows with missing values
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    
    # Add constant for intercept
    X = sm.add_constant(X)
    
    # Fit OLS
    model = sm.OLS(y, X).fit()
    print(model.summary())
    
    # Store results
    results = {
        "model_name": "total_offensive_epa ~ pass_block + run_block",
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "n_obs": int(model.nobs),
        "coefficients": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "std_errors": model.bse.to_dict(),
    }
    
    return results


def run_regression_split_epa(df: pd.DataFrame) -> dict:
    """
    Regression Model 2: Separate regressions for rushing and passing EPA.
    Helps us understand if run/pass blocking have different impacts.
    """
    print("\n" + "=" * 60)
    print("REGRESSION 2a: Run Block Grade -> Rushing EPA")
    print("=" * 60)
    
    results = {}
    
    # Rushing EPA ~ Run Block Grade
    X_rush = df[["team_run_block_grade"]].copy()
    y_rush = df["rushing_epa"].copy()
    mask = X_rush.notna().all(axis=1) & y_rush.notna()
    X_rush = sm.add_constant(X_rush[mask])
    y_rush = y_rush[mask]
    
    model_rush = sm.OLS(y_rush, X_rush).fit()
    print(model_rush.summary())
    
    results["rushing_model"] = {
        "r_squared": model_rush.rsquared,
        "coef_run_block": model_rush.params.get("team_run_block_grade", np.nan),
        "pvalue_run_block": model_rush.pvalues.get("team_run_block_grade", np.nan),
    }
    
    # Passing EPA ~ Pass Block Grade
    print("\n" + "=" * 60)
    print("REGRESSION 2b: Pass Block Grade -> Passing EPA")
    print("=" * 60)
    
    X_pass = df[["team_pass_block_grade"]].copy()
    y_pass = df["passing_epa"].copy()
    mask = X_pass.notna().all(axis=1) & y_pass.notna()
    X_pass = sm.add_constant(X_pass[mask])
    y_pass = y_pass[mask]
    
    model_pass = sm.OLS(y_pass, X_pass).fit()
    print(model_pass.summary())
    
    results["passing_model"] = {
        "r_squared": model_pass.rsquared,
        "coef_pass_block": model_pass.params.get("team_pass_block_grade", np.nan),
        "pvalue_pass_block": model_pass.pvalues.get("team_pass_block_grade", np.nan),
    }
    
    return results


def run_regression_with_pressure(df: pd.DataFrame) -> dict:
    """
    Regression Model 3: Include pressure rate as a predictor.
    Pressure rate might mediate the relationship between grades and EPA.
    """
    print("\n" + "=" * 60)
    print("REGRESSION 3: Including Pressure Rate")
    print("=" * 60)
    
    X_cols = ["team_pass_block_grade", "team_run_block_grade", "pressure_rate"]
    X = df[X_cols].copy()
    y = df["total_offensive_epa"].copy()
    
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())
    
    results = {
        "model_name": "total_epa ~ pass_block + run_block + pressure_rate",
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "coefficients": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
    }
    
    return results


def run_positional_regression(df: pd.DataFrame) -> dict:
    """
    Regression Model 4: Position-specific grades (T, G, C) as separate predictors.
    Tests whether tackles, guards, centers contribute differently.
    """
    print("\n" + "=" * 60)
    print("REGRESSION 4: Position-Specific Grades")
    print("=" * 60)
    
    # Check which position columns exist
    pos_cols = []
    for pos in ["T", "G", "C"]:
        for grade_type in ["pass_block_grade", "run_block_grade"]:
            col = f"{pos}_{grade_type}"
            if col in df.columns:
                pos_cols.append(col)
    
    if not pos_cols:
        print("No position-specific columns found. Skipping this regression.")
        return {}
    
    X = df[pos_cols].copy()
    y = df["total_offensive_epa"].copy()
    
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())
    
    results = {
        "model_name": "total_epa ~ T + G + C grades",
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "coefficients": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
    }
    
    return results


def save_regression_summary(results: dict, save_path: Path):
    """Save regression results to CSV for reference."""
    rows = []
    for model_name, model_results in results.items():
        if isinstance(model_results, dict) and "coefficients" in model_results:
            for var, coef in model_results["coefficients"].items():
                rows.append({
                    "model": model_name,
                    "variable": var,
                    "coefficient": coef,
                    "pvalue": model_results.get("pvalues", {}).get(var, np.nan),
                    "r_squared": model_results.get("r_squared", np.nan),
                })
    
    if rows:
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(save_path, index=False)
        print(f"\nSaved regression summary to: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("O-Line Regression Analysis: Blocking -> Offensive EPA")
    print("=" * 60)
    
    # Load data
    df = load_team_data(OLINE_TEAM_FILE)
    
    # Exploratory correlations
    corr_df = exploratory_correlations(df)
    
    # Correlation plot
    plot_correlations(df, FIGURES_DIR / "oline_epa_correlations.png")

    # Scatter plots with regression lines
    plot_grade_vs_epa_scatter(df, FIGURES_DIR / "oline_grade_vs_epa_scatter.png")
    
    # Run regressions
    all_results = {}
    
    # Model 1: Primary model
    all_results["model1_total_epa"] = run_regression_total_epa(df)
    
    # Model 2: Split by rush/pass
    split_results = run_regression_split_epa(df)
    all_results["model2a_rushing"] = split_results.get("rushing_model", {})
    all_results["model2b_passing"] = split_results.get("passing_model", {})
    
    # Model 3: With pressure rate
    all_results["model3_with_pressure"] = run_regression_with_pressure(df)
    
    # Model 4: Position-specific
    all_results["model4_positional"] = run_positional_regression(df)
    
    # Save summary
    save_regression_summary(all_results, TABLES_DIR / "oline_regression_results.csv")
    
    # Print key takeaways for next step
    print("\n" + "=" * 60)
    print("KEY COEFFICIENTS FOR WAR CALCULATION (from Model 1)")
    print("=" * 60)
    
    if "model1_total_epa" in all_results:
        coefs = all_results["model1_total_epa"].get("coefficients", {})
        print(f"  Pass Block Grade coefficient: {coefs.get('team_pass_block_grade', 'N/A'):.2f}")
        print(f"  Run Block Grade coefficient: {coefs.get('team_run_block_grade', 'N/A'):.2f}")
        print(f"\nInterpretation: A 1-point increase in team pass block grade")
        print(f"predicts {coefs.get('team_pass_block_grade', 0):.2f} more offensive EPA over a season.")
    
    return all_results


if __name__ == "__main__":
    results = main()