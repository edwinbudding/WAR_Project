"""
03_enhanced_epa_per_play.py

This script applies regression coefficients from Step 2
to compute modeled and enhanced EPA/play for each QB-season.
"""

import pandas as pd
import os

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "/Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs/tables"
OUT_PATH = f"{DATA_DIR}/qb_enhanced_epa.csv"


# ============================================================
# MAIN FUNCTION
# ============================================================

def apply_enhanced_epa(qb_data_path, coeffs_path):
    """Compute modeled and enhanced EPA/play for each QB-season."""
    qb_df = pd.read_csv(qb_data_path)
    coeff_df = pd.read_csv(coeffs_path)

    # --- Validate expected columns ---
    required_cols = ["avg_cpoe", "overall_success", "epa_per_play", "season"]
    for col in required_cols:
        if col not in qb_df.columns:
            raise ValueError(f"Missing expected column: {col}")

    # --- Compute modeled EPA/play per season ---
    modeled_epa = []
    for _, row in qb_df.iterrows():
        season_coeffs = coeff_df[coeff_df["season"] == row["season"]]
        if season_coeffs.empty:
            # fallback to pooled
            season_coeffs = coeff_df[coeff_df["season"] == "Pooled"]

        coeff = season_coeffs.iloc[0]
        modeled = (
            coeff["intercept"]
            + coeff["beta_cpoe"] * row["avg_cpoe"]
            + coeff["beta_success"] * row["overall_success"]
        )
        modeled_epa.append(modeled)

    qb_df["modeled_epa_per_play"] = modeled_epa

    # --- Blend actual + modeled (50/50 weighting) ---
    qb_df["enhanced_epa_per_play"] = 0.5 * (
        qb_df["epa_per_play"] + qb_df["modeled_epa_per_play"]
    )

    # --- Compute total enhanced EPA ---
    qb_df["enhanced_epa"] = qb_df["enhanced_epa_per_play"] * qb_df["total_plays"]

    # --- Save ---
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    qb_df.to_csv(OUT_PATH, index=False)
    print(f"✓ Saved enhanced EPA dataset to: {OUT_PATH}")

    return qb_df


# ============================================================
# EXECUTE
# ============================================================

if __name__ == "__main__":
    QB_DATA = f"{DATA_DIR}/qb_war_2021_2024.csv"
    COEFFS = f"{DATA_DIR}/coefficients_per_season.csv"
    df = apply_enhanced_epa(QB_DATA, COEFFS)

    print("\nSample enhanced EPA/play by season:")
    print(df.groupby("season")["enhanced_epa_per_play"].describe())
