# ============================================================
# EMPIRICAL ENHANCEMENT CALIBRATION — 2024 QB DATA (STANDARDIZED)
# ============================================================

import pandas as pd
import statsmodels.api as sm
import os

print("="*70)
print("ESTIMATING EMPIRICAL WEIGHTS FOR CPOE AND SUCCESS RATE (STANDARDIZED)")
print("="*70)

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------
qb_path = "/Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs/tables/qb_war_2024_enhanced.csv"
df = pd.read_csv(qb_path)

# Keep necessary variables
cols = ["passer_player_name", "epa_per_play", "avg_cpoe", "overall_success"]
df = df[cols].dropna()

print(f"Loaded {len(df)} QB entries for regression.\n")

# ------------------------------------------------------------
# 2. Standardize variables (z-score)
# ------------------------------------------------------------
for col in ["avg_cpoe", "overall_success"]:
    mean = df[col].mean()
    std = df[col].std(ddof=0)
    df[f"{col}_z"] = (df[col] - mean) / std
    print(f"{col}: mean={mean:.3f}, std={std:.3f}")

# ------------------------------------------------------------
# 3. Run regression on standardized predictors
# ------------------------------------------------------------
X = df[["avg_cpoe_z", "overall_success_z"]]
y = df["epa_per_play"]

X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()

print(model.summary())

# Extract coefficients
b0 = model.params["const"]
b1 = model.params["avg_cpoe_z"]
b2 = model.params["overall_success_z"]

print(f"\nIntercept: {b0:.4f}")
print(f"Standardized β₁ (CPOE): {b1:.4f}")
print(f"Standardized β₂ (SuccessRate): {b2:.4f}")

# ------------------------------------------------------------
# 4. Compute enhanced EPA (using standardized weights)
# ------------------------------------------------------------
# NOTE: because predictors are standardized, the model explains variance,
# not raw additive EPA/play units. Use this for relative influence, not direct scaling.

df["modeled_epa_per_play_std"] = (
    b0
    + b1 * df["avg_cpoe_z"]
    + b2 * df["overall_success_z"]
)

# Optionally, combine with base EPA to get a blended "enhanced" metric
df["enhanced_epa_per_play_empirical"] = (
    0.5 * df["epa_per_play"]
    + 0.5 * df["modeled_epa_per_play_std"]
)

# ------------------------------------------------------------
# 5. Save results
# ------------------------------------------------------------
out_path = "/Users/anokhpalakurthi/Documents/nfl_offensive_war/outputs/tables/qb_2024_empirical_enhancement_standardized.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False)

print(f"\n✓ Saved standardized enhancement coefficients to: {out_path}")