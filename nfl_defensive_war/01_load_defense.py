"""
01_load_defense.py
-------------------

Purpose:
    Load yearly defensive summary files (2021–2024), standardize schema,
    create engineered features, define positional role flags, and output a
    single engineered defender-season table for downstream WAR modeling.

Output:
    /outputs/defense_engineered_2021_2024.csv
"""

# ============================================================
# Imports & Paths
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("/Users/anokhpalakurthi/Documents/nfl_defensive_war")
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "defense_engineered_2021_2024.csv"

YEAR_FILES = {
    2021: DATA_DIR / "defense_summary_2021.csv",
    2022: DATA_DIR / "defense_summary_2022.csv",
    2023: DATA_DIR / "defense_summary_2023.csv",
    2024: DATA_DIR / "defense_summary_2024.csv",
}

# ============================================================
# Helpers
# ============================================================

def standardize_team_code(code: str) -> str:
    if pd.isna(code):
        return np.nan
    code = str(code).upper().strip()

    TEAM_MAP = {
        "ARZ": "ARI",
        "BLT": "BAL",
        "CLV": "CLE",
        "HST": "HOU",
        "JAX": "JAC",
        "LA":  "LAR",
        "STL": "LAR",
        "SD":  "LAC",
        "OAK": "LV",
        "WSH": "WAS",
    }
    return TEAM_MAP.get(code, code)


def safe_div(n, d):
    return n / d if d != 0 else 0.0


def assign_role(row):
    """
    Assign defensive role based on PFF position code and snap distribution.
    
    PFF Position Codes:
        - DI = Defensive Interior (IDL)
        - ED = Edge Defender
        - LB = Linebacker
        - CB = Cornerback
        - S  = Safety
    
    Role Output:
        - IDL:  Interior defensive linemen (DT, NT, 3-tech)
        - EDGE: Edge rushers (DE, OLB pass rushers)
        - LB:   Off-ball linebackers
        - CB:   Cornerbacks
        - S:    Safeties
        - HYBRID: Coverage/rush tweeners
        - UNKNOWN: Insufficient data
    """
    pos = str(row["position"]).upper()
    pr = row["snap_counts_pass_rush"]
    cov = row["snap_counts_coverage"]
    box = row["snap_counts_box"]
    slot = row["snap_counts_slot"]
    deep = row["snap_counts_fs"]
    dl = row["snap_counts_dl"]
    total = row["snap_counts_defense"]

    if total == 0:
        return "UNKNOWN"

    pr_rate = pr / total
    cov_rate = cov / total
    box_rate = box / total
    slot_rate = slot / total
    deep_rate = deep / total
    dl_rate = dl / total

    # =========================================================
    # IDL FIRST — Check position code before snap distribution
    # PFF uses "DI" for interior defenders
    # This prevents high-pass-rush IDL (like Aaron Donald) from
    # being misclassified as EDGE
    # =========================================================
    if pos in ["DI", "DT", "NT"]:
        return "IDL"
    
    # IDL by snap distribution (backup for any missed codes)
    if dl_rate >= 0.55 and cov_rate <= 0.15 and pos not in ["DE", "ED", "EDGE", "OLB"]:
        return "IDL"

    # =========================================================
    # EDGE — Defensive ends and edge rushers
    # PFF uses "ED" for edge defenders
    # =========================================================
    if pos in ["DE", "ED", "EDGE"]:
        return "EDGE"
    
    # OLB with high pass rush rate = EDGE
    if pos == "OLB" and pr_rate >= 0.35:
        return "EDGE"
    
    # High pass rush rate + DL snaps = EDGE (backup)
    if pr_rate >= 0.45 and dl_rate > 0.2 and pos not in ["DI", "DT", "NT"]:
        return "EDGE"

    # =========================================================
    # COVERAGE POSITIONS
    # =========================================================
    if pos == "CB" and cov_rate >= 0.40:
        return "CB"

    if pos in ["S", "FS", "SS"] and (deep_rate >= 0.25 or cov_rate >= 0.35):
        return "S"

    # =========================================================
    # LINEBACKERS
    # =========================================================
    if pos in ["LB", "ILB", "MLB"]:
        return "LB"
    
    # OLB without high pass rush = off-ball LB
    if pos == "OLB" and pr_rate < 0.35:
        return "LB"

    # =========================================================
    # SLOT CORNERS (often listed differently)
    # =========================================================
    if slot_rate >= 0.25 and cov_rate >= 0.40:
        return "CB"

    # =========================================================
    # HYBRIDS — Coverage/rush tweeners
    # =========================================================
    if cov_rate >= 0.30 and pr_rate >= 0.20:
        return "HYBRID"
    if box_rate >= 0.25 and cov_rate >= 0.25:
        return "HYBRID"

    return "UNKNOWN"


# ============================================================
# Main Load + Engineering
# ============================================================

def load_and_engineer():
    dfs = []

    print("\n============================================")
    print(" Loading & Engineering Defensive Data (2021–2024)")
    print("============================================\n")

    for year, path in YEAR_FILES.items():
        print(f"Loading {year} from {path} ...")
        df = pd.read_csv(path)

        # Lowercase BEFORE any processing
        df.columns = [c.lower() for c in df.columns]

        df["season"] = year

        # Create authoritative TEAM column
        if "team" in df.columns:
            df["team"] = df["team"].apply(standardize_team_code)
        elif "team_name" in df.columns:
            df["team"] = df["team_name"].apply(standardize_team_code)
        else:
            raise ValueError("No team/team_name column found in defensive summary file.")

        # Per-snap and per-target engineering
        df["pressures_per_prsnap"] = df.apply(lambda r: safe_div(r["total_pressures"], r["snap_counts_pass_rush"]), axis=1)
        df["sacks_per_prsnap"]     = df.apply(lambda r: safe_div(r["sacks"], r["snap_counts_pass_rush"]), axis=1)
        df["hits_per_prsnap"]      = df.apply(lambda r: safe_div(r["hits"], r["snap_counts_pass_rush"]), axis=1)
        df["tfl_per_rundef"]       = df.apply(lambda r: safe_div(r["tackles_for_loss"], r["snap_counts_run_defense"]), axis=1)
        df["stops_per_rundef"]     = df.apply(lambda r: safe_div(r["stops"], r["snap_counts_run_defense"]), axis=1)

        df["yards_per_target"] = df.apply(lambda r: safe_div(r["yards"], r["targets"]), axis=1)
        df["tds_per_target"]   = df.apply(lambda r: safe_div(r["touchdowns"], r["targets"]), axis=1)
        df["ints_per_target"]  = df.apply(lambda r: safe_div(r["interceptions"], r["targets"]), axis=1)
        df["pbus_per_target"]  = df.apply(lambda r: safe_div(r["pass_break_ups"], r["targets"]), axis=1)

        df["missed_tackle_pct"] = df["missed_tackle_rate"]
        df["tackles_per_snap"]  = df.apply(lambda r: safe_div(r["tackles"], r["snap_counts_defense"]), axis=1)
        df["stops_per_snap"]    = df.apply(lambda r: safe_div(r["stops"], r["snap_counts_defense"]), axis=1)
        df["penalties_per_snap"] = df.apply(lambda r: safe_div(r["penalties"], r["snap_counts_defense"]), axis=1)

        df["role"] = df.apply(assign_role, axis=1)

        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # Print role distribution for validation
    print("\nRole distribution:")
    print(all_df["role"].value_counts())

    print("\nSaving engineered file to:")
    print(f"  {OUT_FILE}\n")
    all_df.to_csv(OUT_FILE, index=False)

    print("Done.\n")


if __name__ == "__main__":
    load_and_engineer()