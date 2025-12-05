# NFL Defensive WAR (2021-2024)

A Wins Above Replacement (WAR) model for NFL defensive players using publicly available data. Part of the ["Untangling the NFL"](https://bignokh.com) blog series.

## Overview

This project builds a defensive WAR metric using:

- **PFF summary statistics** (snap counts, pressures, stops, coverage metrics)
- **nflfastR play-by-play data** (EPA values for empirical weight derivation)

Unlike offensive WAR where individual contributions are more measurable, defensive WAR faces a fundamental attribution problem: without tracking data, we can't know who was responsible for coverage on completions. This model acknowledges that limitation while still producing reasonable positional valuations.

## Key Findings

| Position | Avg WAR
|----------|--------
| EDGE | 0.121
| IDL | 0.070
| LB | 0.169
| CB | 0.215
| S | 0.232

**Validation:** This model correlates strongly with PFF's independent 2006-2018 WAR methodology (r = 0.95), suggesting both approaches capture real positional value despite different time periods and methods.

## Methodology

### Activity Weights (EPA Magnitude-Based)
- **Coverage:** 55% — Most defensive plays are pass coverage
- **Run Defense:** 25% — Run stops at/behind line of scrimmage
- **Pass Rush:** 10% — Sacks, hits, hurries
- **Turnovers:** 10% — INTs, forced fumbles, recoveries

### Position-Specific Multipliers

**Run Defense** (from tackle EPA by field position):
- EDGE: 1.07x (tackles at -0.6 yards)
- IDL: 1.00x (baseline, tackles at LOS)
- LB: 0.21x (tackles at +1.7 yards, less valuable)
- CB/S: 0.00x (tackles happen too late)

**Coverage** (from catch rate allowed):
- CB: 1.00x (64.6% catch rate, baseline)
- S: 0.94x (68.5% catch rate)
- LB: 0.81x (79.5% catch rate, significant penalty)
- EDGE/IDL: 0.50x (minimal coverage responsibility)

### Allocation Within Activities
- **Pass Rush:** 70% sacks / 20% hits / 10% hurries
- **Run Defense:** 70% stops / 30% TFLs
- **Coverage:** 85% snaps / 15% PBUs
- **Turnovers:** 60% INTs / 25% FF / 15% FR

All weights derived empirically from EPA analysis of 2021-2024 play-by-play data.

## Usage

```bash
# Run the pipeline in order
python src/01_load_defense.py
python src/02_team_aggregation.py
python src/03_war_model.py
python src/04_validation.py
```

## Known Limitations

1. **LB Inflation (~70% above target):** Without tracking data, we can't attribute coverage responsibility on completions. LBs get credit for coverage snaps despite objectively worse coverage outcomes. The 0.81x catch-rate multiplier helps but doesn't fully solve this. Or maybe they're underrated in coverage; I'm not sure.

2. **Team Context Effects:** Players on bad defenses (e.g., 2024 Panthers) get penalized because they're allocated shares of a negative WAR pool. A good player on a bad team will have lower WAR than the same player on a good team. Take these with a grain of salt. 

3. **No Film Grading:** This model uses box score statistics only. It cannot capture technique, assignment execution, or plays where a defender's presence influenced the QB without a stat. I did not want to use PFF grades like I did with O-Linemen; felt like cheating.

## Data Sources

- **PFF Premium Stats:** Defensive summary tables (subscription required)
- **nflfastR:** Play-by-play and roster data (open source)
- **Over The Cap:** Market salary data for validation