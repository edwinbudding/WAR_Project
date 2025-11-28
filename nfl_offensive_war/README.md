NFL Wins Above Replacement: Quantifying Player Value in the Modern NFL with EPA & More

Overview

This project develops a Wins Above Replacement (WAR) framework for NFL offenses, using play-by-play Expected Points Added (EPA) 
Using EPA, I estimate how individual players contribute to team success in terms of 'wins abve replacement" 
This is done for the 2021-2024 seasons

Data Sources

• nflfastR / nflplayR play-by-play (via nfl_data_py)
• Pro Football Reference (player metadata + AV)
• PFF WAR (for external benchmarking)

Methodology Summary

• Each offensive play produces or loses expected points
• Credit is assigned based on involvement roles

Replacement baselines per position

• Modeled using bottom 25 percentile performance pools and customized snap counts
• Standardized to 1 WAR ≈ 1 team win over a season

Validation + Diagnostics

• WAR vs AV correlations (player level)
• WAR vs Wins regressions (team level)
• AV vs Wins regressions as point of comparison

Author: Anokh Palakurthi
Special thanks to Claude/Sonnet 4.5 for helping me jump into programming.

NOTE: Because of the sheer size of play by play data, as well as the CSVs/plots I've built from this project, I cannot share all of them inside GitHub - if you want a CSV, just email me or contact me about it. 




