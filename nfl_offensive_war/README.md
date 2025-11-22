NFL Wins Above Replacement: Quantifying Player Value in the Modern NFL with EPA & More

Overview

This project develops a Wins Above Replacement (WAR) framework for NFL offenses, using play-by-play Expected Points Added (EPA) 
Using EPA, I estimate how individual players contribute to team success in terms of 'wins abve replacement" 
This is largely still a work in progress, but so far I have completed calculating WAR for QBs/RBs/WRs/TEs
Offensive linemen still a WIP as are defensive players

Data Sources

• nflfastR / nflplayR play-by-play (via nfl_data_py)
• Pro Football Reference (player metadata + AV)
• PFF WAR (for external benchmarking)
• Manual postseason win results (2024)

Seasons covered: 2021–2024

⸻

Methodology Summary

EPA-based play credit

• Each offensive play produces or loses expected points
• Credit is assigned based on involvement roles:
	•	QB: passing/rushing on scrambles and designed runs
	•	WR/TE/RB: targeted or responsible ball carrier
	•	Negative credit is applied for turnovers + failed plays

Replacement baselines per position

• Modeled using bottom 25 percentile performance pools
• Standardized to 1 WAR ≈ 1 team win over a season

Validation + Diagnostics

• WAR vs AV correlations (player level)
• WAR vs Wins regressions (team level)
• AV vs Wins regressions as point of comparison

Author: Anokh Palakurthi
Special thanks to Claude/Sonnet 4.5 for helping me jump into programming
