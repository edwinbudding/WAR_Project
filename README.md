Untangling the NFL: My Wins Above Replacement Framework for NFL Players

Overview

- This repository contains my personal take on Wins Above Replacement (WAR) framework for NFL players. 
- I developed this as a graduate project for a football analytics independent study. 
- Data Scope: 2021-2024 NFL seasons
- Companion Blog Series: "Untangling the NFL"

TLDR Methodology:

- The core formula is either using directly attributed EPA, 'predicted' EPA by a few set season-long statistics, or a mix of both for graded players.
- After using these, I estimate the typical EPA per team-win.
- "Replacement level" is the bottom 25 percentile of each position, with customized snap counts for positions. 
- Offensive WAR = (Player EPA - Replacement EPA) / (EPA per Win)
- Defensive WAR = (Player estimated EPA - Replacement EPA) / (Side of the Field Divisor) (NOTE THIS IS VERY MUCH A WORK IN PROGRESS ATM)

Limitations

- Attribution Differences: O-line (and planned defense) use regression-based attribution rather than direct EPA
- Double-Counting Risk: O-line blocking credit may overlap with skill player EPA
- PFF Subjectivity: O-line model relies on PFF grades, which introduce some subjectivity
- Defensive Analysis current exists in one script; struggles to differentiate individual players from team context at cornerback position
- Missing Factors: WAR doesn't capture durability, insurance value, leadership, or locker room impact (AKA "YOUR NUMBERS DON'T MEASURE HEART!")

TBD:

- Year-over-year WAR stability analysis
- Contract value analysis (WAR per dollar)
- Aging curves by position
- Predictive modeling for future WAR
- Draft value chart creation
- Positional breakdown on defense in sep files

References

- Yurko, R., Ventura, S., & Horowitz, M. (2018). nflWAR: A Reproducible Method for Offensive Player Evaluation in Football
- David Drinen's Approximate Value
- PAVing the Way for the Future – A Model That Determines Player Value and Evaluates Trades in the NFL - Atul Venkatesh
- PFF WAR: Modeling Player Value in American Football (2018)
- Over The Cap: Positional Value in the NFL

Author

Anokh Palakurthi

Special thanks to Claude/Opus 4.5 for helping me program/code like any good data analyst. 

License

This project is for educational purposes. Data sourced from nflFastR is open source. PFF data usage subject to PFF Premium terms of service.
