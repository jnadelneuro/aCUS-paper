# Figure S2 (rel. Fig 1) — RI60 training & shock-probe detail (adolescent + adult)

Panels **S2A–S2Q**: nosepoke rate by sex/day, port entries, pokes/entry, reward rate, efficiency,
shock-probe microstructure, poke-rate-vs-shocks correlation, and the adult-stress counterparts.

| Panel(s) | Content | Code | Env |
|---|---|---|---|
| S2A–J | adolescent RI60 + shock-probe detail | `src/analysisapp/RI60_behavior_analysis.py` | `jimmy` |
| S2K–Q | adult-stress RI60 + shock-probe detail | `src/analysisapp/RI60_behavior_analysis.py` | `jimmy` |

## Reproduce

Same pipeline as **Figure 1** — this is the fuller set of behavior metrics from the same run:

1. `conda activate jimmy` → `python src/analysisapp/RI60_behavior_analysis.py` →
   `RI30_RI60_data.csv`, `Shock_Data.csv`, `shock_microstructure.csv`, `shock_slopes.csv`,
   `RI60dataForCorrs.csv`, `forPrism*` (the S2 sources for both adolescent and adult cohorts).

Panels are drawn in **Prism** (`stats/aCUS RI60 Behavior Collective (Not KiR though).prism`, `stats/adult acus RI60.prism`).

## Notes
- S2H (shock-probe nosepokes over time) and S2J (poke-rate vs shocks correlation) come from the same CSVs; the correlation/regression is computed in Prism.
