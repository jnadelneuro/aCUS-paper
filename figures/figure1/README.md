# Figure 1 — Chronic adolescent (not adult) stress increases punishment-resistant reward-seeking

Panels produced by code: **1C, 1D, 1F, 1G, 1H, 1I** (behavior) and **1E–F** (IPI mixture model).
Schematics 1A, 1B and the 1E cartoon are made in BioRender (no code).

| Panel(s) | Content | Code | Env |
|---|---|---|---|
| 1C, 1D | RI60 poke rate + shocks tolerated (adolescent, adult) | `src/analysisapp/RI60_behavior_analysis.py` | `jimmy` |
| 1G, 1H, 1I | last-poke time, return latency, % compulsive returns | `src/analysisapp/RI60_behavior_analysis.py` | `jimmy` |
| 1E, 1F | lognormal IPI mixture weights (c0–c4) | `src/analysisapp/RI60_behavior_modeling_v2.py` | `jimmy` |
| 1F stat | Bayesian Dirichlet on the weights (p-analog = 0.022) | `src/analysisapp/bayesian_dirichlet_for_muPI.R` | R (`renv_stats`) |

## Reproduce

1. **(only if rebuilding from raw)** `conda activate jimmy` → `python src/analysisapp/readMEDPC_ASAP_Jan23.py`
   parses MED-PC files (`data_root/ALL_RI60_PHOTO/behavior/rawMedData`) → `behaviorData.pickle`.
2. `python src/analysisapp/RI60_behavior_analysis.py` → `forPrismRI60.csv`, `forPrismShock.csv`,
   `RI30_RI60_data.csv`, `Shock_Data.csv`, `shock_lastPoke.csv`, `shock_microstructure.csv` (the 1C/D/G/H/I sources).
3. `python src/analysisapp/RI60_behavior_modeling_v2.py` → `component_params.csv`, `session_weights.csv`,
   `prism_weights_by_group.csv` (the 1E/1F weights).
4. `conda activate` your R env, then `Rscript src/analysisapp/bayesian_dirichlet_for_muPI.R`
   (reads `session_weights.csv`) → `bayes_group_effects.csv` = the 1F group contrast / p-analog.

Final panels are drawn in **GraphPad Prism** from the `forPrism*` / `prism_*` CSVs → `stats/aCUS RI60 Behavior Collective (Not KiR though).prism`.

## Notes
- The paper's conventional stats (Welch t, Mann-Whitney) are computed in Prism; the CSVs above are their inputs.
- `n` varies by panel due to Prism ROUT (Q=0.1%) outlier removal, applied **in Prism**.
