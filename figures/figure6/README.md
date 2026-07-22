# Figure 6 — Normalizing CeA→SNL excitability or TS dopamine prevents the stress-induced increase

Panels **6C, 6D** (Kir2.1 cohort) and **6G, 6H** (NpHR cohort). Schematics/histology 6A, 6B, 6E, 6F (no analysis code).

| Panel(s) | Content | Code | Env | Data |
|---|---|---|---|---|
| 6C, 6D | RI60 poke rate + shocks × stress × virus (Kir2.1 vs eYFP) | `src/analysisapp/RI60_behavior_analysis.py` | `jimmy` | Kir cohort |
| 6G, 6H | RI60 poke rate + shocks, NpHR vs YFP | `src/analysisapp/RI60_behavior_analysis.py` | `jimmy` | NpHR cohort |

## Reproduce

Same behavior pipeline as Figure 1, run against the **manipulation cohorts** (separate data trees from the
main RI60 dataset — `KIR ALL` and `TS NpHR ALL` under `data_root`).

1. Point `config.yaml → data_root` (or the relevant sub-path) at the Kir / NpHR cohort's `output_datafiles`.
2. `conda activate jimmy` → `python src/analysisapp/RI60_behavior_analysis.py` →
   `forPrismShock.csv` etc. for that cohort.

Final panels are drawn in **Prism** (`stats/KiR Behavior.prism`, `stats/Opto Behavior.prism`);
the 2-way ANOVA + Tukey stats in the legend are computed there.

## Notes
- 6C/6D and 6G/6H come from **different cohorts** than Fig 1/2/4 — make sure `data_root` points at the
  Kir (`KIR ALL`) or NpHR (`TS NpHR ALL`) tree when reproducing each.
- Histology panels (6B, 6F probe maps) are Keyence images mapped to the Allen atlas — not code.
