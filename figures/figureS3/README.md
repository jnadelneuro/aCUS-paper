# Figure S3 (rel. Fig 1) — lognormal IPI mixture-model detail

Panels **S3A–S3F**: model fit + ΔBIC across component count; pokes/bout; inter-bout interval;
% disengaged (c4) across the session; c4 slope (satiation).

| Panel(s) | Content | Code | Env |
|---|---|---|---|
| S3A–B | model fit / ΔBIC across k (component selection) | `src/analysisapp/RI60_logmixmodeling_K_select.py` | `jimmy` |
| S3C–F | pokes/bout, inter-bout interval, % disengaged, c4 slope | `src/analysisapp/RI60_behavior_modeling_v2.py` | `jimmy` |

## Reproduce

1. `conda activate jimmy` → `python src/analysisapp/RI60_logmixmodeling_K_select.py`
   (the 2–8 component sweep behind choosing k=5) → `K_sweep_summary.csv`, `K_sweep_fits.pkl`.
2. `python src/analysisapp/RI60_behavior_modeling_v2.py` → `interaction_disengagement.csv`,
   `prism_disengagement_by_day.csv`, `prism_time_share_by_group.csv` (the S3C–F sources).

Panels are drawn in **Prism** from the `prism_*` CSVs.

## Notes
- Same mixture run as Fig 1E/F; S3 is the model-diagnostics + bout-structure view.
