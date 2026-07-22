# Figure S7 (rel. Fig 5) — FLMM group-interaction models

Panels **S7A–S7F**: the group-interaction β(t) for each predictor (experience, reward rate, poke rate) on
rewarded/unrewarded nosepokes. These are the models that generate the purple "unstressed ≠ stressed" bars in Fig 5.

| Panel(s) | Content | Code | Env |
|---|---|---|---|
| S7A–F | group × variable interaction β(t) | `src/analysisapp/FFM_modeling_session_vars_zscores.R` → `fastFMM_comps.py` | R (`renv_fastfmm`) + `jimmy` |

## Reproduce

Same FLMM run as Figure 5 — S7 **is** the group-interaction model output (Fig 5 shows the group-specific
fits; S7 shows the interaction). See `figures/figure5/README.md` for the full sequence:

1. `python src/analysisapp/patch_reward_rate_into_DF.py` → `photoDF_R_with_rates.feather`.
2. `Rscript "src/analysisapp/FFM_modeling_session_vars_zscores.R"` (the **group-interaction** model) → `FMM_models/*.csv`.
3. `python src/analysisapp/fastFMM_comps.py` and the `plot_FMMs` path in `RI60 analysis photo.py` render the interaction β(t) panels.

## Notes
- Significant divergence of the interaction β(t) from 0 = the groups differ in encoding → the purple bars in Fig 5.
