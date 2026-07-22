# Figure 5 — FLMM reveals a computational switch in trial-by-trial TS dopamine encoding

Panels **5E–5L** (FLMM β(t)) and the example traces **5D, 5G, 5J**. 5A schematic; **5B, 5C reuse Fig 4C, 4D**.

| Panel(s) | Content | Code | Env |
|---|---|---|---|
| 5D, 5G, 5J | example total-poke / reward-rate / poke-rate traces | `src/analysisapp/RI60_representative_sessions.py` | `jimmy` |
| 5E–L | FLMM β(t): experience / reward-rate / poke-rate → rewarded/unrewarded | R fit + `fastFMM_comps.py` + `RI60 analysis photo.py` | R (`renv_fastfmm`) + `jimmy` |

## Reproduce

1. `conda activate jimmy` → `python src/analysisapp/patch_reward_rate_into_DF.py` computes the leaky-integrator
   predictors (reward rate, poke rate; τ = 60) → `photoDF_R_with_rates.feather`.
2. Switch to R (`renv_fastfmm`) → run the FLMM fits (each fits `photoTrace ~ … + (1|mouse)` with `fastFMM::fui`,
   predictors z-scored):
   - `Rscript "src/analysisapp/FFM_modeling_session_vars_zscores.R"` (group-interaction model → Fig 5 purple bars / **Fig S7**)
   - `..._Re.R`, `..._Un.R`, `..._stress_RevsUn.R` (group-specific fits)
   → `FMM_models/*.csv` (`s`, `beta.hat`, pointwise CIs).
3. Back in `jimmy`: `python src/analysisapp/fastFMM_comps.py` → stress-vs-unstressed Δβ(t) diffs;
   `python "src/analysisapp/RI60 analysis photo.py"` with `doFastFFMPlots=True` renders the paired β(t) panels.
4. Example traces: `python src/analysisapp/RI60_representative_sessions.py` → `fig_out/` (5D/5G/5J).

## Notes
- 5B/5C are the same traces as 4C/4D, reused for visual comparison — no separate computation.
- The **group-interaction** fit is what produces the purple "unstressed ≠ stressed" bars and all of **Fig S7**.
