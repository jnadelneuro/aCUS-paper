# Figure 4 — Potentiated reward-related dopamine in TS but not DMS

Panels **4B–4G**. Schematic 4A (no code).

| Panel(s) | Content | Code | Env |
|---|---|---|---|
| 4B, 4E | TS / DMS transient frequency + amplitude | `src/analysisapp/RI60 analysis photo.py` | `jimmy` |
| 4C, 4D | TS rewarded / unrewarded-nosepoke PSTH (+ sig bars) | `RI60 analysis photo.py` → `RI60_Photo_bootstrapCIs.py` | `jimmy` |
| 4F, 4G | DMS rewarded / unrewarded-nosepoke PSTH | same | `jimmy` |

## Reproduce

1. **(upstream, only if rebuilding)** GuPPy (external) preprocesses raw TDT photometry → `z_score_*.h5`;
   `python src/analysisapp/RI60_photo_baseline_adder.py` writes the baseline-corrected per-session `.h5`;
   the aCUS `photoDataFrame.feather` is built by `setupRI60.py → RI60_analaysis_fx_PHOTO_JIMMYv2.createPhotoDF`
   (folder pickers select the session dirs).
2. `conda activate jimmy` → `python "src/analysisapp/RI60 analysis photo.py"` with `doPSTHs=True` →
   `ReNP_photoTraces.csv`, `UnNP_photoTraces.csv`, `*_photoTraces.csv` (the PSTH sources; rows=mice, cols=time −5…10).
3. `python src/analysisapp/RI60_Photo_bootstrapCIs.py` → `bootstrapping_*.csv` (25,000-resample hierarchical
   bootstrap; the purple significance bars on 4C/4D).

Amplitude/frequency panels (4B, 4E) are drawn in Prism (`stats/DMS vs TS Traces ... .prism`); the PSTH traces come from the `*_photoTraces.csv`.

## Notes
- **PSTH source = `*_photoTraces.csv`**, not the FLMM modeling DFs (`df_TS_*` / `df_DMS_*`).
- Transient detection (MAD) is done inside GuPPy; the repo reads `transientsOccurrences_*.csv`.
