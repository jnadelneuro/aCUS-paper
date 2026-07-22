# Figure 2 — Adolescent stress alters CeA intrinsic properties, projection-specifically

Panels **2D–2N**. Schematics/recording-location maps 2A–2C are drawn from the injection atlas maps (no analysis code).

| Panel(s) | Content | Code | Env |
|---|---|---|---|
| 2D–F | F-I curves (CeA→BNST / PAG / SNL) | `src/ephys/ephysAnalysisAnalyze.py` → `getFiringRateData` | `ephysAnalysis` |
| 2G–I | voltage sag | `ephysAnalysisAnalyze.py` → `getSagData` | `ephysAnalysis` |
| 2J–M | RMP / capacitance / input resistance / rheobase | `getRMPData` / `getCapacitance` / `getInputResistance` / `getRheobase` | `ephysAnalysis` |
| 2N | P40 SNL F-I (immediately post-stress) | `getFiringRateData` (P40 cohort) | `ephysAnalysis` |

## Reproduce

1. **(only if rebuilding the collective from raw)** `conda activate ephysAnalysis` →
   `python src/ephys/ephysAnalysisCreate.py`. Requires **MATLAB + WaveSurfer 0.945** (set `config.yaml → tools.matlab_wavesurfer`);
   reads WaveSurfer `.h5` → `aCUS_intrinsicData_collective.pkl` in `data_root/ALL_INTRINSIC_EPHYS/analysis`.
2. `python src/ephys/ephysAnalysisAnalyze.py` — **enable the `get*` calls for the panels you want** at the
   bottom of the file (by default only `getFiringRateData` + `getPhasePlotV3` are uncommented). Each writes its
   CSV (`baseline Firing Rate.csv`, `sag.csv`, `input_resistance.csv`, `baseline rheobase.csv`, …) plus `prism/*.xlsx`.

Final panels are drawn in **Prism** (`stats/CeA Ephys Collective.prism`, `stats/SNL Ephys Collective.prism`).

## Notes
- Stats are hierarchical **linear mixed-effects models** (statsmodels, Šidák) computed inside `ephysAnalysisAnalyze.py`; the numbers in the legend come from its `models/*.txt` outputs.
- Rheobase is the 5 pA / 50 ms protocol (see Methods); the F-I is 20 pA / 500 ms.
