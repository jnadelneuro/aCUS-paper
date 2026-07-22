# Figure S4 (rel. Fig 2) — spike-train properties (BNST / PAG / SNL)

Panels **S4A–S4I**: adaptation index, latency to spike, and mean ISI for each of the three projections.

| Panel(s) | Content | Code | Env |
|---|---|---|---|
| S4A–C | BNST adaptation / latency / mean ISI | `src/ephys/ephysAnalysisAnalyze.py` → `getTrainProps` | `ephysAnalysis` |
| S4D–F | PAG adaptation / latency / mean ISI | same | `ephysAnalysis` |
| S4G–I | SNL adaptation / latency / mean ISI | same | `ephysAnalysis` |

## Reproduce

1. `conda activate ephysAnalysis` → `python src/ephys/ephysAnalysisAnalyze.py` with the
   `getTrainProps` (and `getSpikeProps`) call enabled at the bottom →
   `spike train properties.csv`, `single spike properties.csv` + `prism/*.xlsx`.

Panels are drawn in **Prism** from the exported workbooks; stats are the hierarchical LMMs computed in the script.

## Notes
- Same collective pickle and pipeline as Figure 2 — just the spike-train feature exports.
