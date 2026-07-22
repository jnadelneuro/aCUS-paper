# Figure S6 (rel. Fig 4) — photometry maps + SNL→TS dopamine-neuron ephys

Panels **S6B, S6H** (photometry) and **S6D–F** (SNL→TS ephys). S6A, S6C, S6G are probe-location maps / schematics.

| Panel(s) | Content | Code | Env |
|---|---|---|---|
| S6B, S6H | TS / DMS dopamine response to rewarded **port entry** | `src/analysisapp/RI60 analysis photo.py` | `jimmy` |
| S6D–F | SNL→TS DA-neuron F-I / RMP / input resistance | `src/ephys/ephysAnalysisAnalyze.py` | `ephysAnalysis` |

## Reproduce — photometry PE responses (S6B, S6H)

Same photometry pipeline as Figure 4, aligned to the **rewarded port-entry** event (rather than nosepoke):
`conda activate jimmy` → `python "src/analysisapp/RI60 analysis photo.py"` (port-entry PSTH) →
`RePE_photoTraces.csv`, `UnPE_photoTraces.csv`.

## Reproduce — SNL→TS ephys (S6D–F)

`conda activate ephysAnalysis` → `python src/ephys/ephysAnalysisAnalyze.py` on the **TS-retrobead SNL/SNc
dopamine-neuron** dataset (`getFiringRateData`, `getRMPData`, `getInputResistance`) →
`baseline Firing Rate.csv`, `baseline RMP.csv`, `input_resistance.csv` for that cohort.

## Notes
- S6D–F use a distinct retrobead cohort (TS-injected) — point `data_root` at that intrinsic dataset when reproducing.
