# Adolescent Stress Sensitizes Amygdalo-Nigro-Striatal Circuitry to Drive Punishment-Resistant Reward-Seeking

Analysis code and derived data for **Nadel et al.**. This repository reproduces the figures of the paper from processed data. Raw data are deposited externally (see **Data availability**).

> **Reproduction verified.** Figures 1, 2, and S5D regenerate **byte-for-byte** from this repository across all three conda environments (`jimmy` / `ephysAnalysis` / `ephys_model`) and three modalities (behavior, intrinsic ephys, scRNA-seq) — reported p-values match the manuscript to the digit (F-I SNL stress 0.0037 / interaction 0.0165; KCNQ Kcnq2 0.0118 / Kcnq3 0.0024).

## How this repo is organized

- **`figures/`** — one folder per figure. Each has a `README.md` recipe: which panels, which command(s) to run, which inputs/outputs. **Start here** to reproduce a specific figure.
- **`src/`** — the analysis code, grouped by its source pipeline (kept together where the original code cross-imports, so it runs unmodified):
  - `src/analysisapp/` — operant behavior, lognormal-IPI mixture model, fiber-photometry PSTH/bootstrap, and FLMM (Python + the R fit scripts). *(One flat package — the scripts import each other.)*
  - `src/ephys/` — intrinsic electrophysiology (ingest, ipfx feature extraction, LMM) + `ClusterDeeZ.R` (CeA→SNL clustering).
  - `src/model/` — NEURON biophysical model (CMA-ES) of the three CeA→SNL subtypes.
  - `src/rnaseq/` — `kcnq.py` scRNA-seq reanalysis (GEO GSE213828).
  - `src/avoidance/` — active-avoidance behavior (Fig S1K-L).
- **`config/config.yaml`** — the single place data paths live (copy from `config.example.yaml`).
- **`env/`** — environment files (see **Environments**).
- **`stats/`** — GraphPad Prism `.pzfx` files: the record for Table S2 and all conventional statistics (t-tests, ANOVAs, correlations). Conventional stats were run in Prism, not in code.
- **`figures/*/data/`** — the small processed tables each panel is built from, tracked in git. Large intermediates (`behaviorData.pickle`, the collective `.pkl`, `photoDataFrame.feather`) are fetched from Zenodo — see **Data availability**.

## Figure → pipeline map

| Figure | Panels | Pipeline | Entry point(s) |
|---|---|---|---|
| 1 | 1C,D,G,H,I | behavior | `src/analysisapp/RI60_behavior_analysis.py` |
| 1 | 1E,F | mixture | `RI60_behavior_modeling_v2.py`, `bayesian_dirichlet_for_muPI.R` |
| 2 | 2D–N | ephys | `src/ephys/ephysAnalysisAnalyze.py` |
| 3 | 3A–J | ephys clustering | `src/ephys/ClusterDeeZ.R`, `ephysAnalysisCluster.py` |
| 3 | 3K–P | model | `src/model/pre-phaseplot/analyze_and_plot.py` |
| 4 | 4B–G | photometry | `src/analysisapp/RI60 analysis photo.py` (PSTH + bootstrap) |
| 5 | 5E–L | FLMM | `FFM_modeling_session_vars_zscores.R`, `fastFMM_comps.py` |
| 6 | 6C,D,G,H | behavior | `src/analysisapp/RI60_behavior_analysis.py` (Kir/NpHR cohorts) |
| S1 | A–I | physiology | Prism / ANY-maze only (no code) — see `figures/figureS1/` |
| S1 | K,L | avoidance | `src/avoidance/extractMed.py` |
| S2 | A–Q | behavior | `src/analysisapp/RI60_behavior_analysis.py` |
| S3 | A–F | mixture | `RI60_logmixmodeling_K_select.py`, `RI60_behavior_modeling_v2.py` |
| S4 | A–I | ephys | `src/ephys/ephysAnalysisAnalyze.py` |
| S5 | C | model | `src/model/pre-phaseplot/analyze_and_plot.py` |
| S5 | D | rnaseq | `src/rnaseq/kcnq.py` |
| S6 | B,H | photometry | `src/analysisapp/RI60 analysis photo.py` |
| S6 | D–F | ephys | `src/ephys/ephysAnalysisAnalyze.py` (SNL→TS) |
| S7 | A–F | FLMM | `FFM_modeling_session_vars_zscores.R` (interaction models) |

## Environments

| Env file | Covers | Create |
|---|---|---|
| `env/jimmy.yml` | **analysisApp** — behavior · mixture · photometry · FLMM (Python) | `conda env create -f env/jimmy.yml` |
| `env/ephysAnalysis.yml` | **EphysAnalysis** — intrinsic ephys + clustering (Python) | `conda env create -f env/ephysAnalysis.yml` |
| `env/ephys_model.yml` | **modeling** — NEURON model + scRNA-seq (`kcnq.py`) | `conda env create -f env/ephys_model.yml` |
| `env/renv_fastfmm.R` | FLMM fits (`fastFMM`) — R | `Rscript env/renv_fastfmm.R` |
| `env/renv_stats.R` | clustering k-means + Bayesian Dirichlet (`brms`) — R | `Rscript env/renv_stats.R` |

Environment files are exported from the working conda environments (`conda env export --no-builds`). Intrinsic-ephys ingest additionally needs MATLAB + WaveSurfer 0.945 (see `config.yaml → tools`).

External tools (documented, not vendored): **MATLAB + WaveSurfer 0.945** (ephys `.h5` ingest), **GuPPy** (photometry preprocessing, upstream), **NEURON** (compile `src/model/mod`).

## Configure paths

All data paths resolve from `config/config.yaml`:

```
cp config/config.example.yaml config/config.yaml   # then edit `data_root`
```

## Data availability

- **Code + all derived/processed data** — this repository, with the large intermediates
  (`behaviorData.pickle`, the collective ephys `.pkl`, `photoDataFrame.feather`, figure-source CSVs)
  deposited at **Zenodo** (10.5281/zenodo.21500217). Figures reproduce from the derived data alone — no raw needed.
- **Raw data** (photometry tanks, slice-ephys recordings, MED-PC behavior, histology) — deposit TBD
  (NWB → DANDI coming soon).
- **scRNA-seq** — public at **GEO GSE213828** (fetched by `src/rnaseq/kcnq.py`).

## Citing

See `CITATION.cff`.
