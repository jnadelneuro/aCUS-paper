# Figure 3 — Hyperexcitability across three CeA→SNL subtypes + biophysical model

Panels **3A–3J** (clustering) and **3K–3P** (NEURON model).

| Panel(s) | Content | Code | Env |
|---|---|---|---|
| 3A–D | z-scored feature heatmap, PCA, cluster counts, per-cluster F-I | `src/ephys/ephysAnalysisCluster.py` + `src/ephys/ClusterDeeZ.R` | `ephysAnalysis` + R (`renv_stats`) |
| 3E–J | per-cluster F-I / rheobase / adaptation / Rin × stress | `ephysAnalysisCluster.py` (cluster stats) | `ephysAnalysis` |
| 3K–M | model vs observed (spike count, latency, mean ISI) | `src/model/analyze_and_plot.py` | `ephys_model` |
| 3N–P | single-conductance fitness per cluster (Im best) | `src/model/analyze_and_plot.py` | `ephys_model` |

## Reproduce — clustering (3A–J)

1. `conda activate ephysAnalysis` → `python src/ephys/ephysAnalysisCluster.py` builds the SNL-only
   feature matrix `clusterPreppedBoi.csv`.
2. Switch to R (`renv_stats`) → `Rscript src/ephys/ClusterDeeZ.R`: stress-LMM residuals → PCA (4 PCs) →
   **k-means k=3, `set.seed(15432)`, nstart=25** → `PC1_PC2_clusters.csv`, `clustered_cells.csv`.
3. Back in `ephysAnalysis`, `ephysAnalysisCluster.py` merges the labels and runs the cluster×stress /
   per-cluster mixed models (3H–J) → `cluster_models/*.txt`, `_clusters_*.csv`.

> ⚠️ **The final Fig-3 cluster labels were manually curated** after `ClusterDeeZ.R`
> (`clustered_cells.csv` is the ground truth; `clustered_cells_backup_pre_manual.csv` is the pre-edit).
> Ship and use the curated `clustered_cells.csv` — the raw script output alone does not reproduce the exact labels.

## Reproduce — model (3K–P)

1. `conda activate ephys_model`, build mechanisms once: `cd src/model && nrnivmodl mod`.
2. Fit (cluster/HPC): edit the SBATCH block in `src/model/run_pipeline.sh` for your cluster, then
   `bash run_pipeline.sh` → optimizer result JSONs in `data_root/EPHYS_MODELING/.../results`.
3. `python src/model/analyze_and_plot.py` renders 3K–M (model vs data) and 3N–P (per-conductance fitness).

## Notes
- The model is the **pre-phase-plot** LiEtAlModels fit (weighted SSE normalized by feature SD; no dV/dt term) — matches the Methods.
