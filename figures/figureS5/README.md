# Figure S5 (rel. Fig 3) — model fitting strategy + scRNA-seq KCNQ

Panels **S5C** (model fitness) and **S5D** (KCNQ by projection). S5A, S5B are strategy schematics (no code).

| Panel(s) | Content | Code | Env |
|---|---|---|---|
| S5C | per-cluster fitness (unstressed / stress-frozen / stress-all-free) | `src/model/analyze_and_plot.py` | `ephys_model` |
| S5D | Kcnq2/3/5 by putative CeA projection group | `src/rnaseq/kcnq.py` | `ephys_model` |

## Reproduce — model fitness (S5C)

Same run as Fig 3K–P — `analyze_and_plot.py` emits the fitness bars (`candidate_fitness_*.csv`) alongside the model-vs-data panels. See `figures/figure3/README.md`.

## Reproduce — scRNA-seq (S5D)

1. `conda activate ephys_model` → `python src/rnaseq/kcnq.py`. On first run it downloads **GEO GSE213828**
   into `src/rnaseq/data/GSE213828/` and builds the AnnData cache; subsequent runs reuse it. Select steps via
   the `RUN = [...]` list at the top of the file (`stats`, `plots`, `glm`, `glm_zinb`).
2. Outputs to `data_root/EPHYS_MODELING/RNA-seq`: `Kcnq{2,3,5}_for_prism.csv`, `kcnq_stats.txt`
   (Kruskal-Wallis + Dunn BH), `kcnq_by_projection_group.png`.

## Notes
- Projection groups by marker: SN = Drd1⁺/Sema3c⁻, BNST = Dlk1⁺, PAG = Crh⁺ (priority PAG > BNST > SN).
- GSE213828 is public and **not** re-archived by us — `kcnq.py` fetches it.
