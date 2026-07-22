# Figure S1 (rel. Fig 1) — physiology, anxiety, and active avoidance

| Panel(s) | Content | Code | Env |
|---|---|---|---|
| S1A–C | weight, blood glucose, corticosterone | **none** — GraphPad Prism / raw exports | — |
| S1E–F, S1H–I | open-field + EPM (distance, center entries, open-arm time) | **none** — ANY-maze exports → Prism | — |
| S1K, S1L | active avoidance: % avoid across days, avoid/escape latencies | `src/avoidance/extractMed.py` | Python (`jimmy`) |

## Reproduce — physiology / anxiety (S1A–I): no code

These panels have **no analysis pipeline** — they were computed in GraphPad Prism from ANY-maze tracking
exports and assay readouts. Source data + Prism files are archived:
`stats/Physiological Data Collective.prism`, `stats/Sucrose Preference.prism`, and the open-field / EPM
ANY-maze CSVs + weight/sucrose spreadsheets under `data/physiology/`.

## Reproduce — active avoidance (S1K, S1L)

1. `conda activate jimmy` → `python src/avoidance/extractMed.py` parses the two-chamber shuttle-box MED-PC
   files (`avoidance_root/behavior/rawMedData`) → `summaryData.csv` (per-session percent avoid, avoid latency,
   escape latency, chronological day). Panels are drawn in Prism from `summaryData.csv`.

## Notes
- ⚠️ `extractMed.py` **mutates its raw directory** (renames files, writes per-session `.csv` caches into `rawMedData`).
  Run it against a **copy** of the raw data, not the archive master.
- The stress-vs-control grouping for S1K/L is applied downstream (Prism / `mouse_info.json`), not inside `extractMed.py`.
