#!/usr/bin/env python3
"""
Build candidates.json for all clusters.

Without results: generates singles only (all non-frozen channels per cluster).
With results: reads singles results, picks top 5 per cluster, adds all pairs.

Free (non-frozen) channels per cluster:
  RS  (cl1): gbar_na, gbar_kdr, gbar_im, gbar_cal, gbar_sahp, gbar_sk, gbar_bk (7)
  LTB (cl2): e_pas, g_pas, gbar_na, gbar_kdr, gbar_im, gbar_cal, gbar_ih, gbar_sahp, gbar_cat, gbar_sk, gbar_bk (11)
  LF  (cl3): gbar_na, gbar_kdr, gbar_im, gbar_cal, gbar_sahp, gbar_ia, gbar_sk, gbar_bk (8)

Usage:
  python3 build_candidates.py            # before stage 1: singles only
  python3 build_candidates.py            # after stage 1: adds top-5 pairs
  python3 build_candidates.py --top 3    # use top 3 instead of 5
"""

import json, glob, os, sys
import numpy as np
from collections import defaultdict
from itertools import combinations

os.chdir(os.path.dirname(os.path.abspath(__file__)))
logdir = 'logs/robust/results'

TOP_N = 5
for i, arg in enumerate(sys.argv[1:], 1):
    if arg == '--top' and i < len(sys.argv) - 1:
        TOP_N = int(sys.argv[i + 1])

# ============================================================
# CHANNEL DEFINITIONS
# ============================================================
FREE_CHANNELS = {
    1: ['gbar_na', 'gbar_kdr', 'gbar_im', 'gbar_cal', 'gbar_sahp', 'gbar_sk', 'gbar_bk'],
    2: ['e_pas', 'g_pas', 'gbar_na', 'gbar_kdr', 'gbar_im', 'gbar_cal', 'gbar_ih', 'gbar_sahp', 'gbar_cat', 'gbar_sk', 'gbar_bk'],
    3: ['gbar_na', 'gbar_kdr', 'gbar_im', 'gbar_cal', 'gbar_sahp', 'gbar_ia', 'gbar_sk', 'gbar_bk'],
}

SHORT = {
    'e_pas': 'epas', 'g_pas': 'gpas', 'gbar_na': 'Na', 'gbar_kdr': 'KDR',
    'gbar_im': 'IM', 'gbar_cal': 'CaL', 'gbar_ih': 'IH',
    'gbar_sahp': 'sAHP', 'gbar_cat': 'CaT', 'gbar_ia': 'IA',
    'gbar_sk': 'SK', 'gbar_bk': 'BK',
}

CLNAMES = {1: 'RS', 2: 'LTB', 3: 'LF'}

# ============================================================
# BUILD SINGLES (always included)
# ============================================================
all_candidates = []

for cl in [1, 2, 3]:
    cn = CLNAMES[cl]
    for ch in FREE_CHANNELS[cl]:
        all_candidates.append({
            "cluster": cl,
            "channels": [ch],
            "label": f"{cn}_{SHORT[ch]}"
        })

n_singles = len(all_candidates)
print(f"Singles: {n_singles} total")
for cl in [1, 2, 3]:
    n = len(FREE_CHANNELS[cl])
    print(f"  {CLNAMES[cl]}: {n} singles ({', '.join(SHORT[c] for c in FREE_CHANNELS[cl])})")

# ============================================================
# CHECK FOR SINGLES RESULTS → BUILD PAIRS
# ============================================================
n_pairs_total = 0

for cl in [1, 2, 3]:
    cn = CLNAMES[cl]
    singles_fits = defaultdict(list)
    for f in sorted(glob.glob(f'{logdir}/cand*_cl{cl}_ns*_ss*.json')):
        with open(f) as fh:
            d = json.load(fh)
        if len(d.get('channels', [])) == 1:
            singles_fits[d['label']].append(d['fitness'])

    if not singles_fits:
        print(f"\n  {cn}: no singles results yet — skipping pairs")
        continue

    ranked = sorted(singles_fits.items(), key=lambda x: np.mean(x[1]))
    print(f"\n  {cn} singles ranking:")
    for rank, (label, fits) in enumerate(ranked, 1):
        marker = " <--" if rank <= TOP_N else ""
        print(f"    {rank}. {label}: {np.mean(fits):.1f} +/- {np.std(fits):.1f} (n={len(fits)}){marker}")

    # Map labels to channel names
    label_to_ch = {}
    for ch in FREE_CHANNELS[cl]:
        label_to_ch[f"{cn}_{SHORT[ch]}"] = ch

    top_channels = []
    for label, _ in ranked[:TOP_N]:
        if label in label_to_ch:
            top_channels.append(label_to_ch[label])

    print(f"    Top {len(top_channels)} for pairs: {[SHORT[c] for c in top_channels]}")

    for a, b in combinations(top_channels, 2):
        all_candidates.append({
            "cluster": cl,
            "channels": [a, b],
            "label": f"{cn}_{SHORT[a]}+{SHORT[b]}"
        })
        n_pairs_total += 1

# ============================================================
# SAVE
# ============================================================
with open('candidates.json', 'w') as f:
    json.dump(all_candidates, f, indent=2)

print(f"\n{'='*60}")
print(f"candidates.json: {len(all_candidates)} total ({n_singles} singles + {n_pairs_total} pairs)")
for cl in [1, 2, 3]:
    cl_cands = [c for c in all_candidates if c['cluster'] == cl]
    cl_s = sum(1 for c in cl_cands if len(c['channels']) == 1)
    cl_p = sum(1 for c in cl_cands if len(c['channels']) == 2)
    print(f"  {CLNAMES[cl]}: {cl_s} singles + {cl_p} pairs = {len(cl_cands)}")
print(f"\nPipeline candidate jobs: {len(all_candidates)} x 10 seeds = {len(all_candidates) * 10}")
print(f"{'='*60}")
