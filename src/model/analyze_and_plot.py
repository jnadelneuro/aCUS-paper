#!/usr/bin/env python3
"""
Full results pipeline: analysis + plots + simulations + Prism-ready CSV export.
Reads logs/robust/results/. Requires NEURON.

Usage: python3 analyze_and_plot.py

Outputs:
  plots/            — 8 static plots + trace PNGs
  csv_data/         — Prism-formatted CSVs:
    naive_params.csv                   — all naive params, all seeds
    candidate_fitness_{CL}.csv         — column per candidate, rows = seeds
    fi_curves.csv                      — XY: current vs spike count (model + target, naive + stress)
    first_spike_latency.csv            — XY: current vs latency (model + target)
    adaptation_index.csv               — XY: current vs adaptation (ipfx norm_diff, model + target)
    mean_isi.csv                       — XY: current vs mean ISI (model + target)
    isi_cv.csv                         — XY: current vs ISI CV (model + target)
    traces_{CL}_{naive|stress}.csv     — XY: time vs voltage at each amplitude
    sag_trace_{CL}_{naive|stress}.csv  — XY: time vs voltage (-150 pA step)
    grouped_RMP.csv                    — rows=clusters, cols=naive/stress
    grouped_Rin.csv
    grouped_sag_ratio.csv
    grouped_threshold.csv
    grouped_peak.csv
    grouped_trough.csv
    grouped_half_width.csv
"""

import json, glob, os, sys
import numpy as np
from collections import defaultdict, Counter
from _config import MODEL_DIR

basedir = MODEL_DIR
os.chdir(basedir)
logdir = os.path.join(basedir, 'logs', 'robust', 'results')
cnames = {1: 'RS', 2: 'LTB', 3: 'LF'}
colors_cl = {1: '#2196F3', 2: '#FF9800', 3: '#4CAF50'}

# ============================================================
# 1. COLLECT ALL RESULTS INTO MEMORY
# ============================================================
naive_all = []  # list of full dicts
naive_fits = defaultdict(list)
naive_best = {}
for f in sorted(glob.glob(f'{logdir}/naive_cl*_s*.json')):
    with open(f) as fh: d = json.load(fh)
    naive_all.append(d)
    cl = d['cluster']; naive_fits[cl].append(d['fitness'])
    if cl not in naive_best or d['fitness'] < naive_best[cl]['fitness']:
        naive_best[cl] = d

frozen_fits = defaultdict(list)
for f in sorted(glob.glob(f'{logdir}/frozen_cl*_ns*.json')):
    with open(f) as fh: d = json.load(fh)
    frozen_fits[d['cluster']].append(d['fitness'])

cand_all = []  # list of full dicts
cand_fits = defaultdict(list)
cand_info = {}
cand_best = {}
for f in sorted(glob.glob(f'{logdir}/cand*_cl*_ns*_ss*.json')):
    with open(f) as fh: d = json.load(fh)
    cand_all.append(d)
    label = d['label']
    cand_fits[label].append(d['fitness'])
    if label not in cand_info:
        cand_info[label] = {'channels': d['channels'], 'cluster': d['cluster']}
    if label not in cand_best or d['fitness'] < cand_best[label]['fitness']:
        cand_best[label] = d

allfree_all = []  # list of full dicts
allfree_fits = defaultdict(list)
allfree_best = {}
for f in sorted(glob.glob(f'{logdir}/allfree_cl*_ns*_ss*.json')):
    with open(f) as fh: d = json.load(fh)
    allfree_all.append(d)
    cl = d['cluster']; allfree_fits[cl].append(d['fitness'])
    if cl not in allfree_best or d['fitness'] < allfree_best[cl]['fitness']:
        allfree_best[cl] = d

print(f"Loaded: {len(naive_all)} naive, {sum(len(v) for v in frozen_fits.values())} frozen, "
      f"{len(cand_all)} candidates, {len(allfree_all)} allfree")

# ============================================================
# 2. CONSOLE ANALYSIS
# ============================================================
print("=" * 85)
print("ROBUST EVALUATION RESULTS")
print("=" * 85)

print(f"\n  Naive fit quality:")
for cl in [1, 2, 3]:
    fits = naive_fits.get(cl, [])
    if fits:
        print(f"    {cnames[cl]}: {np.mean(fits):.1f} +/- {np.std(fits):.1f}  "
              f"(n={len(fits)}, range {np.min(fits):.1f}-{np.max(fits):.1f})")

for cl in [1, 2, 3]:
    cn = cnames[cl]
    bl = frozen_fits.get(cl, []); af = allfree_fits.get(cl, [])
    cl_cands = {k: v for k, v in cand_fits.items() if cand_info.get(k, {}).get('cluster') == cl}
    singles = {k: v for k, v in cl_cands.items() if len(cand_info[k]['channels']) == 1}
    pairs = {k: v for k, v in cl_cands.items() if len(cand_info[k]['channels']) == 2}

    print(f"\n{'='*85}")
    print(f"  Cluster {cl} ({cn})")
    print(f"{'='*85}")
    if bl: print(f"  Frozen:    {np.mean(bl):>8.1f} +/- {np.std(bl):<6.1f}  (n={len(bl)})")
    else: print(f"  Frozen:    (no data)")

    if singles:
        print(f"\n  --- SINGLES ---")
        print(f"  {'Rank':<5} {'Label':<25} {'Mean':>8} {'Std':>8} {'n':>4}")
        print(f"  {'-'*50}")
        for rank, (label, fits) in enumerate(sorted(singles.items(), key=lambda x: np.mean(x[1])), 1):
            print(f"  {rank:<5} {label:<25} {np.mean(fits):>8.1f} {np.std(fits):>8.1f} {len(fits):>4}")
    if pairs:
        print(f"\n  --- PAIRS ---")
        print(f"  {'Rank':<5} {'Label':<25} {'Mean':>8} {'Std':>8} {'n':>4}")
        print(f"  {'-'*50}")
        for rank, (label, fits) in enumerate(sorted(pairs.items(), key=lambda x: np.mean(x[1])), 1):
            print(f"  {rank:<5} {label:<25} {np.mean(fits):>8.1f} {np.std(fits):>8.1f} {len(fits):>4}")
    if af: print(f"\n  All-free:  {np.mean(af):>8.1f} +/- {np.std(af):<6.1f}  (n={len(af)})")

    ranked = sorted(cl_cands.items(), key=lambda x: np.mean(x[1]))
    if bl and af and ranked:
        bl_mean = np.mean(bl); af_mean = np.mean(af); gap = bl_mean - af_mean
        print(f"\n  Gap analysis:")
        print(f"  {'Level':<25} {'Mean':>8} {'% gap':>8}")
        print(f"  {'-'*43}")
        print(f"  {'Frozen':<25} {bl_mean:>8.1f} {'0%':>8}")
        if singles:
            bs = min(singles.items(), key=lambda x: np.mean(x[1]))
            pct = (bl_mean - np.mean(bs[1])) / gap * 100 if gap > 0 else 0
            suf = ' <-- SUFFICIENT' if pct > 80 else ''
            print(f"  {bs[0]:<25} {np.mean(bs[1]):>8.1f} {pct:>7.0f}%{suf}")
        if pairs:
            bp = min(pairs.items(), key=lambda x: np.mean(x[1]))
            pct = (bl_mean - np.mean(bp[1])) / gap * 100 if gap > 0 else 0
            suf = ' <-- SUFFICIENT' if pct > 80 else ''
            print(f"  {bp[0]:<25} {np.mean(bp[1]):>8.1f} {pct:>7.0f}%{suf}")
        print(f"  {'All-free':<25} {af_mean:>8.1f} {'100%':>8}")

    if len(ranked) >= 2:
        l1, f1 = ranked[0]; l2, f2 = ranked[1]
        if len(f1) >= 5 and len(f2) >= 5:
            try:
                from scipy.stats import ttest_ind, mannwhitneyu
                _, p_t = ttest_ind(f1, f2); _, p_mw = mannwhitneyu(f1, f2, alternative='two-sided')
                print(f"\n  #1 vs #2: {l1} ({np.mean(f1):.1f}) vs {l2} ({np.mean(f2):.1f})")
                print(f"    t={p_t:.4f}  MW={p_mw:.4f}  {'SIGNIFICANT' if p_mw < 0.05 else 'not significant'}")
            except ImportError: pass

# ============================================================
# 3. STATIC PLOTS (no NEURON needed)
# ============================================================
has_data = (
    any(len(v) > 0 for v in naive_fits.values()) or
    any(len(v) > 0 for v in cand_fits.values()) or
    any(len(v) > 0 for v in frozen_fits.values()) or
    any(len(v) > 0 for v in allfree_fits.values())
)
if not has_data:
    print("\nNo input data found."); sys.exit(0)
    
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'figure.dpi': 150, 'savefig.dpi': 200})
outdir = os.path.join(basedir, 'plots'); os.makedirs(outdir, exist_ok=True)

# --- PLOT 1: Gap staircase ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, cl in enumerate([1, 2, 3]):
    ax = axes[idx]; cn = cnames[cl]
    bl = frozen_fits.get(cl, []); af = allfree_fits.get(cl, [])
    cl_c = {k: v for k, v in cand_fits.items() if cand_info.get(k, {}).get('cluster') == cl}
    s = {k: v for k, v in cl_c.items() if len(cand_info[k]['channels']) == 1}
    p = {k: v for k, v in cl_c.items() if len(cand_info[k]['channels']) == 2}
    levels = []
    if bl: levels.append(('Frozen', np.mean(bl), np.std(bl), '#E0E0E0'))
    if s:
        bs = min(s.items(), key=lambda x: np.mean(x[1]))
        levels.append((bs[0].split('_',1)[1], np.mean(bs[1]), np.std(bs[1]), '#FFB74D'))
    if p:
        bp = min(p.items(), key=lambda x: np.mean(x[1]))
        levels.append((bp[0].split('_',1)[1], np.mean(bp[1]), np.std(bp[1]), colors_cl[cl]))
    if af: levels.append(('All-free', np.mean(af), np.std(af), '#81C784'))
    if levels:
        ax.bar(range(len(levels)), [l[1] for l in levels], yerr=[l[2] for l in levels],
               capsize=5, color=[l[3] for l in levels], edgecolor='black', linewidth=0.8, alpha=0.85)
        ax.set_xticks(range(len(levels))); ax.set_xticklabels([l[0] for l in levels], rotation=30, ha='right', fontsize=9)
        if bl and af:
            tg = np.mean(bl) - np.mean(af)
            for i, (_, m, st, _) in enumerate(levels):
                if i > 0 and tg > 0:
                    ax.annotate(f'{(np.mean(bl)-m)/tg*100:.0f}%', xy=(i, m+st+20), ha='center', fontsize=9, fontweight='bold')
    ax.set_title(f'{cn} (Cluster {cl})', fontweight='bold')
    ax.set_ylabel('Fitness' if idx == 0 else ''); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle('Gap Analysis: Minimum Sufficient Channel Set', fontweight='bold', fontsize=14)
plt.tight_layout(); plt.savefig(f'{outdir}/robust_gap_analysis.png'); plt.close()
print(f"\nSaved: {outdir}/robust_gap_analysis.png")

# --- PLOT 2: Candidate ranking ---
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for idx, cl in enumerate([1, 2, 3]):
    ax = axes[idx]; cn = cnames[cl]
    cl_c = {k: v for k, v in cand_fits.items() if cand_info.get(k, {}).get('cluster') == cl}
    if not cl_c: ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes); continue
    ranked = sorted(cl_c.items(), key=lambda x: np.mean(x[1]))
    labels = [k.split('_',1)[1] for k,_ in ranked]
    means = [np.mean(v) for _,v in ranked]; stds = [np.std(v) for _,v in ranked]
    n_ch = [len(cand_info[k]['channels']) for k,_ in ranked]
    ax.barh(range(len(ranked)), means, xerr=stds, capsize=3,
            color=['#FFB74D' if n==1 else colors_cl[cl] for n in n_ch],
            edgecolor='black', linewidth=0.5, alpha=0.85, height=0.7)
    ax.set_yticks(range(len(ranked))); ax.set_yticklabels(labels, fontsize=9); ax.invert_yaxis()
    bl = frozen_fits.get(cl, []); af = allfree_fits.get(cl, [])
    if bl: ax.axvline(np.mean(bl), color='red', ls='--', lw=1.5, alpha=0.7, label='Frozen')
    if af: ax.axvline(np.mean(af), color='green', ls='--', lw=1.5, alpha=0.7, label='All-free')
    ax.set_xlabel('Fitness'); ax.set_title(cn, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    if idx == 0: ax.legend(fontsize=8, loc='lower right')
fig.suptitle('Candidate Rankings', fontweight='bold', fontsize=14)
plt.tight_layout(); plt.savefig(f'{outdir}/robust_candidate_ranking.png'); plt.close()
print(f"Saved: {outdir}/robust_candidate_ranking.png")

# --- PLOT 3: Naive variability ---
fig, ax = plt.subplots(figsize=(8, 5))
for cl in [1, 2, 3]:
    fits = naive_fits.get(cl, [])
    if fits:
        j = np.random.RandomState(42).uniform(-0.15, 0.15, len(fits))
        ax.scatter(np.array([cl]*len(fits))+j, fits, color=colors_cl[cl], alpha=0.7, s=60,
                   edgecolors='black', linewidth=0.5, label=cnames[cl], zorder=3)
        ax.errorbar(cl, np.mean(fits), yerr=np.std(fits), fmt='_', color='black',
                    markersize=20, markeredgewidth=2, capsize=8, capthick=2, zorder=4)
ax.set_xticks([1,2,3]); ax.set_xticklabels([f'{cnames[cl]}\n(n={len(naive_fits.get(cl,[]))})' for cl in [1,2,3]])
ax.set_ylabel('Naive Fitness'); ax.set_title('Naive Variability', fontweight='bold')
ax.legend(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.savefig(f'{outdir}/robust_naive_variability.png'); plt.close()
print(f"Saved: {outdir}/robust_naive_variability.png")

# --- PLOT 4: Channel frequency ---
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for idx, cl in enumerate([1, 2, 3]):
    ax = axes[idx]
    cl_c = {k: v for k, v in cand_fits.items() if cand_info.get(k, {}).get('cluster') == cl}
    if not cl_c: continue
    ranked = sorted(cl_c.items(), key=lambda x: np.mean(x[1]))
    ch_scores = Counter()
    for rank, (label, _) in enumerate(ranked):
        for ch in cand_info[label]['channels']: ch_scores[ch] += len(ranked) - rank
    channels = sorted(ch_scores.keys(), key=lambda x: ch_scores[x], reverse=True)
    display = [ch.replace('gbar_','').replace('g_pas','g_leak').replace('e_pas','e_leak') for ch in channels]
    ax.barh(range(len(channels)), [ch_scores[c] for c in channels], color=colors_cl[cl],
            edgecolor='black', linewidth=0.5, alpha=0.85, height=0.6)
    ax.set_yticks(range(len(channels))); ax.set_yticklabels(display, fontsize=10); ax.invert_yaxis()
    ax.set_xlabel('Rank-weighted score'); ax.set_title(cnames[cl], fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle('Channel Importance', fontweight='bold', fontsize=14)
plt.tight_layout(); plt.savefig(f'{outdir}/robust_channel_frequency.png'); plt.close()
print(f"Saved: {outdir}/robust_channel_frequency.png")

# --- PLOT 5: All-free heatmap ---
all_params = ['g_pas','gbar_na','gbar_kdr','gbar_im','gbar_cal','gbar_ih','gbar_sahp','gbar_cat','gbar_ia','gbar_sk','gbar_bk']
disp_params = ['g_leak','Na','KDR','IM','CaL','IH','sAHP','CaT','IA','SK','BK']
fc_mean = np.full((len(all_params), 3), np.nan); fmask = np.zeros((len(all_params), 3), dtype=bool)
for ci, cl in enumerate([1, 2, 3]):
    fd = {}
    for f in sorted(glob.glob(f'{logdir}/allfree_cl{cl}_ns*_ss*.json')):
        with open(f) as fh: d = json.load(fh)
        if 'naive_values' not in d or 'optimized_values' not in d: continue
        for param in d['naive_values']:
            if param == 'e_pas': continue
            nv, sv = d['naive_values'][param], d['optimized_values'][param]
            if nv > 0 and sv > 0: fd.setdefault(param, []).append(np.log2(sv/nv))
    for pi, param in enumerate(all_params):
        if param in fd: fc_mean[pi, ci] = np.mean(fd[param])
        else: fmask[pi, ci] = True
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(np.ma.masked_where(fmask, fc_mean), cmap=plt.cm.RdBu_r, vmin=-2.5, vmax=2.5, aspect='auto')
ax.set_xticks([0,1,2]); ax.set_xticklabels(['RS','LTB','LF'], fontsize=12, fontweight='bold')
ax.set_yticks(range(len(all_params))); ax.set_yticklabels(disp_params, fontsize=11)
for pi in range(len(all_params)):
    for ci in range(3):
        if fmask[pi,ci]: ax.text(ci, pi, 'frozen', ha='center', va='center', fontsize=8, color='gray', style='italic')
        elif not np.isnan(fc_mean[pi,ci]):
            v = fc_mean[pi,ci]
            ax.text(ci, pi, f'{2**v:.2f}x', ha='center', va='center', fontsize=9, fontweight='bold',
                    color='white' if abs(v)>1.2 else 'black')
cb = plt.colorbar(im, ax=ax, shrink=0.8); cb.set_label('log2(stress/naive)')
ax.set_title('All-Free Parameter Changes', fontweight='bold', fontsize=13)
for pi in range(len(all_params)+1): ax.axhline(pi-0.5, color='white', linewidth=0.5)
for ci in range(4): ax.axvline(ci-0.5, color='white', linewidth=0.5)
plt.tight_layout(); plt.savefig(f'{outdir}/robust_allfree_heatmap.png'); plt.close()
print(f"Saved: {outdir}/robust_allfree_heatmap.png")

# --- PLOT 6: Swarm ---
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for idx, cl in enumerate([1, 2, 3]):
    ax = axes[idx]
    cl_c = {k: v for k, v in cand_fits.items() if cand_info.get(k, {}).get('cluster') == cl}
    if not cl_c: continue
    ranked = sorted(cl_c.items(), key=lambda x: np.mean(x[1]))
    rng = np.random.RandomState(42)
    for i, (label, fits) in enumerate(ranked):
        j = rng.uniform(-0.2, 0.2, len(fits)); n_ch = len(cand_info[label]['channels'])
        ax.scatter(fits, [i]*len(fits)+j, color='#FFB74D' if n_ch==1 else colors_cl[cl],
                   alpha=0.5, s=30, edgecolors='black', linewidth=0.3)
        ax.plot(np.mean(fits), i, 'D', color='black', markersize=6, zorder=5)
    bl = frozen_fits.get(cl, []); af = allfree_fits.get(cl, [])
    if bl: ax.axvline(np.mean(bl), color='red', ls='--', lw=1.5, alpha=0.6, label='Frozen')
    if af: ax.axvline(np.mean(af), color='green', ls='--', lw=1.5, alpha=0.6, label='All-free')
    ax.set_yticks(range(len(ranked))); ax.set_yticklabels([k.split('_',1)[1] for k,_ in ranked], fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel('Fitness'); ax.set_title(cnames[cl], fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    if idx == 0: ax.legend(fontsize=8)
fig.suptitle('Individual Seed Results', fontweight='bold', fontsize=14)
plt.tight_layout(); plt.savefig(f'{outdir}/robust_swarm.png'); plt.close()
print(f"Saved: {outdir}/robust_swarm.png")

# --- PLOT 7: Best candidate params ---
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for idx, cl in enumerate([1, 2, 3]):
    ax = axes[idx]; cn = cnames[cl]
    cl_c = {k: v for k, v in cand_fits.items() if cand_info.get(k, {}).get('cluster') == cl}
    if not cl_c: ax.set_title(cn); continue
    best_label, best_fits = min(cl_c.items(), key=lambda x: np.mean(x[1]))
    fold_per_seed = {}
    for f in sorted(glob.glob(f'{logdir}/cand*_cl{cl}_ns*_ss*.json')):
        with open(f) as fh: d = json.load(fh)
        if d['label'] != best_label or 'naive_values' not in d or 'optimized_values' not in d: continue
        for param in d['naive_values']:
            nv, sv = d['naive_values'][param], d['optimized_values'][param]
            if nv > 0 and sv > 0: fold_per_seed.setdefault(param, []).append(np.log2(sv/nv))
    if fold_per_seed:
        params = sorted(fold_per_seed.keys())
        display = [p.replace('gbar_','').replace('g_pas','g_leak') for p in params]
        means = [np.mean(fold_per_seed[p]) for p in params]
        stds = [np.std(fold_per_seed[p]) for p in params]
        colors_bar = ['#EF5350' if m<-0.3 else '#66BB6A' if m>0.3 else '#BDBDBD' for m in means]
        ax.barh(np.arange(len(params)), means, xerr=stds, capsize=3, color=colors_bar,
                edgecolor='black', linewidth=0.5, alpha=0.85, height=0.6)
        ax.axvline(0, color='black', linewidth=1); ax.set_yticks(np.arange(len(params)))
        ax.set_yticklabels(display, fontsize=10); ax.invert_yaxis()
        for i, (m, st) in enumerate(zip(means, stds)):
            ax.text(m+st+0.05 if m>=0 else m-st-0.05, i, f'{2**m:.2f}x', va='center',
                    ha='left' if m>=0 else 'right', fontsize=8)
    ax.set_xlabel('log2(stress/naive)')
    ax.set_title(f'{cn}: {best_label.split("_",1)[1]}\n(fit: {np.mean(best_fits):.0f})', fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle('Best Candidate: Parameter Changes', fontweight='bold', fontsize=14)
plt.tight_layout(); plt.savefig(f'{outdir}/robust_best_candidate_params.png'); plt.close()
print(f"Saved: {outdir}/robust_best_candidate_params.png")

# --- PLOT 8: e_pas shift ---
epas_data = {}
for cl in [1, 2, 3]:
    shifts = []
    for f in sorted(glob.glob(f'{logdir}/allfree_cl{cl}_ns*_ss*.json')):
        with open(f) as fh: d = json.load(fh)
        if 'naive_values' in d and 'optimized_values' in d and 'e_pas' in d['naive_values']:
            shifts.append(d['optimized_values']['e_pas'] - d['naive_values']['e_pas'])
    if shifts: epas_data[cl] = shifts
if epas_data:
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (cl, shifts) in enumerate(sorted(epas_data.items())):
        j = np.random.RandomState(42).uniform(-0.15, 0.15, len(shifts))
        ax.scatter([i]+j, shifts, color=colors_cl[cl], alpha=0.6, s=50, edgecolors='black', linewidth=0.5, zorder=3)
        ax.errorbar(i, np.mean(shifts), yerr=np.std(shifts), fmt='_', color='black',
                    markersize=15, markeredgewidth=2, capsize=6, capthick=2, zorder=4)
    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.set_xticks(range(len(epas_data))); ax.set_xticklabels([cnames[cl] for cl in sorted(epas_data.keys())])
    ax.set_ylabel('delta e_pas (mV)'); ax.set_title('Leak Reversal Shift (All-Free)', fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); plt.savefig(f'{outdir}/robust_epas_shift.png'); plt.close()
    print(f"Saved: {outdir}/robust_epas_shift.png")

print(f"\nAll static plots saved to {outdir}/")

# ============================================================
# 4. SIMULATIONS + CSV EXPORT
# ============================================================
from neuron import h
import neuron
# NEURON mechanisms live in a sibling 'mod/' folder next to this (flattened) script
neuron.load_mechanisms(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mod'))
h.load_file('stdrun.hoc')

print(f"\n{'='*70}")
print("RUNNING SIMULATIONS + EXPORTING DATA")
print(f"{'='*70}")

tracedir = f'{outdir}/traces'; os.makedirs(tracedir, exist_ok=True)
STIM_DELAY = 200; STIM_DUR = 500
AMPS = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
targets_naive = {
    1: {0:0.0, 20:0.0667, 40:1.6, 60:3.9333, 80:6.8667, 100:9.6333, 120:11.8, 140:13.0333, 160:13.7333, 180:14.8, 200:15.7},
    2: {0:0.0, 20:0.2381, 40:1.2857, 60:2.3333, 80:3.2857, 100:3.4286, 120:3.5238, 140:3.8571, 160:3.7619, 180:3.7143, 200:4.2381},
    3: {0:0.0, 20:0.0, 40:0.0, 60:0.1538, 80:1.0513, 100:2.7179, 120:4.4872, 140:6.2821, 160:8.3846, 180:10.2308, 200:11.8205},
}
targets_stress = {
    1: {0:0.0, 20:0.697, 40:3.5455, 60:7.9848, 80:11.9545, 100:14.7727, 120:17.1818, 140:19.1818, 160:19.8636, 180:19.6061, 200:19.2424},
    2: {0:0.0, 20:2.4271, 40:8.0208, 60:10.6562, 80:11.5, 100:10.7396, 120:9.0833, 140:7.1667, 160:5.4896, 180:4.6771, 200:4.0},
    3: {0:0.0, 20:0.0, 40:0.2424, 60:0.9318, 80:2.5227, 100:4.7727, 120:7.2727, 140:9.2727, 160:11.25, 180:12.8939, 200:13.3636},
}

targets_train_naive = {
    'adapt': {
        1: {40: 0.0578, 60: 0.0353, 80: 0.0931, 100: 0.0513, 120: 0.0506, 140: 0.0465, 160: 0.0354, 180: 0.0385, 200: 0.0337},
        2: {80: 0.1549, 100: 0.2551, 120: 0.2401, 140: 0.2578, 160: 0.2218, 180: 0.2154, 200: 0.1883},
        3: {80: 0.1021, 100: 0.0495, 120: 0.0102, 140: 0.0282, 160: 0.025, 180: 0.0261, 200: 0.0394},
    },
    'latency': {
        1: {20: 279.6501, 40: 170.9693, 60: 79.3823, 80: 44.5318, 100: 29.1613, 120: 22.2685, 140: 17.0092, 160: 11.7589, 180: 11.6438, 200: 10.0573},
        2: {60: 64.8416, 80: 38.3769, 100: 24.9977, 120: 18.645, 140: 14.5757, 160: 12.1899, 180: 10.2469, 200: 8.8921},
        3: {60: 168.5056, 80: 169.9168, 100: 146.2916, 120: 139.2036, 140: 127.4855, 160: 86.3585, 180: 71.2016, 200: 52.7571},
    },
    'isi_cv': {
        1: {40: 0.1555, 60: 0.1664, 80: 0.2251, 100: 0.2106, 120: 0.2446, 140: 0.2563, 160: 0.2214, 180: 0.2182, 200: 0.2108},
        2: {80: 0.3031, 100: 0.2937, 120: 0.2213, 140: 0.2773, 160: 0.2537, 180: 0.2714, 200: 0.2684},
        3: {80: 0.2199, 100: 0.1863, 120: 0.1781, 140: 0.1517, 160: 0.1402, 180: 0.1424, 200: 0.1553},
    },
    'mean_isi': {
        1: {40: 116.3904, 60: 118.3048, 80: 75.1735, 100: 54.7746, 120: 43.882, 140: 37.2228, 160: 32.3341, 180: 28.0538, 200: 25.7008},
        2: {80: 51.8724, 100: 38.8014, 120: 31.8438, 140: 29.3788, 160: 24.6987, 180: 22.5124, 200: 21.2092},
        3: {80: 100.1372, 100: 90.6492, 120: 71.6863, 140: 76.751, 160: 61.9155, 180: 64.8953, 200: 47.0588},
    },
}
targets_train_stress = {
    'adapt': {
        1: {20: -0.0265, 40: -0.0163, 60: 0.0689, 80: 0.0372, 100: 0.0333, 120: 0.035, 140: 0.0311, 160: 0.0372, 180: 0.0395, 200: 0.0462},
        2: {20: 0.0506, 40: 0.0782, 60: 0.0566, 80: 0.0512, 100: 0.0551, 120: 0.0732, 140: 0.0957, 160: 0.105, 180: 0.1092, 200: 0.1025},
        3: {60: 0.0538, 80: 0.0694, 100: -0.0043, 120: 0.0107, 140: 0.0122, 160: 0.0171, 180: 0.0172, 200: 0.0195},
    },
    'latency': {
        1: {20: 150.2103, 40: 113.1957, 60: 53.5368, 80: 28.7977, 100: 20.7957, 120: 16.141, 140: 13.709, 160: 11.2847, 180: 9.4285, 200: 8.433},
        2: {20: 160.2733, 40: 57.6779, 60: 33.3652, 80: 22.5647, 100: 17.0842, 120: 13.1163, 140: 10.8547, 160: 9.4098, 180: 7.7639, 200: 6.4857},
        3: {40: 153.5549, 60: 98.1744, 80: 165.4199, 100: 117.6909, 120: 84.3816, 140: 54.3069, 160: 51.7782, 180: 37.4507, 200: 30.819},
    },
    'isi_cv': {
        1: {20: 0.2669, 40: 0.265, 60: 0.2678, 80: 0.2325, 100: 0.2306, 120: 0.2408, 140: 0.2415, 160: 0.2884, 180: 0.2422, 200: 0.2662},
        2: {20: 0.2021, 40: 0.2944, 60: 0.2705, 80: 0.24, 100: 0.2422, 120: 0.2253, 140: 0.2502, 160: 0.1944, 180: 0.1286, 200: 0.1033},
        3: {60: 0.168, 80: 0.1867, 100: 0.1905, 120: 0.1613, 140: 0.1612, 160: 0.1692, 180: 0.16, 200: 0.1583},
    },
    'mean_isi': {
        1: {20: 158.7864, 40: 92.6533, 60: 64.2744, 80: 43.2351, 100: 34.3738, 120: 29.596, 140: 26.0537, 160: 23.9995, 180: 20.5903, 200: 18.8881},
        2: {20: 150.8453, 40: 69.6784, 60: 44.9376, 80: 35.8683, 100: 29.3745, 120: 24.0205, 140: 19.558, 160: 16.354, 180: 13.3545, 200: 12.2737},
        3: {40: 150.6109, 60: 93.5213, 80: 80.6461, 100: 99.5617, 120: 79.668, 140: 58.9701, 160: 50.7744, 180: 40.9222, 200: 36.0578},
    },
}

def build_cell(cluster, p):
    soma = h.Section(name='soma'); dend = h.Section(name='dend')
    soma.L = 15; soma.diam = 15; soma.nseg = 1; dend.L = 300; dend.diam = 5; dend.nseg = 1
    dend.connect(soma, 1, 0)
    for sec in [soma, dend]:
        sec.cm = p['cm']; sec.Ra = 150
        sec.insert('pas'); sec.g_pas = p['g_pas']; sec.e_pas = p['e_pas']
        if cluster == 2: sec.insert('na_lis')
        else: sec.insert('na_li')
        sec.insert('kdr_li'); sec.insert('im_li'); sec.insert('cal_li'); sec.insert('cadyn_li')
        sec.insert('sk_bk_li')
    dend.insert('ih_li'); dend.insert('sahp_li')
    for sec in [soma, dend]: sec.ena = 45; sec.ek = -90; sec.eca = 120
    if cluster == 2: soma.gbar_na_lis = p['gbar_na']; dend.gbar_na_lis = p['gbar_na']/3
    else: soma.gbar_na_li = p['gbar_na']; dend.gbar_na_li = p['gbar_na']/3
    soma.gbar_kdr_li = p['gbar_kdr']; dend.gbar_kdr_li = p['gbar_kdr']*0.3
    soma.gbar_im_li = p['gbar_im']; dend.gbar_im_li = p['gbar_im']
    cal_dr = 2.0 if cluster == 1 else 3.0
    soma.gbar_cal_li = p['gbar_cal']; dend.gbar_cal_li = p['gbar_cal']*cal_dr
    dend.gbar_ih_li = p['gbar_ih']; dend.gbar_sahp_li = p['gbar_sahp']
    if cluster == 2:
        for sec in [soma, dend]: sec.insert('cat_li')
        soma.gbar_cat_li = p['gbar_cat']; dend.gbar_cat_li = p['gbar_cat']*2
    elif cluster == 3:
        for sec in [soma, dend]: sec.insert('ia_li'); sec.gbar_ia_li = p['gbar_ia']
    for sec in [soma, dend]:
        sec.gbar_sk_sk_bk_li = p.get('gbar_sk', 0); sec.gbar_bk_sk_bk_li = p.get('gbar_bk', 0)
    return soma, dend

def run_sweeps(cluster, params):
    se = STIM_DELAY + STIM_DUR; traces = {}; spike_counts = {}
    for amp in AMPS:
        soma, dend = build_cell(cluster, params)
        stim = h.IClamp(soma(0.5)); stim.delay = STIM_DELAY; stim.dur = STIM_DUR; stim.amp = amp/1000
        tv = h.Vector(); tv.record(h._ref_t); vv = h.Vector(); vv.record(soma(0.5)._ref_v)
        h.finitialize(params['e_pas']); h.continuerun(STIM_DELAY + STIM_DUR + 200)
        t = np.array(tv); v = np.array(vv)
        traces[amp] = (t, v)
        spike_counts[amp] = len([t[i] for i in range(1,len(v)) if v[i-1]<0 and v[i]>=0 and STIM_DELAY<t[i]<se])
        del stim, soma, dend
    return traces, spike_counts

def get_full_stress_params(stress_data, cluster):
    ns = stress_data['naive_seed']
    # Find matching naive from in-memory data
    for d in naive_all:
        if d['cluster'] == cluster and d['seed'] == ns:
            params = dict(d['params'])
            if 'optimized_values' in stress_data: params.update(stress_data['optimized_values'])
            return params
    return None

def plot_six_panel(cluster, params, label, traces, spike_counts):
    cn = cnames[cluster]; se = STIM_DELAY + STIM_DUR
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # A: voltage traces
    ax = axes[0,0]; show = [0, 40, 80, 120, 160, 200]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(show)))
    for amp, c in zip(show, colors):
        t, v = traces[amp]; ax.plot(t, v, color=c, lw=0.8, label=f'{amp} pA')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('mV'); ax.set_title('Voltage Traces', fontweight='bold')
    ax.legend(fontsize=7, ncol=2); ax.set_xlim(STIM_DELAY-50, se+50)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    # B: F-I
    ax = axes[0,1]; ax.plot(AMPS, [spike_counts[a] for a in AMPS], 'ko-', lw=2, ms=5)
    ax.set_xlabel('Current (pA)'); ax.set_ylabel('Spike Count'); ax.set_title('F-I Curve', fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    # C: summary
    ax = axes[0,2]; ax.axis('off')
    txt = ''.join(f"\n{a:>3d} pA: {spike_counts[a]}" for a in AMPS if a > 0)
    ax.text(0.1, 0.95, txt, transform=ax.transAxes, fontsize=10, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('Summary', fontweight='bold')
    # D: 80 pA vs 200 pA
    ax = axes[1,0]; t80, v80 = traces[80]; t200, v200 = traces[200]
    ax.plot(t80, v80, color='gray', lw=0.8); ax.plot(t200, v200, 'k-', lw=0.8)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('mV'); ax.set_title('80 pA (gray) vs 200 pA (black)', fontweight='bold')
    ax.set_xlim(STIM_DELAY-50, se+50); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    # E: first spike
    ax = axes[1,1]; spike_amp = None
    for a in [80, 100, 120, 60, 140, 40]:
        if spike_counts.get(a, 0) >= 1: spike_amp = a; break
    if spike_amp:
        t_sp, v_sp = traces[spike_amp]
        for i in range(1, len(v_sp)):
            if v_sp[i-1] < 0 and v_sp[i] >= 0 and t_sp[i] > STIM_DELAY: spike_t = t_sp[i]; break
        mask = (t_sp > spike_t-5) & (t_sp < spike_t+10)
        ax.plot(t_sp[mask]-spike_t, v_sp[mask], 'k-', lw=1.5)
        ax.set_title(f'Spike ({spike_amp} pA)', fontweight='bold')
    else: ax.text(0.5, 0.5, 'No spikes', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('mV')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    # F: blank
    axes[1,2].axis('off')
    fig.suptitle(f'{cn} — {label}', fontweight='bold', fontsize=14)
    plt.tight_layout(); safe = label.replace('/','_').replace(' ','_').replace(':','_')
    fname = f'{tracedir}/traces_{cn}_{safe}.png'
    plt.savefig(fname, dpi=200); plt.close()
    print(f"  Saved: {fname}")

def plot_comparison(cluster, tr_n, sc_n, tr_s, sc_s, stress_label):
    cn = cnames[cluster]; se = STIM_DELAY + STIM_DUR
    tgt_n = targets_naive[cluster]; tgt_s = targets_stress[cluster]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax = axes[0,0]
    ax.plot(AMPS, [sc_n[a] for a in AMPS], 'o-', color='gray', lw=2, ms=5, label='Model naive')
    ax.plot(AMPS, [sc_s[a] for a in AMPS], 's-', color='red', lw=2, ms=5, label='Model stress')
    ax.plot(sorted(tgt_n.keys()), [tgt_n[a] for a in sorted(tgt_n.keys())], 'o--', color='black', alpha=0.4, ms=4, label='Target naive')
    ax.plot(sorted(tgt_s.keys()), [tgt_s[a] for a in sorted(tgt_s.keys())], 's--', color='darkred', alpha=0.4, ms=4, label='Target stress')
    ax.set_xlabel('Current (pA)'); ax.set_ylabel('Spike Count'); ax.set_title('F-I: Naive vs Stress', fontweight='bold')
    ax.legend(fontsize=8); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for col, (tr, lbl, clr) in enumerate([(tr_n, 'Naive', 'gray'), (tr_s, 'Stress', 'red')]):
        ax = axes[0, col+1]; t, v = tr[80]; ax.plot(t, v, color=clr, lw=1)
        ax.set_xlabel('Time (ms)'); ax.set_ylabel('mV'); ax.set_title(f'{lbl} — 80 pA', fontweight='bold')
        ax.set_xlim(STIM_DELAY-50, se+50); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for col, (tr, lbl, clr) in enumerate([(tr_n, 'Naive', 'gray'), (tr_s, 'Stress', 'red')]):
        ax = axes[1, col]; t, v = tr[200]; ax.plot(t, v, color=clr, lw=1)
        ax.set_xlabel('Time (ms)'); ax.set_ylabel('mV'); ax.set_title(f'{lbl} — 200 pA', fontweight='bold')
        ax.set_xlim(STIM_DELAY-50, se+50); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax = axes[1,2]; ax.axis('off')
    txt = f"{'pA':>6} {'Naive':>8} {'Stress':>8} {'Tgt_N':>8} {'Tgt_S':>8}\n"
    for a in AMPS: txt += f"{a:>6} {sc_n[a]:>8} {sc_s[a]:>8} {tgt_n.get(a,0):>8.1f} {tgt_s.get(a,0):>8.1f}\n"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=9, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    fig.suptitle(f'{cn} — Naive vs {stress_label}', fontweight='bold', fontsize=14)
    plt.tight_layout(); safe = stress_label.replace('+','+').replace(' ','_')
    fname = f'{tracedir}/compare_{cn}_{safe}.png'
    plt.savefig(fname, dpi=200); plt.close()
    print(f"  Saved: {fname}")

# ============================================================
# HELPER: sag step
# ============================================================
def run_sag_step(cluster, params, amp_pA=-150):
    soma, dend = build_cell(cluster, params)
    stim = h.IClamp(soma(0.5)); stim.delay = STIM_DELAY; stim.dur = STIM_DUR; stim.amp = amp_pA/1000
    tv = h.Vector(); tv.record(h._ref_t); vv = h.Vector(); vv.record(soma(0.5)._ref_v)
    h.finitialize(params['e_pas']); h.continuerun(STIM_DELAY + STIM_DUR + 200)
    t = np.array(tv); v = np.array(vv)
    pre = (t > STIM_DELAY - 50) & (t < STIM_DELAY)
    rmp = np.mean(v[pre]) if np.any(pre) else params['e_pas']
    ss = (t > STIM_DELAY + STIM_DUR - 50) & (t < STIM_DELAY + STIM_DUR)
    v_ss = np.mean(v[ss]) if np.any(ss) else rmp
    step = (t > STIM_DELAY) & (t < STIM_DELAY + STIM_DUR)
    v_peak = np.min(v[step]) if np.any(step) else v_ss
    sag = (v_ss - v_peak) / (rmp - v_peak) if (rmp - v_peak) != 0 else 0
    rin = abs((v_ss - rmp) / (amp_pA / 1000)) if amp_pA != 0 else 0
    del stim, soma, dend
    return {'rmp': rmp, 'v_ss': v_ss, 'v_peak': v_peak, 'sag': sag, 'rin': rin}, (t, v)

# ============================================================
# HELPER: extract spike features per amplitude
# ============================================================
def extract_features(traces, spike_counts):
    se = STIM_DELAY + STIM_DUR
    out = {}
    for amp in AMPS:
        t, v = traces[amp]; nspk = spike_counts[amp]
        spk_t = [t[i] for i in range(1, len(v)) if v[i-1]<0 and v[i]>=0 and STIM_DELAY<t[i]<se]
        isis = np.diff(spk_t) if len(spk_t) >= 2 else []
        latency = spk_t[0] - STIM_DELAY if spk_t else np.nan
        mean_isi = np.mean(isis) if len(isis) > 0 else np.nan
        isi_cv = np.std(isis)/np.mean(isis) if len(isis) > 1 and np.mean(isis) > 0 else np.nan
        # ipfx adaptation_index: mean((ISI[i] - ISI[i+1]) / (ISI[i] + ISI[i+1]))
        if len(isis) >= 2:
            adapt = np.mean([(isis[i] - isis[i+1]) / (isis[i] + isis[i+1])
                             for i in range(len(isis)-1)])
        else:
            adapt = np.nan
        # Spike shape from first spike
        threshold = peak = trough = width = np.nan
        if len(spk_t) >= 1:
            dt = np.diff(t); dv = np.diff(v); dvdt = dv/dt
            idx0 = np.argmin(np.abs(t[:-1] - spk_t[0]))
            for j in range(idx0, max(0, idx0-200), -1):
                if dvdt[j] < 10: threshold = v[j]; break
            spk_end = min(len(v)-1, idx0+100)
            peak = np.max(v[idx0:spk_end])
            if len(spk_t) >= 2:
                idx1 = np.argmin(np.abs(t - spk_t[1]))
                trough = np.min(v[idx0:idx1])
            else:
                trough = np.min(v[idx0:min(len(v)-1, idx0+int(20/np.mean(dt)))])
            half_v = (threshold + peak) / 2 if not np.isnan(threshold) else (v[idx0] + peak) / 2
            above = v[idx0:spk_end] > half_v
            cup = cdn = None
            for j in range(1, len(above)):
                if above[j] and not above[j-1] and cup is None: cup = t[idx0+j]
                if not above[j] and above[j-1] and cup is not None: cdn = t[idx0+j]; break
            if cup and cdn: width = cdn - cup
        out[amp] = {'spikes': nspk, 'latency': latency, 'mean_isi': mean_isi,
                    'isi_cv': isi_cv, 'adapt': adapt, 'threshold': threshold,
                    'peak': peak, 'trough': trough, 'width': width}
    return out

# ============================================================
# RUN ALL SIMULATIONS, COLLECT DATA
# ============================================================
# Storage: {cl: {condition: {traces, spike_counts, features, passive}}}
sim_data = {}

for cl in [1, 2, 3]:
    cn = cnames[cl]
    if cl not in naive_best: continue
    sim_data[cl] = {}
    naive_params = naive_best[cl]['params']

    # --- NAIVE ---
    print(f"\n  {cn} naive...")
    tr, sc = run_sweeps(cl, naive_params)
    passive, sag_tr = run_sag_step(cl, naive_params)
    feat = extract_features(tr, sc)
    sim_data[cl]['naive'] = {'traces': tr, 'sc': sc, 'feat': feat, 'passive': passive,
                             'sag_trace': sag_tr, 'params': naive_params}
    plot_six_panel(cl, naive_params, f'naive_s{naive_best[cl]["seed"]}', tr, sc)

    # --- BEST CANDIDATE STRESS ---
    cl_c = {k: v for k, v in cand_fits.items() if cand_info.get(k, {}).get('cluster') == cl}
    if cl_c:
        best_label = min(cl_c.keys(), key=lambda k: np.mean(cl_c[k]))
        if best_label in cand_best:
            sp = get_full_stress_params(cand_best[best_label], cl)
            if sp:
                print(f"  {cn} stress ({best_label})...")
                tr_s, sc_s = run_sweeps(cl, sp)
                passive_s, sag_s = run_sag_step(cl, sp)
                feat_s = extract_features(tr_s, sc_s)
                sim_data[cl]['stress'] = {'traces': tr_s, 'sc': sc_s, 'feat': feat_s,
                                          'passive': passive_s, 'sag_trace': sag_s,
                                          'params': sp, 'label': best_label}
                plot_six_panel(cl, sp, best_label, tr_s, sc_s)
                plot_comparison(cl, tr, sc, tr_s, sc_s, best_label)

    # --- ALL-FREE ---
    if cl in allfree_best:
        afp = get_full_stress_params(allfree_best[cl], cl)
        if afp:
            print(f"  {cn} all-free...")
            tr_a, sc_a = run_sweeps(cl, afp)
            plot_six_panel(cl, afp, f'allfree_ns{allfree_best[cl]["naive_seed"]}', tr_a, sc_a)
            plot_comparison(cl, tr, sc, tr_a, sc_a, 'all-free')

# ============================================================
# CSV EXPORTS — Prism-formatted
# ============================================================
csvdir = os.path.join(basedir, 'csv_data'); os.makedirs(csvdir, exist_ok=True)
print(f"\n{'='*70}")
print("EXPORTING CSVs (Prism-formatted)")
print(f"{'='*70}")

all_param_names = {
    1: ['e_pas','g_pas','cm','gbar_na','gbar_kdr','gbar_im','gbar_cal','gbar_ih','gbar_sahp','gbar_sk','gbar_bk'],
    2: ['e_pas','g_pas','cm','gbar_na','gbar_kdr','gbar_im','gbar_cal','gbar_ih','gbar_sahp','gbar_cat','gbar_sk','gbar_bk'],
    3: ['e_pas','g_pas','cm','gbar_na','gbar_kdr','gbar_im','gbar_cal','gbar_ih','gbar_sahp','gbar_ia','gbar_sk','gbar_bk'],
}

# ---- NAIVE PARAMS (all seeds, from memory) ----
union_params = sorted(set(p for plist in all_param_names.values() for p in plist))
with open(f'{csvdir}/naive_params.csv', 'w') as f:
    f.write('cluster,seed,fitness,' + ','.join(union_params) + '\n')
    for d in sorted(naive_all, key=lambda x: (x['cluster'], x['seed'])):
        p = d['params']
        vals = ','.join(str(p.get(k, '')) for k in union_params)
        f.write(f"{cnames[d['cluster']]},{d['seed']},{d['fitness']},{vals}\n")
print(f"  {csvdir}/naive_params.csv  ({len(naive_all)} rows)")

# ---- CANDIDATE FITNESS (Prism column: one col per candidate, rows = seeds) ----
for cl in [1, 2, 3]:
    cn = cnames[cl]
    cl_c = {k: v for k, v in cand_fits.items() if cand_info.get(k, {}).get('cluster') == cl}
    if not cl_c: continue
    ranked = sorted(cl_c.items(), key=lambda x: np.mean(x[1]))
    max_n = max(len(v) for _, v in ranked)
    with open(f'{csvdir}/candidate_fitness_{cn}.csv', 'w') as f:
        f.write(','.join(label for label, _ in ranked) + '\n')
        for i in range(max_n):
            row = []
            for _, fits in ranked:
                row.append(f'{fits[i]:.6f}' if i < len(fits) else '')
            f.write(','.join(row) + '\n')
    print(f"  {csvdir}/candidate_fitness_{cn}.csv")

# ---- XY TABLES WITH TARGETS: F-I, latency, adaptation, mean_ISI, isi_cv ----
# Layout: model table || empty cols || target table (side by side for Prism)
for metric, key in [
    ('fi_curves', 'spikes'),
    ('first_spike_latency', 'latency'),
    ('adaptation_index', 'adapt'),
    ('mean_isi', 'mean_isi'),
    ('isi_cv', 'isi_cv'),
]:
    # Build column lists
    model_cols = []; target_cols = []
    active_cls = [cl for cl in [1, 2, 3] if cl in sim_data]
    for cl in active_cls:
        cn = cnames[cl]
        model_cols.append(f'{cn}_naive_model')
        target_cols.append(f'{cn}_naive_target')
        if 'stress' in sim_data[cl]:
            model_cols.append(f'{cn}_stress_model')
            target_cols.append(f'{cn}_stress_target')

    with open(f'{csvdir}/{metric}.csv', 'w') as f:
        # Header: model side, separator, target side
        sep = ['', '', '']  # 3 empty columns as visual separator
        f.write(','.join(['current_pA'] + model_cols + sep + ['current_pA'] + target_cols) + '\n')

        for amp in AMPS:
            # Model values
            m_vals = []
            for cl in active_cls:
                v = sim_data[cl]['naive']['feat'][amp][key]
                m_vals.append(f'{v}' if not (isinstance(v, float) and np.isnan(v)) else '')
                if 'stress' in sim_data[cl]:
                    v = sim_data[cl]['stress']['feat'][amp][key]
                    m_vals.append(f'{v}' if not (isinstance(v, float) and np.isnan(v)) else '')

            # Target values
            t_vals = []
            for cl in active_cls:
                if key == 'spikes':
                    tgt = targets_naive[cl].get(amp, '')
                    t_vals.append(str(tgt) if tgt != '' else '')
                else:
                    tgt = targets_train_naive.get(key, {}).get(cl, {}).get(amp, '')
                    t_vals.append(f'{tgt}' if tgt != '' else '')
                if 'stress' in sim_data[cl]:
                    if key == 'spikes':
                        tgt = targets_stress[cl].get(amp, '')
                        t_vals.append(str(tgt) if tgt != '' else '')
                    else:
                        tgt = targets_train_stress.get(key, {}).get(cl, {}).get(amp, '')
                        t_vals.append(f'{tgt}' if tgt != '' else '')

            f.write(','.join([str(amp)] + m_vals + ['', '', ''] + [str(amp)] + t_vals) + '\n')
    print(f"  {csvdir}/{metric}.csv")

# ---- TRACES: one file per cluster × condition ----
for cl in [1, 2, 3]:
    if cl not in sim_data: continue
    cn = cnames[cl]
    for cond in ['naive', 'stress']:
        if cond not in sim_data[cl]: continue
        tr = sim_data[cl][cond]['traces']
        ref_t = tr[AMPS[0]][0]
        fname = f'{csvdir}/traces_{cn}_{cond}.csv'
        with open(fname, 'w') as f:
            f.write('time_ms,' + ','.join(f'{a}pA' for a in AMPS) + '\n')
            for i in range(len(ref_t)):
                vals = ','.join(f'{tr[a][1][i]:.3f}' for a in AMPS)
                f.write(f'{ref_t[i]:.3f},{vals}\n')
        print(f"  {fname}")

# ---- SAG TRACES ----
for cl in [1, 2, 3]:
    if cl not in sim_data: continue
    cn = cnames[cl]
    for cond in ['naive', 'stress']:
        if cond not in sim_data[cl]: continue
        t, v = sim_data[cl][cond]['sag_trace']
        with open(f'{csvdir}/sag_trace_{cn}_{cond}.csv', 'w') as f:
            f.write('time_ms,voltage_mV\n')
            for ti, vi in zip(t, v): f.write(f'{ti:.3f},{vi:.3f}\n')
        print(f"  {csvdir}/sag_trace_{cn}_{cond}.csv")

# ---- GROUPED TABLES: one CSV per metric ----
# Prism grouped: rows=clusters, columns=naive/stress
for metric, unit, getter in [
    ('RMP', 'mV', lambda d: d['passive']['rmp']),
    ('Rin', 'MOhm', lambda d: d['passive']['rin']),
    ('sag_ratio', '', lambda d: d['passive']['sag']),
    ('threshold', 'mV', lambda d: d['naive']['feat'][80]['threshold'] if d is None else None),
    ('peak', 'mV', None),
    ('trough', 'mV', None),
    ('half_width', 'ms', None),
]:
    # For spike props, use the lowest spiking amplitude
    with open(f'{csvdir}/grouped_{metric}.csv', 'w') as f:
        f.write(f',naive,stress\n')
        for cl in [1, 2, 3]:
            if cl not in sim_data: continue
            cn = cnames[cl]
            vals = []
            for cond in ['naive', 'stress']:
                if cond not in sim_data[cl]:
                    vals.append('')
                    continue
                sd = sim_data[cl][cond]
                if metric in ('RMP', 'Rin', 'sag_ratio'):
                    key_map = {'RMP': 'rmp', 'Rin': 'rin', 'sag_ratio': 'sag'}
                    v = sd['passive'][key_map[metric]]
                    vals.append(f'{v}')
                else:
                    # Spike property: find lowest amp with spikes
                    spike_amp = None
                    for a in AMPS:
                        if a > 0 and sd['feat'][a]['spikes'] >= 1:
                            spike_amp = a; break
                    if spike_amp is None:
                        vals.append('')
                    else:
                        v = sd['feat'][spike_amp][metric.replace('half_width', 'width')]
                        vals.append(f'{v:.4f}' if not np.isnan(v) else '')
            f.write(f'{cn},{vals[0]},{vals[1]}\n')
    print(f"  {csvdir}/grouped_{metric}.csv")

print(f"\nAll output saved to {outdir}/ and {csvdir}/")
print(f"{'='*85}")