#!/usr/bin/env python3
"""
Check whether cadyn_li produces enough intracellular calcium to activate SK.
Runs a single 80 pA sweep with the best naive RS params, records cai alongside V.

The SK channel's half-activation (from alphaq/betaq in the Aradi mod file)
is around 0.3-0.5 µM (0.0003-0.0005 mM). If cai never gets above ~0.0001 mM
during spikes, SK will never open regardless of gbar_sk.

Run from EPHYS_MODELING directory:
    python3 check_calcium.py
"""

import os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from neuron import h
import neuron
from _config import MODEL_DIR

basedir = MODEL_DIR
os.chdir(basedir)
# NEURON mechanisms live in a sibling 'mod/' folder next to this (flattened) script
neuron.load_mechanisms(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mod'))
h.load_file('stdrun.hoc')

STIM_DELAY = 200
STIM_DUR = 500

def build_cell(cluster, p):
    soma = h.Section(name='soma'); dend = h.Section(name='dend')
    soma.L = 15; soma.diam = 15; soma.nseg = 1
    dend.L = 300; dend.diam = 5; dend.nseg = 1
    dend.connect(soma, 1, 0)
    for sec in [soma, dend]:
        sec.cm = p['cm']; sec.Ra = 150
        sec.insert('pas'); sec.g_pas = p['g_pas']; sec.e_pas = p['e_pas']
        if cluster == 2: sec.insert('na_lis')
        else: sec.insert('na_li')
        sec.insert('kdr_li'); sec.insert('im_li')
        sec.insert('cal_li'); sec.insert('cadyn_li')
        sec.insert('sk_bk_li')
    dend.insert('ih_li'); dend.insert('sahp_li')
    for sec in [soma, dend]:
        sec.ena = 45; sec.ek = -90; sec.eca = 120
    if cluster == 2:
        soma.gbar_na_lis = p['gbar_na']; dend.gbar_na_lis = p['gbar_na']/3
    else:
        soma.gbar_na_li = p['gbar_na']; dend.gbar_na_li = p['gbar_na']/3
    soma.gbar_kdr_li = p['gbar_kdr']; dend.gbar_kdr_li = p['gbar_kdr']*0.3
    soma.gbar_im_li = p['gbar_im']; dend.gbar_im_li = p['gbar_im']
    cal_dr = 2.0 if cluster == 1 else 3.0
    soma.gbar_cal_li = p['gbar_cal']; dend.gbar_cal_li = p['gbar_cal']*cal_dr
    dend.gbar_ih_li = p['gbar_ih']; dend.gbar_sahp_li = p['gbar_sahp']
    for sec in [soma, dend]:
        sec.gbar_sk_sk_bk_li = p.get('gbar_sk', 0)
        sec.gbar_bk_sk_bk_li = p.get('gbar_bk', 0)
    if cluster == 2:
        for sec in [soma, dend]: sec.insert('cat_li')
        soma.gbar_cat_li = p['gbar_cat']; dend.gbar_cat_li = p['gbar_cat']*2
    elif cluster == 3:
        for sec in [soma, dend]: sec.insert('ia_li'); sec.gbar_ia_li = p['gbar_ia']
    return soma, dend

# Load best naive for each cluster
cnames = {1: 'RS', 2: 'LTB', 3: 'LF'}
logdir = os.path.join(basedir, 'logs', 'robust', 'results')

fig, axes = plt.subplots(3, 3, figsize=(16, 12))

for row, cl in enumerate([1, 2, 3]):
    # Find best naive
    best = None
    for f in sorted(glob.glob(f'{logdir}/naive_cl{cl}_s*.json')):
        with open(f) as fh: d = json.load(fh)
        if best is None or d['fitness'] < best['fitness']:
            best = d
    if best is None:
        print(f"No naive results for cluster {cl}")
        continue

    p = best['params']
    p.setdefault('gbar_sk', 0)
    p.setdefault('gbar_bk', 0)
    cn = cnames[cl]
    print(f"\n{cn}: fitness={best['fitness']:.1f}  gbar_sk={p['gbar_sk']:.2e}  gbar_bk={p['gbar_bk']:.2e}  gbar_cal={p['gbar_cal']:.2e}")

    # Run 80 pA sweep, record V, cai, ica, isk, ibk
    soma, dend = build_cell(cl, p)
    stim = h.IClamp(soma(0.5))
    stim.delay = STIM_DELAY; stim.dur = STIM_DUR; stim.amp = 0.08  # 80 pA

    tv = h.Vector(); tv.record(h._ref_t)
    vv = h.Vector(); vv.record(soma(0.5)._ref_v)
    cai_v = h.Vector(); cai_v.record(soma(0.5)._ref_cai)
    isk_v = h.Vector(); isk_v.record(soma(0.5)._ref_isk_sk_bk_li)
    ibk_v = h.Vector(); ibk_v.record(soma(0.5)._ref_ibk_sk_bk_li)

    h.finitialize(p['e_pas'])
    h.continuerun(STIM_DELAY + STIM_DUR + 200)

    t = np.array(tv); v = np.array(vv)
    cai = np.array(cai_v)
    isk = np.array(isk_v)
    ibk = np.array(ibk_v)

    print(f"  cai range: {cai.min():.6f} - {cai.max():.6f} mM")
    print(f"  cai at rest: {cai[0]:.6f} mM")
    print(f"  cai peak during spikes: {cai.max():.6f} mM")
    print(f"  SK half-activation (KD): ~0.0004 mM")
    print(f"  cai reaches KD? {'YES' if cai.max() > 0.0003 else 'NO'}")
    print(f"  isk range: {isk.min():.6f} - {isk.max():.6f} mA/cm2")
    print(f"  ibk range: {ibk.min():.6f} - {ibk.max():.6f} mA/cm2")

    # Plot
    xlim = (STIM_DELAY - 20, STIM_DELAY + STIM_DUR + 50)

    ax = axes[row, 0]
    ax.plot(t, v, 'k-', lw=0.8)
    ax.set_ylabel('V (mV)'); ax.set_title(f'{cn} — voltage (80 pA)')
    ax.set_xlim(xlim)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = axes[row, 1]
    ax.plot(t, cai * 1000, 'b-', lw=0.8)  # convert mM to µM
    ax.axhline(0.4, color='red', ls='--', lw=1, label='SK KD (~0.4 µM)')
    ax.set_ylabel('[Ca²⁺]ᵢ (µM)'); ax.set_title(f'{cn} — intracellular calcium')
    ax.set_xlim(xlim); ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = axes[row, 2]
    ax.plot(t, isk * 1000, 'g-', lw=0.8, label='I_SK')
    ax.plot(t, ibk * 1000, 'm-', lw=0.8, label='I_BK')
    ax.set_ylabel('current (µA/cm²)'); ax.set_title(f'{cn} — SK & BK currents')
    ax.set_xlim(xlim); ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    del stim, soma, dend

for ax in axes[2, :]:
    ax.set_xlabel('Time (ms)')

plt.tight_layout()
out = os.path.join(basedir, 'plots', 'check_calcium_sk.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=180)
print(f"\nSaved: {out}")

print(f"""
INTERPRETATION:
  If cai peak is << 0.0004 mM (0.4 µM), SK channels can never activate
  because the calcium signal from cadyn_li doesn't reach SK's half-activation
  concentration. In that case, the optimizer correctly set gbar_sk to zero
  because the channel is functionally dead regardless of its conductance.

  Fix options:
    1. Increase gbar_cal (more Ca2+ influx per spike)
    2. Adjust cadyn_li parameters (slower clearance, smaller shell volume)
    3. Both

  If cai peak IS above 0.4 µM but SK current is still tiny, the issue is
  that gbar_sk is too small. Widen the upper bound or start it higher.
""")
