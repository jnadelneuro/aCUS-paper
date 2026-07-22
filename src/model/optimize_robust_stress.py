#!/usr/bin/env python3
"""
Unified robust stress optimizer.
Handles frozen, candidate, and all-free for ALL clusters.
Reads candidates from JSON (sys.argv[1]). Uses label-based output naming.
Uses na_lis (slow Na inactivation) for cluster 2, na_li for clusters 1/3.
All output to logs/robust/results/.

Output naming:
  naive:     naive_cl{cl}_s{seed}.json
  frozen:    frozen_cl{cl}_ns{seed}.json
  candidate: cand_{label}_ns{seed}_ss{seed}.json   (label-based, no index collisions)
  allfree:   allfree_cl{cl}_ns{seed}_ss{seed}.json

Usage:
  python3 optimize_robust_stress.py candidates_singles.json
  python3 optimize_robust_stress.py candidates_pairs.json
  MODE=frozen CLUSTER=2 NAIVE_SEED=1 python3 optimize_robust_stress.py
  MODE=allfree CLUSTER=1 NAIVE_SEED=1 python3 optimize_robust_stress.py
"""

import numpy as np
import os, sys, json, time, multiprocessing, warnings
from deap import base, creator, tools, cma
warnings.filterwarnings('ignore', module='efel')

MODE = os.environ.get('MODE', 'candidate')
CLUSTER_TO_FIT = int(os.environ.get('CLUSTER', 1))
CANDIDATE_IDX = int(os.environ.get('CANDIDATE_IDX', 0))
NAIVE_SEED = int(os.environ.get('NAIVE_SEED', 1))
STRESS_SEED = int(os.environ.get('STRESS_SEED', 1))
NGEN = int(os.environ.get('NGEN', 50))
OFFSPRING = int(os.environ.get('OFFSPRING', 500))
NCPUS = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))

# ============================================================
# LOAD CANDIDATES FROM JSON (sys.argv[1])
# ============================================================
def load_candidates():
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        with open(sys.argv[1]) as f:
            cands = json.load(f)
        return cands
    if MODE in ('frozen', 'allfree'):
        return []
    raise FileNotFoundError(f"Usage: python3 {sys.argv[0]} <candidates.json>")

# Empirically frozen (p>=0.10 naive vs stress)
EMPIRICALLY_FROZEN = {
    1: ['e_pas', 'g_pas', 'cm', 'gbar_ih'],
    2: ['cm'],
    3: ['e_pas', 'g_pas', 'cm', 'gbar_ih'],
}

# All param names per cluster
ALL_PARAM_NAMES = {
    1: ['e_pas','g_pas','cm','gbar_na','gbar_kdr','gbar_im','gbar_cal','gbar_ih','gbar_sahp','gbar_sk','gbar_bk'],
    2: ['e_pas','g_pas','cm','gbar_na','gbar_kdr','gbar_im','gbar_cal','gbar_ih','gbar_sahp','gbar_cat','gbar_sk','gbar_bk'],
    3: ['e_pas','g_pas','cm','gbar_na','gbar_kdr','gbar_im','gbar_cal','gbar_ih','gbar_sahp','gbar_ia','gbar_sk','gbar_bk'],
}

# ============================================================
# LOAD NAIVE PARAMS
# ============================================================
def load_naive(cluster, seed):
    for path in [f'logs/robust/results/naive_cl{cluster}_s{seed}.json',
                 f'logs/naive_cl{cluster}_s{seed}.json']:
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            p = d['params'] if 'params' in d else d
            # Backward compat: old naive runs may not have SK/BK
            p.setdefault('gbar_sk', 0.0)
            p.setdefault('gbar_bk', 0.0)
            return p
    raise FileNotFoundError(f"No naive params for cl{cluster} seed {seed}")

def make_bounds(name, naive_val):
    if name == 'e_pas':
        return (naive_val - 8, naive_val + 8)
    else:
        return (naive_val * 0.25, naive_val * 2.5)

# ===================================================================
# TARGET DATA (STRESS)
# ===================================================================
targets_FI = {
    1: {'counts': {0:0.0, 20:0.697, 40:3.5455, 60:7.9848, 80:11.9545, 100:14.7727, 120:17.1818, 140:19.1818, 160:19.8636, 180:19.6061, 200:19.2424}},
    2: {'counts': {0:0.0, 20:2.4271, 40:8.0208, 60:10.6562, 80:11.5, 100:10.7396, 120:9.0833, 140:7.1667, 160:5.4896, 180:4.6771, 200:4.0}},
    3: {'counts': {0:0.0, 20:0.0, 40:0.2424, 60:0.9318, 80:2.5227, 100:4.7727, 120:7.2727, 140:9.2727, 160:11.25, 180:12.8939, 200:13.3636}},
}
passive_props = {
    1: {'rmp': -60.3, 'rmp_std': 8.3, 'Rin': 316.2},
    2: {'rmp': -58.3, 'rmp_std': 8.3, 'Rin': 425.0},
    3: {'rmp': -70.0, 'rmp_std': 8.7, 'Rin': 246.2},
}
spike_props = {
    1: {'threshold': -34.9, 'peak': 26.7, 'trough': -56.8, 'width': 0.95,
        'threshold_std': 3.8, 'peak_std': 13.3, 'trough_std': 10.5, 'width_std': 0.14},
    2: {'threshold': -36.9, 'peak': 32.0, 'trough': -51.3, 'width': 1.06,
        'threshold_std': 4.2, 'peak_std': 8.6, 'trough_std': 9.2, 'width_std': 0.16},
    3: {'threshold': -34.9, 'peak': 35.0, 'trough': -66.5, 'width': 1.12,
        'threshold_std': 6.5, 'peak_std': 7.3, 'trough_std': 9.8, 'width_std': 0.38},
}
train_props = {
    1: {
         20: {'adapt': -0.0265, 'latency': 150.2, 'isi_cv': 0.2669, 'mean_isi': 158.8,
              'adapt_std': 0.0592, 'latency_std': 60.3, 'isi_cv_std': 0.4014, 'mean_isi_std': 44.6},
         40: {'adapt': -0.0163, 'latency': 113.2, 'isi_cv': 0.265, 'mean_isi': 92.7,
              'adapt_std': 0.174, 'latency_std': 104.2, 'isi_cv_std': 0.2409, 'mean_isi_std': 55.1},
         60: {'adapt': 0.0689, 'latency': 53.5, 'isi_cv': 0.2678, 'mean_isi': 64.3,
              'adapt_std': 0.0812, 'latency_std': 37.4, 'isi_cv_std': 0.1622, 'mean_isi_std': 26.6},
         80: {'adapt': 0.0372, 'latency': 28.8, 'isi_cv': 0.2325, 'mean_isi': 43.2,
              'adapt_std': 0.0311, 'latency_std': 12.2, 'isi_cv_std': 0.1312, 'mean_isi_std': 11.9},
        100: {'adapt': 0.0333, 'latency': 20.8, 'isi_cv': 0.2306, 'mean_isi': 34.4,
              'adapt_std': 0.0203, 'latency_std': 8.6, 'isi_cv_std': 0.1215, 'mean_isi_std': 6.8},
        120: {'adapt': 0.035, 'latency': 16.1, 'isi_cv': 0.2408, 'mean_isi': 29.6,
              'adapt_std': 0.018, 'latency_std': 6.7, 'isi_cv_std': 0.1033, 'mean_isi_std': 5.5},
        140: {'adapt': 0.0311, 'latency': 13.7, 'isi_cv': 0.2415, 'mean_isi': 26.1,
              'adapt_std': 0.0163, 'latency_std': 6.3, 'isi_cv_std': 0.114, 'mean_isi_std': 4.6},
        160: {'adapt': 0.0372, 'latency': 11.3, 'isi_cv': 0.2884, 'mean_isi': 24.0,
              'adapt_std': 0.0314, 'latency_std': 4.9, 'isi_cv_std': 0.2177, 'mean_isi_std': 4.4},
        180: {'adapt': 0.0395, 'latency': 9.4, 'isi_cv': 0.2422, 'mean_isi': 20.6,
              'adapt_std': 0.0218, 'latency_std': 3.8, 'isi_cv_std': 0.0857, 'mean_isi_std': 4.6},
        200: {'adapt': 0.0462, 'latency': 8.4, 'isi_cv': 0.2662, 'mean_isi': 18.9,
              'adapt_std': 0.0313, 'latency_std': 4.4, 'isi_cv_std': 0.1458, 'mean_isi_std': 4.9},
    },
    2: {
         20: {'adapt': 0.0506, 'latency': 160.3, 'isi_cv': 0.2021, 'mean_isi': 150.8,
              'adapt_std': 0.0828, 'latency_std': 75.2, 'isi_cv_std': 0.0851, 'mean_isi_std': 108.5},
         40: {'adapt': 0.0782, 'latency': 57.7, 'isi_cv': 0.2944, 'mean_isi': 69.7,
              'adapt_std': 0.0599, 'latency_std': 37.5, 'isi_cv_std': 0.1606, 'mean_isi_std': 41.3},
         60: {'adapt': 0.0566, 'latency': 33.4, 'isi_cv': 0.2705, 'mean_isi': 44.9,
              'adapt_std': 0.0365, 'latency_std': 22.2, 'isi_cv_std': 0.0957, 'mean_isi_std': 18.6},
         80: {'adapt': 0.0512, 'latency': 22.6, 'isi_cv': 0.24, 'mean_isi': 35.9,
              'adapt_std': 0.0272, 'latency_std': 15.3, 'isi_cv_std': 0.1086, 'mean_isi_std': 12.2},
        100: {'adapt': 0.0551, 'latency': 17.1, 'isi_cv': 0.2422, 'mean_isi': 29.4,
              'adapt_std': 0.0251, 'latency_std': 11.2, 'isi_cv_std': 0.1035, 'mean_isi_std': 8.9},
        120: {'adapt': 0.0732, 'latency': 13.1, 'isi_cv': 0.2253, 'mean_isi': 24.0,
              'adapt_std': 0.0384, 'latency_std': 8.7, 'isi_cv_std': 0.0859, 'mean_isi_std': 7.0},
        140: {'adapt': 0.0957, 'latency': 10.9, 'isi_cv': 0.2502, 'mean_isi': 19.6,
              'adapt_std': 0.0335, 'latency_std': 7.3, 'isi_cv_std': 0.1121, 'mean_isi_std': 5.1},
        160: {'adapt': 0.105, 'latency': 9.4, 'isi_cv': 0.1944, 'mean_isi': 16.4,
              'adapt_std': 0.0388, 'latency_std': 5.7, 'isi_cv_std': 0.076, 'mean_isi_std': 4.6},
        180: {'adapt': 0.1092, 'latency': 7.8, 'isi_cv': 0.1286, 'mean_isi': 13.4,
              'adapt_std': 0.041, 'latency_std': 4.7, 'isi_cv_std': 0.0547, 'mean_isi_std': 5.3},
        200: {'adapt': 0.1025, 'latency': 6.5, 'isi_cv': 0.1033, 'mean_isi': 12.3,
              'adapt_std': 0.0357, 'latency_std': 4.3, 'isi_cv_std': 0.0612, 'mean_isi_std': 5.2},
    },
    3: {
         60: {'adapt': 0.0538, 'latency': 98.2, 'isi_cv': 0.168, 'mean_isi': 93.5,
              'adapt_std': 0.1022, 'latency_std': 53.6, 'isi_cv_std': 0.1037, 'mean_isi_std': 35.5},
         80: {'adapt': 0.0694, 'latency': 165.4, 'isi_cv': 0.1867, 'mean_isi': 80.6,
              'adapt_std': 0.0592, 'latency_std': 195.2, 'isi_cv_std': 0.0676, 'mean_isi_std': 27.1},
        100: {'adapt': -0.0043, 'latency': 117.7, 'isi_cv': 0.1905, 'mean_isi': 99.6,
              'adapt_std': 0.1113, 'latency_std': 105.0, 'isi_cv_std': 0.0693, 'mean_isi_std': 61.2},
        120: {'adapt': 0.0107, 'latency': 84.4, 'isi_cv': 0.1613, 'mean_isi': 79.7,
              'adapt_std': 0.0591, 'latency_std': 73.1, 'isi_cv_std': 0.1079, 'mean_isi_std': 30.4},
        140: {'adapt': 0.0122, 'latency': 54.3, 'isi_cv': 0.1612, 'mean_isi': 59.0,
              'adapt_std': 0.0329, 'latency_std': 41.6, 'isi_cv_std': 0.0828, 'mean_isi_std': 21.5},
        160: {'adapt': 0.0171, 'latency': 51.8, 'isi_cv': 0.1692, 'mean_isi': 50.8,
              'adapt_std': 0.0251, 'latency_std': 63.5, 'isi_cv_std': 0.0951, 'mean_isi_std': 19.7},
        180: {'adapt': 0.0172, 'latency': 37.5, 'isi_cv': 0.16, 'mean_isi': 40.9,
              'adapt_std': 0.0218, 'latency_std': 37.4, 'isi_cv_std': 0.0752, 'mean_isi_std': 12.5},
        200: {'adapt': 0.0195, 'latency': 30.8, 'isi_cv': 0.1583, 'mean_isi': 36.1,
              'adapt_std': 0.0257, 'latency_std': 27.9, 'isi_cv_std': 0.0807, 'mean_isi_std': 11.1},
    },
}

sag_targets = {
    1: {'sag': 0.134, 'sag_std': 0.067},
    2: {'sag': 0.102, 'sag_std': 0.059},
    3: {'sag': 0.070, 'sag_std': 0.049},
}
STIM_DELAYS = {1: 200, 2: 200, 3: 200}
STIM_DUR = 500

# ============================================================
# CELL BUILDER
# ============================================================
def build_cell(cluster, p):
    from neuron import h
    soma = h.Section(name='soma'); dend = h.Section(name='dend')
    soma.L = 15; soma.diam = 15; soma.nseg = 1
    dend.L = 300; dend.diam = 5; dend.nseg = 1
    dend.connect(soma, 1, 0)
    for sec in [soma, dend]:
        sec.cm = p['cm']; sec.Ra = 150
        sec.insert('pas'); sec.g_pas = p['g_pas']; sec.e_pas = p['e_pas']
        if cluster == 2:
            sec.insert('na_lis')
        else:
            sec.insert('na_li')
        sec.insert('kdr_li'); sec.insert('im_li')
        sec.insert('cal_li'); sec.insert('cadyn_li')
        sec.insert('sk_bk_li')
    dend.insert('ih_li'); dend.insert('sahp_li')
    for sec in [soma, dend]:
        sec.ena = 45; sec.ek = -90; sec.eca = 120
    if cluster == 2:
        soma.gbar_na_lis = p['gbar_na']; dend.gbar_na_lis = p['gbar_na'] / 3.0
    else:
        soma.gbar_na_li = p['gbar_na']; dend.gbar_na_li = p['gbar_na'] / 3.0
    soma.gbar_kdr_li = p['gbar_kdr']; dend.gbar_kdr_li = p['gbar_kdr'] * 0.3
    soma.gbar_im_li = p['gbar_im']; dend.gbar_im_li = p['gbar_im']
    cal_dr = 2.0 if cluster == 1 else 3.0
    soma.gbar_cal_li = p['gbar_cal']; dend.gbar_cal_li = p['gbar_cal'] * cal_dr
    dend.gbar_ih_li = p['gbar_ih']; dend.gbar_sahp_li = p['gbar_sahp']
    for sec in [soma, dend]:
        sec.gbar_sk_sk_bk_li = p['gbar_sk']; sec.gbar_bk_sk_bk_li = p['gbar_bk']
    if cluster == 2:
        for sec in [soma, dend]: sec.insert('cat_li')
        soma.gbar_cat_li = p['gbar_cat']; dend.gbar_cat_li = p['gbar_cat'] * 2.0
    elif cluster == 3:
        for sec in [soma, dend]: sec.insert('ia_li'); sec.gbar_ia_li = p['gbar_ia']
    return soma, dend

# ============================================================
# EVALUATE (stress targets)
# ============================================================
FAIL_PENALTY = 250.0
_NAIVE_P = None  # set in main
_FREE_NAMES = None

def evaluate(individual):
    import numpy as np, warnings
    warnings.filterwarnings('ignore', module='efel')
    from neuron import h; h.load_file('stdrun.hoc')
    try: import efel; efel.set_setting('Threshold', 0)
    except: import efel; efel.api.setThreshold(0)

    cluster = CLUSTER_TO_FIT
    p = dict(_NAIVE_P)
    for i, name in enumerate(_FREE_NAMES):
        p[name] = individual[i]
    try: soma, dend = build_cell(cluster, p)
    except: return (1e6,)

    delay = STIM_DELAYS[cluster]; se = delay + STIM_DUR; total_err = 0.0

    # RMP
    pp = passive_props[cluster]
    stim = h.IClamp(soma(0.5)); stim.delay = delay; stim.dur = STIM_DUR; stim.amp = 0
    tv = h.Vector(); tv.record(h._ref_t); vv = h.Vector(); vv.record(soma(0.5)._ref_v)
    h.finitialize(p['e_pas']); h.continuerun(delay + STIM_DUR + 200)
    t = np.array(tv); v = np.array(vv)
    rmp = np.mean(v[(t > delay - 50) & (t < delay - 5)])
    total_err += ((rmp - pp['rmp']) / max(pp['rmp_std'], 2.0)) ** 2
    del stim

    # Sag
    st = sag_targets[cluster]
    stim = h.IClamp(soma(0.5)); stim.delay = delay; stim.dur = STIM_DUR; stim.amp = -0.15
    tv = h.Vector(); tv.record(h._ref_t); vv = h.Vector(); vv.record(soma(0.5)._ref_v)
    h.finitialize(p['e_pas']); h.continuerun(delay + STIM_DUR + 200)
    t = np.array(tv); v = np.array(vv)
    v_rest = np.mean(v[(t > delay - 50) & (t < delay - 5)])
    v_trough = np.min(v[(t > delay) & (t < delay + 200)])
    v_ss = np.mean(v[(t > delay + STIM_DUR - 100) & (t < delay + STIM_DUR)])
    sag = (v_ss - v_trough) / (v_rest - v_trough) if abs(v_rest - v_trough) > 1.0 else 0.0
    total_err += ((sag - st['sag']) / max(st['sag_std'], 0.02)) ** 2
    del stim

    # F-I + spike data
    fi = targets_FI[cluster]['counts']; amps = sorted(fi.keys()); spike_data = {}
    for amp in amps:
        stim = h.IClamp(soma(0.5)); stim.delay = delay; stim.dur = STIM_DUR; stim.amp = amp/1000.0
        tv = h.Vector(); tv.record(h._ref_t); vv = h.Vector(); vv.record(soma(0.5)._ref_v)
        h.finitialize(p['e_pas']); h.continuerun(delay + STIM_DUR + 200)
        t = np.array(tv); v = np.array(vv)
        spk = [t[i] for i in range(1, len(v)) if v[i-1] < 0 and v[i] >= 0 and delay < t[i] < se]
        spike_data[amp] = (t, v, spk)
        n = len(spk); target_n = fi[amp]
        std = 0.3 if target_n < 2 else (0.5 if target_n < 5 else 1.0)
        total_err += ((n - target_n) / std) ** 2
        del stim

    # Spike shape
    sp = spike_props[cluster]; shape_amp = None
    for amp in reversed(amps):
        if amp > 0 and len(spike_data[amp][2]) >= 2: shape_amp = amp; break
    if shape_amp is not None:
        t_sh, v_sh, _ = spike_data[shape_amp]
        trace = {'T': t_sh, 'V': v_sh, 'stim_start': [delay], 'stim_end': [se]}
        try: features = efel.get_feature_values([trace], ['peak_voltage','min_AHP_values','AP_begin_voltage'])
        except: features = efel.getFeatureValues([trace], ['peak_voltage','min_AHP_values','AP_begin_voltage'])
        f = features[0]
        if f.get('peak_voltage') is not None and len(f['peak_voltage']) > 0:
            total_err += ((f['peak_voltage'][0] - sp['peak']) / max(sp['peak_std'], 2.0)) ** 2
        if f.get('min_AHP_values') is not None and len(f['min_AHP_values']) > 0:
            total_err += ((f['min_AHP_values'][0] - sp['trough']) / max(sp['trough_std'], 3.0)) ** 2
        if f.get('AP_begin_voltage') is not None and len(f['AP_begin_voltage']) > 0:
            total_err += ((f['AP_begin_voltage'][0] - sp['threshold']) / max(sp['threshold_std'], 2.0)) ** 2
    else:
        total_err += FAIL_PENALTY * 3

    # Train props
    tp = train_props.get(cluster, {})
    for amp in tp:
        if amp not in spike_data: continue
        _, _, spk_tr = spike_data[amp]; target = tp[amp]
        if len(spk_tr) >= 2:
            lat = spk_tr[0] - delay
            total_err += ((lat - target['latency']) / max(target['latency_std'], 5.0)) ** 2
            isis = np.diff(spk_tr); mean_isi = np.mean(isis)
            total_err += ((mean_isi - target['mean_isi']) / max(target['mean_isi_std'], 5.0)) ** 2
            if len(isis) >= 2:
                total_err += ((np.std(isis)/mean_isi - target['isi_cv']) / max(target['isi_cv_std'], 0.05)) ** 2
            if len(isis) >= 3:
                adapt = np.mean([(isis[i] - isis[i+1]) / (isis[i] + isis[i+1]) for i in range(len(isis)-1)])
                total_err += ((adapt - target['adapt']) / max(target['adapt_std'], 0.03)) ** 2
        elif fi.get(amp, 0) >= 2:
            total_err += FAIL_PENALTY

    # Cluster-specific extras
    if cluster == 2:
        for amp in [120, 160, 200]:
            if amp in spike_data and fi.get(amp, 0) >= 3:
                t_db, v_db, _ = spike_data[amp]
                v_late = v_db[(t_db > se - 200) & (t_db < se)]
                if len(v_late) > 0:
                    total_err += ((np.mean(v_late) - (-40.0)) / 15.0) ** 2
                    late_spk = [t_db[i] for i in range(1, len(v_db))
                                if v_db[i-1] < 0 and v_db[i] >= 0 and se - 200 < t_db[i] < se]
                    total_err += (len(late_spk) / 0.5) ** 2
    if cluster == 3:
        for amp in [80, 100]:
            if amp in spike_data:
                _, _, spk_lf = spike_data[amp]
                if len(spk_lf) > 0:
                    lat = spk_lf[0] - delay
                    tgt = train_props[3].get(amp, {}).get('latency', 100)
                    std_lat = train_props[3].get(amp, {}).get('latency_std', 30)
                    total_err += ((lat - tgt) / max(std_lat, 10.0)) ** 2
    del soma, dend
    return (total_err,)

def worker_init():
    import warnings; warnings.filterwarnings('ignore', module='efel')
    from neuron import h; import neuron
    neuron.load_mechanisms('mod'); h.load_file('stdrun.hoc')

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    from neuron import h; import neuron; import efel
    try: efel.set_setting('Threshold', 0)
    except: efel.api.setThreshold(0)
    neuron.load_mechanisms('mod'); h.load_file('stdrun.hoc')

    outdir = 'logs/robust/results'; os.makedirs(outdir, exist_ok=True)
    cl = CLUSTER_TO_FIT
    cname = {1:'RS', 2:'LTB', 3:'LF'}[cl]

    # Determine mode, cluster, free channels
    if MODE == 'candidate':
        cands = load_candidates()
        cand = cands[CANDIDATE_IDX]
        cl = cand['cluster']; CLUSTER_TO_FIT = cl
        free_ch_names = cand['channels']
        label = cand['label']
        cname = {1:'RS', 2:'LTB', 3:'LF'}[cl]
    elif MODE == 'allfree':
        free_ch_names = [n for n in ALL_PARAM_NAMES[cl] if n not in EMPIRICALLY_FROZEN[cl]]
        label = f'allfree_{cname}'
    elif MODE == 'frozen':
        free_ch_names = []
        label = f'frozen_{cname}'
    else:
        raise ValueError(f"Unknown MODE: {MODE}")

    # Load naive
    naive_p = load_naive(cl, NAIVE_SEED)
    _NAIVE_P = naive_p
    _FREE_NAMES = free_ch_names

    print(f"{'='*60}")
    print(f"ROBUST STRESS | {MODE} | {label}")
    print(f"  Cluster {cl} ({cname}) | naive_seed={NAIVE_SEED} stress_seed={STRESS_SEED}")
    print(f"  Free: {free_ch_names}")
    print(f"  NGEN={NGEN} OFFSPRING={OFFSPRING} NCPUS={NCPUS}")
    print(f"{'='*60}"); sys.stdout.flush()

    ndim = len(free_ch_names)

    if ndim == 0:
        # Frozen: just evaluate once
        _FREE_NAMES = []
        fitness = evaluate([])[0]
        print(f"  FITNESS: {fitness:.4f}")
        result = {
            'mode': 'frozen', 'cluster': cl, 'cluster_name': cname,
            'label': label, 'naive_seed': NAIVE_SEED,
            'fitness': fitness, 'naive_params': naive_p,
        }
        fname = f'{outdir}/frozen_cl{cl}_ns{NAIVE_SEED}.json'
        with open(fname, 'w') as f: json.dump(result, f, indent=2)
        print(f"Saved: {fname}"); sys.exit(0)

    # Build param defs with bounds
    centers = [naive_p[n] for n in free_ch_names]
    bounds = [make_bounds(n, naive_p[n]) for n in free_ch_names]
    lbounds = [b[0] for b in bounds]
    ubounds = [b[1] for b in bounds]

    for n, c, lo, hi in zip(free_ch_names, centers, lbounds, ubounds):
        print(f"  {n:<15} naive={c:<12.4g} [{lo:.4g}, {hi:.4g}]")
    sys.stdout.flush()

    # CMA-ES with seed
    rng = np.random.RandomState(STRESS_SEED)
    jittered = [c * (1 + rng.uniform(-0.05, 0.05)) for c in centers]
    for i in range(ndim):
        jittered[i] = np.clip(jittered[i], lbounds[i], ubounds[i])

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    sigmas = [(hi - lo) * 0.2 for lo, hi in bounds]
    strategy = cma.Strategy(centroid=jittered, sigma=np.mean(sigmas), lambda_=OFFSPRING)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)
    toolbox.register("evaluate", evaluate)
    pool = multiprocessing.Pool(processes=NCPUS, initializer=worker_init)
    toolbox.register("map", pool.map)

    best_ever = None; best_fitness = float('inf'); t0 = time.time()
    stall_count = 0; STALL_LIMIT = max(20, 4 * ndim)
    for gen in range(NGEN):
        population = toolbox.generate()
        for ind in population:
            for i in range(ndim): ind[i] = np.clip(ind[i], lbounds[i], ubounds[i])
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses): ind.fitness.values = fit
        try: toolbox.update(population)
        except np.linalg.LinAlgError:
            strategy = cma.Strategy(centroid=best_ever if best_ever else jittered,
                                   sigma=np.mean(sigmas)*0.5, lambda_=OFFSPRING)
            toolbox.register("generate", strategy.generate, creator.Individual)
            toolbox.register("update", strategy.update)
        gen_best = min(population, key=lambda x: x.fitness.values[0])
        improved_significantly = gen_best.fitness.values[0] < best_fitness - 0.5
        if gen_best.fitness.values[0] < best_fitness:
            best_fitness = gen_best.fitness.values[0]; best_ever = list(gen_best)
        if improved_significantly:
            stall_count = 0
        else:
            stall_count += 1
        if stall_count >= STALL_LIMIT:
            print(f"  Converged at gen {gen+1} (no improvement for {STALL_LIMIT} gens)"); sys.stdout.flush()
            break
        if (gen+1) % 10 == 0 or gen == 0:
            avg = np.mean([ind.fitness.values[0] for ind in population])
            print(f"  gen {gen+1:>4}  min={best_fitness:.2f}  avg={avg:.1f}"); sys.stdout.flush()
    pool.close(); pool.join(); elapsed = time.time() - t0

    best_vals = {n: best_ever[i] for i, n in enumerate(free_ch_names)}
    pct_changes = {}
    for n in free_ch_names:
        nv = naive_p[n]
        pct_changes[n] = (best_vals[n] - nv) / abs(nv) * 100 if nv != 0 else 0

    print(f"\n  FITNESS: {best_fitness:.4f}  ({elapsed/60:.1f} min)")
    for n in free_ch_names:
        print(f"  {n:<15} {naive_p[n]:<12.4g} -> {best_vals[n]:<12.4g} ({pct_changes[n]:+.1f}%)")

    result = {
        'mode': MODE, 'cluster': cl, 'cluster_name': cname,
        'label': label, 'naive_seed': NAIVE_SEED, 'stress_seed': STRESS_SEED,
        'channels': free_ch_names, 'fitness': best_fitness,
        'naive_values': {n: naive_p[n] for n in free_ch_names},
        'optimized_values': best_vals, 'pct_changes': pct_changes,
        'elapsed_min': elapsed / 60,
    }
    if MODE == 'candidate':
        fname = f'{outdir}/cand{CANDIDATE_IDX}_cl{cl}_ns{NAIVE_SEED}_ss{STRESS_SEED}.json'
    else:
        fname = f'{outdir}/allfree_cl{cl}_ns{NAIVE_SEED}_ss{STRESS_SEED}.json'
    with open(fname, 'w') as f: json.dump(result, f, indent=2)
    print(f"Saved: {fname}")
