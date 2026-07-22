#!/usr/bin/env python3
"""
Robust naive optimizer — seeded for reproducibility.
CLUSTER_TO_FIT set by sed. SEED from environment.
Saves: logs/robust/results/naive_cl{cl}_s{seed}.json
"""

import numpy as np
import os, sys, json, time, multiprocessing, warnings
from deap import base, creator, tools, cma
warnings.filterwarnings('ignore', module='efel')

CLUSTER_TO_FIT = 1
SEED = int(os.environ.get('SEED', 1))
NGEN = int(os.environ.get('NGEN', 200))
OFFSPRING = int(os.environ.get('OFFSPRING', 1000))
NCPUS = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))

# ============================================================
# NAIVE TARGETS (copied from optimize_cea_naive.py)
# ============================================================
targets_FI = {
    1: {'counts': {0:0, 20:0.07, 40:1.6, 60:3.93, 80:6.87, 100:9.63, 120:11.8, 140:13.03, 160:13.73, 180:14.8, 200:15.7}},
    2: {'counts': {0:0, 20:0, 40:1.3, 60:2.24, 80:3.33, 100:3.43, 120:3.52, 140:3.86, 160:3.76, 180:3.76, 200:4.24}},
    3: {'counts': {0:0, 20:0, 40:0, 60:0, 80:1.05, 100:2.7, 120:4.5, 140:6.2, 160:8.4, 180:10.0, 200:11.8}},
}
passive_props = {
    1: {'rmp': -58.0, 'rmp_std': 8.9, 'Rin': 269.0},
    2: {'rmp': -64.6, 'rmp_std': 4.7, 'Rin': 280.0},
    3: {'rmp': -75.0, 'rmp_std': 6.6, 'Rin': 329.0},
}
spike_props = {
    1: {'threshold': -35.9, 'peak': 32.4, 'trough': -60.5, 'width': 1.08,
        'threshold_std': 4.0, 'peak_std': 4.8, 'trough_std': 8.2, 'width_std': 0.26},
    2: {'threshold': -34.6, 'peak': 31.4, 'trough': -61.2, 'width': 1.02,
        'threshold_std': 2.4, 'peak_std': 4.0, 'trough_std': 5.2, 'width_std': 0.08},
    3: {'threshold': -33.0, 'peak': 33.8, 'trough': -64.5, 'width': 1.09,
        'threshold_std': 5.3, 'peak_std': 10.5, 'trough_std': 10.5, 'width_std': 0.18},
}
train_props = {
    1: {
        60:  {'adapt': 0.035, 'latency': 79.4, 'isi_cv': 0.166, 'mean_isi': 118.3,
              'adapt_std': 0.070, 'latency_std': 47.2, 'isi_cv_std': 0.165, 'mean_isi_std': 74.1},
        80:  {'adapt': 0.093, 'latency': 44.5, 'isi_cv': 0.225, 'mean_isi': 75.2,
              'adapt_std': 0.136, 'latency_std': 24.6, 'isi_cv_std': 0.160, 'mean_isi_std': 24.7},
        100: {'adapt': 0.051, 'latency': 29.2, 'isi_cv': 0.211, 'mean_isi': 54.8,
              'adapt_std': 0.057, 'latency_std': 23.8, 'isi_cv_std': 0.095, 'mean_isi_std': 16.0},
        120: {'adapt': 0.051, 'latency': 22.3, 'isi_cv': 0.245, 'mean_isi': 43.9,
              'adapt_std': 0.052, 'latency_std': 13.7, 'isi_cv_std': 0.121, 'mean_isi_std': 11.1},
        200: {'adapt': 0.034, 'latency': 10.1, 'isi_cv': 0.211, 'mean_isi': 25.7,
              'adapt_std': 0.019, 'latency_std':  5.7, 'isi_cv_std': 0.107, 'mean_isi_std': 6.1},
    },
    2: {
        80:  {'adapt': 0.155, 'latency': 50.0, 'isi_cv': 0.303, 'mean_isi': 51.9,
              'adapt_std': 0.12, 'latency_std': 30.0, 'isi_cv_std': 0.14, 'mean_isi_std': 30.7},
        120: {'adapt': 0.240, 'latency': 28.0, 'isi_cv': 0.221, 'mean_isi': 31.8,
              'adapt_std': 0.10, 'latency_std': 15.0, 'isi_cv_std': 0.12, 'mean_isi_std': 13.5},
        200: {'adapt': 0.188, 'latency': 15.0, 'isi_cv': 0.268, 'mean_isi': 21.2,
              'adapt_std': 0.08, 'latency_std': 10.0, 'isi_cv_std': 0.13, 'mean_isi_std': 11.1},
    },
    3: {
        120: {'adapt': 0.033, 'latency': 139.2, 'isi_cv': 0.160, 'mean_isi': 71.7,
              'adapt_std': 0.07, 'latency_std': 50.0, 'isi_cv_std': 0.10, 'mean_isi_std': 25.3},
        160: {'adapt': 0.033, 'latency':  86.4, 'isi_cv': 0.140, 'mean_isi': 61.9,
              'adapt_std': 0.07, 'latency_std': 35.0, 'isi_cv_std': 0.09, 'mean_isi_std': 32.1},
        200: {'adapt': 0.033, 'latency':  52.8, 'isi_cv': 0.130, 'mean_isi': 47.1,
              'adapt_std': 0.07, 'latency_std': 25.0, 'isi_cv_std': 0.08, 'mean_isi_std': 21.3},
    },
}
sag_targets = {
    1: {'sag': 0.128, 'sag_std': 0.081},
    2: {'sag': 0.158, 'sag_std': 0.032},
    3: {'sag': 0.049, 'sag_std': 0.037},
}
STIM_DELAYS = {1: 200, 2: 200, 3: 200}
STIM_DUR = 500

PARAM_DEFS = {
    1: [
        ('e_pas',     -70.0,   -75.0,   -65.0),
        ('g_pas',     3.33e-5, 1.0e-5,  1.5e-4),
        ('cm',        1.0,     0.8,     1.2),
        ('gbar_na',   0.120,   0.04,    0.5),
        ('gbar_kdr',  0.010,   0.003,   0.03),
        ('gbar_im',   1.5e-4,  1.0e-5,  5.0e-4),
        ('gbar_cal',  1.0e-4,  5.0e-5,  3.0e-4),
        ('gbar_ih',   3.0e-5,  5.0e-6,  1.5e-4),
        ('gbar_sahp', 1.5e-3,  5.0e-4,  5.0e-3),
        ('gbar_sk',   1.0e-3,  1.0e-5,  1.0e-2),
        ('gbar_bk',   1.0e-4,  1.0e-5,  5.0e-3),
    ],
    2: [
        ('e_pas',     -70.0,   -80.0,   -55.0),
        ('g_pas',     3.33e-5, 1.0e-5,  1.5e-4),
        ('cm',        1.0,     0.8,     1.2),
        ('gbar_na',   0.120,   0.04,    0.4),
        ('gbar_kdr',  0.010,   0.003,   0.03),
        ('gbar_im',   1.0e-4,  1.0e-5,  5.0e-4),
        ('gbar_cal',  1.0e-4,  5.0e-5,  3.0e-4),
        ('gbar_ih',   3.0e-5,  5.0e-6,  1.5e-4),
        ('gbar_sahp', 5.0e-4,  5.0e-5,  2.0e-3),
        ('gbar_cat',  1.0e-4,  2.0e-5,  3.0e-4),
        ('gbar_sk',   1.0e-3,  1.0e-5,  1.0e-2),
        ('gbar_bk',   1.0e-4,  1.0e-5,  5.0e-3),
    ],
    3: [
        ('e_pas',     -72.0,   -80.0,   -65.0),
        ('g_pas',     3.33e-5, 1.0e-5,  1.5e-4),
        ('cm',        1.0,     0.8,     1.2),
        ('gbar_na',   0.120,   0.04,    0.5),
        ('gbar_kdr',  0.010,   0.003,   0.03),
        ('gbar_im',   1.0e-4,  1.0e-5,  5.0e-4),
        ('gbar_cal',  1.0e-4,  5.0e-5,  3.0e-4),
        ('gbar_ih',   3.0e-5,  5.0e-6,  1.5e-4),
        ('gbar_sahp', 3.0e-4,  5.0e-5,  2.0e-3),
        ('gbar_ia',   5.7e-3,  1.0e-3,  2.0e-2),
        ('gbar_sk',   1.0e-3,  1.0e-5,  1.0e-2),
        ('gbar_bk',   1.0e-4,  1.0e-5,  5.0e-3),
    ],
}
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
# EVALUATE (naive targets)
# ============================================================
FAIL_PENALTY = 250.0

def evaluate(individual):
    import numpy as np, warnings
    warnings.filterwarnings('ignore', module='efel')
    from neuron import h; h.load_file('stdrun.hoc')
    try: import efel; efel.set_setting('Threshold', 0)
    except: import efel; efel.api.setThreshold(0)

    cluster = CLUSTER_TO_FIT
    param_defs = PARAM_DEFS[cluster]
    p = {name: individual[i] for i, (name, _, _, _) in enumerate(param_defs)}
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

    # F-I
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

    # Cluster-specific
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

    cl = CLUSTER_TO_FIT
    cname = {1:'RS', 2:'LTB', 3:'LF'}[cl]
    outdir = 'logs/robust/results'; os.makedirs(outdir, exist_ok=True)
    param_defs = PARAM_DEFS[cl]
    ndim = len(param_defs)
    centers = [c for _, c, _, _ in param_defs]
    lbounds = [lo for _, _, lo, _ in param_defs]
    ubounds = [hi for _, _, _, hi in param_defs]

    print(f"{'='*60}")
    print(f"ROBUST NAIVE | Cluster {cl} ({cname}) | seed={SEED}")
    print(f"  {ndim} params | NGEN={NGEN} OFFSPRING={OFFSPRING} NCPUS={NCPUS}")
    print(f"{'='*60}"); sys.stdout.flush()

    # Seeded jitter
    rng = np.random.RandomState(SEED)
    jittered = [c * (1 + rng.uniform(-0.1, 0.1)) for c in centers]
    for i in range(ndim):
        jittered[i] = np.clip(jittered[i], lbounds[i], ubounds[i])

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    sigmas = [(hi - lo) * 0.2 for _, _, lo, hi in param_defs]
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

    best_params = {name: best_ever[i] for i, (name, _, _, _) in enumerate(param_defs)}
    print(f"\n  FITNESS: {best_fitness:.4f}  ({elapsed/60:.1f} min)")
    for name, val in best_params.items():
        print(f"  {name:<15} {val:.6g}")

    result = {
        'cluster': cl, 'cluster_name': cname, 'seed': SEED,
        'fitness': best_fitness, 'params': best_params,
        'elapsed_min': elapsed / 60,
    }
    fname = f'{outdir}/naive_cl{cl}_s{SEED}.json'
    with open(fname, 'w') as f: json.dump(result, f, indent=2)
    print(f"Saved: {fname}")
