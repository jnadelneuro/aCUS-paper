"""
getPhasePlotV3  –  Phase-plot extraction, QC, functional PCA, scalar
features, and hierarchical mixed-effects modelling for SNL CeA neurons.

Design
------
1.  For each cell, group firing-rate sweeps by current and scan ascending.
    A current qualifies only if EVERY sweep at it fired >=N spikes: first
    current with every sweep >=5 -> positions 1, 3, 5; else >=3 -> 1, 3;
    else >=1 -> 1.
2.  Extract those positions from ALL sweeps at the chosen current (averaged
    per position at the cell level downstream).
3.  QC: compute group-level (cluster × stress) peak_v distribution; reject
    spikes whose peak_v falls outside mean ± n_sd of their group.
4.  Per-cell, per-spike-position: keep dV/dt(t) and V(t) arrays (100 pts
    each, ±50 samples around peak).
5.  Scalar features per spike: max_dvdt, min_dvdt, threshold_v, peak_v,
    trough_v, upstroke_v, downstroke_v, width, upstroke_downstroke_ratio.
6.  Functional PCA on dV/dt waveforms (pooled across cells/positions).
7.  Mixed-effects models:
        scalar ~ stress * spike_position,  random = 1|mouse/cell
        PC_score ~ stress * spike_position, random = 1|mouse/cell
    Run per-cluster.
8.  Export CSVs + summary text files.

Intended to be called from ephysAnalysisAnalyze.py after cluster labels
have been assigned (via ephysAnalysisCluster.py / addToCSVs).
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# =====================================================================
# 1.  EXTRACTION
# =====================================================================

# Spike-count thresholds -> which spike positions to keep, in priority
# order. The first current (ascending) where EVERY sweep reaches a tier's
# spike count wins, and that tier's positions are the ones extracted.
SELECTION_TIERS = [
    (5, [1, 3, 5]),   # >=5 spikes -> 1st, 3rd, 5th
    (3, [1, 3]),      # else >=3 spikes -> 1st, 3rd
    (1, [1]),         # else >=1 spike  -> 1st only
]


def _sweep_spikes(row):
    """Per-spike data for one firing-rate sweep (one spikeAnalysis row).

    Returns (n_detected, spikes): n_detected is the number of action
    potentials detected in the sweep (len of the peak_v list); spikes is
    the list of per-spike dicts with a valid 100-pt phase-plot waveform,
    each tagged with its true 1-indexed position in the train.

    The per-spike validity checks and scalar-field handling are identical
    to V2's _extract_spikes_at_current; only the packaging differs.
    """
    pp_data = row['phasePlotData']
    peak_vs = row['peak_v']
    # Both being lists marks a sweep with detected spikes; spikeless
    # sweeps store np.nan for peak_v (see ephysAnalysisSetup) -> 0 spikes.
    if not isinstance(pp_data, list) or not isinstance(peak_vs, list):
        return 0, []
    n_detected = len(peak_vs)

    def _as_list(val):
        return val if isinstance(val, list) else [np.nan] * len(peak_vs)

    threshold_vs = _as_list(row.get('threshold_v', np.nan))
    trough_vs = _as_list(row.get('trough_v', np.nan))
    upstrokes = _as_list(row.get('upstroke', np.nan))
    downstrokes = _as_list(row.get('downstroke', np.nan))
    upstroke_vs = _as_list(row.get('upstroke_v', np.nan))
    downstroke_vs = _as_list(row.get('downstroke_v', np.nan))
    widths = _as_list(row.get('width', np.nan))

    spikes = []
    for i, (pp, pv) in enumerate(zip(pp_data, peak_vs)):
        # pp is (dV_dt_array, V_array) tuple from isolateAP
        if isinstance(pp, float) and np.isnan(pp):
            continue
        dV_dt_arr, V_arr = pp
        if (not isinstance(dV_dt_arr, np.ndarray) or
                not isinstance(V_arr, np.ndarray)):
            continue
        if len(dV_dt_arr) != 100 or len(V_arr) != 100:
            continue

        spikes.append({
            'spike_pos': i + 1,  # 1-indexed position in train
            'dV_dt': dV_dt_arr.copy(),
            'V': V_arr.copy(),
            'peak_v': pv,
            'threshold_v': threshold_vs[i] if i < len(threshold_vs) else np.nan,
            'trough_v': trough_vs[i] if i < len(trough_vs) else np.nan,
            'upstroke': upstrokes[i] if i < len(upstrokes) else np.nan,
            'downstroke': downstrokes[i] if i < len(downstrokes) else np.nan,
            'upstroke_v': upstroke_vs[i] if i < len(upstroke_vs) else np.nan,
            'downstroke_v': downstroke_vs[i] if i < len(downstroke_vs) else np.nan,
            'width': widths[i] if i < len(widths) else np.nan,
        })
    return n_detected, spikes


def _select_current_and_positions(cell):
    """Pick the firing-rate current and spike positions for one cell.

    Groups firing-rate sweeps by injected current and scans currents in
    ascending order. A current qualifies for a tier only if EVERY sweep at
    that current fired >=N detected APs (the strict "every time" rule).
    Applies SELECTION_TIERS: the first current where every sweep has >=5
    spikes is used with positions [1, 3, 5]; failing that, every sweep >=3
    with [1, 3]; failing that, every sweep >=1 with [1]. "Fired N spikes"
    counts detected APs in the sweep (len of peak_v).

    Returns (current, sweeps, positions), or None if no current qualifies,
    where sweeps is the list of per-sweep dicts {'sweep', 'n_detected',
    'spikes'} at the chosen current (kept in full and averaged per position
    downstream by average_to_cell_level).
    """
    if not hasattr(cell, 'spikeAnalysis'):
        return None
    sa = cell.spikeAnalysis
    if isinstance(sa, dict):
        sa = pd.DataFrame(sa)
    if 'protocol' not in sa.columns:
        return None
    fr = sa[sa['protocol'] == 'firing rate']
    if fr.empty:
        return None

    # Group sweeps by injected current so a current can be judged across
    # all of its (possibly replicate) sweeps.
    by_current = {}
    for _, srow in fr.iterrows():
        cur = srow['current injected']
        n_detected, spikes = _sweep_spikes(srow)
        by_current.setdefault(cur, []).append({
            'sweep': srow['sweep'],
            'n_detected': n_detected,
            'spikes': spikes,
        })

    for thresh, positions in SELECTION_TIERS:
        for cur in sorted(by_current.keys()):  # ascending current
            sweeps = by_current[cur]
            # "every time": every sweep at this current must reach thresh.
            if sweeps and all(sw['n_detected'] >= thresh for sw in sweeps):
                return cur, sweeps, positions
    return None


def extract_phase_plots(mouseList, qc_n_sd=1.0):
    """Main extraction function.

    Selection (V3): for each cell, group firing-rate sweeps by current and
    scan ascending; a current qualifies only if EVERY sweep at it fired
    >=N spikes. First current with every sweep >=5 -> positions 1, 3, 5;
    else >=3 -> 1, 3; else >=1 -> 1. All sweeps at the chosen current are
    kept and averaged per position downstream. See
    _select_current_and_positions / SELECTION_TIERS.

    Parameters
    ----------
    mouseList : dict
        {mouse_name: EphysMouse} — the standard ephysMouseDict.
    qc_n_sd : float
        Reject spikes whose peak_v is > qc_n_sd SDs from their
        group (cluster × stress) mean peak_v. (Same strict QC as V2.)

    Returns
    -------
    spike_df : DataFrame
        One row per spike, with columns: cell_name, mouse, sex, stress,
        cluster, target_current, sweep, spike_pos, peak_v, threshold_v,
        trough_v, upstroke, downstroke, upstroke_v, downstroke_v, width,
        dV_dt (array), V (array), plus computed scalar features.
    """
    rows = []

    for mouse in mouseList.values():
        if mouse.proj != 'SNL':
            continue
        for cell in mouse.cells:
            if not hasattr(cell, 'cluster') or np.isnan(cell.cluster):
                continue

            selection = _select_current_and_positions(cell)
            if selection is None:
                print(f'  [phasePlotV3] {cell.name}: no current where every '
                      f'sweep fired enough spikes, skipping')
                continue
            current, sweeps, positions = selection

            # Keep the requested positions from EVERY sweep at the chosen
            # current; average_to_cell_level averages them per (cell,
            # spike_pos) downstream.
            keep = [(sw['sweep'], sp) for sw in sweeps for sp in sw['spikes']
                    if sp['spike_pos'] in positions]
            if not keep:
                print(f'  [phasePlotV3] {cell.name}: current {current} pA '
                      f'qualified but no valid waveform at positions '
                      f'{positions}, skipping')
                continue

            for sweep_id, sp in keep:
                row = {
                    'cell_name': cell.name,
                    'mouse': mouse.name,
                    'sex': mouse.sex,
                    'stress': mouse.stressCon,
                    'cluster': int(cell.cluster),
                    'target_current': current,
                    'sweep': sweep_id,
                    'spike_pos': sp['spike_pos'],
                    'dV_dt': sp['dV_dt'],
                    'V': sp['V'],
                    'peak_v': sp['peak_v'],
                    'threshold_v': sp['threshold_v'],
                    'trough_v': sp['trough_v'],
                    'upstroke': sp['upstroke'],
                    'downstroke': sp['downstroke'],
                    'upstroke_v': sp['upstroke_v'],
                    'downstroke_v': sp['downstroke_v'],
                    'width': sp['width'],
                }
                rows.append(row)

    spike_df = pd.DataFrame(rows)
    if spike_df.empty:
        print('[phasePlotV3] No spikes extracted!')
        return spike_df

    print(f'[phasePlotV3] Extracted {len(spike_df)} spikes from '
          f'{spike_df["cell_name"].nunique()} cells')

    # ── QC: group-level peak_v filtering ──────────────────────────────
    n_before = len(spike_df)
    group_stats = (spike_df.groupby(['cluster', 'stress'])['peak_v']
                   .agg(['mean', 'std']).reset_index())
    group_stats.columns = ['cluster', 'stress', 'group_peak_mean',
                           'group_peak_std']
    spike_df = spike_df.merge(group_stats, on=['cluster', 'stress'])
    spike_df['peak_v_zscore'] = (
        (spike_df['peak_v'] - spike_df['group_peak_mean']).abs()
        / spike_df['group_peak_std']
    )
    spike_df = spike_df[spike_df['peak_v_zscore'] <= qc_n_sd].copy()
    spike_df.drop(columns=['group_peak_mean', 'group_peak_std',
                           'peak_v_zscore'], inplace=True)
    n_after = len(spike_df)
    print(f'[phasePlotV3] QC: {n_before - n_after}/{n_before} spikes '
          f'rejected (peak_v > {qc_n_sd} SD from group mean)')

    # ── Computed scalar features ──────────────────────────────────────
    spike_df['max_dvdt'] = spike_df['dV_dt'].apply(lambda x: np.max(x))
    spike_df['min_dvdt'] = spike_df['dV_dt'].apply(lambda x: np.min(x))
    spike_df['upstroke_downstroke_ratio'] = (
        spike_df['upstroke'].abs() / spike_df['downstroke'].abs()
    )
    # Convert dV/dt from mV/s to V/s for consistency
    spike_df['max_dvdt_Vs'] = spike_df['max_dvdt'] / 1000.0
    spike_df['min_dvdt_Vs'] = spike_df['min_dvdt'] / 1000.0

    # Loop area: approximate as sum of |dV/dt| * dV across the waveform
    def _loop_area(dv_dt, v):
        dv = np.diff(v)
        avg_dvdt = (np.abs(dv_dt[:-1]) + np.abs(dv_dt[1:])) / 2
        return float(np.sum(avg_dvdt * np.abs(dv)))
    spike_df['loop_area'] = spike_df.apply(
        lambda r: _loop_area(r['dV_dt'], r['V']), axis=1)

    return spike_df


# =====================================================================
# 2.  CELL-LEVEL AVERAGING
# =====================================================================

def average_to_cell_level(spike_df):
    """Average spike waveforms and scalars per (cell, spike_position).

    Returns
    -------
    cell_df : DataFrame
        One row per (cell × spike_position). dV_dt and V are the
        within-cell mean waveforms. Scalar features are means.
    """
    meta_cols = ['cell_name', 'mouse', 'sex', 'stress', 'cluster',
                 'target_current']
    scalar_cols = ['peak_v', 'threshold_v', 'trough_v', 'upstroke',
                   'downstroke', 'upstroke_v', 'downstroke_v', 'width',
                   'max_dvdt', 'min_dvdt', 'max_dvdt_Vs', 'min_dvdt_Vs',
                   'upstroke_downstroke_ratio', 'loop_area']

    grouped = spike_df.groupby(meta_cols + ['spike_pos'])
    rows = []
    for keys, grp in grouped:
        row = dict(zip(meta_cols + ['spike_pos'], keys))
        # Average waveforms
        dv_dts = np.vstack(grp['dV_dt'].values)
        vs = np.vstack(grp['V'].values)
        row['dV_dt'] = np.mean(dv_dts, axis=0)
        row['V'] = np.mean(vs, axis=0)
        row['n_spikes_averaged'] = len(grp)
        # Average scalars
        for sc in scalar_cols:
            if sc in grp.columns:
                row[sc] = grp[sc].mean()
        rows.append(row)

    cell_df = pd.DataFrame(rows)
    print(f'[phasePlotV3] Cell-level: {len(cell_df)} rows '
          f'({cell_df["cell_name"].nunique()} cells, '
          f'max spike_pos = {cell_df["spike_pos"].max()})')
    return cell_df


# =====================================================================
# 3.  FUNCTIONAL PCA
# =====================================================================

def run_functional_pca(cell_df, n_components=4):
    """Run PCA on the dV/dt waveforms.

    Parameters
    ----------
    cell_df : DataFrame from average_to_cell_level
    n_components : int

    Returns
    -------
    cell_df : DataFrame with PC score columns added (PC1, PC2, ...)
    pca : fitted PCA object
    scaler : fitted StandardScaler
    """
    waveforms = np.vstack(cell_df['dV_dt'].values)

    scaler = StandardScaler()
    waveforms_scaled = scaler.fit_transform(waveforms)

    n_components = min(n_components, waveforms_scaled.shape[0],
                       waveforms_scaled.shape[1])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(waveforms_scaled)

    for i in range(n_components):
        cell_df[f'PC{i+1}'] = scores[:, i]

    print(f'[phasePlotV3] fPCA: {n_components} components, '
          f'explained variance = '
          f'{pca.explained_variance_ratio_.cumsum()[-1]*100:.1f}%')
    for i in range(n_components):
        print(f'  PC{i+1}: {pca.explained_variance_ratio_[i]*100:.1f}%')

    return cell_df, pca, scaler


# =====================================================================
# 4.  MIXED-EFFECTS MODELS
# =====================================================================

def _get_significance_stars(p):
    if p < 0.0001:
        return '****'
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'ns'


def run_phase_plot_models(cell_df, output_dir='.'):
    """Run mixed-effects models on scalar features and PC scores.

    Models per cluster:
        DV ~ stress * spike_pos,  random = cell nested in mouse

    Parameters
    ----------
    cell_df : DataFrame with scalar features and PC scores.
    output_dir : str, path for output files.

    Returns
    -------
    results_dict : nested dict of model results
    """
    os.makedirs(output_dir, exist_ok=True)

    scalar_features = ['max_dvdt_Vs', 'min_dvdt_Vs', 'peak_v',
                        'threshold_v', 'trough_v',
                        'upstroke_downstroke_ratio', 'width', 'loop_area']
    pc_cols = [c for c in cell_df.columns if c.startswith('PC')]

    all_dvs = scalar_features + pc_cols
    clusters = sorted(cell_df['cluster'].unique())

    results_dict = {}

    for cl in clusters:
        cl_df = cell_df[cell_df['cluster'] == cl].copy()

        # Filter out spike positions where either stress group has < min_per_group cells
        min_per_group = 3
        valid_positions = []
        for pos in sorted(cl_df['spike_pos'].unique()):
            pos_data = cl_df[cl_df['spike_pos'] == pos]
            n_naive = len(pos_data[pos_data['stress'] == 'naive'])
            n_stress = len(pos_data[pos_data['stress'] == 'stress'])
            if n_naive >= min_per_group and n_stress >= min_per_group:
                valid_positions.append(pos)

        cl_df = cl_df[cl_df['spike_pos'].isin(valid_positions)].copy()

        n_cells = cl_df['cell_name'].nunique()
        n_mice = cl_df['mouse'].nunique()
        n_pos = len(valid_positions)
        all_pos = sorted(cell_df[(cell_df['cluster'] == cl)]['spike_pos'].unique())
        dropped = [p for p in all_pos if p not in valid_positions]

        print(f'\n=== Cluster {cl} ({n_cells} cells, {n_mice} mice, '
              f'positions {valid_positions[0]}-{valid_positions[-1]}'
              f'{f", dropped {dropped} (n<{min_per_group})" if dropped else ""}) ===')

        # Check if we have multiple spike positions
        has_spike_pos = len(valid_positions) > 1

        results_dict[cl] = {}
        summaries = []

        for dv in all_dvs:
            model_df = cl_df.dropna(subset=[dv]).copy()
            if len(model_df) < 5:
                continue

            # Build formula
            if has_spike_pos:
                formula = f'{dv} ~ C(stress) * spike_pos'
            else:
                formula = f'{dv} ~ C(stress)'

            vc = {'cell_name': '0 + cell_name:mouse'}

            try:
                model = sm.MixedLM.from_formula(
                    formula, data=model_df, vc_formula=vc,
                    groups=model_df['mouse'])
                result = model.fit(reml=True)
                results_dict[cl][dv] = result

                # Extract key p-values
                pvals = result.pvalues
                stress_p = pvals.get('C(stress)[T.stress]', np.nan)
                summary_line = (
                    f'{dv:30s}  stress p={stress_p:.4f} '
                    f'{_get_significance_stars(stress_p)}'
                )
                if has_spike_pos:
                    interaction_keys = [k for k in pvals.index
                                        if 'stress' in k and 'spike_pos' in k]
                    if interaction_keys:
                        int_p = pvals[interaction_keys[0]]
                        summary_line += (
                            f'  |  stress×pos p={int_p:.4f} '
                            f'{_get_significance_stars(int_p)}'
                        )

                summaries.append(summary_line)
                print(f'  {summary_line}')

            except Exception as e:
                print(f'  {dv}: model failed — {e}')

        # Write summary
        summary_path = os.path.join(
            output_dir, f'phasePlotV3_model_summary_C{cl}.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f'Phase Plot V3 - Cluster {cl}\n')
            f.write(f'{n_cells} cells from {n_mice} mice\n')
            f.write(f'Model: DV ~ stress * spike_pos, '
                    f'random = cell nested in mouse\n')
            f.write('=' * 70 + '\n\n')
            for line in summaries:
                f.write(line + '\n')
            f.write('\n\n')
            # Full summaries
            for dv, res in results_dict[cl].items():
                f.write(f'\n{"-" * 70}\n{dv}\n{"-" * 70}\n')
                f.write(res.summary().as_text())
                f.write('\n')

    return results_dict


def run_first_spike_models(cell_df, output_dir='.'):
    """Run mixed-effects models on first spike only (spike_pos == 1).

    Models per cluster:
        DV ~ stress,  random = 1|mouse

    Uses simple random intercept for mouse (no cell nesting since
    there's only one row per cell at spike_pos == 1).
    """
    os.makedirs(output_dir, exist_ok=True)

    scalar_features = ['max_dvdt_Vs', 'min_dvdt_Vs', 'peak_v',
                        'threshold_v', 'trough_v',
                        'upstroke_downstroke_ratio', 'width', 'loop_area']
    pc_cols = [c for c in cell_df.columns if c.startswith('PC')]
    all_dvs = scalar_features + pc_cols

    sp1 = cell_df[cell_df['spike_pos'] == 1].copy()
    if sp1.empty:
        print('[phasePlotV3] No first-spike data for first-spike models')
        return {}

    clusters = sorted(sp1['cluster'].unique())
    results_dict = {}

    for cl in clusters:
        cl_df = sp1[sp1['cluster'] == cl].copy()
        n_cells = cl_df['cell_name'].nunique()
        n_mice = cl_df['mouse'].nunique()
        print(f'\n=== First spike models: Cluster {cl} '
              f'({n_cells} cells, {n_mice} mice) ===')

        results_dict[cl] = {}
        summaries = []

        for dv in all_dvs:
            model_df = cl_df.dropna(subset=[dv]).copy()
            if len(model_df) < 5:
                continue

            formula = f'{dv} ~ C(stress)'

            # Effect size (computed regardless of model)
            naive_m = model_df[model_df['stress'] == 'naive'][dv].mean()
            stress_m = model_df[model_df['stress'] == 'stress'][dv].mean()
            delta = stress_m - naive_m

            # Try mixed model first; fall back to OLS with clustered SEs
            import warnings as _warnings
            model_type = 'mixed'
            try:
                with _warnings.catch_warnings(record=True) as caught:
                    _warnings.simplefilter('always')
                    model = sm.MixedLM.from_formula(
                        formula, data=model_df,
                        groups=model_df['mouse'])
                    result = model.fit(reml=True)

                # Check for convergence problems
                hessian_bad = any('Hessian' in str(w.message) for w in caught)
                grad_bad = any('Gradient optimization failed' in str(w.message)
                               for w in caught)

                if hessian_bad or grad_bad:
                    raise ValueError('Mixed model convergence issues')

                results_dict[cl][dv] = result
                pvals = result.pvalues
                stress_p = pvals.get('C(stress)[T.stress]', np.nan)

            except Exception:
                # Fallback: OLS with clustered standard errors by mouse
                model_type = 'OLS-cl'
                try:
                    import statsmodels.formula.api as smf
                    ols = smf.ols(formula, data=model_df).fit(
                        cov_type='cluster',
                        cov_kwds={'groups': model_df['mouse']})
                    results_dict[cl][dv] = ols
                    stress_p = ols.pvalues.get('C(stress)[T.stress]', np.nan)
                except Exception as e2:
                    print(f'  {dv}: both models failed - {e2}')
                    continue

            summary_line = (
                f'{dv:30s}  p={stress_p:.4f} '
                f'{_get_significance_stars(stress_p):4s}  '
                f'naive={naive_m:+10.3f}  stress={stress_m:+10.3f}  '
                f'delta={delta:+10.3f}'
                f'  [{model_type}]'
            )
            summaries.append(summary_line)
            print(f'  {summary_line}')

        # Write summary
        summary_path = os.path.join(
            output_dir, f'phasePlotV3_firstSpike_summary_C{cl}.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f'Phase Plot V3 - First Spike Only - Cluster {cl}\n')
            f.write(f'{n_cells} cells from {n_mice} mice\n')
            f.write(f'Model: DV ~ stress, random = 1|mouse\n')
            f.write('=' * 70 + '\n\n')
            for line in summaries:
                f.write(line + '\n')
            f.write('\n\n')
            for dv, res in results_dict[cl].items():
                f.write(f'\n{"-" * 70}\n{dv}\n{"-" * 70}\n')
                f.write(res.summary().as_text())
                f.write('\n')

    return results_dict


# =====================================================================
# 5.  PLOTTING
# =====================================================================

def plot_phase_plots(cell_df, output_dir='.'):
    """Generate per-cluster phase plot figures (mean ± SEM, naive vs stress).

    Creates:
    - Per-cluster, first spike only
    - Per-cluster, spike 1 vs spike 3 vs spike 5 overlay
    - Accommodation plot: max_dvdt vs spike_pos by stress
    """
    os.makedirs(output_dir, exist_ok=True)
    clusters = sorted(cell_df['cluster'].unique())

    # ── 1. First-spike phase plots per cluster ────────────────────────
    fig, axes = plt.subplots(1, len(clusters), figsize=(5 * len(clusters), 5),
                             sharey=True)
    if len(clusters) == 1:
        axes = [axes]

    first_spikes = cell_df[cell_df['spike_pos'] == 1]

    for i, cl in enumerate(clusters):
        ax = axes[i]
        cl_data = first_spikes[first_spikes['cluster'] == cl]

        for stress, color, ls in [('naive', 'black', '-'),
                                   ('stress', 'red', '--')]:
            sub = cl_data[cl_data['stress'] == stress]
            if sub.empty:
                continue
            vs = np.vstack(sub['V'].values)
            dvs = np.vstack(sub['dV_dt'].values) / 1000  # → V/s
            mean_v = np.mean(vs, axis=0)
            mean_dv = np.mean(dvs, axis=0)
            sem_dv = np.std(dvs, axis=0) / np.sqrt(len(sub))

            ax.plot(mean_v, mean_dv, color=color, ls=ls, lw=2,
                    label=f'{stress} (n={len(sub)})')
            ax.fill_between(mean_v, mean_dv - sem_dv, mean_dv + sem_dv,
                            color=color, alpha=0.2)

        ax.set_title(f'Cluster {cl}')
        ax.set_xlabel('V (mV)')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel('dV/dt (V/s)')
    plt.suptitle('First spike phase plots (first current >=5/>=3/>=1 spikes)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phasePlotV3_firstSpike.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # ── 2. Spike position overlay per cluster ─────────────────────────
    positions_to_show = [1, 3, 5]
    positions_available = sorted(cell_df['spike_pos'].unique())
    positions_to_show = [p for p in positions_to_show
                         if p in positions_available]

    if len(positions_to_show) > 1:
        fig, axes = plt.subplots(len(clusters), 2,
                                 figsize=(12, 5 * len(clusters)),
                                 sharey='row')
        if len(clusters) == 1:
            axes = axes.reshape(1, -1)

        cmap = plt.cm.viridis
        pos_colors = {p: cmap(j / max(1, len(positions_to_show) - 1))
                      for j, p in enumerate(positions_to_show)}

        for i, cl in enumerate(clusters):
            for j, stress in enumerate(['naive', 'stress']):
                ax = axes[i, j]
                cl_stress = cell_df[(cell_df['cluster'] == cl) &
                                    (cell_df['stress'] == stress)]

                for pos in positions_to_show:
                    sub = cl_stress[cl_stress['spike_pos'] == pos]
                    if sub.empty:
                        continue
                    vs = np.vstack(sub['V'].values)
                    dvs = np.vstack(sub['dV_dt'].values) / 1000
                    mean_v = np.mean(vs, axis=0)
                    mean_dv = np.mean(dvs, axis=0)
                    ax.plot(mean_v, mean_dv, color=pos_colors[pos],
                            lw=2, label=f'spike {pos} (n={len(sub)})')

                ax.set_title(f'C{cl} — {stress}')
                ax.set_xlabel('V (mV)')
                ax.legend(fontsize=8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(alpha=0.3)

            axes[i, 0].set_ylabel('dV/dt (V/s)')

        plt.suptitle('Phase plot accommodation (spike 1 vs 3 vs 5)',
                     fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                                 'phasePlotV3_accommodation.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # ── 3. Accommodation trajectories: all scalar features vs spike pos ─
    trajectory_features = [
        ('max_dvdt_Vs', 'Max dV/dt (V/s)'),
        ('min_dvdt_Vs', 'Min dV/dt (V/s)'),
        ('peak_v', 'Peak voltage (mV)'),
        ('trough_v', 'Trough voltage (mV)'),
        ('threshold_v', 'Threshold voltage (mV)'),
        ('upstroke_downstroke_ratio', 'Upstroke/downstroke ratio'),
        ('width', 'Width (s)'),
        ('loop_area', 'Loop area'),
    ]

    min_per_group = 3

    for feat, ylabel in trajectory_features:
        fig, axes = plt.subplots(1, len(clusters),
                                 figsize=(5 * len(clusters), 4), sharey=True)
        if len(clusters) == 1:
            axes = [axes]

        for i, cl in enumerate(clusters):
            ax = axes[i]
            cl_data = cell_df[cell_df['cluster'] == cl]

            for stress, color, marker in [('naive', 'black', 'o'),
                                           ('stress', 'red', 's')]:
                sub = cl_data[cl_data['stress'] == stress]
                if sub.empty:
                    continue

                # Filter positions with enough cells in BOTH groups
                g = sub.groupby('spike_pos')[feat]
                counts = g.count()
                # Also check the other group's counts at each position
                other_stress = 'stress' if stress == 'naive' else 'naive'
                other_sub = cl_data[cl_data['stress'] == other_stress]
                other_counts = other_sub.groupby('spike_pos')[feat].count()

                valid_pos = [p for p in counts.index
                             if counts.get(p, 0) >= min_per_group
                             and other_counts.get(p, 0) >= min_per_group]

                sub_valid = sub[sub['spike_pos'].isin(valid_pos)]
                if sub_valid.empty:
                    continue

                g = sub_valid.groupby('spike_pos')[feat]
                means = g.mean().reset_index(name='mean')
                means['sem'] = g.apply(
                    lambda x: x.std() / np.sqrt(len(x))).values

                ax.errorbar(means['spike_pos'], means['mean'],
                            yerr=means['sem'], color=color, marker=marker,
                            capsize=3, lw=1.5, label=f'{stress}')

            ax.set_title(f'Cluster {cl}')
            ax.set_xlabel('Spike position in train')
            ax.legend(fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(alpha=0.3)

        axes[0].set_ylabel(ylabel)
        plt.suptitle(f'{ylabel} across spike train', fontsize=13)
        plt.tight_layout()
        feat_fname = feat.replace('/', '_')
        plt.savefig(os.path.join(output_dir,
                                 f'phasePlotV3_trajectory_{feat_fname}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    print(f'[phasePlotV3] Plots saved to {output_dir}')


def plot_spike_waveforms(cell_df, output_dir='.'):
    """Generate per-cluster spike waveform figures (V vs time).

    Mirrors plot_phase_plots but plots voltage waveform instead of
    dV/dt vs V. Time axis is ±5 ms around peak (100 pts at 10 kHz).

    Creates:
    - Per-cluster, first spike only (naive vs stress)
    - Per-cluster, spike 1 vs 3 vs 5 overlay
    - Accommodation plot: peak_v vs spike_pos by stress
    """
    os.makedirs(output_dir, exist_ok=True)
    clusters = sorted(cell_df['cluster'].unique())
    t_ms = np.linspace(-5, 5, 100)

    # ── 1. First-spike waveforms per cluster ──────────────────────────
    fig, axes = plt.subplots(1, len(clusters),
                             figsize=(5 * len(clusters), 5), sharey=True)
    if len(clusters) == 1:
        axes = [axes]

    first_spikes = cell_df[cell_df['spike_pos'] == 1]

    for i, cl in enumerate(clusters):
        ax = axes[i]
        cl_data = first_spikes[first_spikes['cluster'] == cl]

        for stress, color, ls in [('naive', 'black', '-'),
                                   ('stress', 'red', '--')]:
            sub = cl_data[cl_data['stress'] == stress]
            if sub.empty:
                continue
            vs = np.vstack(sub['V'].values)
            mean_v = np.mean(vs, axis=0)
            sem_v = np.std(vs, axis=0) / np.sqrt(len(sub))

            ax.plot(t_ms, mean_v, color=color, ls=ls, lw=2,
                    label=f'{stress} (n={len(sub)})')
            ax.fill_between(t_ms, mean_v - sem_v, mean_v + sem_v,
                            color=color, alpha=0.2)

        ax.set_title(f'Cluster {cl}')
        ax.set_xlabel('Time from peak (ms)')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel('V (mV)')
    plt.suptitle('First spike waveforms (first current >=5/>=3/>=1 spikes)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phasePlotV3_firstSpike_waveform.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # ── 2. Spike position overlay per cluster ─────────────────────────
    positions_to_show = [1, 3, 5]
    positions_available = sorted(cell_df['spike_pos'].unique())
    positions_to_show = [p for p in positions_to_show
                         if p in positions_available]

    if len(positions_to_show) > 1:
        fig, axes = plt.subplots(len(clusters), 2,
                                 figsize=(12, 5 * len(clusters)),
                                 sharey='row')
        if len(clusters) == 1:
            axes = axes.reshape(1, -1)

        cmap = plt.cm.viridis
        pos_colors = {p: cmap(j / max(1, len(positions_to_show) - 1))
                      for j, p in enumerate(positions_to_show)}

        for i, cl in enumerate(clusters):
            for j, stress in enumerate(['naive', 'stress']):
                ax = axes[i, j]
                cl_stress = cell_df[(cell_df['cluster'] == cl) &
                                    (cell_df['stress'] == stress)]

                for pos in positions_to_show:
                    sub = cl_stress[cl_stress['spike_pos'] == pos]
                    if sub.empty:
                        continue
                    vs = np.vstack(sub['V'].values)
                    mean_v = np.mean(vs, axis=0)
                    ax.plot(t_ms, mean_v, color=pos_colors[pos],
                            lw=2, label=f'spike {pos} (n={len(sub)})')

                ax.set_title(f'C{cl} - {stress}')
                ax.set_xlabel('Time from peak (ms)')
                ax.legend(fontsize=8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(alpha=0.3)

            axes[i, 0].set_ylabel('V (mV)')

        plt.suptitle('Spike waveform accommodation (spike 1 vs 3 vs 5)',
                     fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                                 'phasePlotV3_accommodation_waveform.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    print(f'[phasePlotV3] Waveform plots saved to {output_dir}')


def plot_pca_loadings(pca, scaler, cell_df, output_dir='.'):
    """Plot PCA component loadings and perturbation waveforms.

    For each PC, shows:
    - Left: the loading vector (which time points matter)
    - Right: the mean waveform perturbed by +/- 2 SD along that PC,
      so you can see what "high PC2" vs "low PC2" looks like as an
      actual dV/dt waveform.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_components = pca.n_components_

    # Mean waveform in scaled space
    mean_scaled = np.zeros(pca.components_.shape[1])
    # Mean waveform in original units (V/s)
    mean_orig = scaler.mean_ / 1000.0
    scale_factor = scaler.scale_ / 1000.0

    # Time axis (sample index — ±50 around peak at 10 kHz = ±5 ms)
    t_ms = np.linspace(-5, 5, 100)

    fig, axes = plt.subplots(n_components, 2,
                             figsize=(12, 3 * n_components))
    if n_components == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_components):
        loading = pca.components_[i]
        var_pct = pca.explained_variance_ratio_[i] * 100

        # Left: loading vector
        ax_l = axes[i, 0]
        ax_l.plot(t_ms, loading, color='#2c7fb8', lw=1.5)
        ax_l.fill_between(t_ms, 0, loading, alpha=0.3, color='#2c7fb8')
        ax_l.axhline(0, color='gray', lw=0.5)
        ax_l.set_ylabel(f'PC{i+1} loading')
        ax_l.set_title(f'PC{i+1} loading ({var_pct:.1f}% var)')
        ax_l.spines['top'].set_visible(False)
        ax_l.spines['right'].set_visible(False)

        # Right: mean waveform ± 2 SD perturbation along this PC
        ax_r = axes[i, 1]
        for sign, label, color, ls in [(0, 'mean', 'black', '-'),
                                        (+2, '+2 SD', '#e41a1c', '--'),
                                        (-2, '-2 SD', '#2c7fb8', '--')]:
            perturbed_scaled = mean_scaled + sign * np.sqrt(
                pca.explained_variance_[i]) * loading
            perturbed_orig = (perturbed_scaled * scaler.scale_ +
                              scaler.mean_) / 1000.0
            ax_r.plot(t_ms, perturbed_orig, color=color, ls=ls, lw=1.5,
                      label=label)

        ax_r.set_ylabel('dV/dt (V/s)')
        ax_r.set_title(f'PC{i+1}: what it captures')
        ax_r.legend(fontsize=8)
        ax_r.spines['top'].set_visible(False)
        ax_r.spines['right'].set_visible(False)

    axes[-1, 0].set_xlabel('Time from peak (ms)')
    axes[-1, 1].set_xlabel('Time from peak (ms)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phasePlotV3_PCA_loadings.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[phasePlotV3] PCA loading plots saved to {output_dir}')


# =====================================================================
# 6.  CSV EXPORT
# =====================================================================

def export_csvs(spike_df, cell_df, output_dir='.'):
    """Write analysis-ready CSVs."""
    os.makedirs(output_dir, exist_ok=True)

    # Scalar features per spike (raw, before cell averaging)
    scalar_cols = ['cell_name', 'mouse', 'sex', 'stress', 'cluster',
                   'target_current', 'sweep', 'spike_pos', 'peak_v',
                   'threshold_v', 'trough_v', 'upstroke', 'downstroke',
                   'upstroke_v', 'downstroke_v', 'width',
                   'max_dvdt', 'min_dvdt', 'max_dvdt_Vs', 'min_dvdt_Vs',
                   'upstroke_downstroke_ratio', 'loop_area']
    spike_scalar = spike_df[[c for c in scalar_cols if c in spike_df.columns]]
    spike_scalar.to_csv(os.path.join(output_dir,
                                     'phasePlotV3_spikes_scalar.csv'),
                        index=False)

    # Cell-level scalar + PC scores
    exclude = ['dV_dt', 'V']
    cell_scalar = cell_df[[c for c in cell_df.columns if c not in exclude]]
    cell_scalar.to_csv(os.path.join(output_dir,
                                    'phasePlotV3_cell_level.csv'),
                       index=False)

    # Cell-level waveforms (exploded to columns for Prism etc.)
    dv_expanded = pd.DataFrame(
        np.vstack(cell_df['dV_dt'].values),
        columns=[f'dV_dt_{i}' for i in range(100)])
    v_expanded = pd.DataFrame(
        np.vstack(cell_df['V'].values),
        columns=[f'V_{i}' for i in range(100)])
    waveform_df = pd.concat([
        cell_df[['cell_name', 'mouse', 'stress', 'cluster',
                 'spike_pos']].reset_index(drop=True),
        dv_expanded, v_expanded
    ], axis=1)
    waveform_df.to_csv(os.path.join(output_dir,
                                    'phasePlotV3_waveforms.csv'),
                       index=False)

    print(f'[phasePlotV3] CSVs saved to {output_dir}')


# =====================================================================
# 7.  MAIN ENTRY POINT
# =====================================================================

def getPhasePlotV3(mouseList, qc_n_sd=1.0, n_pca_components=4,
                   do_pca=False, output_dir='.'):
    """Run the full phase plot v3 pipeline.

    Identical to V2 except for current/spike selection: per cell, group
    firing-rate sweeps by current and scan ascending; the first current
    where EVERY sweep fired >=5 spikes is used with positions 1, 3, 5;
    else every sweep >=3 with 1, 3; else every sweep >=1 with 1. All
    sweeps at that current are averaged per position (as in V2). There is
    no rheobase/offset or max_spike_pos parameter — the spike-count tiers
    drive both the chosen current and the spike positions.

    Parameters
    ----------
    mouseList : dict
        The standard ephysMouseDict.
    qc_n_sd : float
        Reject spikes with peak_v > this many SDs from group mean
        (cluster x stress). Same strict QC as V2 (default 1.0 SD).
    n_pca_components : int
        Number of PCA components to extract (only used if do_pca=True).
    do_pca : bool
        If True, run functional PCA, include PC scores in models and
        CSVs, and generate PCA loading plots. Default False.
    output_dir : str
        Directory for output files.

    Returns
    -------
    spike_df, cell_df, pca, results_dict
    """
    print('=' * 60)
    print('Phase Plot V3 Pipeline')
    print('=' * 60)

    # Step 1-2: Select sweep/positions, extract, and QC
    spike_df = extract_phase_plots(mouseList, qc_n_sd=qc_n_sd)
    if spike_df.empty:
        return spike_df, None, None, None

    # Step 3: Cell-level averaging
    cell_df = average_to_cell_level(spike_df)

    # Step 4: Functional PCA (optional)
    pca = None
    scaler = None
    if do_pca:
        cell_df, pca, scaler = run_functional_pca(cell_df,
                                                   n_components=n_pca_components)

    # Step 5: Export CSVs
    export_csvs(spike_df, cell_df, output_dir=output_dir)

    # Step 6: Plots
    plot_phase_plots(cell_df, output_dir=output_dir)
    plot_spike_waveforms(cell_df, output_dir=output_dir)
    if do_pca and pca is not None:
        plot_pca_loadings(pca, scaler, cell_df, output_dir=output_dir)

    # Step 7: Models (full train: stress * spike_pos)
    results_dict = run_phase_plot_models(cell_df, output_dir=output_dir)

    # Step 8: First-spike-only models (stress only)
    first_spike_results = run_first_spike_models(cell_df,
                                                  output_dir=output_dir)

    print('\n' + '=' * 60)
    print('Phase Plot V3 Pipeline - Complete')
    print('=' * 60)

    return spike_df, cell_df, pca, results_dict