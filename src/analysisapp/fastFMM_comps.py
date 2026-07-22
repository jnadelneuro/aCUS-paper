"""
Compute between-group difference β(t) curves and pointwise CIs from separate
fastFMM fits (naive vs stress).

Assumes each fit CSV has columns:
    s                   (timepoint index, 1..N)
    beta.hat            (point estimate at s)
    CI.lower.pointwise  (95% pointwise lower)
    CI.upper.pointwise  (95% pointwise upper)

And filenames follow the convention:
    {event}_FMM_{group}_{region}_{predictor}_zscored_{component}.csv

For each (event, region, predictor) combo, pairs the stress and naive files,
computes Δβ(t) = β_stress(t) - β_naive(t) with pointwise SE combined in
quadrature (assumes independence between the two group-specific fits, which
is approximately true since they share no mice and no random effects), and
writes one diff CSV per combo plus a summary of significant windows.

Usage
-----
python compute_diff_beta.py /path/to/fmm_csv_dir /path/to/output_dir

Or from a notebook:
    from compute_diff_beta import compute_all_diffs
    compute_all_diffs('fmm_csv_dir', 'output_dir')
"""
# ============================================================================
# EDIT THESE PATHS
# ============================================================================
import os
from _config import RI60_DIR
CSV_DIR = os.path.join(RI60_DIR, "FMM_models")
OUT_DIR = os.path.join(RI60_DIR, "FMM_models", "diffs")
# ============================================================================

import os
import re
import numpy as np
import pandas as pd
from scipy.stats import norm

# ---- config ------------------------------------------------------------------

# Time axis of the fits. The R script uses x_rescale = 51 and align_x = 2,
# so sample index s = 1..N maps to t = (s - 1) / 51 - 2.
X_RESCALE = 51.0
ALIGN_X   = 2.0

ALPHA = 0.05
Z_CRIT = norm.ppf(1 - ALPHA / 2)  # 1.959964 for alpha = 0.05

# Minimum run length (in samples) for a significant stretch to count.
CONSEC_THRESH = 10

# Minimum consecutive samples for a significant stretch to count.
CONSEC_THRESH = 10


# ---- file parsing ------------------------------------------------------------

def parse_filename(fname):
    """Extract (event, group, region, predictor, is_zscored) from an FMM csv name.

    Handles predictor names with embedded underscores (e.g. reward_rate) by
    splitting suffix_parts on the 'zscored' token.
    """
    base = fname.replace('.csv', '')
    parts = base.split('_')

    anchor_idx = None
    for i in range(len(parts) - 1):
        if parts[i] in ('naive', 'stress') and parts[i + 1] in ('TS', 'DMS'):
            anchor_idx = i
            break
    if anchor_idx is None:
        return None

    prefix_parts = parts[:anchor_idx]
    group  = parts[anchor_idx]
    region = parts[anchor_idx + 1]
    suffix_parts = parts[anchor_idx + 2:]

    is_zscored = 'zscored' in suffix_parts
    if is_zscored:
        z = suffix_parts.index('zscored')
        predictor_parts = suffix_parts[:z]
    else:
        # Without zscored marker we don't know where predictor ends and
        # component begins unambiguously. Assume the standard fastFMM export
        # pattern where component == predictor (so suffix_parts is
        # [predictor_parts..., component_parts..] with component == predictor).
        # Take first half as predictor.
        half = len(suffix_parts) // 2
        predictor_parts = suffix_parts[:half] if half else suffix_parts

    predictor = '_'.join(predictor_parts)
    event     = '_'.join(prefix_parts).replace('_FMM', '').replace('FMM', '')
    # Event is the very first token in practice (ReNP / UnNP / etc.)
    event = prefix_parts[0] if prefix_parts else ''

    return {
        'event': event,
        'group': group,
        'region': region,
        'predictor': predictor,
        'is_zscored': is_zscored,
        'filename': fname,
    }


# ---- diff computation --------------------------------------------------------

def load_fit(path):
    """Load a single FMM fit CSV and compute pointwise SE from the CI."""
    df = pd.read_csv(path)
    # SE = (upper - lower) / (2 * z_crit)  — more robust to asymmetric CIs
    # than (beta - lower) / z_crit since fastFMM can produce slightly
    # asymmetric CIs at boundary timepoints.
    df['se'] = (df['CI.upper.pointwise'] - df['CI.lower.pointwise']) / (2 * Z_CRIT)
    df['t']  = (df['s'] - 1) / X_RESCALE - ALIGN_X
    return df[['s', 't', 'beta.hat', 'CI.lower.pointwise', 'CI.upper.pointwise', 'se']].copy()


def compute_diff(naive_df, stress_df):
    """Compute Δβ(t) = β_stress - β_naive with pointwise 95% CI.

    Assumes independence between the two fits: Var(Δβ) = Var(β_s) + Var(β_n).
    """
    # Align on sample index
    n = naive_df.set_index('s')
    s = stress_df.set_index('s')
    common = n.index.intersection(s.index)
    n = n.loc[common]; s = s.loc[common]

    diff = pd.DataFrame({
        's': common,
        't': n['t'].values,
        'beta_naive':  n['beta.hat'].values,
        'beta_stress': s['beta.hat'].values,
        'se_naive':    n['se'].values,
        'se_stress':   s['se'].values,
    })
    diff['delta_beta'] = diff['beta_stress'] - diff['beta_naive']
    diff['se_delta']   = np.sqrt(diff['se_stress']**2 + diff['se_naive']**2)
    diff['CI_lower']   = diff['delta_beta'] - Z_CRIT * diff['se_delta']
    diff['CI_upper']   = diff['delta_beta'] + Z_CRIT * diff['se_delta']
    diff['z_stat']     = diff['delta_beta'] / diff['se_delta']
    diff['sig']        = (diff['CI_lower'] > 0) | (diff['CI_upper'] < 0)

    # Also flag within-group significance for convenience
    diff['naive_sig']  = (n['CI.lower.pointwise'].values > 0) | (n['CI.upper.pointwise'].values < 0)
    diff['stress_sig'] = (s['CI.lower.pointwise'].values > 0) | (s['CI.upper.pointwise'].values < 0)

    return diff


def find_sig_runs(sig_bool, t, min_len=CONSEC_THRESH):
    """Return (t_start, t_end, n_samples) for each run of consecutive True
    values in `sig_bool` with length >= min_len.
    """
    sig = np.asarray(sig_bool, dtype=bool)
    if sig.size == 0:
        return []
    padded = np.concatenate([[False], sig, [False]])
    edges = np.diff(padded.astype(int))
    starts = np.where(edges == 1)[0]
    ends   = np.where(edges == -1)[0]
    return [(float(t[a]), float(t[b - 1]), int(b - a))
            for a, b in zip(starts, ends) if (b - a) >= min_len]


def apply_consec_filter(sig_bool, min_len=CONSEC_THRESH):
    """Zero out any significant stretch shorter than min_len samples."""
    sig = np.asarray(sig_bool, dtype=bool)
    if sig.size == 0:
        return sig.copy()
    padded = np.concatenate([[False], sig, [False]])
    edges = np.diff(padded.astype(int))
    starts = np.where(edges == 1)[0]
    ends   = np.where(edges == -1)[0]
    out = np.zeros_like(sig)
    for a, b in zip(starts, ends):
        if (b - a) >= min_len:
            out[a:b] = True
    return out


def summarize_curve(diff_df):
    """Compact summary of a Δβ(t) curve. Uses CONSEC_THRESH to filter short
    stretches of pointwise significance.
    """
    t = diff_df['t'].values

    n_sig = apply_consec_filter(diff_df['naive_sig'].values)
    s_sig = apply_consec_filter(diff_df['stress_sig'].values)
    d_sig = apply_consec_filter(diff_df['sig'].values)

    n_runs = find_sig_runs(n_sig, t)
    s_runs = find_sig_runs(s_sig, t)
    d_runs = find_sig_runs(d_sig, t)

    delta = diff_df['delta_beta'].values
    if d_sig.any():
        idx_max = np.argmax(np.abs(np.where(d_sig, delta, 0)))
        peak_t = float(t[idx_max])
        peak_delta = float(delta[idx_max])
    else:
        peak_t = np.nan
        peak_delta = np.nan

    def runs_to_str(runs):
        if not runs:
            return ''
        return '; '.join(f'[{a:.2f},{b:.2f}]s ({n})' for a, b, n in runs)

    return {
        'n_samples': len(diff_df),
        'naive_sig_runs':  runs_to_str(n_runs),
        'stress_sig_runs': runs_to_str(s_runs),
        'diff_sig_runs':   runs_to_str(d_runs),
        'frac_naive_sig':  float(n_sig.mean()),
        'frac_stress_sig': float(s_sig.mean()),
        'frac_diff_sig':   float(d_sig.mean()),
        'diff_peak_t': peak_t,
        'diff_peak_value': peak_delta,
    }


def plot_pair(diff_df, title, out_path):
    """Plot naive + stress β(t) with pointwise CIs and pointwise significance
    bars. Blue = naive, red = stress, purple = stress ≠ naive.
    """
    import matplotlib.pyplot as plt

    t = diff_df['t'].values

    naive_lo = diff_df['beta_naive'].values  - Z_CRIT * diff_df['se_naive'].values
    naive_hi = diff_df['beta_naive'].values  + Z_CRIT * diff_df['se_naive'].values
    stress_lo = diff_df['beta_stress'].values - Z_CRIT * diff_df['se_stress'].values
    stress_hi = diff_df['beta_stress'].values + Z_CRIT * diff_df['se_stress'].values

    n_sig = apply_consec_filter(diff_df['naive_sig'].values)
    s_sig = apply_consec_filter(diff_df['stress_sig'].values)
    d_sig = apply_consec_filter(diff_df['sig'].values)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, diff_df['beta_naive'].values,  color='blue', linewidth=2, label='naive β(t)')
    ax.plot(t, diff_df['beta_stress'].values, color='red',  linewidth=2, label='stress β(t)')
    ax.fill_between(t, naive_lo,  naive_hi,  color='blue', alpha=0.2)
    ax.fill_between(t, stress_lo, stress_hi, color='red',  alpha=0.2)

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    h1 = y_max + y_range * 0.05
    h2 = y_max + y_range * 0.10
    h3 = y_max + y_range * 0.15

    ax.plot(t, np.where(n_sig, h1, np.nan), '|', color='blue',   markersize=3, label='naive ≠ 0')
    ax.plot(t, np.where(s_sig, h2, np.nan), '|', color='red',    markersize=3, label='stress ≠ 0')
    ax.plot(t, np.where(d_sig, h3, np.nan), '|', color='purple', markersize=3, label='stress ≠ naive')

    ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('β(t)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.set_ylim(y_min, y_max + y_range * 0.20)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


# ---- orchestrator ------------------------------------------------------------

def compute_all_diffs(csv_dir, out_dir, only_zscored=True, verbose=True):
    os.makedirs(out_dir, exist_ok=True)

    # Index all the fit files
    all_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv') and 'Intercept' not in f]
    parsed = []
    for f in all_files:
        p = parse_filename(f)
        if p is None:
            continue
        if only_zscored and not p['is_zscored']:
            continue
        parsed.append(p)

    if verbose:
        print(f'Parsed {len(parsed)} fit files '
              f'({"zscored only" if only_zscored else "all"})')

    # Group by (event, region, predictor, is_zscored) so stress and naive pair up
    from collections import defaultdict
    pairs = defaultdict(dict)
    for p in parsed:
        key = (p['event'], p['region'], p['predictor'], p['is_zscored'])
        pairs[key][p['group']] = p['filename']

    # Compute diffs, plot, and collect summaries
    all_summaries = []
    n_pairs = 0
    for key, groupdict in sorted(pairs.items()):
        event, region, predictor, is_zscored = key
        if 'naive' not in groupdict or 'stress' not in groupdict:
            if verbose:
                missing = {'naive', 'stress'} - set(groupdict.keys())
                print(f'  SKIP {key}: missing {missing}')
            continue

        naive_path  = os.path.join(csv_dir, groupdict['naive'])
        stress_path = os.path.join(csv_dir, groupdict['stress'])
        try:
            nf = load_fit(naive_path)
            sf = load_fit(stress_path)
        except Exception as e:
            if verbose:
                print(f'  ERROR loading {key}: {e}')
            continue

        diff = compute_diff(nf, sf)

        zs_tag = '_zscored' if is_zscored else ''
        base = f'{event}_{region}_{predictor}{zs_tag}'
        diff.to_csv(os.path.join(out_dir, f'{base}_diff.csv'), index=False)

        # Plot
        title = f'{event}  {region}  {predictor}{zs_tag}'
        plot_pair(diff, title, os.path.join(out_dir, f'{base}_diff.png'))

        # Continuous summary
        summ = summarize_curve(diff)
        summ.update({
            'event': event,
            'region': region,
            'predictor': predictor,
            'is_zscored': is_zscored,
        })
        all_summaries.append(summ)
        n_pairs += 1

        if verbose:
            runs = summ['diff_sig_runs'] or '(none)'
            print(f'  {event:5s} {region:3s} {predictor:16s}  stress≠naive: {runs}')

    # Combined summary
    if all_summaries:
        combined = pd.DataFrame(all_summaries)
        cols = ['event', 'region', 'predictor', 'is_zscored', 'n_samples',
                'naive_sig_runs', 'stress_sig_runs', 'diff_sig_runs',
                'frac_naive_sig', 'frac_stress_sig', 'frac_diff_sig',
                'diff_peak_t', 'diff_peak_value']
        combined = combined[cols]
        combined_path = os.path.join(out_dir, 'ALL_diffs_summary.csv')
        combined.to_csv(combined_path, index=False)
        if verbose:
            print(f'\nWrote {n_pairs} diff CSVs + {n_pairs} plots + summary to {out_dir}')
            print(f'Combined summary: {combined_path}')

    return all_summaries


if __name__ == '__main__':
    compute_all_diffs(CSV_DIR, OUT_DIR)