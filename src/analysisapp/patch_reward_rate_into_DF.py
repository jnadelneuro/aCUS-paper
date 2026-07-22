"""
Patch `reward_rate` (and `instant_poke_rate`) in the existing photoDF feathers
WITHOUT re-running createPhotoDF / JIMMYv2.

Pipeline assumed:
    behaviorData.pickle
      -> create_RR_arrays() rewrites reward_rate_trace + poke_rate_trace
      -> behaviorData_RR.pickle
      -> patched into each photoDF feather via (mouse, date, timestamp) lookup

Targets (any that exist are patched in place; a .bak copy is written first):
    photoDataFrame_baseLineCorr.feather
    photoDF_R.feather
    photoDF_R_with_rates.feather
    photoDF_R_with_weights.feather

Run from this folder:
    python patch_reward_rate.py
"""

import os
import shutil
import pickle
import numpy as np
import pandas as pd
import pyarrow.feather as feather

from RI60_behavior_functions import create_RR_arrays
from _config import RI60_DIR

DATA_DIR = RI60_DIR
PICKLE   = 'behaviorData_RR.pickle'

BASE_FEATHER = 'photoDataFrame_baseLineCorr.feather'
R_FEATHERS = [
    'photoDF_R.feather',
    'photoDF_R_with_rates.feather',
    'photoDF_R_with_weights.feather',
]

# (mouse, date, recordingLoc, event, event_number) uniquely identifies a row
# across the base feather and all R-export feathers (they are 1:1 derived).
JOIN_KEYS = ['mouse', 'date', 'recordingLoc', 'event', 'event_number']

POKE_EVENTS = {'ReNP', 'UnNP'}


def _norm_mouse(subject):
    s = str(subject)
    return s.replace('.', '_') if '.' in s else s


def _norm_date(start_date):
    s = str(start_date)
    return s.split('.')[0] if '.' in s else s


def _active_side(session):
    if float(session['Right Rewards'][0]) > 0:
        return 'Right'
    if float(session['left rewards'][0]) > 0:
        return 'Left'
    return None


def build_lookup(behaviorData):
    """(mouse, date) -> (allNP_TS np.ndarray, rr np.ndarray, pr np.ndarray)."""
    lookup = {}
    skipped = 0
    for s in behaviorData:
        active = _active_side(s)
        if active is None:
            skipped += 1
            continue
        ts_col = f'{active}_nose_timestamps'
        if ts_col not in s.columns:
            skipped += 1
            continue
        ts = s[ts_col].dropna().values
        rr_raw = s.get('reward_rate_trace', None)
        pr_raw = s.get('poke_rate_trace',  None)
        if rr_raw is None or not isinstance(rr_raw, (np.ndarray, pd.Series)):
            skipped += 1
            continue
        rr = np.asarray(rr_raw)
        rr = rr[~pd.isna(rr)] if rr.dtype.kind == 'f' else rr
        if pr_raw is not None and isinstance(pr_raw, (np.ndarray, pd.Series)):
            pr = np.asarray(pr_raw)
            pr = pr[~pd.isna(pr)] if pr.dtype.kind == 'f' else pr
        else:
            pr = np.full_like(rr, np.nan, dtype=float)
        n = min(len(ts), len(rr), len(pr))
        if n == 0:
            skipped += 1
            continue
        key = (_norm_mouse(s['Subject'][0]), _norm_date(s['Start Date'][0]))
        lookup[key] = (ts[:n].astype(float), rr[:n].astype(float), pr[:n].astype(float))
    print(f'  built lookup for {len(lookup)} sessions (skipped {skipped})')
    return lookup


def patch_df(df, lookup):
    """Return (new_rr_series, new_pr_series, stats_dict)."""
    new_rr = np.full(len(df), np.nan)
    new_pr = np.full(len(df), np.nan)

    matched = 0
    no_session = 0
    no_ts_hit = 0
    not_poke = 0

    mouse_arr = df['mouse'].astype(str).values
    date_arr  = df['date'].astype(str).values
    ts_arr    = df['timestamp'].astype(float).values
    ev_arr    = df['event'].astype(str).values

    for i in range(len(df)):
        if ev_arr[i] not in POKE_EVENTS:
            not_poke += 1
            continue
        key = (mouse_arr[i], date_arr[i])
        pair = lookup.get(key)
        if pair is None:
            no_session += 1
            continue
        ts, rr, pr = pair
        hits = np.where(ts == ts_arr[i])[0]
        if len(hits) == 0:
            no_ts_hit += 1
            continue
        j = hits[0]
        new_rr[i] = rr[j]
        new_pr[i] = pr[j]
        matched += 1

    return new_rr, new_pr, {
        'rows': len(df),
        'matched': matched,
        'no_session': no_session,
        'no_ts_hit': no_ts_hit,
        'not_poke': not_poke,
    }


def main(regenerate_pickle=True, backup=True):
    os.chdir(DATA_DIR)

    if regenerate_pickle:
        print('regenerating behaviorData_RR.pickle ...')
        create_RR_arrays()

    print(f'loading {PICKLE} ...')
    with open(PICKLE, 'rb') as f:
        behaviorData = pickle.load(f)

    print('building (mouse, date) -> (ts, rr, pr) lookup ...')
    lookup = build_lookup(behaviorData)

    # ---- Pass 1: patch the base feather using timestamp lookup ----
    if not os.path.exists(BASE_FEATHER):
        raise FileNotFoundError(
            f'{BASE_FEATHER} not found in {DATA_DIR} — '
            'cannot patch downstream feathers without it'
        )

    print(f'\npatching {BASE_FEATHER} (timestamp lookup) ...')
    base = feather.read_feather(BASE_FEATHER)
    for col in ('mouse', 'date', 'timestamp', 'event'):
        if col not in base.columns:
            raise RuntimeError(f'{BASE_FEATHER} is missing required column `{col}`')

    new_rr, new_pr, stats = patch_df(base, lookup)
    print(f'  rows={stats["rows"]}  matched={stats["matched"]}  '
          f'no_session={stats["no_session"]}  no_ts_hit={stats["no_ts_hit"]}  '
          f'not_poke={stats["not_poke"]}')

    _backup_once(BASE_FEATHER, backup)
    base['reward_rate'] = new_rr
    if 'instant_poke_rate' in base.columns:
        base['instant_poke_rate'] = new_pr
    base.reset_index(drop=True).to_feather(BASE_FEATHER)
    print(f'  wrote {BASE_FEATHER}')

    # ---- Pass 2: propagate to R-export feathers via JOIN_KEYS ----
    rates_map = (
        base[JOIN_KEYS + ['reward_rate', 'instant_poke_rate']]
        if 'instant_poke_rate' in base.columns
        else base[JOIN_KEYS + ['reward_rate']]
    ).copy()
    # Coerce join cols to consistent types for reliable merging
    for k in JOIN_KEYS:
        rates_map[k] = rates_map[k].astype(str)
    # Dedup defensively — base should already be unique on these keys
    rates_map = rates_map.drop_duplicates(subset=JOIN_KEYS, keep='first')

    for fname in R_FEATHERS:
        if not os.path.exists(fname):
            print(f'\nskip (missing): {fname}')
            continue
        print(f'\npatching {fname} (join on {JOIN_KEYS}) ...')
        df = feather.read_feather(fname)

        missing = [c for c in JOIN_KEYS if c not in df.columns]
        if missing:
            print(f'  WARNING: {fname} missing join cols {missing}; skipping')
            continue

        df_keys = df[JOIN_KEYS].astype(str)
        merged = df_keys.merge(rates_map, on=JOIN_KEYS, how='left')
        matched = merged['reward_rate'].notna().sum()
        unmatched = len(merged) - matched
        print(f'  rows={len(df)}  matched={matched}  unmatched/NaN={unmatched}')

        _backup_once(fname, backup)
        df['reward_rate'] = merged['reward_rate'].values
        if 'instant_poke_rate' in df.columns and 'instant_poke_rate' in merged.columns:
            df['instant_poke_rate'] = merged['instant_poke_rate'].values

        df.reset_index(drop=True).to_feather(fname)
        print(f'  wrote {fname}')

    print('\ndone.')


def _backup_once(fname, backup):
    if not backup:
        return
    bak = fname + '.bak'
    if not os.path.exists(bak):
        shutil.copy2(fname, bak)
        print(f'  backup -> {bak}')
    else:
        print(f'  backup already exists: {bak} (left untouched)')


if __name__ == '__main__':
    main(regenerate_pickle=True, backup=True)
