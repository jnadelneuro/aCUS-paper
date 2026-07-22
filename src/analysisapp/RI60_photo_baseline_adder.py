import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from _config import DATA_ROOT

# ============== USER PARAMETERS ==============
PROCESSED_DIR = os.path.join(DATA_ROOT, "ALL_RI60_PHOTO", "photometry", "processed")

EVENTS = {"InactiveNP", "RewardPE", "UnrewardedNP", "RewardNP"}
SIGNAL_TYPES = {"dff", "z_score"}
REGIONS = {"DMS", "TS", "DA"}

BASELINE_START = -10
BASELINE_END = -5

# Output naming tag (keeps you from overwriting GuPPy's existing files)
OUT_TAG = f"baselineCorrected"
# =============================================


def baseline_correction_like_guppy(arr, time_axis, baseline_start, baseline_end):
    """Matches GuPPy's baselineCorrection(): mean over [start, end) then subtract."""
    if baseline_start == 0 and baseline_end == 0:
        return arr.copy()

    start_idx = np.where(time_axis >= baseline_start)[0][0]
    end_idx = np.where(time_axis >= baseline_end)[0][0]

    baseline = np.nanmean(arr[start_idx:end_idx])
    return arr - baseline


def correct_psth_h5(in_path, out_path):
    df = pd.read_hdf(in_path, key="df", mode="r")

    required = {"timestamps", "mean", "err"}
    if not required.issubset(df.columns):
        raise Warning(
            f"{in_path} is missing expected columns {required}. "
            f"Found columns: {list(df.columns)}"
        )

    time_axis = df["timestamps"].to_numpy()

    # event occurrence columns are everything except the trailing summary columns
    trial_cols = [c for c in df.columns if c not in ("timestamps", "mean", "err")]

    start_idx = np.where(time_axis >= BASELINE_START)[0][0]
    end_idx = np.where(time_axis >= BASELINE_END)[0][0]

    out_df = df.copy()

    # baseline correct each trial/occurrence column
    for c in trial_cols:
        arr = out_df[c].to_numpy()
        if BASELINE_START == 0 and BASELINE_END == 0:
            corrected = arr
        else:
            baseline = np.nanmean(arr[start_idx:end_idx])
            corrected = arr - baseline
        out_df[c] = corrected

    # recompute mean and err from corrected trials
    trials_mat = out_df[trial_cols].to_numpy()
    out_df["mean"] = np.nanmean(trials_mat, axis=1)

    # GuPPy labels this "err"; typically SEM is std/sqrt(n)
    # We'll compute SEM using the non-nan count per row.

    n = np.sum(~np.isnan(trials_mat), axis=1)
    std = np.nanstd(trials_mat, axis=1)
    out_df["err"] = np.divide(std, np.sqrt(n), out=np.full_like(std, np.nan), where=n > 0)

    out_df.to_hdf(out_path, key="df", mode="w")
def parse_uncorrected_filename(fname):
    """
    Expected pattern (region always last):
      {event}_{region? optional stuff}_baselineUncorrected_{sig}_{region}.h5

    But per your constraint, region is ALWAYS last and in {DMS,TS,DA}, so we only rely on the tail:
      *_baselineUncorrected_{sig}_{region}.h5
    """
    if not fname.endswith(".h5"):
        return None

    stem = fname[:-3]  # strip ".h5"
    parts = stem.split("_")
    if len(parts) < 4:
        return None

    region = parts[-1]
    if "z_score" in fname:
        sig = "z_score"
    elif "dff" in fname:
        sig = "dff"
    else:
        return None
        sig = "z_score"


    if region not in REGIONS:
        return None
    if sig not in SIGNAL_TYPES:
        return None
    if "baselineUncorrected" not in stem:
        return None

    event = parts[0]
    if event not in EVENTS:
        return None

    # Confirm the exact marker is present in expected position-ish
    # (We won't over-enforce; just ensure it exists.)
    return event, sig, region


def process_output_dir(output_dir):
    for fname in os.listdir(output_dir):
        parsed = parse_uncorrected_filename(fname)
        if not parsed:
            continue

        event, sig, region = parsed
        in_path = os.path.join(output_dir, fname)

        out_name = f"{event}_{region}_{OUT_TAG}_{sig}_{region}.h5"
        out_path = os.path.join(output_dir, out_name)

        # if os.path.exists(out_path):
        #     continue

        correct_psth_h5(in_path, out_path)
        print(f"Saved: {out_path}")


session_dirs = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*")))

for session_dir in session_dirs:
    output_dirs = glob.glob(os.path.join(session_dir, "*_output_*"))
    for output_dir in output_dirs:
        process_output_dir(output_dir)

print("Done.")


