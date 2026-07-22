"""
Representative-session traces for the RI60 figure.

Plots three per-poke measures for representative RI60 sessions, x = session
time in minutes (NO downsampling -- one point per poke):
  1. cumulative active pokes   -> a median-#-pokes session
  2. reward rate (rewards/min) -> a high-dynamism session
  3. instantaneous poke rate   -> the same high-dynamism session
     (pokes/min)

Reward rate and instantaneous poke rate use the lab's canonical leaky-integrator
definition (calculate_reward_rate, tau = 60 s) -- identical to create_RR_arrays
in RI60_behavior_functions.py. Only the data/IO layer differs: traces are
computed straight from RI30_RI60_dataFrame.feather instead of behaviorData.pickle.

Outputs PNGs + Prism-ready XY CSVs (X = time_min, Y = measure) to the Fig 5 folder.
VSCode play button.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _config import RI60_DIR, FIG_OUT

# ===========================================================
# Config
# ===========================================================
FEATHER_RI = os.path.join(RI60_DIR, "RI30_RI60_dataFrame.feather")
OUT_DIR = FIG_OUT

TAU = 60.0                 # leaky-integrator time constant (s), matches create_RR_arrays
MAX_DAY_INCLUSIVE = 14     # analysis window, matches RI60_behavior_modeling_v2
GRID_DT = 0.5              # s; dense grid for rendering the continuous decay of rate traces

# Representative sessions, identified by row index into the RI60-only frame
# (see RI60_representative_sessions selection: median pokes / high dynamism).
IDX_CUM = 365   # 821-L  naive  day13   484 pokes  (== median across sessions)
IDX_DYN = 891   # 578-1  stress day11   930 pokes  (high reward+poke rate dynamism)


# ===========================================================
# Canonical reward-rate definition (verbatim from RI60_behavior_functions.py)
# ===========================================================
def calculate_reward_rate(all_pokes, rewarded_pokes, tau=180.0):
    """Decaying rate (events/min) for every poke via a causal leaky integrator."""
    pokes = np.sort(np.array(all_pokes))
    rewards_ts = set(rewarded_pokes)
    is_rewarded = np.array([1.0 if t in rewards_ts else 0.0 for t in pokes])
    rates = np.zeros(len(pokes))
    current_rate = 0.0
    for i in range(1, len(pokes)):
        dt = pokes[i] - pokes[i - 1]
        current_rate = current_rate * np.exp(-dt / tau)
        if is_rewarded[i - 1]:
            current_rate += 1.0
        rates[i] = current_rate
    return (rates / tau) * 60


# ===========================================================
# Helpers
# ===========================================================
def load_ri60():
    ri = pd.read_feather(FEATHER_RI)
    return ri[ri['sesType'] == 'RI60'].copy().reset_index(drop=True)


def continuous_rate(pokes, rates_at_pokes, t_grid, tau):
    """Render the leaky integrator on a dense grid.

    The integrator is only *defined* at poke times by calculate_reward_rate;
    between poke i and i+1 its value is exactly rates_at_pokes[i]*exp(-(t-t_i)/tau)
    (pure exponential decay -- see RI60_behavior_functions recurrence). Sampling
    on a dense grid reproduces the per-poke values and fills the gaps with the
    true decay, instead of the straight lines a poke-only plot draws. Before the
    first poke the value is 0.
    """
    i = np.searchsorted(pokes, t_grid, side='right') - 1   # last poke at or before t
    out = np.zeros_like(t_grid, dtype=float)
    valid = i >= 0
    out[valid] = rates_at_pokes[i[valid]] * np.exp(-(t_grid[valid] - pokes[i[valid]]) / tau)
    return out


def session_traces(row):
    """Per-poke cumulative trace + dense-grid reward / instant-poke rate traces."""
    pokes = np.sort(np.array(row['allPokeTimestamps'], dtype=float))
    rewards = np.array(row['rewPokeTimestamps'], dtype=float)

    # cumulative pokes: one step per poke (x = poke time, no grid needed)
    t_poke_min = pokes / 60.0
    cum = np.arange(1, len(pokes) + 1)

    # rate traces: canonical per-poke values, then rendered on a dense grid
    rr_pokes = calculate_reward_rate(pokes, rewards, TAU)          # rewards/min
    pr_pokes = calculate_reward_rate(pokes, pokes, TAU)            # pokes/min
    t_grid = np.arange(0.0, pokes[-1] + GRID_DT, GRID_DT)
    reward_rate = continuous_rate(pokes, rr_pokes, t_grid, TAU)
    poke_rate = continuous_rate(pokes, pr_pokes, t_grid, TAU)
    t_grid_min = t_grid / 60.0
    return t_poke_min, cum, t_grid_min, reward_rate, poke_rate


def tag(row):
    return f"{row['mouse']}_{row['group']}_day{int(float(row['dayOnType']))}_{row['date']}"


def save_xy(fname, x, xname, y, yname):
    path = os.path.join(OUT_DIR, fname)
    pd.DataFrame({xname: x, yname: y}).to_csv(path, index=False)
    print(f"  wrote {fname}  (n={len(x)})")


def save_plot(fname, x, y, ylabel, title, color):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(x, y, color=color, lw=1.0)
    ax.set_xlabel('time (min)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  wrote {fname}")


# ===========================================================
# Main
# ===========================================================
def main():
    d = load_ri60()

    # ---- sanity: report where the cumulative session sits in the pokes distribution ----
    valid = d[d['dayOnType'].apply(lambda v: 1 <= float(v) <= MAX_DAY_INCLUSIVE
                                   if pd.notna(v) else False)]
    npokes = valid['allPokeTimestamps'].apply(len)
    print(f"Sessions in window days 1-{MAX_DAY_INCLUSIVE}: {len(valid)}  "
          f"median pokes = {npokes.median():.0f}")

    cum_row = d.iloc[IDX_CUM]
    dyn_row = d.iloc[IDX_DYN]
    print(f"Cumulative session  [idx {IDX_CUM}]: {tag(cum_row)}  "
          f"n_pokes={len(cum_row['allPokeTimestamps'])}")
    print(f"Rate session        [idx {IDX_DYN}]: {tag(dyn_row)}  "
          f"n_pokes={len(dyn_row['allPokeTimestamps'])}, "
          f"n_rew={len(dyn_row['rewPokeTimestamps'])}")

    # ---- 1. cumulative pokes (median-pokes session) ----
    t_c, cum, _, _, _ = session_traces(cum_row)
    ctag = tag(cum_row)
    save_xy(f'rep_cumulative_pokes_{ctag}.csv', t_c, 'time_min', cum, 'cumulative_pokes')
    save_plot(f'rep_cumulative_pokes_{ctag}.png', t_c, cum,
              'cumulative pokes', f'Cumulative active pokes  ({ctag})', 'k')

    # ---- 2 & 3. reward rate + instantaneous poke rate (dynamic session) ----
    _, _, t_d, rr, pr = session_traces(dyn_row)
    dtag = tag(dyn_row)
    save_xy(f'rep_reward_rate_{dtag}.csv', t_d, 'time_min', rr, 'reward_rate_rew_per_min')
    save_plot(f'rep_reward_rate_{dtag}.png', t_d, rr,
              'reward rate (rewards/min)', f'Reward rate  ({dtag})', 'tab:green')

    save_xy(f'rep_instant_poke_rate_{dtag}.csv', t_d, 'time_min', pr, 'poke_rate_pokes_per_min')
    save_plot(f'rep_instant_poke_rate_{dtag}.png', t_d, pr,
              'instantaneous poke rate (pokes/min)',
              f'Instantaneous poke rate  ({dtag})', 'tab:purple')

    print("\nDone. Outputs in:\n  " + OUT_DIR)


if __name__ == '__main__':
    main()
