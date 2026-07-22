import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import scipy.stats as stats
from hmmlearn import hmm
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Mixture Model for IRT Distributions
# ---------------------------------------------------------
from scipy.stats import lognorm
from scipy.optimize import minimize
from scipy.special import logsumexp

# NOTE: The IRT / bout / mixture / HMM model functions that used to live here
# (mixture_exponential_loglik, fit_mixture_exponential, get_mixture_irt_params,
#  shull_equation, fit_shull_params, analyze_time_budget, get_split_half_metrics,
#  calculate_rundown_slope, fit_gamma, fit_exgaussian, get_irt_dist_params,
#  fit_lognormal_mixture_4, fit_hmm_states) were moved to RI60_behavior_archive.py
# together with the analysis blocks (doBoutAnalysis, IRT_analysis, HMM_analysis,
# mixture_IRT_analysis, lognormal_mixture_analysis) that were their only callers.
# Imports above are kept unchanged so `from RI60_behavior_modeling import *`
# elsewhere still re-exports them.


def calculate_local_reward_rate(row, tau=180.0):
    """
    Calculates a decaying reward rate for every poke.

    params:
    tau: Decay time constant in seconds.
         180s is good for RI60 (smoothing over ~3 rewards).
    """
    pokes = np.sort(np.array(row['allPokeTimestamps']))
    # Assuming 'rewardPokeTimestamps' exists or you have a way to know which pokes were rewarded
    # Create a reward vector (1 if rewarded, 0 if not) matched to pokes
    rewards_ts = set(np.sort(np.array(row.get('rewardEntryTS', [])))) # Or relevant reward column

    # Vectors
    is_rewarded = np.array([1.0 if t in rewards_ts else 0.0 for t in pokes])
    rates = np.zeros(len(pokes))

    current_rate = 0.0

    # Iterate through pokes (Standard Leaky Integrator)
    # We update rate based on time passed since last event
    for i in range(1, len(pokes)):
        dt = pokes[i] - pokes[i-1]

        # 1.Decay the previous rate over the time gap
        current_rate = current_rate * np.exp(-dt / tau)

        # 2.Add the reward (if any) from the PREVIOUS poke
        # (The rate at the moment of Poke i depends on history up to i)
        if is_rewarded[i-1]:
            # Add impulse.Magnitude can be 1.0 or scaled by 1/tau for "rate per second" units
            current_rate += 1.0

        rates[i] = current_rate

    # Normalize to "Rewards per Minute" for readability
    # (Optional: depends on how you want to interpret the y-axis)
    # If using pure impulse addition, divide by tau to get rate density, then * 60 for per minute
    rates_per_min = (rates / tau) * 60

    return rates_per_min
