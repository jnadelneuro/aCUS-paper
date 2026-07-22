import os
import numpy as np
import pandas as pd
import pickle
import pyarrow.feather as feather
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import re
import statsmodels.api as sm
import json
from scipy.stats import linregress
from RI60_behavior_modeling import *
from RI60_behavior_functions import *
from _config import RI60_DIR

output_file_folder = RI60_DIR
#? output_file_folder = r'R:\Basic_Sciences\Phys\Lerner_Lab_tnl2633\Jacob\aCUS\KIR ALL\behavior\output_datafiles'
#? output_file_folder = r"R:\Basic_Sciences\Phys\Lerner_Lab_tnl2633\Jacob\aCUS\TS NpHR ALL\behavior\output_datafiles"
# output_file_folder = r'R:\Basic_Sciences\Phys\Lerner_Lab_tnl2633\Jacob\aCUS\ALL_RI60_ADULT\behavior\output_datafiles'
# analyzeBehavior(output_file_folder)
os.chdir(output_file_folder)
analyzeBehavior(output_file_folder)
shockHistogram(output_file_folder)
# df_training = feather.read_feather('RI30_RI60_dataFrame.feather')
# df_training['reward_rate_trace'] = df_training.apply(calculate_local_reward_rate, axis=1)
# df_training.to_feather('RI30_RI60_dataFrame.feather')
# add_cumulative_poke_count()

create_RR_arrays()
# with open('behaviorData_RR.pickle', 'rb') as file:
#     behaviorData = pickle.load(file)


# NOTE: doBoutAnalysis, IRT_analysis, HMM_analysis, mixture_IRT_analysis, and
# lognormal_mixture_analysis (and the model functions they call) were moved to
# RI60_behavior_archive.py. Run that script to use them.
shockMicroAnalysis = True
bodyWeightReg = False

# 1. Load your data (assuming a CSV structure)
shockDF = feather.read_feather('Shock_dataFrame.feather')
# shockDF[shockDF['']]
 # Assuming this CSV has 'mouse' and 'body_weight' columns
# Creating dummy data for demonstration


# 2. Define the Formula
# "Shocks depends on Sex, Stress, their Interaction, AND BodyWeight"
formula = "numShock ~ C(group) + pct_OG_BW"

# 3. Fit the Negative Binomial Model
# We use NegativeBinomial because shock data usually has high variance
model = smf.glm(formula=formula, data=shockDF, family=sm.families.NegativeBinomial()).fit()

# 4. View Results
print(model.summary())


if shockMicroAnalysis == True:
    df_shock = feather.read_feather('shock_dataFrame.feather')
    df_shock = df_shock[df_shock['dayOnType'] != 'nan']


    df_shock['dayOnType'] = df_shock['dayOnType'].astype(float).astype(int).astype(str)
    df_shock = df_shock[df_shock['dayOnType'] == '2']  # Focus on Day 2 sessions
    # 1.Load Data
    # Assuming df_shock is your shock dataframe (already in memory or loaded)
    # Load the averaged training metrics we calculated earlier
    # df_pheno = pd.read_csv('RI60_mouse_phenotype_averages.csv')
    # df_dist = pd.read_csv('IRT_distribution_params.csv').groupby('mouse', as_index=False).median(numeric_only=True)
    # df_hmm = pd.read_csv('HMM_metrics.csv').groupby('mouse', as_index=False).median(numeric_only=True)

    # # Merge Training Phenotypes into one master predictor DF
    # df_training_metrics = df_pheno.merge(df_dist, on='mouse').merge(df_hmm, on='mouse')

    # 2.Extract Shock Microstructure
    shock_micro_data = []

    print(f"Analyzing microstructure for {len(df_shock)} shock sessions...")

    for index, row in df_shock.iterrows():
        # Timestamps
        pokes = np.sort(np.array(row['pokeTimestamps']))
        shocks = np.sort(np.array(row['shockTimestamps']))

        if (len(shocks) == 0) or (len(shocks) > 100):
            continue

        # A.Post-Shock Latency (How fast do they return?)
        latencies = []
        for s_time in shocks:
            future_pokes = pokes[pokes > s_time]
            if len(future_pokes) > 0:
                lat = future_pokes[0] - s_time
                if lat < 600: # Filter 10m timeouts
                    latencies.append(lat)

        avg_latency = np.median(latencies) if latencies else np.nan

        # B."Compulsive" Returns (Percentage of returns < 5s)
        n_compulsive = sum(np.array(latencies) < 5.0)
        pct_compulsive = (n_compulsive / len(latencies)) * 100 if latencies else 0
        if 'virus' in row:
            shock_micro_data.append({
                'mouse': row['mouse'],
                'group': row['group'],
                'virus': row['virus'],
                'total_shocks': len(shocks),
                'median_post_shock_latency': avg_latency,
                'pct_compulsive_returns': pct_compulsive
            }

        )
        else:
            shock_micro_data.append({
                'mouse': row['mouse'],
                'group': row['group'],
                'total_shocks': len(shocks),
                'median_post_shock_latency': avg_latency,
                'pct_compulsive_returns': pct_compulsive
            }

        )

    df_shock_micro = pd.DataFrame(shock_micro_data)

    df_shock_micro = df_shock_micro.drop_duplicates(subset=['mouse'])
    df_shock_micro.to_csv('shock_microstructure.csv', index=False)
    print("Shock Microstructure Analysis Complete.Saved to shock_microstructure.csv")



    a=1

# if compareModels == True:
#     comparison_results = []
#     for index, row in df_training.iterrows():
#         results = compare_irt_models(row)
#         results['mouse'] = row['mouse']
#         results['date'] = row['date']
#         comparison_results.append(results)

#     df_comparison = pd.DataFrame(comparison_results)
#     df_comparison.to_csv('model_comparison_results.csv')

print("Analysis complete.")
