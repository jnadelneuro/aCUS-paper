#%%
import numpy as np
import pandas as pd
import os
import scipy.stats as stats
import pyarrow.feather as feather
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from _config import RI60_DIR


def boot_diffCI(data1, data2, num_boots, sig):
    """
    Bootstraps CI for difference between two conditions.

    Parameters:
    data1 : numpy.ndarray
        Data array for the first condition (rows = trials, columns = time relative to event)
    data2 : numpy.ndarray
        Data array for the second condition (rows = trials, columns = time relative to event)
    num_boots : int
        Number of bootstrap iterations
    sig : float
        Significance level (e.g., 0.05 for 95% confidence interval)

    Returns:
    bootCI : numpy.ndarray
        Lower and upper confidence intervals (LCI, UCI)
    """
    
    num_trials1, window = data1.shape
    num_trials2, _ = data2.shape

    # Minimum 2 trials for each condition
    if num_trials1 > 2 and num_trials2 > 2:
        
        # Prepare bootstrapping variables (one row for each bootstrap)
        data_boots = np.zeros((num_boots, window))
        bootCI = np.zeros((2, window))

        for b in range(num_boots):
            # Bootstrap data across all trials
            trial_array1 = np.random.choice(num_trials1, num_trials1, replace=True)
            trial_array2 = np.random.choice(num_trials2, num_trials2, replace=True)
            data_boots[b, :] = np.mean(data1[trial_array1, :], axis=0) - np.mean(data2[trial_array2, :], axis=0)
        
        # Calculate bootstrap CI
        data_boots = np.sort(data_boots, axis=0)

        lower_conf_index = int(np.ceil(num_boots * (sig / 2)))
        upper_conf_index = int(np.floor(num_boots * (1 - sig / 2)))

        bootCI[0, :] = data_boots[lower_conf_index, :]
        bootCI[1, :] = data_boots[upper_conf_index, :]

    else:
        print('Less than 3 trials for either condition - bootstrapping skipped')
        bootCI = np.nan

    return bootCI

def boot_CI(data, num_boots, sig):
    """
    Bootstraps CI for mean population waveform.
    
    Parameters:
    data : numpy.ndarray
        Data array (rows = trials, columns = time relative to event)
    num_boots : int
        Number of bootstrap iterations
    sig : float
        Significance level (e.g., 0.05 for 95% confidence interval)
    
    Returns:
    bootCI : numpy.ndarray
        Lower and upper confidence intervals (LCI, UCI)
    """
    
    num_trials, window = data.shape
    
    # Minimum 2 trials
    if num_trials > 2:
        
        # Prep bootstrapping variables (one row for each bootstrap)
        data_boots = np.zeros((num_boots, window))
        bootCI = np.zeros((2, window))
        
        for b in range(num_boots):
            # Bootstrap data across all trials
            trial_array = np.random.randint(0, num_trials, num_trials)
            data_boots[b, :] = np.mean(data[trial_array, :], axis=0)
        
        # Calculate bootstrap CI
        data_boots = np.sort(data_boots, axis=0)
        
        lower_conf_index = int(np.ceil(num_boots * (sig / 2)))
        upper_conf_index = int(np.floor(num_boots * (1 - sig / 2)))
        
        bootCI[0, :] = data_boots[lower_conf_index, :]
        bootCI[1, :] = data_boots[upper_conf_index, :]
        
    else:
        print('Less than 3 trials - bootstrapping skipped')
        bootCI = np.nan

    return bootCI

def CIadjust(LCI, UCI, est, n, adj_type):
    # Adjusts CI according to adj_type
    # Type 1 = extend CI from reference by sqrt(n/(n-1))
    # Type 2 = expand CI by sqrt(n/(n-1)) (doesn't need mean)

    # CI_fix: adjustment factor
    CI_fix = np.sqrt(n / (n - 1))
    
    if adj_type == 1:
        # Extend CI from reference
        print(f'CI extended from reference by {CI_fix * 100:.2f}%')
        adjUCI = (UCI - est) * CI_fix + est
        adjLCI = est - (est - LCI) * CI_fix
    
    elif adj_type == 2:
        # Expand CI
        print(f'CI expanded by {CI_fix * 100:.2f}%')
        CIchange = ((UCI - LCI) * CI_fix - (UCI - LCI)) / 2
        adjUCI = UCI + CIchange
        adjLCI = LCI - CIchange
    
    return adjLCI, adjUCI

def apply_consecutive_threshold(ci_array, consec_thresh):
    """
    Apply consecutive threshold filter to confidence intervals.
    Regions with fewer than consec_thresh consecutive significant points are set to include 0.
    
    Parameters:
    ci_array : numpy.ndarray
        2D array with lower CI in row 0, upper CI in row 1
    consec_thresh : int
        Minimum number of consecutive significant points
        
    Returns:
    ci_array : numpy.ndarray
        Filtered confidence intervals
    """
    window = ci_array.shape[1]
    
    # Identify significant regions (where CI doesn't include 0)
    is_sig = (ci_array[0, :] > 0) | (ci_array[1, :] < 0)
    
    # Find consecutive runs of significance
    filtered_sig = np.zeros(window, dtype=bool)
    
    # Pad to detect boundaries
    padded = np.pad(is_sig.astype(int), (1, 1), mode='constant', constant_values=0)
    diff = np.diff(padded)
    
    # Find start and end indices of significant runs
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    # Filter runs that meet the consecutive threshold
    for start, end in zip(starts, ends):
        if (end - start) >= consec_thresh:
            filtered_sig[start:end] = True
    
    # For non-significant regions, adjust CI to include 0
    ci_array[0, ~filtered_sig] = np.minimum(ci_array[0, ~filtered_sig], 0)
    ci_array[1, ~filtered_sig] = np.maximum(ci_array[1, ~filtered_sig], 0)
    
    return ci_array

def make_SigArray(ci_array):
    # ci_array should be a 2 by x numpy array
    lower_bound = ci_array[0, :]
    upper_bound = ci_array[1, :]

    # Create an array where 1 indicates that 0 is outside the bounds
    significant = np.where((lower_bound > 0) | (upper_bound < 0), 3, np.nan)
    
    return significant

def filter_and_pad(times, trace, time_min, time_max, max_length):
    # Filter time and corresponding correlation values
    mask = (times.astype(float) >= time_min) & (times.astype(float) <= time_max)
    filtered_times = times[mask]
    filtered_trace = trace[mask]
    
    # Calculate how many NA values are needed at the beginning
    
    return filtered_times, filtered_trace
def bootstrap_analysis(photoDataFrame, criteria1, criteria2, comparison={'group' : ['naive', 'stress']}):
    #criteria1 is a tuple with 'column name', value
    # criteria2 is same 
    comparison_key = list(comparison.keys())[0]
    comparison_values = list(comparison[comparison_key])
    
    criteria1_col = criteria1[0]
    criteria1_val = criteria1[1]
    criteria2_col = criteria2[0]
    criteria2_val = criteria2[1]

    targetData = photoDataFrame[(photoDataFrame[criteria1[0]] == criteria1[1]) & (photoDataFrame[criteria2[0]] == criteria2[1]) & (photoDataFrame['sesType'] == 'RI60')]

    def has_nan(arr):
        return any(np.isnan(arr))

    # apply the has_nan function to column 'C' and drop any rows where the value in column 'C' contains NaN
    dropped_rows = targetData.loc[targetData['photoTrace'].apply(has_nan)].shape[0]
    print(f"Dropped {dropped_rows} row(s) with NaN in column 'photoTrace.' That's " +
        str(np.round(100*(dropped_rows/targetData.shape[0]), 2)) + "%!")
    targetData = targetData[~targetData['photoTrace'].apply(has_nan)]

    
    groupBys = ['mouse', comparison_key, criteria1[0], criteria2[0]]

    groupMeansMouse = targetData.groupby(groupBys)['photoTrace'].mean().reset_index()

    
    traces1 = groupMeansMouse[groupMeansMouse[comparison_key] == comparison_values[0]]
    array1 = np.vstack(traces1['photoTrace'].to_numpy())

    traces2 = groupMeansMouse[groupMeansMouse[comparison_key] == comparison_values[1]]
    array2 = np.vstack(traces2['photoTrace'].to_numpy())

    sig = .05 # significance level, being strict
    consec_thresh = 25 # number of consecutive significant points to consider a real effect--we're sampling at ~100 Hz
    n_boot = 25000

    n_1 = np.shape(array1)[0]
    n_2, ev_win = np.shape(array2)
   

    Cp_t_crit = stats.t.ppf(1 - sig / 2, n_2 - 1)
    Cm_t_crit = stats.t.ppf(1 - sig / 2, n_1 - 1)

    means_1 = np.mean(array1, axis=0)
    SEMs_1 = stats.sem(array1, axis=0)
    bCI_1 = boot_CI(array1, n_boot, sig)
    [adjLCI, adjUCI] = CIadjust(bCI_1[0,:], bCI_1[1,:], [], n_1, 2)
    bCIexp_1 = np.vstack((adjLCI, adjUCI))
    bCIexp_1 = apply_consecutive_threshold(bCIexp_1, consec_thresh)


    means_2 = np.mean(array2, axis=0)
    SEMs_2 = stats.sem(array2, axis=0)
    bCI_2 = boot_CI(array2, n_boot, sig)
    [adjLCI, adjUCI] = CIadjust(bCI_2[0,:], bCI_2[1,:], [], n_2, 2)
    bCIexp_2 = np.vstack((adjLCI, adjUCI))
    bCIexp_2 = apply_consecutive_threshold(bCIexp_2, consec_thresh)


    diff_bCI = boot_diffCI(array2, array1, n_boot, sig)
    [adjLCI, adjUCI] = CIadjust(diff_bCI[0,:], diff_bCI[1,:], [], n_1, 2)
    diff_bCIexp = np.vstack((adjLCI, adjUCI))
    diff_bCIexp = apply_consecutive_threshold(diff_bCIexp, consec_thresh)

    diff_sigArray = make_SigArray(diff_bCIexp)
    sigArray_1 = make_SigArray(bCIexp_1)
    sigArray_2 = make_SigArray(bCIexp_2)
    

    # time = targetEvent['time'].iloc[0]
    time = np.linspace(-5, 10, len(array1[0,:]))

    export_1 = pd.DataFrame(np.vstack((time, means_1, SEMs_1, sigArray_1)))
    export_2 = pd.DataFrame(np.vstack((time, means_2, SEMs_2, sigArray_2)))
    diff_export = pd.DataFrame(np.vstack((time, diff_sigArray)))

    export_1.to_csv(f'bootstrapping_{comparison_values[0]}_{criteria1[1]}_{criteria2[1]}.csv', index=False)
    export_2.to_csv(f'bootstrapping_{comparison_values[1]}_{criteria1[1]}_{criteria2[1]}.csv', index=False)
    diff_export.to_csv(f'bootstrapping_difference_{criteria1[1]}_{criteria2[1]}_{comparison_values[0]}_vs_{comparison_values[1]}.csv', index=False)
    #%% Plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the main traces
    ax.plot(time, means_1, label=f'{comparison_values[0]}', color='blue', linewidth=2)
    ax.plot(time, means_2, label=f'{comparison_values[1]}', color='red', linewidth=2)

    # Optional: Add confidence intervals as shaded regions
    ax.fill_between(time, bCIexp_1[0,:], bCIexp_1[1,:], alpha=0.2, color='blue')
    ax.fill_between(time, bCIexp_2[0,:], bCIexp_2[1,:], alpha=0.2, color='red')

    # Get y-axis limits for positioning significance markers
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    # Position significance markers above the plot
    sig_height_1 = y_max + y_range * 0.05  # Naive sig
    sig_height_2 = y_max + y_range * 0.10  # Stress sig
    sig_height_3 = y_max + y_range * 0.15  # Difference sig

    # Plot significance markers
    # Naive different from 0
    ax.plot(time, np.where(~np.isnan(sigArray_1), sig_height_1, np.nan), 
            '|', color='blue', markersize=3, label=f'{comparison_values[0]} ≠ 0')

    # Stress different from 0
    ax.plot(time, np.where(~np.isnan(sigArray_2), sig_height_2, np.nan), 
            '|', color='red', markersize=3, label=f'{comparison_values[1]} ≠ 0')

    # Stress different from Naive
    ax.plot(time, np.where(~np.isnan(diff_sigArray), sig_height_3, np.nan), 
            '|', color='purple', markersize=3, label=f'{comparison_values[0]} ≠ {comparison_values[1]}')

    # Add labels and formatting
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Signal', fontsize=12)
    ax.set_title(f'{comparison_key}, in {criteria1[1]}/{criteria2[1]} - {comparison_values[0]} vs {comparison_values[1]} Comparison with Significance', fontsize=14)
    ax.legend(loc='best')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Adjust ylim to accommodate significance markers
    ax.set_ylim(y_min, y_max + y_range * 0.20)

    plt.tight_layout()
    plt.savefig(f'{criteria1[1]}_{criteria2[1]}_{comparison_values[0]}_vs_{comparison_values[1]}_bootstrap_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # %%
def findClosestIdx(timeArray, timePoint):
    closestIdx = (np.abs(timeArray[0] - float(timePoint))).argmin()
    return closestIdx

def make_timeseries(photoDF, numOfSamples, time_start, time_end):
    x = []
    for i in range(len(photoDF)):
        x.append(np.linspace(time_start, time_end, num=numOfSamples))

    return x

def AUC_finder(photoDF, AUCWindow, AUCName):
    time = make_timeseries(photoDF, 1526, -5, 10)
    AUCidx = [findClosestIdx(time, numb) for numb in AUCWindow]

    def findAUC(array, time, start, end):
        sliced_array = array[start:end+1]
        sliced_time = time[start:end+1]
        auc = np.trapz(sliced_array, sliced_time)
        return auc

    photoDF[AUCName] = photoDF['photoTrace'].apply(
        lambda x: pd.Series(findAUC(x, time[0], AUCidx[0], AUCidx[1])))

    return photoDF

def corrParams(df, param1, param2, id=None):

    # Ensure parameters exist in dataframe
    if param1 not in df.columns or param2 not in df.columns:
        print(f"Error: {param1} or {param2} not found in dataframe columns")
        return
    
    # Remove rows with NaN values in either parameter
    df_clean = df[[param1, param2, 'mouse', 'group']].dropna()
    
    # Create figure with subplots for each group
    groups = df_clean['group'].unique()
    fig, axes = plt.subplots(1, len(groups), figsize=(6*len(groups), 5))
    if len(groups) == 1:
        axes = [axes]
    
    # Store results
    model_results = {}
    
    for idx, group in enumerate(groups):
        group_data = df_clean[df_clean['group'] == group]
        
        # Fit mixed effects model with random intercept for mouse
        formula = f'{param2} ~ {param1}'
        model = smf.mixedlm(formula, group_data, groups=group_data['mouse'])
        result = model.fit()
        
        # Extract fixed effect (slope) and p-value
        slope = result.fe_params[param1]
        pval = result.pvalues[param1]
        
        # Get unique mice
        mice = group_data['mouse'].unique()
        
        # Plot individual trials as transparent points
        axes[idx].scatter(group_data[param1], group_data[param2], 
                         alpha=0.1, s=20, color='gray', label='Individual trials')
        
        # Plot mouse averages as larger points
        mouse_avgs = group_data.groupby('mouse')[[param1, param2]].mean()
        axes[idx].scatter(mouse_avgs[param1], mouse_avgs[param2], 
                         alpha=0.7, s=100, label='Mouse averages')
        
        # Add regression line
        x_line = np.linspace(group_data[param1].min(), group_data[param1].max(), 100)
        y_line = result.fe_params['Intercept'] + slope * x_line
        axes[idx].plot(x_line, y_line, "r--", alpha=0.8, linewidth=2)
        
        axes[idx].set_xlabel(param1)
        axes[idx].set_ylabel(param2)
        axes[idx].set_title(f'{group}\nslope={slope:.3f}, p={pval:.4f}, n={len(mice)} mice')
        axes[idx].legend()
        
        model_results[group] = {'slope': slope, 'p_value': pval, 'n_mice': len(mice)}
    
    plt.tight_layout()
    plt.savefig(f'{id}_{param1}_vs_{param2}_mixed_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    
os.chdir(RI60_DIR)

print('loading photoDF')
photoDataFrame = pd.read_feather('photoDataFrame.feather')
print('loaded!')

photoDataFrame['earlyOrLate'] = np.where(
    photoDataFrame['dayOnType'] <= 4, 'early',
    np.where(photoDataFrame['dayOnType'] >= 9, 'late', 'mid')
)




bootstrapping = True
corr_w_AUC = False
corr_w_AUC_timeSince = False
#%%

if bootstrapping == True:
    event = ['UnNP', 'ReNP', 'RePE']
    area = ['TS', 'DMS']
    group = ['naive', 'stress']

    # boutStatusDataFrame = photoDataFrame[(photoDataFrame['recordingLoc'] == 'TS')
    #                                      & (photoDataFrame['bout_status'].notna())]
    # for gr in group:
    #     for ev in ['ReNP', 'UnNP']:
    #         bootstrap_analysis(boutStatusDataFrame, ('group', gr), ('event', ev),
    #                            comparison={'bout_status': ['within_bout', 'disengaged']})

    reNPDataFrame = photoDataFrame[(photoDataFrame['event'] == 'ReNP') & (photoDataFrame['sesType'] == 'RI60')]
    timeSinceRew_ReNP_DataFrame = photoDataFrame
    timeSinceRew_ReNP_DataFrame = timeSinceRew_ReNP_DataFrame[(timeSinceRew_ReNP_DataFrame['time_since_reward_poke'] >= 0)\
                                                         & (timeSinceRew_ReNP_DataFrame['time_since_reward_poke'] <= 200)]
    

    timeSinceRew_ReNP_DataFrame['timeSinceRew_cat'] = np.where(
        timeSinceRew_ReNP_DataFrame['time_since_reward_poke'] <= timeSinceRew_ReNP_DataFrame['time_since_reward_poke'].quantile(0.5), 'short',
        np.where(timeSinceRew_ReNP_DataFrame['time_since_reward_poke'] >= timeSinceRew_ReNP_DataFrame['time_since_reward_poke'].quantile(0.5), 'long', 'mid')
    )

    event = ['UnNP', 'ReNP', 'RePE']
    area = ['TS', 'DMS']
    group = ['naive', 'stress']
    for ev in event:
        for ar in area:
            # print(f'Bootstrapping CIs for event: {ev}, area: {ar}')
            bootstrap_analysis(photoDataFrame, ('event', ev), ('recordingLoc', ar), comparison={'group': ['naive', 'stress']})

    for gr in group:
        for ar in area:
            bootstrap_analysis(photoDataFrame, ('group', gr), ('recordingLoc', ar), comparison={'event': ['UnNP', 'ReNP']})

    for gr in group:
        for ar in area:
            bootstrap_analysis(reNPDataFrame, ('group', gr), ('recordingLoc', ar), comparison={'earlyOrLate': ['early', 'late']})
            bootstrap_analysis(reNPDataFrame, ('group', gr), ('recordingLoc', ar), comparison={'sex': ['M', 'F']})
                        

    area = ['TS']
    timeSinceRew_ReNP_DataFrame = timeSinceRew_ReNP_DataFrame[timeSinceRew_ReNP_DataFrame['recordingLoc'] == area[0]]
    for gr in group:
        for ev in event:
            bootstrap_analysis(timeSinceRew_ReNP_DataFrame, ('group', gr), ('event', ev), comparison={'timeSinceRew_cat': ['short', 'long']})

    # --- Within-bout vs disengaged pokes (bout_status from lognormal mixture) ---
    # TS only (do NOT pool TS + DMS). One comparison per group x event.
    boutStatusDataFrame = photoDataFrame[(photoDataFrame['recordingLoc'] == 'TS')
                                         & (photoDataFrame['bout_status'].notna())]
    for gr in group:
        for ev in ['ReNP', 'UnNP']:
            bootstrap_analysis(boutStatusDataFrame, ('group', gr), ('event', ev),
                               comparison={'bout_status': ['within_bout', 'disengaged']})

if corr_w_AUC == True:
    area = ['TS', 'DMS']
    
    photoDF_ReNP = photoDataFrame[(photoDataFrame['event'] == 'ReNP') & (photoDataFrame['sesType'] == 'RI60')]
    photoDF_AUC = AUC_finder(photoDF_ReNP, [0, 3], 'AUC_0_3')
    for ar in area:
        corrParams(photoDF_AUC[photoDF_AUC['recordingLoc'] == ar], 'dayOnType', 'AUC_0_3', id=ar)

if corr_w_AUC_timeSince == True:
    area = ['TS']
    photoDF_ReNP = photoDataFrame[(photoDataFrame['event'] == 'ReNP') & (photoDataFrame['sesType'] == 'RI60')]
    photoDF_ReNP = photoDF_ReNP[(photoDF_ReNP['time_since_reward_poke'] >= 0) & (photoDF_ReNP['time_since_reward_poke'] <= 200)]
    photoDF_ReNP['time_diff_from_60'] = abs(photoDF_ReNP['time_since_reward_poke'] - 60)


    photoDF_AUC = AUC_finder(photoDF_ReNP, [0, 3], 'AUC_0_3') 
    # photoDF_AUC = photoDF_AUC[photoDF_AUC['time_since_reward_poke_exp'] <= 300]

    for ar in area:
        # corrParams(photoDF_AUC[photoDF_AUC['recordingLoc'] == ar], 'time_since_reward_poke', 'AUC_0_3', id=ar)
        corrParams(photoDF_AUC[photoDF_AUC['recordingLoc'] == ar], 'time_diff_from_60', 'AUC_0_3', id=ar)