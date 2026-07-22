import os
import numpy as np
import pandas as pd
import pickle
import pyarrow.feather as feather
# import kaleido
from typing import Optional, Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import scipy
from scipy.optimize import curve_fit, OptimizeWarning
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import tkinter as tk
import json
# import itertools
import h5py
from general_JIMMY_functions import *
from RI60_analaysis_fx_PHOTO_JIMMYv2 import line
import statsmodels.api as sm
from scipy import stats
import statsmodels.formula.api as smf
pio.renderers.default = 'browser'

import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import sem
import pandas as pd
from _config import RI60_DIR


def plot_dataset(ax, df, title, color='black'):
    t = np.linspace(-2, 5, len(df))
    beta = df['beta.hat']
    error = df['error']
    upper = beta + error
    lower = beta - error

    # Plot Error Band
    ax.fill_between(t, lower, upper, color='#D3D3D3', alpha=0.7, linewidth=0)
    
    # Reference lines
    ax.axhline(0, color='gray', linestyle=':', alpha=0.6)
    ax.axvline(0, color='black', linestyle=':', alpha=0.8, linewidth=1.5)

    # Main Line
    ax.plot(t, beta, color='black', linewidth=1)

    # Significance highlight
    is_significant = (lower > 0) | (upper < 0)
    beta_sig = beta.copy()
    beta_sig[~is_significant] = np.nan 
    ax.plot(t, beta_sig, color='red', linewidth=1.2)

    ax.set_title(title)
    ax.set_xlabel('time (s)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def line(error_y_mode=None, **kwargs):
    """Extension of `plotly.express.line` to use error bands."""
    ERROR_MODES = {'bar', 'band', 'bars', 'bands', None}
    if error_y_mode not in ERROR_MODES:
        raise ValueError(
            f"'error_y_mode' must be one of {ERROR_MODES}, received {repr(error_y_mode)}.")
    if error_y_mode in {'bar', 'bars', None}:
        fig = px.line(**kwargs)
    elif error_y_mode in {'band', 'bands'}:
        if 'error_y' not in kwargs:
            raise ValueError(
                f"If you provide argument 'error_y_mode' you must also provide 'error_y'.")
        figure_with_error_bars = px.line(**kwargs)
        fig = px.line(
            **{arg: val for arg, val in kwargs.items() if arg != 'error_y'})
        for data in figure_with_error_bars.data:
            x = list(data['x'])
            y_upper = list(data['y'] + data['error_y']['array'])
            y_lower = list(data['y'] - data['error_y']['array'] if data['error_y']
                           ['arrayminus'] is None else data['y'] - data['error_y']['arrayminus'])
            color = f"rgba({tuple(int(data['line']['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))},.3)".replace(
                '((', '(').replace('),', ',').replace(' ', '')
            fig.add_trace(
                go.Scatter(
                    x=x+x[::-1],
                    y=y_upper+y_lower[::-1],
                    fill='toself',
                    fillcolor=color,
                    line=dict(
                        color='rgba(255,255,255,0)'
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=data['legendgroup'],
                    xaxis=data['xaxis'],
                    yaxis=data['yaxis'],
                )
            )
        # Reorder data as said here: https://stackoverflow.com/a/66854398/8849755
        reordered_data = []
        for i in range(int(len(fig.data)/2)):
            reordered_data.append(fig.data[i+int(len(fig.data)/2)])
            reordered_data.append(fig.data[i])
        fig.data = tuple(reordered_data)
    return fig


# def downsample_traces(array):
#     factor = 10
#     return array[::factor]

def findClosestIdx(timeArray, timePoint):
    closestIdx = (np.abs(timeArray[0] - float(timePoint))).argmin()
    return closestIdx


def peakFinder(photoDF, peakWindow):
    # peakWindow should be a list of start, end (int)
    x = make_timeseries(photoDF, 1526, -5, 10)
    pkWindow_idx = [findClosestIdx(x, numb) for numb in peakWindow]

    def findMax(array, start, end):
        sliced_arary = array[start:end+1]
        max_val = sliced_arary.max()
        max_idx = sliced_arary.argmax() + start
        return max_val, max_idx

    photoDF[['peak', 'peak_idx']] = photoDF['photoTrace'].apply(
        lambda x: pd.Series(findMax(x, pkWindow_idx[0], pkWindow_idx[1])))

    return photoDF


def AUC_finder(photoDF, AUCWindow, rBool=False):
    if rBool:
        time = make_timeseries(photoDF, 712, -2, 7)
    else:
        time = make_timeseries(photoDF, 1526, -5, 10)
    AUCidx = [findClosestIdx(time, numb) for numb in AUCWindow]

    def findAUC(array, time, start, end):
        sliced_array = array[start:end+1]
        sliced_time = time[start:end+1]
        auc = np.trapz(sliced_array, sliced_time)
        return auc

    photoDF['AUC'] = photoDF['photoTrace'].apply(
        lambda x: pd.Series(findAUC(x, time[0], AUCidx[0], AUCidx[1])))

    return photoDF

def kinetics_finder(photoDF, baseline_window=(-5, 0), stimulus_time=0, peak_search_window=(0, 5)):
    """
    Add kinetic measurements to photoDF similar to AUC_finder
    """
    time = make_timeseries(photoDF, 1526, -5, 10)
    
    def extract_kinetics_for_trace(trace):
        """Extract kinetics for a single trace"""
        signal = np.array(trace)
        time_array = time[0]  # Assuming time is consistent across traces
        
        # 1. Baseline
        baseline_mask = (time_array >= baseline_window[0]) & (time_array <= baseline_window[1])
        baseline = np.mean(signal[baseline_mask])
        baseline_std = np.std(signal[baseline_mask])
        
        # Subtract baseline
        signal_corrected = signal - baseline
        
        # 2. Peak analysis
        peak_mask = (time_array >= peak_search_window[0]) & (time_array <= peak_search_window[1])
        if not np.any(peak_mask):
            return {col: np.nan for col in ['baseline', 'peak_amplitude', 'peak_time', 'time_to_peak', 
                                          'rise_time_10_90', 'rise_time_20_80', 'max_rise_slope', 
                                          'half_decay_time', 'quarter_decay_time', 'decay_90_time', 'auc']}
        
        peak_idx = np.argmax(signal_corrected[peak_mask])
        peak_idx_global = np.where(peak_mask)[0][peak_idx]
        
        peak_amplitude = signal_corrected[peak_idx_global]
        peak_time = time_array[peak_idx_global]
        
        # 3. Rise kinetics
        rise_mask = (time_array >= stimulus_time) & (time_array <= peak_time)
        rise_times = time_array[rise_mask]
        rise_signal = signal_corrected[rise_mask]
        
        if len(rise_signal) > 1:
            # 10-90% rise time
            rise_10 = 0.1 * peak_amplitude
            rise_90 = 0.9 * peak_amplitude
            
            try:
                idx_10 = np.argmin(np.abs(rise_signal - rise_10))
                idx_90 = np.argmin(np.abs(rise_signal - rise_90))
                rise_time_10_90 = rise_times[idx_90] - rise_times[idx_10]
            except:
                rise_time_10_90 = np.nan
            
            # 20-80% rise time
            try:
                rise_20 = 0.2 * peak_amplitude
                rise_80 = 0.8 * peak_amplitude
                idx_20 = np.argmin(np.abs(rise_signal - rise_20))
                idx_80 = np.argmin(np.abs(rise_signal - rise_80))
                rise_time_20_80 = rise_times[idx_80] - rise_times[idx_20]
            except:
                rise_time_20_80 = np.nan
            
            # Maximum rise slope
            rise_slopes = np.gradient(rise_signal, rise_times)
            max_rise_slope = np.max(rise_slopes)
        else:
            rise_time_10_90 = np.nan
            rise_time_20_80 = np.nan
            max_rise_slope = np.nan
        
        # 4. Decay kinetics
        decay_mask = time_array >= peak_time
        decay_times = time_array[decay_mask]
        decay_signal = signal_corrected[decay_mask]
        
        # Half decay time
        try:
            half_amplitude = peak_amplitude / 2
            idx = np.where(decay_signal <= half_amplitude)[0][0]
            half_decay_time = decay_times[idx] - peak_time
        except:
            half_decay_time = np.nan
        
        # Quarter decay time
        try:
            quarter_amplitude = peak_amplitude / 4
            idx = np.where(decay_signal <= quarter_amplitude)[0][0]
            quarter_decay_time = decay_times[idx] - peak_time
        except:
            quarter_decay_time = np.nan
        
        # 90% decay time
        try:
            decay_90_amplitude = 0.1 * peak_amplitude
            idx = np.where(decay_signal <= decay_90_amplitude)[0][0]
            decay_90_time = decay_times[idx] - peak_time
        except:
            decay_90_time = np.nan
        
        # AUC
        try:
            auc = np.trapz(signal_corrected[signal_corrected > 0], 
                          time_array[signal_corrected > 0]) if np.any(signal_corrected > 0) else 0
        except:
            auc = np.nan
        
        return {
            'baseline': baseline,
            'peak_amplitude': peak_amplitude,
            'peak_time': peak_time,
            'time_to_peak': peak_time - stimulus_time,
            'rise_time_10_90': rise_time_10_90,
            'rise_time_20_80': rise_time_20_80,
            'max_rise_slope': max_rise_slope,
            'half_decay_time': half_decay_time,
            'quarter_decay_time': quarter_decay_time,
            'decay_90_time': decay_90_time,
            'auc': auc
        }
    
    # Apply to each trace and add columns to photoDF
    kinetics_results = photoDF['photoTrace'].apply(extract_kinetics_for_trace)
    
    # Add each kinetic parameter as a new column
    for param in ['baseline', 'peak_amplitude', 'peak_time', 'time_to_peak', 
                  'rise_time_10_90', 'rise_time_20_80', 'max_rise_slope', 
                  'half_decay_time', 'quarter_decay_time', 'decay_90_time', 'auc']:
        photoDF[param] = kinetics_results.apply(lambda x: x[param])
    
    return photoDF

def make_timeseries(photoDF, numOfSamples, time_start, time_end):
    x = []
    for i in range(len(photoDF)):
        x.append(np.linspace(time_start, time_end, num=numOfSamples))

    return x

# def findTimeSinceLastReward(photoDF, behaviorData):
#     # this will basically use event timestamps to find the time the mouse last got rewarded (both rewarded poke and port entry). Needs the photoDataFrame
#     rewardEntry_timestamps = photoDF[photoDF['event'] == 'RePE']['timestamp']
#     rewardPoke_timestamps = photoDF[photoDF['event'] == 'ReNP']['timestamp']

def prepFor_R(df, target_col, time_min, time_max, groupBys, getMeans=True):
    def filter_and_pad(times, trace, time_min, time_max, max_length):
        # Filter time and corresponding correlation values
        mask = (times.astype(float) >= time_min) & (
            times.astype(float) <= time_max)
        filtered_times = times[mask]
        filtered_trace = trace[mask]

        # Calculate how many NA values are needed at the beginning

        return filtered_times, filtered_trace

    x = make_timeseries(df, 1526, -5, 10)
    df['time'] = x
    # Apply the function to each row and find the max length for padding
    max_length = len(df['time'][0])

    # Initialize new columns for padded times and correlation values
    df['padded_time'] = [None] * len(df)
    df['padded_trace'] = [None] * len(df)

    # Filter and pad each row
    for i in range(len(df)):
        padded_times, padded_trace = filter_and_pad(
            df.at[i, 'time'], df.at[i, target_col], time_min, time_max, max_length)
        df.at[i, 'padded_time'] = padded_times
        df.at[i, 'padded_trace'] = padded_trace

    # Convert list columns to DataFrame for better readability
    df_padded = pd.DataFrame({
        'mouse': df['mouse'],
        'sex': df['sex'],
        'group': df['group'],
        'sesType': df['sesType'],
        'date': df['date'],
        'timestamp': df['timestamp'],
        'timeSince_ReNP' : df['time_since_reward_poke'],
        'timeSince_RePE' : df['time_since_reward_entry'],
        'event_number' : df['event_number'],
        # 'sensor' : df['sensor'],
        'recordingLoc': df['recordingLoc'],
        'dayOnType': df['dayOnType'],
        'event': df['event'],
        'reward_rate': df['reward_rate'],
        'instant_poke_rate' : df['instant_poke_rate'],
        'trimTime': df['padded_time'],
        'trimTrace': df['padded_trace'],
        'cumulative_poke': df['cumulative_poke']  
    })
    # if not getMeans:
    #     df_padded.insert(2, 'sensor', df['sensor'])

    # Function to calculate SEM

    def calc_sem(x):
        n = np.sum(~np.isnan(x))  # Count non-NA values
        if n < 2:
            return np.nan  # Return NA if fewer than 2 non-NA values
        return scipy.stats.sem(x, nan_policy='omit')

    # Summarize mean and SEM

    if getMeans:
        # df_padded = df_padded.drop(['sensor'], axis=1)
        result = df_padded.groupby(groupBys).apply(
            lambda g: pd.Series({
                'trace_mean': np.nanmean(np.vstack(g['padded_trace'].values), axis=0),
                'trace_sem': np.apply_along_axis(calc_sem, 0, np.vstack(g['padded_trace'].values))
            })
        ).reset_index()
        return result, padded_times
    else:
        return df_padded


def prep4csv(photoDF, targetColumn, pivot_index):
    # column is a string of the key column (usually 'photoTrace' or 'sem')
    # pivot_index is the groupbys--a list of strings

    x = make_timeseries(photoDF, 1526, -5, 10)
    photoDF['x'] = x

    # photoDF['x'] = photoDF['x'].apply(downsample_traces)
    photoDF = photoDF.reset_index()

    photoDF = photoDF.explode([targetColumn, 'x'])

    photoDF = photoDF.pivot_table(
        index=pivot_index, columns='x', values=targetColumn, aggfunc='mean').reset_index()

    return photoDF


def has_nan(arr):
    return any(np.isnan(arr))


def loadPhotoDF(fileName='photoDataFrame.feather'):
    os.chdir(RI60_DIR)

    print('loading photoDF')
    photoDataFrame = pd.read_feather(fileName)
    print('loaded!')
    
    # * I used different group names (stress or aCUS) for cohort 2 and cohort 4 so this is just to standardize that
    photoDataFrame['group'] = photoDataFrame['group'].str.replace(
        'aCUS', 'stress')
    photoDataFrame['group'] = photoDataFrame['group'].str.replace(
        'control', 'naive')
    photoDataFrame['recordingLoc'] = photoDataFrame['recordingLoc'].str.replace(
        'DA', 'TS')
    return photoDataFrame


def destroyNaNs(photoDF):
    dropped_rows = photoDF.loc[photoDF['photoTrace'].apply(
        has_nan)].shape[0]
    print(f"Dropped {dropped_rows} row(s) with NaN in column 'photoTrace.' That's " +
          str(np.round(100*(dropped_rows/photoDF.shape[0]), 2)) + "%!")
    photoDF_noNaN = photoDF[~photoDF['photoTrace'].apply(
        has_nan)]

    return photoDF_noNaN


def generateCSVs(photoDataFrame, event, numberGroups_forSplit, plotOpt=False, only_R = False):
    # event = 'UnNP'
    # subregion = 'TS'
    # numberGroups_forSplit = 3

    # plotOpt = False

    # photoDataFrame = loadPhotoDF('photoDataFrame.feather')

    photoDF_forR = prepFor_R(photoDataFrame, 'photoTrace', -2, 5, None, False)
    feather.write_feather(photoDF_forR, 'photoDF_R.feather')

    if only_R == True:
        return

    targetEvent = photoDataFrame[(photoDataFrame['event'] == event) & (
        photoDataFrame['sesType'] != 'FR1')]

    def assign_session_epoch(photoDF, splitNumb):
        # Define the maximum value in dayOnType
        max_value = np.max(photoDF['dayOnType'].unique())

        # Calculate the size of each chunk
        chunk_size = max_value / splitNumb

        # Create conditions and choices for numpy.select
        conditions = [(photoDF['dayOnType'] > chunk_size * i) &
                      (photoDF['dayOnType'] <= chunk_size * (i + 1)) for i in range(splitNumb)]
        choices = list(range(1, splitNumb + 1))

        # Assign the new column using numpy.select
        photoDF['sessionEpoch'] = np.select(conditions, choices, default=None)

        return photoDF

    targetEvent = assign_session_epoch(targetEvent, numberGroups_forSplit)

    targetEvent = destroyNaNs(targetEvent)

    # apply the has_nan function to column 'C' and drop any rows where the value in column 'C' contains NaN

    
    groupMeansMouse = targetEvent.groupby(['mouse', 'group', 'recordingLoc'])[
        'photoTrace'].mean().reset_index()

    groupMeansEpochs_mouse = targetEvent.groupby(
        ['mouse', 'sessionEpoch', 'group', 'recordingLoc'])['photoTrace'].mean().reset_index()
    groupMeansEpochs = groupMeansEpochs_mouse.groupby(
        ['sessionEpoch', 'group', 'recordingLoc'])['photoTrace'].mean().reset_index()
    groupMeansEpochsSEM = groupMeansEpochs_mouse.groupby(['sessionEpoch', 'group', 'recordingLoc'])['photoTrace'].apply(
        lambda x: np.std(x.tolist(), axis=0) / np.sqrt(len(x))).reset_index()

    groupMeansEpochs['sem'] = groupMeansEpochsSEM['photoTrace']

    # os.chdir(r'D:\Data_analysis\Lerner_Lab\aCUS\_aCUS_Jan2024\for JIMMY\behavior\output_datafiles')
    os.chdir(RI60_DIR)

    groupMeansMouse = peakFinder(groupMeansMouse, [-2, 5])
    groupMeansMouse = AUC_finder(groupMeansMouse, [-0.5, 4])
    groupMeansMouse = kinetics_finder(groupMeansMouse, baseline_window=(-5, 0), stimulus_time=0, peak_search_window=(0, 5))

    peakData_forCSV = groupMeansMouse.copy()
    peakData_forCSV = peakData_forCSV.drop(columns='photoTrace')

    peakCSVName = event+'_photoPeaks.csv'
    peakData_forCSV.to_csv(peakCSVName)

    # groupMeansMouse = groupMeansMouse.reset_index()

    # mouseTracesforCSV = groupMeansMouse.copy()
    # epochTracesforCSV = groupMeansEpochs.copy()
    # epochTracesforCSV_SEMs = groupMeansEpochs.copy()

    #! downsample the photoTraces

    mouseTracesforCSV = prep4csv(groupMeansMouse, 'photoTrace', [
                                 'mouse', 'group', 'recordingLoc'])
    epochTracesforCSV = prep4csv(groupMeansEpochs, 'photoTrace', [
                                 'sessionEpoch', 'group', 'recordingLoc'])
    epochTracesforCSV_SEMs = prep4csv(groupMeansEpochs, 'sem', [
                                      'sessionEpoch', 'group', 'recordingLoc'])

    # mouseTracesforCSV = mouseTracesforCSV.pivot_table(index=['mouse', 'group'], columns='x', values='photoTrace', aggfunc='mean').reset_index()

    # epochTracesforCSV = epochTracesforCSV.pivot_table(index=['sessionEpoch', 'group'], columns='x', values='photoTrace', aggfunc='mean').reset_index()
    # epochTracesforCSV.insert(epochTracesforCSV.columns.get_loc('group') + 1, 'type', 'mean')

    # epochTracesforCSV_SEMs = epochTracesforCSV_SEMs.pivot_table(index=['sessionEpoch', 'group'], columns='x', values='sem', aggfunc='mean').reset_index()
    # epochTracesforCSV_SEMs.insert(epochTracesforCSV_SEMs.columns.get_loc('group') + 1, 'type', 'sem')

    # forCSV_exportEpochs = pd.concat([epochTracesforCSV, pivotedSEMsEpochs], axis=0, ignore_index=True)

    csvName = event+'_photoTraces.csv'

    csvNameEpochs = event+'_epochSplit_photoTraces_Means.csv'
    csvNameEpochs_SEMs = event+'_epochSplit_photoTraces_SEMs.csv'

    mouseTracesforCSV.to_csv(csvName)
    epochTracesforCSV.to_csv(csvNameEpochs)
    epochTracesforCSV_SEMs.to_csv(csvNameEpochs_SEMs)

    groupMeans = groupMeansMouse.groupby(
        ['group'])['photoTrace'].mean().reset_index()

    groupSEMs = groupMeansMouse.groupby(['group'])['photoTrace'].apply(
        lambda x: np.std(x.tolist(), axis=0) / np.sqrt(len(x))).reset_index()
    groupMeans['sem'] = groupSEMs['photoTrace']
    
    # color_map = {
    #      ('stress'): '#8B0000',
    #     #  ('aCUS','inhibition'): '#FF6961',
    #      ('control'): '#00008B',
    #     #  ('control','inhibition'): '#ADD8E6',
    # }

    # groupMeans['groupandDrug'] = groupMeans['group']
    # groupMeans['color'] = groupMeans['group'].map(color_map)

    xList = []
    for i in range(len(groupMeans)):
        xList.append(np.linspace(-5, 10, num=1526))

    groupMeans['x'] = xList

    groupMeans = groupMeans.reset_index(drop=True)

    groupMeans = groupMeans.explode(['photoTrace', 'sem', 'x'])
    colormap = {
        'stress': '#FF0000',
        'control': '#0000FF'
    }
    fig_args = {
        'data_frame': groupMeans,
        'x': 'x',
        'y': 'photoTrace',
        'error_y': 'sem',
        'error_y_mode': 'band',
        'color': 'group',
        # 'facet_row' : 'drug',
        # 'line_group':'groupandDrug',
        'color_discrete_map': colormap
        # {'aCUS + YFP': '#8B0000',
        #                        'aCUS + h4MDi': '#FF6961',
        #                         'control + YFP': '#00008B',
        #                         'control + h4MDi' : '#ADD8E6'}
        # 'showLegend' = 'True'
    }

    print('csv generated!')
    if plotOpt:
        print('plotting...')
        fig = line(**fig_args)

        fig.show()
        print('done!')

#! eventually also be a function, just to do for debugging now
def modelAUC(photoDataFrame, event):
    targetEvent = photoDataFrame[(photoDataFrame['event'] == event) & (
        photoDataFrame['sesType'] != 'FR1')]

    targetEvent = destroyNaNs(targetEvent)

    targetEvent = AUC_finder(targetEvent, [-0.5, 4])

    subregions = targetEvent['recordingLoc'].unique()
    for region in subregions:
        modelDF = targetEvent[targetEvent['recordingLoc'] == region]
        # * and now for the model!
        formula = "AUC ~ C(group) * C(sex)"

        # Fit the mixed-effects model with random effects (mice and repeated measure)
        model = sm.MixedLM.from_formula(
            formula, data=modelDF, groups=modelDF["mouse"])
        result = model.fit()
        print('for ' + region)
        print(result.summary())
        a = 1

    a = 1

def modelFreqAmp():
    freqAmpDF = loadPhotoDF('photoFreqAmpData.feather')
    freqAmpGroupBys = ['mouse', 'group', 'recordingLoc']
    freqAmpGrouped = freqAmpDF.groupby(freqAmpGroupBys).mean().reset_index()

    freqAmpGrouped.to_csv('photoFreqAmp.csv')
    # freqAmpDF = freqAmpDF.rename(columns={'freq (events/min)': 'frequency'})

    freqAmpDF_TS = freqAmpDF[freqAmpDF['recordingLoc'] == 'TS'].reset_index(drop=True)
    freqAmpDF_DMS = freqAmpDF[freqAmpDF['recordingLoc'] == 'DMS'].reset_index(drop=True)




    model = sm.MixedLM.from_formula(
        "transient_frequency ~ C(group)", data=freqAmpDF_TS, groups=freqAmpDF_TS["mouse"])
    result = model.fit()
    print('for TS')
    print(result.summary())

    model_amplitude = sm.MixedLM.from_formula(
        "transient_amplitude ~ C(group)", data=freqAmpDF_TS, groups=freqAmpDF_TS["mouse"])
    result_amplitude = model_amplitude.fit()    
    print('for TS amplitude')
    print(result_amplitude.summary())

    model_DMS = sm.MixedLM.from_formula(
        "transient_frequency ~ C(group)", data=freqAmpDF_DMS, groups=freqAmpDF_DMS["mouse"])
    result_DMS = model_DMS.fit()
    print('for DMS')
    print(result_DMS.summary())

    model_amplitude_DMS = sm.MixedLM.from_formula(
        "transient_amplitude ~ C(group)", data=freqAmpDF_DMS, groups=freqAmpDF_DMS["mouse"])
    result_amplitude_DMS = model_amplitude_DMS.fit()
    print('for DMS amplitude')
    print(result_amplitude_DMS.summary())

    a=1

def createShockDict(output_path):
    os.chdir(output_path)
    
    shockData = pd.read_feather('Shock_dataFrame.feather')
    shockData['dayOnType'] = shockData['dayOnType'].astype(float).astype(int)
    shockDataLate = shockData[shockData['dayOnType'] == 2]
    photoDataFrame = loadPhotoDF('photoDataFrame_baseLineCorr.feather')
    photoMice = photoDataFrame['mouse'].unique().tolist()

    shockDict = {}
    for mouse in photoMice:
        mouseShocks = shockDataLate[shockDataLate['mouse'] == mouse]

        if len(mouseShocks) > 1:
            raise ValueError(f'Multiple shock entries for mouse {mouse}')
        if mouseShocks.empty:
            mouse = mouse.replace('_', '.')
            mouseShocks = shockDataLate[shockDataLate['mouse'] == mouse]

        if (mouseShocks.empty) & (mouse == '725-R'):
            mouse = '725-T'
            mouseShocks = shockDataLate[shockDataLate['mouse'] == mouse]

        if mouseShocks.empty:
            print(f'No shock data for mouse {mouse}')
            shockDict[mouse] = np.nan
            continue



        
        shockDict[mouse] = int(mouseShocks['numShock'])

    return shockDict
# eventList = ['UnNP', 'ReNP', 'UnPE', 'RePE']



def putShocksInPhotoDF(shockDict):
    photoDF = loadPhotoDF('photoDataFrame_baseLineCorr.feather')
    photoDF_for_R = loadPhotoDF(fileName='photoDF_R.feather')
    def map_shock_count(row):
        mouse = row['mouse']
        return shockDict.get(mouse, np.nan)

    photoDF_for_R['numShocks'] = photoDF_for_R.apply(map_shock_count, axis=1)
    photoDF['numShocks'] = photoDF.apply(map_shock_count, axis=1)
    photoDF.to_feather('photoDataFrame_with_shocks.feather')
    photoDF_for_R.to_feather('photoDF_R_with_shocks.feather')
    # return photoDF

def addCumulPokeCount(photo_df, behavior_data):
    """
    add_cumulative_pokes.py

    Adds a `cumulative_poke` column to photoDataFrame.feather by matching each
    active-side nosepoke event (ReNP, UnNP) to its corresponding entry in
    behaviorData.pickle's `total_poke_overall` column.

    Matching key:  (mouse, sesType, dayOnType, timestamp)
    - mouse     -> behavior 'Subject' (with '.' -> '_' normalization, as Jimmy does)
    - sesType   -> from MSN substring ('_D_'=FR1, '_E_'=RI30, '_F_'=RI60)
    - dayOnType -> behavior 'Experiment'
    - timestamp -> matched against active-side nose_timestamps within TOL seconds

    Rows that are not active-side nosepokes (InNP, RePE, UnPE) get NaN because
    `total_poke_overall` is only defined for active pokes in the behavior pipeline.

    Usage:
        python add_cumulative_pokes.py \
            --photo  photoDataFrame.feather \
            --behav  behaviorData.pickle \
            --out    photoDataFrame_withCumPoke.feather
    """
    import argparse
    import pickle
    from collections import defaultdict

    import numpy as np
    import pandas as pd

    TS_TOL = 0.005  # seconds; MedPC timestamps are centiseconds, so 5 ms is safe


    def ses_type_from_msn(msn: str):
        if "MagTraining" in msn:
            return None
        if "_D_" in msn:
            return "FR1"
        if "_E_" in msn:
            return "RI30"
        if "_F_" in msn:
            return "RI60"
        return None


    def normalize_mouse(subject_raw) -> str:
        s = str(subject_raw)
        if "." in s:
            s = s.replace(".", "_")
        return s


    def active_side(session) -> Optional[str]:
        """Return 'Left' or 'Right' for the active side, matching Jimmy's logic."""
        try:
            if float(session["Right Rewards"].iloc[0]) > 0:
                return "Right"
        except (KeyError, ValueError, TypeError):
            pass
        try:
            if float(session["left rewards"].iloc[0]) > 0:
                return "Left"
        except (KeyError, ValueError, TypeError):
            pass
        return None


    def build_lookup(behavior_data):
        """
        Returns dict keyed by (mouse, sesType, dayOnType) ->
            (timestamps_array, total_poke_overall_array)   (both 1-D, same length)
        If multiple behavior sessions collide on the same key, the last one wins
        and a warning is printed.
        """
        lookup: Dict[tuple, List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
        collisions = 0
        skipped_no_tpo = 0
        skipped_no_side = 0

        for sess in behavior_data:
            msn = str(sess["MSN"].iloc[0])
            sesType = ses_type_from_msn(msn)
            if sesType is None:
                continue

            if "total_poke_overall" not in sess.columns:
                skipped_no_tpo += 1
                continue

            side = active_side(sess)
            if side is None:
                skipped_no_side += 1
                continue

            ts_col = f"{side}_nose_timestamps"
            if ts_col not in sess.columns:
                continue

            ts = pd.to_numeric(sess[ts_col], errors="coerce").to_numpy()
            tpo = pd.to_numeric(sess["total_poke_overall"], errors="coerce").to_numpy()

            mask = ~np.isnan(ts) & ~np.isnan(tpo)
            ts = ts[mask]
            tpo = tpo[mask]
            if ts.size == 0:
                continue

            order = np.argsort(ts, kind="stable")
            ts = ts[order]
            tpo = tpo[order]

            mouse = normalize_mouse(sess["Subject"].iloc[0])
            try:
                day = int(sess["Experiment"].iloc[0])
            except (ValueError, TypeError):
                continue

            key = (mouse, sesType, day)
            if key in lookup:
                collisions += 1
            lookup[key].append((ts, tpo))

        if skipped_no_tpo:
            print(f"  [warn] {skipped_no_tpo} behavior sessions had no total_poke_overall column")
        if skipped_no_side:
            print(f"  [warn] {skipped_no_side} behavior sessions had no resolvable active side")
        if collisions:
            print(f"  [warn] {collisions} key collisions on (mouse, sesType, dayOnType); last wins")

        return lookup


    def merge_cumulative_pokes(photo_df: pd.DataFrame, lookup: dict) -> pd.DataFrame:
        photo_df = photo_df.copy()
        n = len(photo_df)
        cum = np.full(n, np.nan, dtype=np.float64)

        active_events = {"ReNP", "UnNP"}
        mouse_arr = photo_df["mouse"].astype(str).to_numpy()
        sesType_arr = photo_df["sesType"].astype(str).to_numpy()
        day_arr = photo_df["dayOnType"].astype(int).to_numpy()
        event_arr = photo_df["event"].astype(str).to_numpy()
        ts_arr = photo_df["timestamp"].astype(float).to_numpy()

        # Group row indices by key so we only look up each session once
        groups: Dict[tuple, List[int]] = defaultdict(list)
        for i in range(n):
            if event_arr[i] not in active_events:
                continue
            groups[(mouse_arr[i], sesType_arr[i], day_arr[i])].append(i)

        missing_keys = 0
        unmatched_ts = 0
        matched = 0

        for key, row_idxs in groups.items():
            candidates_list = lookup.get(key)
            if candidates_list is None:
                alt_key = (key[0].replace("_", "."), key[1], key[2])
                candidates_list = lookup.get(alt_key)
            if not candidates_list:
                missing_keys += len(row_idxs)
                continue

            for i in row_idxs:
                t = ts_arr[i]
                best_val = np.nan
                best_d = np.inf
                for behav_ts, behav_tpo in candidates_list:
                    j = np.searchsorted(behav_ts, t)
                    for c in (j, j - 1):
                        if 0 <= c < behav_ts.size:
                            d = abs(behav_ts[c] - t)
                            if d < best_d:
                                best_d = d
                                best_val = behav_tpo[c]
                if best_d <= TS_TOL:
                    cum[i] = best_val
                    matched += 1
                else:
                    unmatched_ts += 1

        photo_df["cumulative_poke"] = cum

        n_active = int(sum(event_arr[i] in active_events for i in range(n)))
        print(f"  active-side event rows: {n_active}")
        print(f"  matched: {matched}")
        print(f"  unmatched timestamp (no behav row within {TS_TOL}s): {unmatched_ts}")
        print(f"  rows whose (mouse,sesType,dayOnType) not in behavior lookup: {missing_keys}")
        return photo_df



    print(f"  {len(photo_df)} rows, {len(photo_df.columns)} cols")


    print(f"  {len(behavior_data)} sessions")

    print("Building (mouse, sesType, dayOnType) -> (ts, tpo) lookup...")
    lookup = build_lookup(behavior_data)
    print(f"  {len(lookup)} session keys")

    print("Merging cumulative pokes into photoDF...")
    out = merge_cumulative_pokes(photo_df, lookup)
    return out



def removeMouse(photoDF, mice_to_remove):
    """
    Remove specific mouse/recording location combinations from photoDF.
    
    Parameters:
    -----------
    photoDF : DataFrame
        The photo dataframe
    mice_to_remove : list of tuples
        List of (mouseID, recLoc) tuples. If recLoc is None, removes all entries for that mouse.
    
    Returns:
    --------
    DataFrame with specified mice/locations removed
    """
    photoDF_filtered = photoDF.copy()
    for mouseID, recLoc in mice_to_remove:
        if recLoc is not None:
            mask = (photoDF_filtered['mouse'] == mouseID) & (photoDF_filtered['recordingLoc'] == recLoc)
        else:
            mask = photoDF_filtered['mouse'] == mouseID
        photoDF_filtered = photoDF_filtered[~mask]
    return photoDF_filtered

def addPctOriginalBW(photoDataFrame, xlsx_path='JN_Weight_logs.xlsx'):
    """
    Add `pct_original_bw` to photoDataFrame by linking on (mouse, date) to BW
    entries in the JN weight-log workbook. Sessions with no BW that day -> NaN.

    Requires `date` column in photoDataFrame (YYMMDD strings, as in
    photoDF_R_with_rates.feather).
    """
    import re, datetime as dt
    import pandas as pd

    def _norm_label(x):
        return '' if pd.isna(x) else re.sub(r'\s+', ' ', str(x)).strip()

    def _to_date(x):
        return pd.Timestamp(x) if isinstance(x, (pd.Timestamp, dt.datetime, dt.date)) else None

    def _norm_mouse(m):
        # strip strain prefix ('WT 859-L' -> '859-L'), parentheticals, unify '.' -> '_'
        s = re.sub(r'\(.*\)', '', str(m)).strip()
        if ' ' in s:
            s = s.split()[-1]
        return s.replace('.', '_')

    def _parse_sheet(sheet):
        raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)
        if raw.shape[0] < 3:
            return pd.DataFrame()
        labels = raw.iloc[1].map(_norm_label)
        obw_idx = [i for i, l in enumerate(labels) if l == 'Original BW']
        if not obw_idx:
            return pd.DataFrame()
        obw_col = obw_idx[0]
        bw_cols = [(i, _to_date(raw.iat[0, i]))
                   for i, l in enumerate(labels) if l == 'BW']
        bw_cols = [(i, d) for i, d in bw_cols if d is not None]
        if not bw_cols:
            return pd.DataFrame()
        rows = []
        for r in range(2, raw.shape[0]):
            m = raw.iat[r, 0]
            if pd.isna(m) or not str(m).strip():
                continue
            o = raw.iat[r, obw_col]
            if pd.isna(o):
                continue
            try:
                o = float(o)
            except (ValueError, TypeError):
                continue
            for ci, d in bw_cols:
                v = raw.iat[r, ci]
                if pd.isna(v):
                    continue
                try:
                    vf = float(v)
                except (ValueError, TypeError):
                    continue  # handles '-', 'v', etc.
                rows.append({
                    'mouse_norm':  _norm_mouse(m),
                    'date':        d.strftime('%y%m%d'),
                    'original_bw': o,
                    'bw':          vf,
                })
        return pd.DataFrame(rows)

    # Build long weights table from all cohort sheets
    xl = pd.ExcelFile(xlsx_path)
    parts = [_parse_sheet(s) for s in xl.sheet_names if s.lower() != 'template']
    weights = pd.concat([p for p in parts if len(p)], ignore_index=True)

    # Collapse any (mouse, date) duplicates (none expected for the photo cohort,
    # but this guards against future cohort additions).
    weights = (weights
               .groupby(['mouse_norm', 'date'], as_index=False)
               .agg(original_bw=('original_bw', 'first'),
                    bw=('bw', 'mean')))

    # Merge
    out = photoDataFrame.copy()
    out['_mouse_norm'] = out['mouse'].map(_norm_mouse)

    n_before = len(out)
    out = out.merge(
        weights,
        left_on=['_mouse_norm', 'date'],
        right_on=['mouse_norm', 'date'],
        how='left',
        validate='many_to_one',
    )
    assert len(out) == n_before, f"merge changed row count {n_before} -> {len(out)}"

    out['pct_original_bw'] = out['bw'] / out['original_bw'] * 100
    out = out.drop(columns=['_mouse_norm', 'mouse_norm', 'original_bw', 'bw'])

    n_ok = out['pct_original_bw'].notna().sum()
    print(f'pct_original_bw: {n_ok}/{len(out)} rows ({n_ok/len(out):.1%})')
    return out

def mergePhotoAndComp(option):
    if option == 'comp':
        photoDataFrame = loadPhotoDF('photoDF_R_with_rates.feather')
        df_training = feather.read_feather('RI30_RI60_dataFrame.feather')

        df_irt_session = pd.read_csv('IRT_distribution_per_session.csv')
        df_hmm_session = pd.read_csv('HMM_metrics_per_session.csv')

        # ensure dayOnType is the same type across all dataframes
        photoDataFrame['dayOnType'] = photoDataFrame['dayOnType'].astype(int)
        df_irt_session['dayOnType'] = df_irt_session['dayOnType'].astype(int)
        df_hmm_session['dayOnType'] = df_hmm_session['dayOnType'].astype(int)
        df_training['dayOnType'] = df_training['dayOnType'].astype(float).astype(int)

        # link the two dataframes to the photoDataFrame
        photoDataFrame = photoDataFrame.merge(df_irt_session, on=['mouse', 'dayOnType'])
        photoDataFrame = photoDataFrame.merge(df_hmm_session, on=['mouse', 'dayOnType'])
        photoDataFrame = photoDataFrame.merge(df_training[['mouse', 'dayOnType', 'pokeRate', 'entryRate']].drop_duplicates(), on=['mouse', 'dayOnType'])
        photoDataFrame = photoDataFrame.drop(columns=['group_y', 'sex_y', 'group'])
        photoDataFrame = photoDataFrame.rename(columns={'group_x': 'group', 'sex_x': 'sex'})
        
        photoDataFrame.to_feather('photoDF_R_with_IRT_HMM.feather')
        print('merged!')
    elif option == 'rates':
        photoDataFrame = loadPhotoDF('photoDF_R.feather')
        df_training = feather.read_feather('RI30_RI60_dataFrame.feather')

        photoDataFrame['dayOnType'] = photoDataFrame['dayOnType'].astype(int)
        df_training['dayOnType'] = df_training['dayOnType'].astype(float).astype(int)

        # Build the right-hand rates table with EXACTLY ONE row per
        # (mouse, sesType, dayOnType). If duplicates exist upstream
        # (e.g. same dayOnType assigned to two different recording dates for
        # some mice), this collapses them. Using .first() is deterministic but
        # arbitrary — the real fix is upstream in the dayOnType labeling.
        rates = (
            df_training[['mouse', 'sesType', 'dayOnType', 'pokeRate', 'entryRate']]
            .groupby(['mouse', 'sesType', 'dayOnType'], as_index=False)
            .first()
        )

        # Sanity check: verify the right side is now unique on the merge key
        assert not rates.duplicated(subset=['mouse', 'sesType', 'dayOnType']).any(), \
            "rates table still has duplicate (mouse, sesType, dayOnType) — check upstream"

        n_before = len(photoDataFrame)
        photoDataFrame = photoDataFrame.merge(
            rates,
            on=['mouse', 'sesType', 'dayOnType'],
            how='left',
            validate='many_to_one',  # will raise if right side is non-unique
        )
        weight_path = os.path.join(RI60_DIR, "JN_Weight_logs.xlsx")

        photoDataFrame = addPctOriginalBW(photoDataFrame, weight_path)   # <- add
        photoDataFrame.to_feather('photoDF_R_with_rates.feather')
        n_after = len(photoDataFrame)
        assert n_before == n_after, f"merge changed row count {n_before} -> {n_after} (should be equal with many_to_one)"

        photoDataFrame.to_feather('photoDF_R_with_rates.feather')
        print('merged!')



    
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Helper function assuming it exists in your scope or imported
# def plot_dataset(...): 
#     pass

def plot_FMMs(output_path):
    output_path_FMM = os.path.join(output_path, 'FMM_models')
    os.chdir(output_path_FMM)

    # get a list of all the csv files in the directory
    all_files = [f for f in os.listdir(output_path_FMM) if f.endswith('.csv')]
    # Exclude Intercept files
    FMM_files = [f for f in all_files if 'Intercept' not in f]

    # delete intercept files from the directory
    for f in all_files:
        if 'Intercept' in f:
            os.remove(f)
    
    # Parse files to identify components dynamically
    parsed_files = []
    
    for f in FMM_files:
        parts = f.replace('.csv', '').split('_')
        
        # Search for the anchor pattern: group ('naive'/'stress') followed by region ('DMS'/'TS')
        anchor_idx = -1
        for i in range(len(parts) - 1):
            if parts[i] in ['naive', 'stress'] and parts[i+1] in ['DMS', 'TS']:
                anchor_idx = i
                break
        
        if anchor_idx != -1:
            # Extract components based on anchor position
            prefix_parts = parts[:anchor_idx]
            group = parts[anchor_idx]
            region = parts[anchor_idx+1]
            suffix_parts = parts[anchor_idx+2:]
            
            # Detect zscored and split the suffix cleanly around it.
            # Filenames look like: {prefix}_{group}_{region}_{predictor}_{zscored}_{component}
            # where {predictor} may itself contain underscores (e.g. "reward_rate")
            # and {component} is the fastFMM-exported object name (also possibly
            # containing underscores, e.g. "reward_rate"). Splitting on zscored
            # as a delimiter gives us predictor_parts and component_parts
            # unambiguously. For non-zscored files we fall back to the old
            # behavior where suffix_parts is treated as-is.
            is_zscored = 'zscored' in suffix_parts
            if is_zscored:
                z_idx = suffix_parts.index('zscored')
                predictor_parts = suffix_parts[:z_idx]
                component_parts = suffix_parts[z_idx+1:]
                # Canonicalized suffix used for pair matching and plot grouping:
                # predictor + component, with zscored removed.
                canonical_suffix_parts = predictor_parts + component_parts
            else:
                predictor_parts = []  # unused for non-zscored path
                component_parts = []
                canonical_suffix_parts = suffix_parts
            
            parsed_files.append({
                'filename': f,
                'prefix': '_'.join(prefix_parts),
                'prefix_parts': prefix_parts,
                'group': group,
                'region': region,
                'suffix': '_'.join(canonical_suffix_parts),
                'suffix_parts': canonical_suffix_parts,
                'predictor_parts': predictor_parts,
                'component_parts': component_parts,
                'anchor_idx': anchor_idx,
                'is_zscored': is_zscored
            })

    # Find pairs
    paired_data = []
    seen_pairs = set()

    for item in parsed_files:
        current_group = item['group']
        target_group = 'stress' if current_group == 'naive' else 'naive'
        
        # Reconstruct target filename. For zscored files we rebuild the full
        # suffix using the original {predictor}_zscored_{component} layout so
        # the partner filename on disk is matched exactly. For non-zscored we
        # use the suffix as-is.
        if item['is_zscored']:
            target_suffix_parts = item['predictor_parts'] + ['zscored'] + item['component_parts']
        else:
            target_suffix_parts = item['suffix_parts']
        target_parts = item['prefix_parts'] + [target_group, item['region']] + target_suffix_parts
        target_filename = '_'.join(target_parts) + '.csv'
        
        # Define pair key (naive first) to avoid duplicates
        if target_group == 'naive':
            pair_key = (target_filename, item['filename'])
        else:
            pair_key = (item['filename'], target_filename)
            
        if pair_key in seen_pairs:
            continue
            
        if target_filename in FMM_files:
            paired_data.append({
                'naive_file': pair_key[0],
                'stress_file': pair_key[1],
                'prefix': item['prefix'],
                'suffix': item['suffix'],
                'region': item['region'],
                'is_zscored': item['is_zscored']
            })
            seen_pairs.add(pair_key)
        else:
            # Check if we've already warned about this missing partner (checking the reverse lookup)
            reverse_pair_check = (target_filename, item['filename']) if current_group == 'naive' else (item['filename'], target_filename)
            if reverse_pair_check not in seen_pairs:
                print(f"No match found for {item['filename']} (looking for {target_filename})")

    # Load and process data
    # Group by (prefix, suffix, is_zscored) to keep raw and zscored variants
    # in separate plot groups so they each get their own figure.
    plot_groups = {} 

    for pair in paired_data:
        try:
            df_naive = pd.read_csv(pair['naive_file'])
            df_stress = pd.read_csv(pair['stress_file'])
            
            # Processing
            df_naive = df_naive.drop(columns=['CI.lower.joint', 'CI.upper.joint'], errors='ignore')
            df_stress = df_stress.drop(columns=['CI.lower.joint', 'CI.upper.joint'], errors='ignore')
            df_naive['error'] = df_naive['beta.hat'] - df_naive['CI.lower.pointwise']
            df_stress['error'] = df_stress['beta.hat'] - df_stress['CI.lower.pointwise']
            
            # Organize for plotting
            group_key = (pair['prefix'], pair['suffix'], pair['is_zscored'])
            if group_key not in plot_groups:
                plot_groups[group_key] = {}
            
            plot_groups[group_key][pair['region']] = {
                'naive': df_naive,
                'stress': df_stress
            }
            
        except Exception as e:
            print(f"Error loading {pair['naive_file']} or {pair['stress_file']}: {e}")

    # Plotting
    for (prefix, suffix, is_zscored), regions_data in plot_groups.items():
        has_dms = 'DMS' in regions_data
        has_ts = 'TS' in regions_data
        
        # Format strings for titles/files
        title_prefix = prefix.replace('_', ' ') + " " if prefix else ""
        file_prefix = prefix + "_" if prefix else ""
        
        suffix_title = suffix.replace('_', ' ')
        zscore_title_tag = " (z-scored)" if is_zscored else ""
        zscore_file_tag = "_zscored" if is_zscored else ""

        fig_title = f"{file_prefix}DMS_TS_{suffix}{zscore_file_tag}_Combined_FMM.png"
        if fig_title in os.listdir(output_path_FMM):
            print(f"Plot {fig_title} already exists, skipping...")
            continue
        else:
            print(f"Plotting {fig_title}...")

        if has_dms and has_ts:
            # 2x2 Plot
            fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
            
            # Row 1: DMS
            plot_dataset(axes[0, 0], regions_data['DMS']['naive'], 'DMS Naive', color='black')
            plot_dataset(axes[0, 1], regions_data['DMS']['stress'], 'DMS Stress', color='red')
            
            # Row 2: TS
            plot_dataset(axes[1, 0], regions_data['TS']['naive'], 'TS Naive', color='black')
            plot_dataset(axes[1, 1], regions_data['TS']['stress'], 'TS Stress', color='red')
            
            # Add labels for rows
            axes[0, 0].set_ylabel('DMS\nbeta')
            axes[1, 0].set_ylabel('TS\nbeta')
            
            fig.suptitle(f"{title_prefix}{suffix_title}{zscore_title_tag} Comparison")
            plt.tight_layout()
            plt.savefig(fig_title)
            plt.close()
            
        else:
            # Fallback to individual 1x2 plots for whatever regions exist
            for region, data in regions_data.items():
                fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
                
                plot_dataset(axes[0], data['naive'], f'{region} Naive', color='black')
                plot_dataset(axes[1], data['stress'], f'{region} Stress', color='red')
                
                axes[0].set_ylabel('beta')
                
                fig.suptitle(f"{title_prefix}{suffix_title}{zscore_title_tag} ({region}) Comparison")
                plt.tight_layout()
                plt.savefig(f"{file_prefix}{region}_{suffix}{zscore_file_tag}_FMM.png")
                plt.close()

    print('all done!')