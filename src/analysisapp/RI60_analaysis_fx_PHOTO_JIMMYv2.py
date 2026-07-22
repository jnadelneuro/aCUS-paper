# a repository for ASAP analysis functions PHOTOMETRY
import os
import numpy as np
import pandas as pd
import pickle
import pyarrow.feather as feather
# import kaleido
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import scipy
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import tkinter as tk
import json
# import itertools
import h5py
from general_JIMMY_functions import *
from scipy.stats import mode


pio.renderers.default = 'browser'


def trimArray(data, timeArr, min_time, max_time):
    minIdx = (np.abs(timeArr - float(min_time))).argmin()
    maxIdx = (np.abs(timeArr - float(max_time))).argmin()

    dataTrim = data[minIdx:maxIdx]
    return dataTrim

def floatify(list):
    float_list = [float(item) for item in list]
    return float_list


def match_timestamps(set_a, set_b):
    """
    Matches timestamps from set_a to set_b by determining the constant offset via mode.

    Parameters:
    - set_a: List of timestamps (reference set with all values)
    - set_b: List of timestamps (set with potential missing values)

    Returns:
    - matched_pairs: List of tuples (timestamp_from_set_a, matched_timestamp_from_set_b)
    """
    # Compute all possible offsets
    set_a = floatify(set_a)
    set_b = floatify(set_b)

    offsets = [ts_b - ts_a for ts_b in set_b for ts_a in set_a]

    offsets = [a for a in offsets if a >= 0]  # * get rid of negative offsets!

    # Calculate the mode of the offsets
    offset_mode = mode(np.round(offsets)).mode

    # Adjust set_b using the mode offset
    adjusted_set_b = [ts_b - offset_mode for ts_b in set_b]

    # Match timestamps
    matched_pairs = []
    used_set_b = []
    for ts_a in set_a:
        # Find the closest timestamp in the adjusted set_b
        closest_match = None
        min_diff = offset_mode + 1.5
        # max_diff = offset_mode + 3
        smallest_abs_diff = float('inf')
        min_diff = offset_mode - 1.5
        max_diff = offset_mode + 1.5
        for ts_b in set_b:
            if ts_b in used_set_b:
                continue
            diff = ts_b - ts_a
            if min_diff < diff < max_diff:
                if abs(diff) < smallest_abs_diff:
                    closest_match = ts_b
                    smallest_abs_diff = abs(diff)
                    used_set_b.append(ts_b)

        # Append the match
        matched_pairs.append(
            (ts_a, closest_match if min_diff != float('inf') else None))

    return matched_pairs


def downsampleArray(arr, factor):
    if arr is None:
        return np.nan
    elif np.size(arr) == 1:
        return np.nan
    else:
        return arr[::factor]


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


def load_mouseInfo_json(inputParameters):
    try:
        folderSelected = inputParameters['folderNames'][0]
    except:
        a = 1
    try:
        folderSelected = sorted(inputParameters['photoFolderNames'])[0]
    except:
        raise (Exception('what the heck! check your folders!'))
    #! check if there's an output_datafiles folder
    output_file_folder = os.path.join(folderSelected, 'output_datafiles')
    os.chdir(output_file_folder)
    with open("mouse_info.json", "r") as f:
        data = json.load(f)
    return data


def createPhotoDF(inputParameters):
    # %%
    mouseInfoDict = load_mouseInfo_json(inputParameters)
    dsRate = 10
    PSTH_start = -5  # seconds
    PSTH_end = 10  # seconds
    # %%
    # * going to start by defining the folder that should have a "behavior" and "photo" subfolder. I'll do this manually for now, eventually integrate into JIMMY
    # ? folderPath = r'C:\Users\jan7154\Documents\ASAP_Analysis'
    folderPathBeh = sorted(inputParameters['photoFolderNames'])[0]
    folderPathPho = sorted(inputParameters['photoFolderNames'])[1]

    # TODO: add checks to make sure these folders exist and raise exceptions if not
    try:
        behaviorPath = os.path.normcase(
            os.path.join(folderPathBeh, 'output_datafiles'))
        photoPath = os.path.normcase(os.path.join(folderPathPho, 'processed'))
    except:
        raise (Exception(
            'you selected something wrong, give me your "behavior" and "photo" folders!'))
    # * extract behavior data
    # TODO: add check that makes sure the files I'm looking for actually exist
    # * go to the directory whooooop whoop
    os.chdir(behaviorPath)
    # * load .csv file of good SNc recordings

    # * load up that diggity dang dataframe
    with open('behaviorData_RR.pickle', 'rb') as file:
        behaviorData = pickle.load(file)

    # ---- Incremental processing: skip sessions already in the EVENT feather ----
    # The skip is keyed on the EVENT feather only (identity = mouse, sesType,
    # dayOnType, date). A session missing from it -- brand new, OR one whose event
    # extraction produced nothing last time -- is (re)processed, so event-failed
    # sessions retry automatically instead of being stuck. The freq/amp feather is
    # per-session too, but it does NOT carry the identity: it only needs to exist
    # (so skipped sessions' freq/amp rows are preserved on append) and is deduped
    # on write. If the event feather lacks the identity or the freq/amp feather is
    # missing, we full-rebuild both. To force a full rebuild, delete both feathers.
    def _session_keys(df):
        need = {'mouse', 'sesType', 'dayOnType', 'date'}
        if df is None or df.empty or not need.issubset(df.columns):
            return None
        return set(zip(df['mouse'].astype(str).tolist(),
                       df['sesType'].astype(str).tolist(),
                       df['dayOnType'].astype(float).astype(int).tolist(),
                       df['date'].astype(str).tolist()))

    existing_event = feather.read_feather('photoDataFrame_baseLineCorr.feather') \
        if os.path.exists('photoDataFrame_baseLineCorr.feather') else None
    existing_freqamp = feather.read_feather('photoFreqAmpData.feather') \
        if os.path.exists('photoFreqAmpData.feather') else None
    event_keys = _session_keys(existing_event)
    if event_keys is not None and existing_freqamp is not None and not existing_freqamp.empty:
        processed_sessions = event_keys
        print(f'[incremental] {len(processed_sessions)} session(s) already in the event '
              f'feather will be skipped; sessions missing from it (new, or previously '
              f'event-failed) are (re)processed. freq/amp is deduped on write.')
    else:
        # Event feather missing/lacks identity, or freq/amp feather missing ->
        # full rebuild that overwrites both feathers (keeps them in sync).
        processed_sessions = set()
        existing_event = None
        existing_freqamp = None
        print('[incremental] full rebuild: both feathers will be regenerated.')

    # ---- Force-reprocess override (optional) ----
    # Sessions missing from the event feather already retry automatically, so this
    # is only needed to force a mouse that IS already in the event feather to be
    # rebuilt. List mouse ids (as in the 'mouse' column) to drop them from the skip
    # set and remove their existing rows from both feathers. Empty = normal run.
    FORCE_REPROCESS_MICE = set()
    if FORCE_REPROCESS_MICE:
        processed_sessions = {k for k in processed_sessions
                              if k[0] not in FORCE_REPROCESS_MICE}
        if existing_event is not None:
            existing_event = existing_event[
                ~existing_event['mouse'].astype(str).isin(FORCE_REPROCESS_MICE)]
        if existing_freqamp is not None:
            existing_freqamp = existing_freqamp[
                ~existing_freqamp['mouse'].astype(str).isin(FORCE_REPROCESS_MICE)]
        print(f'[incremental] force-reprocessing mice {sorted(FORCE_REPROCESS_MICE)}: '
              f'removed from skip set and dropped from existing feathers so they rebuild fresh.')

    def make_range(n):
        return list(range(1, n+1))

    # b

    # %%
    #! need to make function n

    # behaviorData = behaviorData.assign(sinceBlockChange=behaviorData['blockTypeArray'].apply(
    #     lambda x: time_since_last_change_calc(x)))

    # OK so I'm gonna have to translate the info from this dataFrame into info I can use to match to photometry
    # What do I need: I need to classify pokes as LPUn, LPRew, etc
    # %%
    # * initialize lists for dataframe columns
    mouseList = []
    sexList = []
    groupList = []
    # drugList = []
    sesTypeList = []
    dayOnTypeList = []
    eventList = []
    photoArrayList = []
    recLocList = []
    timestampList = []
    evNumList = []
    timeSinceRewPokeList = []
    timeSinceRewEntryList = []
    rewardRateList = []
    instantPR_list = []
    cumulative_poke_list = []
    dateList = []



    #* session based, not trial based
    transientFreqList = []
    transientAmpList = []
    mouseList_ses = []
    sexList_ses = []
    groupList_ses = []
    sesTypeList_ses = []
    dayOnTypeList_ses = []
    recLocList_ses = []

    #! making mouse dict here where I say who has good signal where. Position 0 is DMS, 1 is SNc

    # %%
    # * get the photometry
    os.chdir(photoPath)
    sessionFolders = [d for d in os.listdir(
        photoPath) if os.path.isdir(os.path.join(photoPath, d))]

    # Get the length of the subdirs list to get the number of subdirectories in the directory
    numberofSessions = len(sessionFolders)
    # %%

    def separate_reward_and_not(allTimestamps, rewardTimestamps):
        # purpose of this is to extract only the unrewarded TS's from all timestamps
        unrewTS = [ts for ts in allTimestamps if ts not in rewardTimestamps]

        return unrewTS

    def stripNaN(timestampArray):
        timestampArray_stripped = timestampArray[~timestampArray.isna()]
        return timestampArray_stripped
    sesCounter = 0
    # failCount=0
    for idxNum in range(len(behaviorData)):
        sessionData = behaviorData[idxNum]
        if "_D_" in sessionData['MSN'][0]:
            sesType = "FR1"
        elif "_E_" in sessionData['MSN'][0]:
            sesType = "RI30"
        elif "_F_" in sessionData['MSN'][0]:
            sesType = "RI60"
        if 'MagTraining' in sessionData['MSN'][0]:
            continue
        # Incremental skip BEFORE the expensive per-session work below (the two
        # O(n*m) separate_reward_and_not calls and the per-iteration os.listdir).
        # Uses only cheap identity fields. The Experiment access is guarded so a
        # missing/non-numeric value on a photometry-less session can't abort the
        # whole run -- it just isn't skipped here and falls through as before.
        mouseDot = str(sessionData['Subject'][0])
        mouse = mouseDot.replace('.', "_") if "." in mouseDot else mouseDot
        sesDate = str(sessionData['Start Date'][0])
        if '.' in sesDate:
            sesDate = sesDate.split('.')[0]
        try:
            this_dayOnType = int(float(sessionData['Experiment'][0]))
        except (KeyError, ValueError, TypeError):
            this_dayOnType = None
        if this_dayOnType is not None and \
                (str(mouse), str(sesType), this_dayOnType, sesDate) in processed_sessions:
            print(f'session {mouse} {sesType} day {this_dayOnType} {sesDate} already processed, skipping!')
            continue
        # set column names for extracting timestamps later on
        if float(sessionData['Right Rewards'][0]) > 0:
            inactive_TSName = 'Left_nose_timestamps'
            active_TSName = 'Right_nose_timestamps'
            rewPoke_TSName = 'Right_reward_timestamps'

        elif float(sessionData['left rewards'][0]) > 0:
            inactive_TSName = 'Right_nose_timestamps'
            active_TSName = 'Left_nose_timestamps'
            rewPoke_TSName = 'Left_reward_timestamps'
        ReNP_TS = stripNaN(sessionData[rewPoke_TSName])
        UnNP_TS = separate_reward_and_not(
            stripNaN(sessionData[active_TSName]), stripNaN(sessionData[rewPoke_TSName]))
        InNP_TS = stripNaN(sessionData[inactive_TSName])
        RePE_TS = stripNaN(sessionData['Rewarded_port_entry_timestamps'])
        UnPE_TS = separate_reward_and_not(stripNaN(sessionData['Port_entry_timestamps']), stripNaN(
            sessionData['Rewarded_port_entry_timestamps']))
        allNP_TS = stripNaN(sessionData[active_TSName])
        if 'reward_rate_trace' in sessionData.columns:
            rewardRates = stripNaN(sessionData['reward_rate_trace'])
        else: rewardRates = []
        if 'poke_rate_trace' in sessionData.columns:
            instant_pokerate = stripNaN(sessionData['poke_rate_trace'])
        else: instant_pokerate = []
        if 'total_poke_overall' in sessionData.columns:
            cumulative_pokes = stripNaN(sessionData['total_poke_overall'])
        else: cumulative_pokes = []

        timestampDict = {'ReNP': ReNP_TS,
                         'UnNP': UnNP_TS,
                         'InNP': InNP_TS,
                         'RePE': RePE_TS,
                         'UnPE': UnPE_TS}

        # make the all timestamp lists into just unrewarded

        # ? jesus christ this is gonna be complicated...
        # ? I guess for now just start with one session
        # idxNum = 95  # ! make this into a for loop eventually to get every session
        sesPhotoID = (mouse + '_JN-' + sesDate)
        sesPhotoID1 = ("JN_WT_" + mouse + '-' + sesDate)
        # print(sesPhotoID)

        # * now check if there's a photometry recording corresponding to the behavior session
        matching_dir = next((d for d in os.listdir(photoPath) if os.path.isdir(
            os.path.join(photoPath, d)) and (sesPhotoID in d or sesPhotoID1 in d)), None)

        # Check if a matching subdirectory was found
        if matching_dir is None:
            continue        # print(idxNum)
        elif mouseDot not in mouseInfoDict.keys():
            continue
        else:
            sesCounter = sesCounter+1
            # If a matching subdirectory was found, open it
            matching_dir_path = os.path.join(photoPath, matching_dir)
            output_path = os.path.join(
                matching_dir_path, (matching_dir + '_output_1'))
            print("Loading " + matching_dir +
                  ', mouse ' + mouse + ' , date '+sesDate)
            if os.path.exists(output_path) == False:
                # * added this because sometimes I add 'g_' to the front of the main folder and it doesn't also have the g_ in front of the output folder
                matching_dir = matching_dir[2:]
                output_path = os.path.join(
                    matching_dir_path, (matching_dir + '_output_1'))
            if os.path.exists(output_path) == False:
                print(
                    'Are you sure this session has been processed? Maybe output is not labeled right (has to be output_1). Skipping!')
                continue

            # Now you can do whatever you want with the matching directory, for example:

            # print(f"No subdirectory containing the string '{sesPhotoID}' was found.")

            a = 1  # goodSNcRecs

        os.chdir(output_path)



        matchingDict = {'InactiveNP': 'InNP',
                        'RewardNP': 'ReNP',
                        'RewardPE': 'RePE',
                        'UnrewardedNP': 'UnNP',
                        'UnrewardedPE': 'UnPE'
                        }

        listofBois = []
        listofevents = []
        if '_' in mouse:
            newMouse = mouse.replace('_', '.')
        # * load up the arrays for each target event
        try:
            recordingLocations = mouseInfoDict[mouse]['recLoc'].copy()
        except:
            try:
                recordingLocations = mouseInfoDict[newMouse]['recLoc'].copy()
            except:
                recordingLocations = mouseInfoDict[mouse]['recLoc']
        for recLoc in recordingLocations:            



            ampFile = f'transientsOccurrences_z_score_{recLoc}.csv'
            if not os.path.exists(ampFile):
                ampFile = f'transientsOccurrences_z_score_DA.csv'  # older 'DA' naming
            # Always append exactly one value so this list stays aligned with the
            # per-session metadata lists below (NaN if neither TS nor DA exists).
            if os.path.exists(ampFile):
                transientAmp = pd.read_csv(ampFile)
                transientAmpList.append(float(np.mean(transientAmp['amplitude'].values)))
            else:
                transientAmpList.append(np.nan)

            freqFile = f'freqAndAmp_z_score_{recLoc}.csv'
            if not os.path.exists(freqFile):
                freqFile = f'freqAndAmp_z_score_DA.csv'  # older 'DA' naming
            # Always append exactly one value so this list stays aligned (NaN if
            # neither TS nor DA exists).
            if os.path.exists(freqFile):
                transientFreq = pd.read_csv(freqFile)
                transientFreqList.append(float(transientFreq['freq (events/min)'].values))
            else:
                transientFreqList.append(np.nan)

            mouseList_ses.append(mouse)
            sexList_ses.append(mouseInfoDict[mouseDot]['sex'])
            groupList_ses.append(mouseInfoDict[mouseDot]['group'])
            sesTypeList_ses.append(sesType)
            dayOnTypeList_ses.append(sessionData['Experiment'][0])
            recLocList_ses.append(recLoc)
            for eventType in matchingDict:
                targetFile = f'{eventType}_{recLoc}_z_score_{recLoc}.h5'
                targetFile = f'{eventType}_{recLoc}_baselineCorrected_z_score_{recLoc}.h5'
                if (recLoc == 'TS') & (os.path.exists(targetFile) == False):
                    targetFile = f'{eventType}_DA_z_score_DA.h5'
                # targetFile = (eventType+'_'+recLoc + ''+recLoc+'.h5')
                # targetEvTimestamps = (ttl_label + '.hdf5')
                targetEvTimestamps = (eventType + '_' + recLoc + '.hdf5')
                if (recLoc == 'TS') & (os.path.exists(targetEvTimestamps) == False):
                    targetEvTimestamps = (eventType + '_DA.hdf5')
                sampRateFile = '405A.hdf5'

                timestamp_and_trace = []
                if not os.path.exists(targetFile):
                    print('Event file ' + targetFile +
                          ' doesn\'t seem to exist in ' + matching_dir)
                    continue
                with h5py.File(sampRateFile) as h5Read:
                    sampling_rate = h5Read[('sampling_rate')][()]
                with h5py.File(targetEvTimestamps) as h5Read:
                    # * this is to match traces to med timestamps to make sure events are assigned correctly
                    event_timestamps = h5Read['/ts'][:]
                    timestamp_and_trace.append(event_timestamps)
                with h5py.File(targetFile) as h5Read:
                    data = h5Read['/df/block0_values'][:, 0:-3]
                    timestamps = h5Read['/df/block0_values'][:, -3]
                    
                    trimData = trimArray(data, timestamps, PSTH_start, PSTH_end)
                    dsTrimData = downsampleArray(
                        trimData, dsRate)

                    timestamp_and_trace.append(dsTrimData)
                    #! -3 because the last 3 bois are timestamps, mean, err. Remove!

                    listofevents.append(recLoc + '_' +
                                        matchingDict[eventType])
                    

                listofBois.append(timestamp_and_trace)
                # listofBois.append(timestamp_and_trace)
        photo_and_events = dict(zip(listofevents, listofBois))
        numEvents = [np.shape(photo_and_events[eN][1])[1]
                     for eN in list(photo_and_events.keys())]
        eventNumDict = dict(zip(listofevents, numEvents))

        #! NOTE!! IN the guppy data, the arrays of each event are in an h5 file, reading it then ['/df/block0_values'][:] will extract!!
        # * I guess start laying out the dataframe?
        # ? so start with the first trial? I guess go through each event and find the
        # ? trial it aligns to, then use that to fill in information

        # ? I guess the plan is to go through each event type, then within the event go through
        # ? each event, then ID the shit I need to and put it in the dataframe lists

        #! to avoid for loops for now, I'll start with DMS_LPUn
        for evNum in range(len(listofevents)):
            # for evName,evArray in photo_and_events
            evName = listofevents[evNum]  # ! eventually replace 0 w/ loop
            evNameUse = evName.split("_", 1)[1]
            recLoc = evName.split("_", 1)[0]

            synapseTS = listofBois[evNum][0]
            synapseTS = [ts for ts in synapseTS if ts != 0]

            behaviorTS = timestampDict[evNameUse]
            behaviorTS = [float(ts) for ts in behaviorTS if float(ts) != 0.0]

            matched_times = match_timestamps(behaviorTS, synapseTS)

            if recLoc == 'SNc':
                a = 1
            evArray = listofBois[evNum][1]  # ! eventually replace 0 w/ loop

            if len(behaviorTS) != len(synapseTS):
                print('synapse values are missing for ' +
                      evName + ', wish us luck with matching :(')
                # print(matched_times[?0:10])
                # failCount = failCount+1
                # print(Warning('your event array and timestamps are not the same lengths! To quote Mario, "oh no"!'))
                # print(f'skipping {evNameUse} for {sesPhotoID} because of unmatched bois. This is the {failCount} time this has happened.')
                # continue
            # evArray = evArray.transpose()

            failCount = 0
            for eventNumb in range(np.shape(evArray)[1]):
                evTs = synapseTS[eventNumb]

                if any(evTs == t[1] for t in matched_times) == False:
                    failCount = failCount+1
                    continue
                matching_medTs = next(
                    (pair for pair in matched_times if pair[1] == evTs), None)[0]
                
                # * find how long it's been since rewarded NP and PE
                sinceNP = abs(max([num for num in np.array(ReNP_TS) - matching_medTs if num < 0], default=np.nan))
                sincePE = abs(max([num for num in np.array(RePE_TS) - matching_medTs if num < 0], default=np.nan))
                mouseList.append(mouse)
                # drugList.append(mouseInfoDict[mouseDot]['drug'])
                sexList.append(mouseInfoDict[mouseDot]['sex'])
                groupList.append(mouseInfoDict[mouseDot]['group'])
                sesTypeList.append(sesType)
                dayOnTypeList.append(sessionData['Experiment'][0])
                recLocList.append(recLoc)
                eventList.append(evNameUse)
                dateList.append(sesDate)

                timestampList.append(matching_medTs)
                timeSinceRewPokeList.append(sinceNP)
                timeSinceRewEntryList.append(sincePE)
                evNumList.append(eventNumb)
                if (evNameUse == 'ReNP') or (evNameUse == 'UnNP'):
                    if rewardRates[allNP_TS == matching_medTs].empty == False:
                        rr_at_poke = float(rewardRates[allNP_TS == matching_medTs])
                    else:
                        rr_at_poke = np.nan
                else:
                    rr_at_poke = np.nan

                if (evNameUse == 'ReNP') or (evNameUse == 'UnNP'):
                    if instant_pokerate[allNP_TS == matching_medTs].empty == False:
                        pr_at_poke = float(instant_pokerate[allNP_TS == matching_medTs])
                    else:
                        pr_at_poke = np.nan
                else:
                    pr_at_poke = np.nan
                if len(cumulative_pokes) != 0:
                    if cumulative_pokes[allNP_TS == matching_medTs].empty == False:
                        cum_poke_at_poke = float(cumulative_pokes[allNP_TS == matching_medTs])  
                        cumulative_poke_list.append(cum_poke_at_poke)
                    else:
                        cumulative_poke_list.append(np.nan)
                instantPR_list.append(pr_at_poke)
                rewardRateList.append(rr_at_poke)
                photoArrayList.append(evArray[:, eventNumb])
            if failCount > 0:
                print(f'{evNameUse} had a fail percent of {failCount/len(behaviorTS)}. Fails were {failCount} and diff was {len(behaviorTS)-len(synapseTS)}.')

    print(sesCounter)
    bigBoi = {'mouse': mouseList,
              'sex': sexList,
              'group': groupList,
              #   'drug' : drugList,
              'sesType': sesTypeList,
              'dayOnType': dayOnTypeList,
              'event': eventList,
              'event_number': evNumList,
              'timestamp': timestampList,
              'time_since_reward_poke': timeSinceRewPokeList,
              'time_since_reward_entry': timeSinceRewEntryList,
              'reward_rate': rewardRateList,
              'instant_poke_rate': instantPR_list,
              'date': dateList,
            #   'cumPoke' : cumulative_poke_list,
              # 'WS_LS' : WS_LS_ID_List,
              'photoTrace': photoArrayList,
              'recordingLoc': recLocList
              }
    bigBoiDtype = {'mouse': 'str',
                   'sex': 'str',
                   'group': 'str',
                   'sesType': 'str',
                   'dayOnType': 'int16',
                   'timestamp': 'float64',
                   'time_since_reward_poke': 'float64',
                   'time_since_reward_entry': 'float64',
                   'event': 'str',
                   'event_number': 'int64',
                   'reward_rate': 'float64',
                   'photoTrace': 'object',
                   'recordingLoc': 'str'}
    bigBoi = pd.DataFrame(bigBoi)
    bigBoi = bigBoi.astype(bigBoiDtype)
    os.chdir(behaviorPath)
    # Append the newly-processed sessions to whatever was already in the feather.
    if existing_event is not None and not existing_event.empty:
        extra_cols = [c for c in existing_event.columns if c not in bigBoi.columns]
        if extra_cols:
            print(f'[incremental] existing feather has post-hoc columns not produced by '
                  f'createPhotoDF ({extra_cols}); appended rows will be NaN there until you '
                  f're-run the step that adds them (e.g. addCumCount for cumulative_poke).')
        bigBoi = pd.concat([existing_event, bigBoi], ignore_index=True)
    print('...saving')
    feather.write_feather(bigBoi, 'photoDataFrame_baseLineCorr.feather')

    freq_amp_dict = {'transient_frequency': transientFreqList,
                     'transient_amplitude': transientAmpList,
                     'mouse': mouseList_ses,
                     'sex': sexList_ses,
                     'group': groupList_ses,
                     'sesType': sesTypeList_ses,
                     'dayOnType': dayOnTypeList_ses,
                     'recordingLoc': recLocList_ses}
    freq_amp_dict = pd.DataFrame(freq_amp_dict)
    if existing_freqamp is not None and not existing_freqamp.empty:
        freq_amp_dict = pd.concat([existing_freqamp, freq_amp_dict], ignore_index=True)
    # A (re)processed session re-appends its freq/amp row; keep the newest so it
    # replaces the old one rather than duplicating it (one row per session x recLoc).
    freq_amp_dict = freq_amp_dict.drop_duplicates(
        subset=['mouse', 'sesType', 'dayOnType', 'recordingLoc'], keep='last')

    feather.write_feather(freq_amp_dict, 'photoFreqAmpData.feather')
    with open('PSTH_info.txt', 'w') as fp:
        fp.write(f'PSTH_start: {PSTH_start}, PSTH_end: {PSTH_end}, dsRate: {dsRate}')
    print('done!')
    # bigBoi.write_feather()
    # %%
    # print(bigBoi['mouse'].unique())


def analyzePhotoDF(inputParameters):

    # TODO: add checks to make sure these folders exist and raise exceptions if not
    behaviorPath = sorted(inputParameters['photoFolderNames'])[0]
    photoPath = sorted(inputParameters['photoFolderNames'])[1]

    # * extract behavior data
    # TODO: add check that makes sure the files I'm looking for actually exist
    # * go to the directory whooooop whoop
    os.chdir(os.path.normcase(os.path.join(behaviorPath, 'output_datafiles')))
    print('loading dataframe...')
    try:
        photoDF = feather.read_feather('photoDataFrame.feather')
    except:
        raise (Exception('no photo DataFrame found, try creating it!'))
    # %%
    print('loaded!')
    groupMeans = pd.DataFrame()

    # * recording location options from input params
    if inputParameters['photo_fig_specs']['loc_sel'] == 'DMS':
        recLocLogic = photoDF['recordingLoc'] == 'DMS'
    elif inputParameters['photo_fig_specs']['loc_sel'] == 'DLS':
        recLocLogic = photoDF['recordingLoc'] == 'DLS'
    elif inputParameters['photo_fig_specs']['loc_sel'] == 'both':
        recLocLogic = photoDF['recordingLoc'] != 'SNc'
    elif inputParameters['photo_fig_specs']['loc_sel'] == 'SNc':
        recLocLogic = photoDF['recordingLoc'] == 'SNc'

    if inputParameters['photo_fig_specs']['event_sel'] == 'rewarded NP':
        eventTypeLogic = (photoDF['outcome'] == 1) & (
            photoDF['event'] != 'PrtR')
    elif inputParameters['photo_fig_specs']['event_sel'] == 'unrew NP':
        eventTypeLogic = (photoDF['outcome'] == 0) & (
            photoDF['event'] != 'PrtR')
    elif inputParameters['photo_fig_specs']['event_sel'] == 'reward retrieval':
        eventTypeLogic = (photoDF['outcome'] == 1) & (
            photoDF['event'] == 'PrtR')
    conditionsDict = {'conditions': (photoDF['dayOnType'] > inputParameters['photo_fig_specs']['days_to_trim']) & (eventTypeLogic) & (recLocLogic),
                      'fig_args': {
        'data_frame': groupMeans,
        'x': 'x',
        'y': 'photoTrace',
        'error_y': 'sem',
        'error_y_mode': 'band',
        'color': inputParameters['photo_fig_specs']['split_sel'],
        'title': inputParameters['photo_fig_specs']['event_sel'],
        'color_discrete_map': {'asyn': '#FF4500',
                               'control': '#4169E1'}

    },
        'groupBys': [inputParameters['photo_fig_specs']['split_rows'], inputParameters['photo_fig_specs']['split_cols'], inputParameters['photo_fig_specs']['split_sel']]
    }

    if inputParameters['photo_fig_specs']['event_sel'] != 'None':
        # %%
        selected_subset = photoDF[conditionsDict['conditions']]

        #! this is how I'm getting rid of traces w/ NaNs for now--just remove. But maybe there's a better way?
        def has_nan(arr):
            return any(np.isnan(arr))

        # apply the has_nan function to column 'C' and drop any rows where the value in column 'C' contains NaN
        dropped_rows = selected_subset.loc[selected_subset['photoTrace'].apply(
            has_nan)].shape[0]
        groupBys = conditionsDict['groupBys']
        groupBys = [x for x in groupBys if x != 'None']
        # print the number of dropped rows and the updated dataframe
        print(f"Dropped {dropped_rows} row(s) with NaN in column 'photoTrace.' That's " +
              str(np.round(100*(dropped_rows/selected_subset.shape[0]), 2)) + "%!")
        selected_subset = selected_subset[~selected_subset['photoTrace'].apply(
            has_nan)]
        groupMeans = selected_subset.groupby(['mouse'] + groupBys, as_index=False)[
            'photoTrace'].mean().reset_index()
        groupSEMs = groupMeans.groupby(groupBys)['photoTrace'].apply(
            lambda x: np.std(x.tolist(), axis=0) / np.sqrt(len(x))).reset_index()
        groupMeans = groupMeans.groupby(
            groupBys)['photoTrace'].mean().reset_index()

        groupMeans['sem'] = groupSEMs['photoTrace']

        # %%
        xList = []
        for i in range(len(groupMeans)):
            xList.append(np.linspace(-10, 20, num=30519))

        groupMeans['x'] = xList
        groupMeans = groupMeans.apply(pd.Series.explode)
        conditionsDict['fig_args']['data_frame'] = groupMeans
        fig_args = conditionsDict['fig_args']

        if inputParameters['photo_fig_specs']['split_rows'] != 'None':
            fig_args['facet_row'] = inputParameters['photo_fig_specs']['split_rows']
        if inputParameters['photo_fig_specs']['split_cols'] != 'None':
            fig_args['facet_col'] = inputParameters['photo_fig_specs']['split_cols']
        print('plotting... (takes a bit!)')
        fig = line(**fig_args)

        # fig.update_layout(
        #     shapes=[
        #         dict(
        #             type='line',
        #             xref='x',
        #             yref='y',
        #             x0=0,
        #             y0=-5,
        #             x1=0,
        #             y1=5,
        #             line=dict(color='black', width=0.5, dash='dash')
        #         )
        #     ]
        # )
        rows, cols = fig._get_subplot_rows_columns()
        for ii in rows:
            fig.update_yaxes(title_text='z-score', row=ii, col=1)
            # fig.update_yaxes(title_text='z-score', row=ii, col=1)
        for ii in cols:
            fig.update_xaxes(title_text='time (s)', row=1, col=ii)
            # fig.update_xaxes(title_text='time (s)', row=2, col=1)

        # fig.update_xaxes(title_text='time (s)')
        # fig.update_yaxes(title_text='z-score')
        figSpecDict = inputParameters['photo_fig_specs']
        #!NEED TO FIGURE OUT WHY TF KALEIDO ISN'T WORKING ON WINDOWS!
        #! write_fig_svg(fig,figSpecDict['event_sel'],figSpecDict['loc_sel'],figSpecDict['split_rows'],figSpecDict['split_cols'],figSpecDict['split_sel'],photoPath)
        fig.show()


# %%
def write_fig_svg(fig, event, recLoc, rowSplit, colSplit, lineSplit, outputPath):
    output_file_path = os.path.join(outputPath, 'photo_figures')
    if not os.path.exists(output_file_path):
        os.mkdir(output_file_path)
    os.chdir(output_file_path)
    if (rowSplit == 'None') & (colSplit == 'None'):
        fig.write_image(event + '_' + recLoc +
                        ' split by ' + lineSplit + '.svg')
    elif (rowSplit == 'None') & (colSplit != 'None'):
        fig.write_image(event + '_' + recLoc + ' split by ' +
                        lineSplit + ' lines and ' + colSplit + ' columns' + '.svg')
    elif (rowSplit != 'None') & (colSplit == 'None'):
        fig.write_image(event + '_' + recLoc + ' split by ' +
                        lineSplit + ' lines and ' + rowSplit + ' rows' + '.svg')
    elif (rowSplit != 'None') & (colSplit != 'None'):
        fig.write_image(event + '_' + recLoc + ' split by ' + lineSplit +
                        ' lines, ' + colSplit + ' columns and ' + rowSplit + ' rows.svg')
