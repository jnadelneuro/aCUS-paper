#!/usr/bin/env python
# coding: utf-8
#* FULLY WORKS FOR RI60 AND ASAP
# In[1]:

import h5py
import os
import numpy as np
import re
import glob
import json
import pandas as pd
from collections import OrderedDict
import re
import toolz
import pickle
from create_operantData_h5 import *

# def find_nearest_after(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return idx


def paramSelection(inputParameters):
    inputParameters = inputParameters

    if inputParameters['behaviorParamSelect'] == 0:  # * if it's ASAP
        parameters = {"A": "Counters",
                      "B": "Left_unreward_timestamps",
                      "C": "Left_reward_timestamps",
                      "D": "Right_unreward_timestamps",
                      "E": "Right_reward_timestamps",
                      "F": "Right_poke_timestamps",
                      "G": "Left_poke_timestamps",
                      "H": "Rewarded_PE_timestamps",
                      "I": "Unrewarded_PE_timestamps",
                      "J": "All_port_entry_timestamps",
                      "K": "Trial_start_timestamps",
                      "L": "Timer_for_timestamps",
                      "M": "number_of_trials_per_block",
                      "N": "block transition times",
                      "O": "block transition type",
                      "P": "trial choice",
                      "Q": "blockTypeArray",
                      "R": "reward_history",
                      "T": "Array for times/clocks",
                      "V": "List of required times in port for trial initiation",
                      "W": "list of right (0) or left (1) for first block",
                      "X": "Array TTL state keepers",
                      "Y": "List to pull from for high reward probability",
                      "Z": "List to pull from for low reward probability"}
    elif inputParameters['behaviorParamSelect'] == 1:  # * if it's on tab 2 and thus RI60
        parameters = {"A": "Left_nose_timestamps",
                      "B": "Left_reward_timestamps",
                      "C": "Right_nose_timestamps",
                      "D": "Right_reward_timestamps",
                      "E": "Port_entry_duration",
                      "F": "Rewarded_port_entry_timestamps",
                      "G": "Port_entry_timestamps",
                      "H": "Number drawn/Timestamps for shocks",
                      "I": "timestamps_5_drawn",
                      "J": "Duration_left_nosepokes",
                      "K": "Timer for timestamps",
                      "L": "Total Time of Session",
                      "M": "Max Session Time",
                      "N": "Duration_right_nosepokes or scrambled timestamps",
                      "O": "Right Rewards",
                      "P": "Port Entries",
                      "Q": "Counter/Shocks",
                      "R": "Right nosepokes",
                      "S": "left rewards",
                      "W": "Left nose pokes"}
    return parameters


def closest_greater_index(array, value):
    array = array.astype(float)
    positive_numbers = array[np.where((array >= value) & (array > 0))]
    if len(positive_numbers) == 0:
        return None
    index = np.argmin(positive_numbers)
    return np.where((array >= value) & (array > 0))[0][index]




def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def stripLine(line, index, inputParameters):
    arr = []
    # print(index)
    res = line[index].strip(":\n").split(": ")
    # print(res)
    key = None
    value = None

    if len(res) > 1:
        key = res[0]
        value = res[1]
        if res[0] in inputParameters.keys():
            key = inputParameters[res[0]]
        index += 1
    else:

        if res[0] in inputParameters.keys():
            key = inputParameters[res[0]]
        else:
            key = res[0]

        index += 1

        value = []
        #print('Index: ', index)
        strip = line[index].strip().split(" ")
        # print(strip)
        while len(strip) > 1 and index < len(line)-1:
            for i in strip[1:]:
                if len(i) > 0:
                    value.append(i)
            index += 1

            strip = line[index].strip().split(" ")

    if key:
        if type(value) is list:
            if 'timestamp' in key:
                value = np.asarray(value, dtype=np.float32)
                value = value[value != 0]
            elif 'block transition times' in key:
                value = np.asarray(value, dtype=np.float32)
                value = value[value != 0]
                value = list(value)
                value.append(5400)
            else:
                value = np.asarray(value, dtype=np.float32)

            return key, value, index
        else:
            return key, value, index
    else:
        return 0, 0, index


def findStartLocation(data):
    location = []
    for i in range(len(data)):
        if i == len(data)-1:
            location.append(i)
            break
        if "Start Date" in data[i]:
            location.append(i)

    # print(location)
    return location


def removeUnrewardedNonTrialPokes(df):
    a = 1

def findNaNsNeeded(array,df):
    nansNeeded = len(df) - len(array)
    array = np.array(array)
    new_array = np.append(array, np.zeros(nansNeeded) + np.nan)
    return new_array


def closest_greater_value(array, value):
    value = float(value)
    closest_greater = float('inf')
    array = array.astype(float)
    for val in array:
        if val > value and val < closest_greater:
            closest_greater = val
            keep = 1
    if closest_greater == float('inf'):
        closest_greater = 'NaN'
        keep = 0
    
    return closest_greater, keep

def getRewardHistory(df):
    if '_B_' in df['MSN'][0] or '_C_' in df['MSN'][0] or '_D_' in df['MSN'][0] or '_F_' in df['MSN'][0]:
        # so I'm a dumbass and didn't think to include a column for if they were rewarded or not for the first round (Jan 2023)
        # basically this will find the closest poke timestamp to trial start then figure out if that poke was left or right and rewarded or unrewarded
        trialStartTimes = np.array(df['Trial_start_timestamps'].dropna())

        # combine timestamps of L and R all pokes to get that
        pokeTimes = np.append(
            df['Right_poke_timestamps'].dropna(), df['Left_poke_timestamps'].dropna())

        # combine timestamps of L and R rewards to get that
        rewardedPokeTimes = np.append(
            df['Right_reward_timestamps'].dropna(), df['Left_reward_timestamps'].dropna())

        left_unrew = df['Left_unreward_timestamps'].dropna()
        right_unrew = df['Right_unreward_timestamps'].dropna()
        unrewardedPokeTimes = np.append(left_unrew, right_unrew)
        choice_history = df['trial choice'].dropna()
        if '_B_' in df['MSN'][0] or '_C_' in df['MSN'][0]:
            rightPokeTimes = df['Right_reward_timestamps'].dropna()
            leftPokeTimes =  df['Left_reward_timestamps'].dropna()
            choice_history = []
            for startTime in trialStartTimes:
                rightValue, keep = closest_greater_value(rightPokeTimes, startTime)
                leftValue, keep = closest_greater_value(leftPokeTimes, startTime)
             
                if rightValue == 'NaN':
                    choice_history.append(1)
                elif leftValue == 'NaN':
                    choice_history.append(0)
                else:
                    rightVsStart = rightValue - startTime
                    leftVsStart = leftValue - startTime
                    if rightVsStart > leftVsStart: #if it's a left poke
                        choice_history.append(1)
                    elif leftVsStart > rightVsStart:
                        choice_history.append(0)   
            
        # to find first poke after trial start
        firstPokeTimes = []
        keeplist = []
        for startTime in trialStartTimes:
            value, keep = closest_greater_value(pokeTimes, startTime)

            firstPokeTimes.append(value)
            keeplist.append(keep)


        reward_history = []
        firstPokeTimes_trim = [firstPokeTimes[i] for i in range(len(firstPokeTimes)) if keeplist[i] == 1]
        # now see if it's in a rewarded list or unrewarded list
        unrew_to_keep = []
       
        for pokeCheck in firstPokeTimes_trim:
            if pokeCheck in rewardedPokeTimes:
                reward_history.append(1)
            elif pokeCheck in unrewardedPokeTimes:
                reward_history.append(0)
                unrew_to_keep.append(pokeCheck)
            else:
                reward_history.append(3)
                Warning('check your poke times, something going on w/ trial alignment')
        # * if there's unrewarded pokes to get rid of
        if len(unrew_to_keep) > 0:
            # * need to figure out which timestamps are L and which are R and split them appropriately
            L_unrewarded_new = [i for i in unrew_to_keep if i in np.array(left_unrew)]
            R_unrewarded_new = [i for i in unrew_to_keep if i in np.array(right_unrew)]
            L_unrewarded_new_naned = findNaNsNeeded(L_unrewarded_new,df)
            R_unrewarded_new_naned = findNaNsNeeded(R_unrewarded_new,df)
            df['Left_unreward_timestamps'] = L_unrewarded_new_naned
            df['Right_unreward_timestamps'] = R_unrewarded_new_naned



        
        # choice_history = df['trial choice'].dropna()
        # lenRewH = len(reward_history)
        # lenChH = len(choice_history)
        # if len1 < len2:
        #     del reward_history[-2:]
        #     del df[]

        # insert nans to get it to fit in the dataframe
        # nansNeeded = len(df) - len(reward_history)
        if (0 in keeplist) | ('_B_' in df['MSN'][0] or '_C_' in df['MSN'][0]):
            choice_history_trim = [choice_history[i] for i in range(len(choice_history)) if keeplist[i] == 1]
            choice_history = findNaNsNeeded(choice_history_trim,df)
            df['trial choice'] = choice_history
        reward_history = findNaNsNeeded(reward_history,df)
        # reward_history = np.array(reward_history)
        # reward_history = np.append(reward_history, np.zeros(nansNeeded_rew) + np.nan)



        # insert this array after the trial choice

        if 'reward_history' not in df.columns: 
            df.insert(30, 'reward_history', reward_history)
        elif 'reward_history' in df.columns:
            df['reward_history'] = reward_history
        else:
            print('what the heck. Check reward history thingies')
    return df


def findBlockTransitionTrials(df):
    blockTransTimes = df['block transition times'].dropna()
    blockTransArray = np.array(blockTransTimes)
    blockSwTrials = []
    trialStartArray = np.array(df['Trial_start_timestamps'].dropna())
    for trns in blockTransArray[0:-1]:
        blockSwTrials.append(closest_greater_index(
            trialStartArray, float(trns)))

    # fill out the nans
    nansNeeded = len(df) - len(blockSwTrials)
    blockSwTrials = np.array(blockSwTrials)
    blockSwTrials = np.append(blockSwTrials, np.zeros(nansNeeded) + np.nan)

    # ['block transition trials'] = blockSwTrials
    df.insert(27, 'block transition trials', blockSwTrials)
    return df


def readMEDPC(filepath, medParams, inputParameters):
    dir_path = os.path.dirname(filepath)

    res = OrderedDict()

    print(filepath)
    with open(filepath, "r") as in_file:
        line = in_file.readlines()
    if ('3087_609' in filepath) & ('2023-12-11' in filepath):
        a=1
    location = findStartLocation(line)
    for i in range(1, len(location)):
        j = location[i-1]
        while j >= location[i-1] and j < location[i]:
            # for j in range(location[i-1], location[i]):
            key, value, index = stripLine(line, j, medParams)
            res[key] = value
            j = index
            # print(j)
        # get rid of the 0s in timestamp variables
        #res = destroyZeros(res,['timestamps'])
        if inputParameters['behaviorParamSelect'] == 1: #* if it's from RI60, get rid of column Z because it's unnecessary and thicc
            del res['Z']

        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in res.items()]))
        # date_cols = [col for col in df.columns if 'Date' in col]
        for colName in df.columns:
            if 'Date' in str(colName):
                oldDate = df[colName][0]
                # I'm also going to rejigger the date format to match Guppy's
                mm = oldDate[0:2]
                dd = oldDate[3:5]
                yy = oldDate[6:8]
                newDate = yy+mm+dd
                df.loc[0, colName] = str(newDate)
        if inputParameters['behaviorParamSelect'] == 0:
            df = findBlockTransitionTrials(df) 
            if 'reward_history' not in df.columns or len(df['reward_history'].dropna()) < 2:
                df = getRewardHistory(df)


        sesName = filepath.split('\\')[-1]
        a = len(sesName.split('.')) #* have to do this because some of my old mice names have . in them :(
        if a > 2:
            #* if it's one of them there mice
            sesName = '.'.join(sesName.split('.')[0:2]) 
        else:
            sesName = sesName.split('.')[0]
        df.to_csv(sesName+'.csv')
        # * add it to the list of bois
        return df


# In[ ]:


# Three parameters should be provided by the user
# 1) folderNames
# 2) abspath
# 3) parameters

# Provide all the folder names corresponding to each animal where your MEDPC files are located
# folderNames = ["531-1", "531-2", "531-3", "531-4",
#               "531-5", "532-1", "532-2", "532-3", "532-4", "532-5"]


# provide the absoute path where all the animal folders are present
# abspath = '/Users/jnadel0/Desktop/NEUROSCIENCE_STUFF/Lerner_Lab/ASAP/Jan2023/Behavior/rawMedData'

# create a parameters file like the one in the folder and give the parameters file path over here
# parameters = '/Users/jnadel0/Desktop/NEUROSCIENCE_STUFF/Lerner_Lab/ASAP/Jan2023/Behavior/parameters.JSON'
def med2csv(inputParameters):
    # with open(parameters) as f:
    #    inputParameters = json.load(f)
    filePath = inputParameters['folderNames']
    abspath = inputParameters['abspath']
    medParams = paramSelection(inputParameters)
    dataFrameList = []
    # * find folder in the current directory that contains 'medData' in some permutation
    os.chdir(filePath[0])
    rawDataFolderCheck = inputParameters['rawMedName']
    if not os.path.exists(os.path.join(filePath[0], rawDataFolderCheck)):
        raise Exception('There is no raw data folder in the selected folder!')

    folderNames = os.path.join(filePath[0], rawDataFolderCheck)
    # for i in range(len(folderNames)):
    folder = os.path.join(abspath, folderNames)
    filepath = glob.glob(os.path.join(folder, "*"))
    os.chdir(folder)
    # print(filepath)
    for j in range(len(filepath)):

        # if j==1:
        #    break
        
        sesName = filepath[j].split('\\')[-1]
        csvCheck=None
        if filepath[j].endswith('.csv') == False:
            csvCheck = filepath[j].replace('.txt', '.csv')
            if '.csv' not in csvCheck[-4:]:
                csvCheck = csvCheck+'.csv' # sometimes there's no extension at all on the bois so this is to deal w that
            if os.path.exists(csvCheck):
                continue
        elif filepath[j].endswith('.csv'):
            csvCheck = filepath[j]
        if os.path.exists(csvCheck):
            print('skipping ' + sesName)
            name = csvCheck.split('\\')[-1]
            df = pd.read_csv(name)
            dataFrameList.append(df)
        elif filepath[j].endswith('.csv') == False:
            df = readMEDPC(filepath[j], medParams, inputParameters)
            dataFrameList.append(df)
    for i in range(len(dataFrameList) - 1):
        df = dataFrameList[i]
        
        # Check if the column has a length less than 2
        if inputParameters['behaviorParamSelect'] == 0:
            if len(df['blockTypeArray']) < 2:
                # Remove the dataframe from the list
                del dataFrameList[i]
    
            # dataFrameList = [df for df in dataFrameList if len(df['blockTypeArray']) > 2]
    all_data = dataFrameList
    os.chdir(filePath[0])
    folder_name = 'output_datafiles'

    output_file_path = os.path.join(filePath[0], folder_name)
    if not os.path.exists(output_file_path):
        os.mkdir(output_file_path)

    CSVFile = "behaviorData.pickle"
    if inputParameters['behaviorParamSelect'] == 1:
        # for session in all_data:
# Create a new list keeping only the sessions that DON'T match the condition
        all_data = [session for session in all_data 
                if 'JN' not in session['MSN'][0]]

    newFilePath = os.path.join(output_file_path, CSVFile)

    if os.path.exists(newFilePath):
        print(
            f"File '{CSVFile}' already exists within '{folder_name}'. Overwriting.")
        os.chdir(output_file_path)
        with open('behaviorData.pickle', 'wb') as file:
            pickle.dump(all_data, file)
    else:
        print(f"Creating '{CSVFile}' within '{folder_name}'.")
        os.chdir(output_file_path)
        with open('behaviorData.pickle', 'wb') as file:
            pickle.dump(all_data, file)
    if inputParameters['h5_check'] == True:
        h5creation(inputParameters)
    
    print('done with extraction!')
# %%
