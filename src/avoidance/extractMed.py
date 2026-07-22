"""
extractMed.py — Stage 1 of pipeline: parse raw MED-PC text files.

Reads MED-PC operant-box output for the Active Avoidance task and
turns it into tidy per-trial pandas DataFrames + a behaviorData.pickle.

Key pieces:
  - `parameters` dict documents the MED-PC D() array layout
    (avoid/escape tags, latencies, crossings, ITI shocks, and the
    timestamps for avoid / escape / shock / cue / cue-end events).
  - stripLine(): parses one line of a MED-PC dump into a named array.
  - readMEDPC(): walks a whole .txt file -> session dict.
  - calcEscapeAvoid(): derives avoid vs escape trial labels and
    latencies from raw timestamps.
  - findStartLocation(): finds where the numeric data block begins.
  - At the bottom, iterates every mouse/session, writes per-session
    CSVs, then calls AA_h5.create_h5() to consolidate into HDF5.

Hard-coded Windows lab paths at top — edit `analysisPath` to run.
"""

import os
import numpy as np
import json
import glob
import pandas as pd
import pickle
from collections import OrderedDict
import h5py
from AA_h5 import create_h5
from _config import AVOIDANCE_ROOT

analysisPath = r'R:\Basic_Sciences\Phys\Lerner_Lab_tnl2633\Evan\fall rotation\aCUS_C7_Active_Avoidance\official_analysis'
analysisPath = AVOIDANCE_ROOT
behaviorPath = os.path.normcase(os.path.join(analysisPath, 'behavior'))
medDataPath = os.path.normcase(os.path.join(behaviorPath, 'rawMedData'))
outputDataPath = os.path.normcase(os.path.join(behaviorPath, 'output_datafiles'))
photoPath = os.path.normcase(os.path.join(analysisPath, 'photometry'))
# mouseInfoPath = r'R:\Basic_Sciences\Phys\Lerner_Lab_tnl2633\Evan\aCUS_C7_Active_Avoidance\output_datafiles'
# behaviorDataPath = r'R:\Basic_Sciences\Phys\Lerner_Lab_tnl2633\Evan\aCUS_C7_Active_Avoidance\raw_behavior'


os.chdir(outputDataPath)
with open("mouse_info.json", "r") as f:
    mouseData = json.load(f)

parameters = {'D': 'dataTable',
              # * D() = Trial by Trial Data Array
              # * D(I) = Trial Number
              # * D(I+1) = Avoid Tag
              # * D(I+2) = Avoid Latency
              # * D(I+3) = Escape Tag
              # * D(I+4) = Escape Latency
              # * D(I+5) = Left Movement Activity
              # * D(I+6) = Right Movement Activity
              # * D(I+7) = Crossings
              # * D(I+8) = ITI Shocks
              # * D(I+9) = Avoid timestamp \ RK: taking over 9:13 for new timestamp values
              # * D(I+10) = Escape timestamp
              # * D(I+11) = Shock timestamp
              # * D(I+12) = Cue timestamp
              # * D(I+13) = Cue End timestamp
              # * D(I+14) = Reserved
              }



def stripLine(line, index, medParams):
    arr = []
    # print(index)
    res = line[index].strip(":\n").split(": ")
    # print(res)
    key = None
    value = None

    if len(res) > 1:
        key = res[0]
        value = res[1]
        if res[0] in medParams.keys():
            key = medParams[res[0]]
        index += 1
    else:

        if res[0] in medParams.keys():
            key = medParams[res[0]]
        else:
            key = res[0]

        index += 1

        value = []
        # print('Index: ', index)
        strip = line[index].strip().split(" ")
        # print(strip)
        while len(strip) > 1 and index < len(line)-1:
            for i in strip[1:]:
                if len(i) > 0:
                    value.append(i)
            index += 1

            strip = line[index].strip().split(" ")

    if res[0] == 'D':
        reshaped_data = [value[i:i + 15] for i in range(0, len(value), 15)]
        headers = ['trialNum', "avoidTag", "avoidLat", "escapeTag", "escapeLat", "leftMvmntAct", "rightMvmntAct", "crossings",
               "ITIShks", "avoidTimestamp", "escapeTimestamp", "shockTimestamp", "cueTimestamp", "cueEndTimestamp", "reserved"]
        
        value = pd.DataFrame(reshaped_data, columns=headers)
    

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

def calcEscapeAvoid(df):
    df['avoidTag'] = df['avoidTag'].astype(float)
    df['escapeTag'] = df['escapeTag'].astype(float)
    df['avoidLat'] = df['avoidLat'].astype(float)
    df['escapeLat'] = df['escapeLat'].astype(float)

    numAvoid = int(sum(df['avoidTag']))
    numEscape = int(sum(df['escapeTag']))
    avoidPct = numAvoid/(numEscape + numAvoid)

    avgAvoidLatency = np.mean(df[df['avoidTag'] == 1.0]['avoidLat'])
    avgEscapeLatency = np.mean(df[df['escapeTag'] == 1.0]['escapeLat'])

    crossings = int(sum(df['crossings'].astype(float)))

    summaryDict = {'avoids' : numAvoid,
                   'escapes' : numEscape,
                   'percent avoid' : avoidPct,
                   'avoid latency' : avgAvoidLatency,
                   'escape latency' : avgEscapeLatency,
                   'trial crossings' : crossings,
                   'mouse' : df['mouse'][0],
                   'dayOnType' : df['dayOnType'][0],
                #    'group' : df['group'][0],
                   'sex' : df['sex'][0]}
    
    return summaryDict

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


def readMEDPC(filepath, medParams, mouseInfo):
    dir_path = os.path.dirname(filepath)

    res = OrderedDict()

    print(filepath)
    with open(filepath, "r") as in_file:
        line = in_file.readlines()
    if ('3087_609' in filepath) & ('2023-12-11' in filepath):
        a = 1
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
        # res = destroyZeros(res,['timestamps'])        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in res.items()]))
        # date_cols = [col for col in df.columns if 'Date' in col]
        for key, value in res.items():
            if 'Date' in str(key):
                oldDate = value
                # I'm also going to rejigger the date format to match Guppy's
                mm = oldDate[0:2]
                dd = oldDate[3:5]
                yy = oldDate[6:8]
                newDate = yy+mm+dd
                res[key] = int(newDate)

        if res['Subject'] not in mouseInfo.keys():
            print(f"Mouse {res['Subject']} not found in mouse info json file.")
            return None
        sesName = filepath.split('\\')[-1]
        # sesName = '_'.join(sesName.split('.'))
        

        dataTable = res["dataTable"]

# Get the 'Start Date' value (this is a single value, not a list)
        dataTable['date'] = res['Start Date']
        dataTable['mouse'] = res['Subject']
        dataTable['dayOnType'] = sesName.split('_')[5]
        # dataTable['group'] = mouseInfo[res['Subject']]['group']
        dataTable['sex'] = mouseInfo[res['Subject']]['sex']

        # Add 'Start Date' to every row in the dataTable

        dataTable = dataTable.drop(['leftMvmntAct', 'rightMvmntAct', 'ITIShks',
        'reserved'], axis=1)



        csvName = sesName[:-4] + '.csv' if sesName.endswith('.txt') else sesName + '.csv'
        dataTable.to_csv(csvName, index=False)        # * add it to the list of bois
        return dataTable


filePath = medDataPath
dataFrameList = []

# for i in range(len(folderNames)):
folder = os.path.join(filePath)
filepath = glob.glob(os.path.join(folder, "*"))
os.chdir(folder)
# print(filepath)
sessionData_list = []
summaryData_list = []


for j in range(len(filepath)):

    # if j==1:
    #    break

    sesName = filepath[j].split('\\')[-1]
    if '.' in sesName and not sesName.endswith('.csv'):
        if sesName.endswith('.txt'):
            newName = sesName[:-4].replace('.', '_') + '.txt'
        else:
            newName = sesName.replace('.', '_')

        if newName != sesName:
            newPath = os.path.join(folder, newName)
            os.rename(filepath[j], newPath)
            filepath[j] = newPath
            sesName = newName
    csvCheck = None
    if filepath[j].endswith('.csv') == False:
        csvCheck = filepath[j].replace('.txt', '.csv')
        if '.csv' not in csvCheck[-4:]:
            # sometimes there's no extension at all on the bois so this is to deal w that
            csvCheck = csvCheck+'.csv'
        if os.path.exists(csvCheck):
            continue
    elif filepath[j].endswith('.csv'):
        csvCheck = filepath[j]
    if os.path.exists(csvCheck):
        print('skipping ' + sesName)
        name = csvCheck.split('\\')[-1]
        df = pd.read_csv(name)
        sessionData_list.append(df)

        summaryDict = calcEscapeAvoid(df)
        summaryDF = pd.DataFrame([summaryDict])
        summaryData_list.append(summaryDF)

    elif filepath[j].endswith('.csv') == False:

        df = readMEDPC(filepath[j], parameters, mouseData)
        if df is None:
            continue
        # sessionData_list.append(df)
        summaryDict = calcEscapeAvoid(df)
        summaryDF = pd.DataFrame([summaryDict])

        sessionData_list.append(df)
        summaryData_list.append(summaryDF)

        a=1
        
CSVFile = "behaviorData.pickle"
newFilePath = os.path.join(outputDataPath, CSVFile)

if os.path.exists(newFilePath):
    print(
        f"File '{CSVFile}' already exists within. Overwriting.")
    os.chdir(outputDataPath)
    with open('behaviorData.pickle', 'wb') as file:
        pickle.dump(sessionData_list, file)
else:
    print(f"Creating '{CSVFile}'.")
    os.chdir(outputDataPath)
    with open('behaviorData.pickle', 'wb') as file:
        pickle.dump(sessionData_list, file)

combined_df = pd.concat(sessionData_list, ignore_index=True)
summaryDF = pd.concat(summaryData_list, ignore_index=True)

os.chdir(outputDataPath)
combined_df.to_csv('behaviorData.csv', index=False)
summaryDF.to_csv('summaryData.csv', index=False)

os.chdir(medDataPath)

# create_h5(outputDataPath, medDataPath)

a=1