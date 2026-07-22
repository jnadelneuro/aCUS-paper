"""
AA_h5.py — Build/append a master HDF5 store of Active-Avoidance behavior.

Exposes create_h5(outputPath, medPath): walks medPath for per-session
CSVs (already produced by extractMed.py), and writes each session as
a group inside `OperantData.h5py` under the top-level "Avoid" group.
Group naming follows date / subject / session-number conventions and
is structured to be matched later against Guppy photometry output.
Called by extractMed.py at the end of MED-PC parsing.
"""

import pandas as pd
import h5py
import os
import numpy as np
from collections import OrderedDict

def create_h5(outputPath, medPath):
    h5Check =  outputPath
    dataset = 'Avoid'
    os.chdir(outputPath)
    if 'OperantData.h5' not in h5Check:
        hf = h5py.File("OperantData.h5py", "w")
    else:
        hf = h5py.File("OperantData.h5py", "a")
    
    # ok so now we have our h5 file. So it's time to check if there's an h5 file already
    # is there a group for our dataset? if not, make it!
    if dataset not in hf.keys():
        hf.create_group(dataset)
    # CSVFileLocs =  medPath
    csvFilePath = medPath
    
    for root, directories, files in os.walk(csvFilePath, topdown=True):
        #filepath = glob.glob(os.path.join(folder, "*"))
        
        for file in files:
            if not file.endswith('.csv'):
                continue
            sessionData = pd.read_csv(os.path.join(root,file))
            
            #create dict with variables relevant for structure and guppy-matching
                # date, subject, experiment, and then create an if statement for sesType
            # first let's do the sesType thingydoo
            # THIS WILL CHANGE BASED ON TASK                    
            sesNum = sessionData['dayOnType'].astype(str).replace('\.0', '', regex=True)
            if type(sessionData['mouse'][0]) is not str:
                sessionData.loc[0, 'mouse'] = str(int(sessionData['mouse'][0].copy()))

                
            sessionDict = OrderedDict([('dataset' , dataset),
                                    ('subject', sessionData['mouse'][0]),
                                    ('paradigm', sesNum[0])
                                    ])
            
            
            # {'dataset' : dataset,
            #                'subject' : sessionData['Subject'][0],
            #                'paradigm' : sesType + str(sessionData['Experiment'][0])
            #                }
            
            # ok so now check if the subject has a group under the dataset
            subjPath = dataset + '/' + str(sessionData['mouse'][0])
            if subjPath not in hf:
                hf.create_group(subjPath)
                print('adding ' + str(sessionDict['subject']))

            # MOVED THIS PART TO THE READMED FILE SO IT'S OBSOLETE        
            # I'm also going to rejigger the date format to match Guppy's
            sesDate = str(sessionData['date'][0])

            # mm = sesDate[0:2]
            # dd = sesDate[3:5]
            # yy = sesDate[6:8]
            # sesDateGuppy = yy+mm+dd
            
            
            # same with day on task (experiment)
            expPath = subjPath + '/' + str(sesDate)
            if expPath not in hf:
                hf.create_group(expPath)
                print('adding ' + sessionDict['paradigm'] + ' ' + str(sesDate))
            
            # okay now it's time for the events. Everything w/ timestamps and the date (for guppy matching)

            # create event dictionary to be fed into the h file
            # eventDict = {'date' : sesDateGuppy
            
            # check if date is altready stored to avoid error            }
            sesPath = expPath + '/' + 'sessionType'
            if sesPath not in hf:
                hf.create_dataset(expPath + '/' + 'sessionType', data = sesNum[0])
            
            # get all the timestamps

            # also prepare to change medPC names to TTL names!
            matchingDict = {
                "avoidTimestamp": "avoid",
                "escapeTimestamp": "escape",
                "shockTimestamp": "shock",
                "cueTimestamp": "cue on",
                "cueEndTimestamp": "cue off",
                    }


            # matchingDict = {
            #     "Left_unreward_timestamps": "LPUn",
            #     "Left_reward_timestamps": "LRew",
            #     "Unrewarded_PE_timestamps": "PrtN",
            #     "Rewarded_PE_timestamps": "PrtR",
            #     "Right_unreward_timestamps": "RPUn",
            #     "Right_reward_timestamps": "RRew",
            #     "Trial_start_timestamps": "TlSt"
            #         }

            for key in sessionData.keys():
                data = pd.Series(dtype=object)
                if 'Timestamp' in key:
                    data = sessionData[key]
                    data = data[data != 0.0]
                    # need to do the following to get rid of "timer from timestamps" key
                    if len(data.dropna()) > 0:
                        # now see if it's in the matchingDict
                        if key in matchingDict.keys():
                            eventPath = expPath + '/' + matchingDict[key]
                            # check if it exists
                            if  eventPath not in hf:
                                a=1
                                hf.create_dataset(eventPath, data=np.array(data.dropna()))
                                # ..print('creating dataset storing ' + key + ' from ' + sessionDict['subject'] + ' ' + sesDate)
                            
                            
            

                
    hf.close() 
