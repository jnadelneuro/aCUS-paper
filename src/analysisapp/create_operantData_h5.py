"""
Created on Tue Jan 31 18:34:46 2023

@author: jacob
"""
import os
import glob
import pandas as pd
import numpy as np
import h5py
from collections import OrderedDict



# this boi will transfer everything from a bulk of .csv files to a single h5 file 
# compatible with Ryan's Guppy alignment system. Structure:
    # Dataset
    # Subject
    # Paradigm
    # Event timestamps
    # so e.g., 'ASAP_Jan2023/531-1/100_0_D1/Right Rewards
    
# # for pretty much all analysis codes, dataset will be static
# dataset = 'ASAP' # eventually user input option

# # now define the source file path with .csvs of behavior data
# csvPath = r'C:\Users\jan7154\Documents\ASAP_Analysis\behavior\rawMedData' # eventually user input option
# h5Path =  r'C:\Users\jan7154\Documents\ASAP_Analysis\behavior\output_datafiles' # eventually user input option

#check if there's already an h5 file named "OperantData.h5". if not, make one

def h5creation(inputParameters):
    if inputParameters['behaviorParamSelect'] == 0:
        dataset = 'ASAP'
    elif inputParameters['behaviorParamSelect'] == 1:
        dataset = 'RI60'
    
    folder_name = 'output_datafiles'
    filePath = inputParameters['folderNames'][0]
    output_file_path = os.path.join(filePath[0],folder_name)
    h5Check =  output_file_path
    if 'OperantData.h5' not in h5Check:
        hf = h5py.File("OperantData.h5py", "w")
    else:
        hf = h5py.File("OperantData.h5py", "a")
    
    # ok so now we have our h5 file. So it's time to check if there's an h5 file already
    # is there a group for our dataset? if not, make it!
    if dataset not in hf.keys():
        hf.create_group(dataset)
    CSVFileLocs =  inputParameters['rawMedName']
    csvFilePath = os.path.join(filePath, CSVFileLocs)
    
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
            if inputParameters['behaviorParamSelect'] == 0: 
                sesType = sessionData['MSN'][0] 
                if "_D_" in sesType:
                    sesType = '100_0_'
                elif "C" in sesType:
                    sesType = "FR1_"
                elif '_F_' in sesType:
                    sesType = '90_10_'
                elif '_G_' in sesType:
                    sesType = '75_25'
                else:
                    continue
            if inputParameters['behaviorParamSelect'] == 1: 
                sessionTitle = sessionData['MSN'][0] 
                if "D" in sessionTitle:
                    sesType = "FR1_"
                elif '_E_' in sessionTitle:
                    sesType = 'RI30_'
                elif '_F_' in sessionTitle:
                    sesType = 'RI60_'
                else:
                    continue
                if "LEFT" in sessionTitle:
                    sessionData['Activepoke_unrewarded_timestamps'] = sessionData['Left_nose_timestamps'][
                        sessionData['Left_nose_timestamps'] != sessionData['Left_reward_timestamps']
                    ]
                    sessionData['Unrewarded_Port_timestamps'] = sessionData['Port_entry_timestamps'][
                        sessionData['Port_entry_timestamps'] != sessionData['Rewarded_port_entry_timestamps']
                    ]
                    column_mapping = {
                        'Left_reward_timestamps' : 'Activepoke_rewarded_timestamps',
                        'Right_nose_timestamps' : "Inactive_poke_timestamps"
                    }
                    sessionData.rename(columns=column_mapping,inplace=True)
                elif "RIGHT" in sessionTitle:
                    sessionData['Activepoke_unrewarded_timestamps'] = sessionData['Right_nose_timestamps'][
                        sessionData['Right_nose_timestamps'] != sessionData['Right_reward_timestamps']
                    ]
                    sessionData['Unrewarded_Port_timestamps'] = sessionData['Port_entry_timestamps'][
                        sessionData['Port_entry_timestamps'] != sessionData['Rewarded_port_entry_timestamps']
                    ]
                    column_mapping = {
                        'Right_reward_timestamps' : 'Activepoke_rewarded_timestamps',
                        'Left_nose_timestamps' : "Inactive_poke_timestamps"
                    }
                    sessionData.rename(columns=column_mapping,inplace=True)
                    
            sesNum = sessionData['Experiment'].astype(str).replace('\.0', '', regex=True)
            if type(sessionData['Subject'][0]) is not str:
                sessionData.loc[0, 'Subject'] = str(int(sessionData['Subject'][0].copy()))

                
            sessionDict = OrderedDict([('dataset' , dataset),
                                    ('subject', sessionData['Subject'][0]),
                                    ('paradigm', sesType + sesNum[0])
                                    ])
            
            
            # {'dataset' : dataset,
            #                'subject' : sessionData['Subject'][0],
            #                'paradigm' : sesType + str(sessionData['Experiment'][0])
            #                }
            
            # ok so now check if the subject has a group under the dataset
            subjPath = dataset + '/' + str(sessionData['Subject'][0])
            if subjPath not in hf:
                hf.create_group(subjPath)
                print('adding ' + str(sessionDict['subject']))


            sesDate = str(sessionData['Start Date'][0])
            if '.' in sesDate:
                sesDate = sesDate.split('.')[0]

            
            
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
                hf.create_dataset(expPath + '/' + 'sessionType', data = sesType + sesNum[0])
            
            # get all the timestamps

            # also prepare to change medPC names to TTL names!
            if inputParameters['behaviorParamSelect'] == 0: 
                matchingDict = {
                    "Left_unreward_timestamps": "Left poke unrewarded",
                    "Left_reward_timestamps": "Left poke rewarded",
                    "Unrewarded_PE_timestamps": "Port unrewarded",
                    "Rewarded_PE_timestamps": "Port rewarded",
                    "Right_unreward_timestamps": "Right poke unrewarded",
                    "Right_reward_timestamps": "Right poke rewarded",
                    "Trial_start_timestamps": "Trial start"
                        }
            if inputParameters['behaviorParamSelect'] == 1:
                 matchingDict = {
                    "Inactive_poke_timestamps": "InactiveNP",
                    "Activepoke_rewarded_timestamps": "RewardNP",
                    "Rewarded_port_entry_timestamps": "RewardPE",
                    "Activepoke_unrewarded_timestamps": "UnrewardedNP",
                    "Unrewarded_Port_timestamps": "UnrewardedPE"
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
                if 'timestamp' in key:
                    # need to do the following to get rid of "timer from timestamps" key
                    if len(sessionData[key].dropna()) > 2:
                        # now see if it's in the matchingDict
                        if key in matchingDict.keys():
                            eventPath = expPath + '/' + matchingDict[key]
                            # check if it exists
                            if  eventPath not in hf:
                                a=1
                                hf.create_dataset(eventPath, data=np.array(sessionData[key].dropna()))
                                # ..print('creating dataset storing ' + key + ' from ' + sessionDict['subject'] + ' ' + sesDate)
                            
                            
            

                
    hf.close() 