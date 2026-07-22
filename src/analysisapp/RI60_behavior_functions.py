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

def stripZero(string):
    splitStr = string.split('.')
    if '0' not in splitStr:
        return string
    newStr = splitStr[0]

    return newStr

def calculate_reward_rate(all_pokes, rewarded_pokes, tau=180.0):
    """
    Calculates a decaying reward rate for every poke using a leaky integrator.
    
    Args:
        all_pokes (array-like): Timestamps of every nose poke.
        rewarded_pokes (array-like): Timestamps of rewarded pokes.
        tau (float): Decay time constant in seconds.
                     180s is good for RI60 (smoothing over ~3 rewards).
                     
    Returns:
        np.array: Reward rate (rewards/min) aligned to each timestamp in all_pokes.
    """
    # specific logic requires sorted timestamps for dt calculation
    pokes = np.sort(np.array(all_pokes))
    
    # Create a set for O(1) lookup
    rewards_ts = set(rewarded_pokes)
    
    # Boolean vector matching pokes (1.0 if that specific poke was rewarded)
    is_rewarded = np.array([1.0 if t in rewards_ts else 0.0 for t in pokes])
    
    rates = np.zeros(len(pokes))
    current_rate = 0.0
    
    # Iterate through pokes starting from the second one
    for i in range(1, len(pokes)):
        dt = pokes[i] - pokes[i-1]
        
        # 1.Decay the previous rate over the time gap
        current_rate = current_rate * np.exp(-dt / tau)
        
        # 2.Add the reward impulse from the PREVIOUS poke (causal filtering)
        if is_rewarded[i-1]:
            current_rate += 1.0 
            
        rates[i] = current_rate

    # Normalize to "Rewards per Minute"
    # rate density = accumulated_impulse / tau
    rates_per_min = (rates / tau) * 60
    
    return rates_per_min

def create_RR_arrays():
    with open('behaviorData.pickle', 'rb') as file:
        behaviorData = pickle.load(file)

    for session in behaviorData:
        # if 'RI60' not in session['MSN'][0]:
        #     continue
        if float(session['Right Rewards'][0]) > 0:
            active = 'Right'
        elif float(session['left rewards'][0]) > 0:
            active = 'Left'
        else:
            session['reward_rate_trace'] = np.nan
            continue
        
        allTimestamps = session[f'{active}_nose_timestamps'].dropna().values
        rewardedTimestamps = session[f'{active}_reward_timestamps'].dropna().values

        rewardRateArray = calculate_reward_rate(allTimestamps, rewardedTimestamps, 60.0)
        pokeRateArray = calculate_reward_rate(allTimestamps, allTimestamps, 60.0)
        pad_width = len(session) - len(rewardRateArray)
        if pad_width > 0:
            rewardRateArray = np.pad(rewardRateArray, (0, pad_width), constant_values=np.nan)
            pokeRateArray = np.pad(pokeRateArray, (0, pad_width), constant_values=np.nan)
        session['reward_rate_trace'] = rewardRateArray
        session['poke_rate_trace'] = pokeRateArray

    with open('behaviorData_RR.pickle', 'wb') as file:
        pickle.dump(behaviorData, file)
        # a=1

def add_cumulative_poke_count(filename='behaviorData.pickle'):
    """
    Adds a 'total_poke_overall' column to each session dataframe in behaviorData.pickle.
    The array contains the cumulative active nose pokes for that mouse across all sessions up to that specific poke.
    Sorted by date.
    """
    if not os.path.exists(filename):
        print(f"{filename} not found.")
        return

    print(f"Loading {filename}...")
    with open(filename, 'rb') as file:
        behaviorData = pickle.load(file)

    # Create a new list keeping only the sessions that DON'T match the condition
    behaviorData = [
        session for session in behaviorData 
        if 'RK' in session['MSN'][0]
    ]

    # 1. Group sessions by mouse
    sessions_by_mouse = {}
    for session in behaviorData:
        subj = str(session['Subject'][0])
        if '.' in subj: 
            subj = stripZero(subj)
            
        if subj not in sessions_by_mouse:
            sessions_by_mouse[subj] = []
        sessions_by_mouse[subj].append(session)

    # 2. Sort sessions and calculate cumulative pokes
    for mouse, sessions in sessions_by_mouse.items():
        # Sort by Start Date
        def parse_date(s):
            # Handle formats like '05.01.2023' -> '05/01/2023' for pandas
            d_str = stripZero(str(s['Start Date'][0]))
            # If d_str is 6 digits (e.g., 241106), assume simple YYMMDD
            if len(d_str) == 6 and d_str.isdigit():
                return pd.to_datetime(d_str, format='%y%m%d')
            return pd.to_datetime(d_str)

        sessions.sort(key=parse_date)
        
        cumulative_pokes = 0
        
        for session in sessions:
            # Determine active side (consistent with create_RR_arrays logic)
            active = None
            if 'Right Rewards' in session and float(session['Right Rewards'][0]) > 0:
                active = 'Right'
            elif 'left rewards' in session and float(session['left rewards'][0]) > 0:
                active = 'Left'
            else:
                # Fallback: assume the side with more pokes is active
                r_pokes = float(session['Right nosepokes'][0]) if 'Right nosepokes' in session else 0
                l_pokes = float(session['Left nose pokes'][0]) if 'Left nose pokes' in session else 0
                active = 'Right' if r_pokes >= l_pokes else 'Left'

            # Get timestamps
            ts_col = f'{active}_nose_timestamps'
            
            if ts_col in session:
                timestamps = session[ts_col].dropna().values
                n_pokes = len(timestamps)
                
                if n_pokes > 0:
                    # Create array: [prev_total + 1, ..., prev_total + n_pokes]
                    current_indices = np.arange(1, n_pokes + 1)
                    session_cumulative = current_indices + cumulative_pokes
                    
                    # Pad to match DataFrame length (standard for this pipeline)
                    pad_width = len(session) - len(session_cumulative)
                    if pad_width > 0:
                        session_cumulative = session_cumulative.astype(float)
                        session_cumulative = np.pad(session_cumulative, (0, pad_width), constant_values=np.nan)
                    
                    session['total_poke_overall'] = session_cumulative
                    
                    # Update global counter for this mouse
                    cumulative_pokes += n_pokes
                else:
                    session['total_poke_overall'] = np.full(len(session), np.nan)
            else:
                session['total_poke_overall'] = np.full(len(session), np.nan)

    # 3. Save back to file
    with open(filename, 'wb') as file:
        pickle.dump(behaviorData, file)
    
    print(f"Updated {filename} with 'total_poke_overall' column.")


def analyzeBehavior(output_file_folder):
    os.chdir(output_file_folder)
    with open("mouse_info.json", "r") as f:
        mouseInfoDict = json.load(f)
    # * for may2023 acus cohort 2


    # * for jan2024 acus cohort 2
    

    # output_file_folder = r'C:\Users\jan7154\Documents\aCUS_analysis\adultMar2023\behavior\output_datafiles'
    # output_file_folder = r'D:\Data_analysis\Lerner_Lab\aCUS\adultCUS\behavior\output_datafiles'
    # output_file_folder = r'D:\Data_analysis\Lerner_Lab\aCUS\_May2023_Cohort2\behavior\output_datafiles'
    # output_file_folder = r'D:\Data_analysis\Lerner_Lab\aCUS\_aCUS_Jan2024\for JIMMY\behavior\output_datafiles'
    # output_file_folder = r'C:\Users\jacob\iCloudDrive\Desktop\NEUROSCIENCE_STUFF\Lerner_Lab\aCUS\_aCUS June 2024 (Cohort 6)\behavior\output_datafiles'
    # output_file_folder = r'C:\Users\jan7154\iCloudDrive\Desktop\NEUROSCIENCE_STUFF\Lerner_Lab\aCUS\_aCUS June 2024 (Cohort 6)\behavior\output_datafiles'


    def drugCheck(outer_dict, target_key):  # is the given experiment using drug/kir condition?
        second_level_keys = [key for inner_dict in outer_dict.values(
        ) if isinstance(inner_dict, dict) for key in inner_dict]
        return target_key in second_level_keys


    #! check if there's a behaviorData.pickle file in the output_datafiles folder
    if not os.path.exists(os.path.join(output_file_folder, 'behaviorData.pickle')):
        raise Exception(
            'There is no behaviorData.pickle folder in the selected folder! Extract med pc data!')

    os.chdir(output_file_folder)
    with open('behaviorData.pickle', 'rb') as file:
        behaviorData = pickle.load(file)
    
    # * get rid of the mag training bois
    behaviorDataRI_only = []
    behaviorDataRI_shock = []
    behaviorData_FRonly = []
    for sessionData in behaviorData:
        if sessionData['Subject'][0] not in mouseInfoDict.keys():
            continue
        if "FR1" in sessionData['MSN'][0]:
            behaviorData_FRonly.append(sessionData)
        if ("RI30" in sessionData['MSN'][0]) | ("RI60" in sessionData['MSN'][0]):
            behaviorDataRI_only.append(sessionData)
        if "Footshock" in sessionData['MSN'][0]:
            behaviorDataRI_shock.append(sessionData)

    
    mouseList = []
    sexList = []
    stressList = []
    if drugCheck(mouseInfoDict, 'virus'):
        virusList = []
    sesDateList = []
    sesTypeList = []
    dayOnTypeList = []
    pokeRateList = []
    inactiveRateList = []
    entryRateList = []
    pokeEntryRatioList = []
    efficiencyList = []
    rewRateList = []
    rewardPokeTimestamp_list = []
    allPokeTimestamp_list = []
    rewardEntryTimestamp_list = []
    allEntryTimestamp_list = []
    allEntryDurations_list = []

    mouseList_shock = []
    sexList_shock = []
    stressList_shock = []
    if drugCheck(mouseInfoDict, 'virus'):
        virusList_shock = []
    sesDateList_shock = []
    sesTypeList_shock = []
    pct_OG_BW_shock = []
    dayOnTypeList_shock = []
    numofshocksList = []
    numofrew_shock_list = []
    pokeTS_shockList = []
    rewardPokeTS_shockList = []
    shockTS_shockList = []

    mouseList_FR1 = []
    sexList_FR1 = []
    stressList_FR1 = []
    if drugCheck(mouseInfoDict, 'virus'):
        virusList_FR1 = []
    sesDateList_FR1 = []
    sesTypeList_FR1 = []
    dayOnTypeList_FR1 = []
    rewRateList_FR1 = []

    # pokeRateList_shock = []
    # entryRateList_shock = []
    # pokeEntryRatioList_shock = []
    for sessionData in behaviorData_FRonly:
        sesDate = str(sessionData['Start Date'][0])
        subj = str(sessionData['Subject'][0])
        if '.' in sesDate:
            sesDate = stripZero(sesDate)
        if'.' in subj:
            subj = stripZero(subj)
        
        mouseList_FR1.append(subj)
        sexList_FR1.append(mouseInfoDict[sessionData['Subject'][0]]['sex'])
        stressList_FR1.append(mouseInfoDict[sessionData['Subject'][0]]['group'])
        if drugCheck(mouseInfoDict, 'virus'):
            virusList_FR1.append(mouseInfoDict[sessionData['Subject'][0]]['virus'])
        sesDateList_FR1.append(sesDate)
        sesTypeList_FR1.append("FR1")
        dayOnTypeList_FR1.append(sessionData['Experiment'][0])
        if float(sessionData['Right Rewards'][0]) > 0:
            rewRate = (float(sessionData['Right Rewards'][0])) / \
                ((float(sessionData['Timer for timestamps'][0])/60))

        elif float(sessionData['left rewards'][0]) > 0:
            rewRate = (float(sessionData['left rewards'][0])) / \
                ((float(sessionData['Timer for timestamps'][0])/60))

        rewRateList_FR1.append(rewRate)


    for sessionData in behaviorDataRI_only:

        subj = str(sessionData['Subject'][0])
        sesDate = str(sessionData['Start Date'][0])
        dayOnType = str(sessionData['Experiment'][0])
        if '.' in sesDate:
            sesDate = stripZero(sesDate)
        if'.' in subj:
            subj = stripZero(subj)

        if ('use' in mouseInfoDict[subj]):
            if mouseInfoDict[subj]['use'] == 'n':
                continue    
        if ('use' in mouseInfoDict[subj]):    
            if mouseInfoDict[subj]['use'] == 'u':
                continue
        if float(sessionData['Right Rewards'][0]) > 0:
            if int(float(sessionData['Right nosepokes'][0])) < 30:
                # a=1
                print(f'skipping {sessionData["Subject"][0]} on {sessionData["Start Date"][0]} for low poke count')
                continue
            pokeRate = (float(sessionData['Right nosepokes'][0])) / \
                ((float(sessionData['Timer for timestamps'][0])/60))
            inactivePoke = (float(sessionData['Left nose pokes'][0])) / \
                ((float(sessionData['Timer for timestamps'][0])/60))
            entryRate = (float(sessionData['Port Entries'][0])) / \
                ((float(sessionData['Timer for timestamps'][0])/60))
            ratioList = (float(sessionData['Right nosepokes'][0])
                        ) / float((sessionData['Port Entries'][0]))
            efficiency = (float(
                float(sessionData['Right Rewards'][0]) / int(float(sessionData['Right nosepokes'][0]))))
            rewRate = (float(sessionData['Right Rewards'][0])) / \
                ((float(sessionData['Timer for timestamps'][0])/60))
            activePoke_timestamps = list(
                sessionData[~sessionData['Right_reward_timestamps'].isna()]['Right_reward_timestamps'])
            allPoke_timestamps = list(
                sessionData[~sessionData['Right_nose_timestamps'].isna()]['Right_nose_timestamps'])

        elif float(sessionData['left rewards'][0]) > 0:
            if int(float(sessionData['Left nose pokes'][0])) < 30:
                print(f'skipping {sessionData["Subject"][0]} on {sessionData["Start Date"][0]} for low poke count')
                continue
            pokeRate = (float(sessionData['Left nose pokes'][0])) / \
                ((float(sessionData['Timer for timestamps'][0])/60))
            inactivePoke = (float(sessionData['Right nosepokes'][0])) / \
                ((float(sessionData['Timer for timestamps'][0])/60))
            entryRate = (float(sessionData['Port Entries'][0])) / \
                ((float(sessionData['Timer for timestamps'][0])/60))
            ratioList = (float(sessionData['Left nose pokes'][0])
                        ) / float((sessionData['Port Entries'][0]))
            efficiency = (
                float(sessionData['left rewards'][0]) / float(sessionData['Left nose pokes'][0]))
            rewRate = (float(sessionData['left rewards'][0])) / \
                ((float(sessionData['Timer for timestamps'][0])/60))
            activePoke_timestamps = list(
                sessionData[~sessionData['Left_reward_timestamps'].isna()]['Left_reward_timestamps'])
            allPoke_timestamps = list(
                sessionData[~sessionData['Left_nose_timestamps'].isna()]['Left_nose_timestamps'])
            
        
        mouseList.append(sessionData['Subject'][0])
        sexList.append(mouseInfoDict[sessionData['Subject'][0]]['sex'])
        stressList.append(mouseInfoDict[sessionData['Subject'][0]]['group'])
        if drugCheck(mouseInfoDict, 'virus'):
            virusList.append(mouseInfoDict[sessionData['Subject'][0]]['virus'])
        sesDateList.append(sesDate)
        if "RI30" in sessionData['MSN'][0]:
            sesTypeList.append("RI30")
        elif "RI60" in sessionData['MSN'][0]:
            sesTypeList.append("RI60")
        else:
            sesTypeList.append('WHAT')
        dayOnTypeList.append(dayOnType)
        rewardEntryTimestamp_list.append(list(sessionData[~sessionData['Rewarded_port_entry_timestamps'].isna(
        )]['Rewarded_port_entry_timestamps']))
        allEntryTimestamp_list.append(list(sessionData[~sessionData['Port_entry_timestamps'].isna()]['Port_entry_timestamps']))
        allEntryDurations_list.append(list(sessionData[~sessionData['Port_entry_duration'].isna()]['Port_entry_duration']))



        rewardPokeTimestamp_list.append(activePoke_timestamps)
        pokeRateList.append(pokeRate)
        inactiveRateList.append(inactivePoke)
        entryRateList.append(entryRate)
        pokeEntryRatioList.append(ratioList)
        efficiencyList.append(efficiency)
        rewRateList.append(rewRate)
        allPokeTimestamp_list.append(allPoke_timestamps)

    bodyWeightInfo = pd.read_csv('weight_at_probe.csv')
    bodyWeightInfo['% at shock'] = bodyWeightInfo['% at shock'].astype(float)
    # shockDF = shockDF[shockDF['mouse'] != '725-RL']
    # feather.write_feather(shockDF, 'Shock_dataFrame.feather')
    # Rename the column in the source dataframe before merging
    bodyWeightInfo = bodyWeightInfo.rename(columns={'% at shock': 'pct_OG_BW'})

    for sessionData in behaviorDataRI_shock:
        if (float(sessionData['Right Rewards'][0]) == 0) & (float(sessionData['left rewards'][0]) == 0):
            continue
        if int(float(sessionData['Experiment'][0])) == 1:
            continue
        mouse = sessionData['Subject'][0]
        if bodyWeightInfo[bodyWeightInfo['Mouse'] == mouse]['pct_OG_BW'].iloc[0] < 0.8:
            print(f'skipping {mouse} on {sessionData["Start Date"][0]} for low body weight')
            continue
        sesDate = str(sessionData['Start Date'][0])
        dayOnType = str(sessionData['Experiment'][0])
        if '.' in sesDate:
            sesDate = stripZero(sesDate)

        if ('use' in mouseInfoDict[mouse]):
            if mouseInfoDict[mouse]['use'] == 'n':
                continue    
        if ('use' in mouseInfoDict[mouse]):    
            if mouseInfoDict[mouse]['use'] == 'u':
                continue
        if (mouse not in mouseInfoDict.keys()):
            continue
        mouseList_shock.append(sessionData['Subject'][0])
        sexList_shock.append(mouseInfoDict[sessionData['Subject'][0]]['sex'])
        stressList_shock.append(mouseInfoDict[sessionData['Subject'][0]]['group'])
        if drugCheck(mouseInfoDict, 'virus'):
            virusList_shock.append(
                mouseInfoDict[sessionData['Subject'][0]]['virus'])
        sesDateList_shock.append(sesDate)
        pct_OG_BW_shock.append(bodyWeightInfo[bodyWeightInfo['Mouse'] == mouse]['pct_OG_BW'].iloc[0])
        dayOnTypeList_shock.append(dayOnType)
        numofshocksList.append(int(float(sessionData['Counter/Shocks'][0])))

        if float(sessionData['Right Rewards'][0]) > 0:
            numofrew_shock_list.append(int(float(sessionData['Right Rewards'][0])))
            activePoke_timestamps_shock = list(
                sessionData[~sessionData['Right_nose_timestamps'].isna()]['Right_nose_timestamps'])
            rewardPoke_timestamps_shock = list(
                sessionData[~sessionData['Right_reward_timestamps'].isna()]['Right_reward_timestamps'])
        #     pokeRate = (float(sessionData['Right nosepokes'][0])) / ((float(sessionData['Timer for timestamps'][0])/60))
        #     entryRate = (float(sessionData['Port Entries'][0])) / ((float(sessionData['Timer for timestamps'][0])/60))
        #     ratioList = (float(sessionData['Right nosepokes'][0])) / float((sessionData['Port Entries'][0]))
        elif float(sessionData['left rewards'][0]) > 0:
            numofrew_shock_list.append(int(float(sessionData['left rewards'][0])))
            activePoke_timestamps_shock = list(
                sessionData[~sessionData['Left_nose_timestamps'].isna()]['Left_nose_timestamps'])
            rewardPoke_timestamps_shock = list(
                sessionData[~sessionData['Left_reward_timestamps'].isna()]['Left_reward_timestamps'])
        
        shock_timestamps = list(
            sessionData[~sessionData['Number drawn/Timestamps for shocks'].isna()]['Number drawn/Timestamps for shocks'])
        shock_timestamps = [x for x in shock_timestamps if x != 0.0] 
        if len(shock_timestamps) > 100:
            print(f'session {sessionData["Subject"][0]} on {sessionData["Start Date"][0]} has more than 100 shocks, something wrong')

        pokeTS_shockList.append(activePoke_timestamps_shock)
        rewardPokeTS_shockList.append(rewardPoke_timestamps_shock)
        shockTS_shockList.append(shock_timestamps)
        #     pokeRate = (float(sessionData['Left nose pokes'][0])) / ((float(sessionData['Timer for timestamps'][0])/60))
        #     entryRate = (float(sessionData['Port Entries'][0])) / ((float(sessionData['Timer for timestamps'][0])/60))
        #     ratioList = (float(sessionData['Left nose pokes'][0])) / float((sessionData['Port Entries'][0]))

        # pokeRateList_shock.append(pokeRate)
        # entryRateList_shock.append(entryRate)
        # pokeEntryRatioList_shock.append(ratioList)
    
    df_training_dict = {"mouse": mouseList,
                        "sex": sexList,
                        "group": stressList,
                        #    'drug': drugList,
                        'date': sesDateList,
                        'sesType': sesTypeList,
                        'dayOnType': dayOnTypeList,
                        'pokeRate': pokeRateList,
                        'inactiveRate': inactiveRateList,
                        'entryRate': entryRateList,
                        'P/E ratio': pokeEntryRatioList,
                        'efficiency': efficiencyList,
                        'rewardRate': rewRateList,
                        'rewPokeTimestamps': rewardPokeTimestamp_list,
                        'allPokeTimestamps': allPokeTimestamp_list,
                        'rewardEntryTS': rewardEntryTimestamp_list,
                        'allEntryTS': allEntryTimestamp_list,
                        'allEntryDurations': allEntryDurations_list


                        }


    df_FR1_dict = {"mouse": mouseList_FR1,
                "sex": sexList_FR1,
                "group": stressList_FR1,
                #    'virus': virusList_FR1,
                'date': sesDateList_FR1,
                'sesType': sesTypeList_FR1,
                'dayOnType': dayOnTypeList_FR1,
                'rewardRate': rewRateList_FR1
                }

    df_shock_dict = {"mouse": mouseList_shock,
                    "sex": sexList_shock,
                    "group": stressList_shock,
                    #    'drug': drugList_shock,
                    'date': sesDateList_shock,
                    #    'sesType' : sesTypeList,
                    'dayOnType': dayOnTypeList_shock,
                    'numShock': numofshocksList,
                    'numRew': numofrew_shock_list,
                    'numPoke': [len(lst) for lst in pokeTS_shockList],
                    'pct_OG_BW': pct_OG_BW_shock,
                    'pokeTimestamps': pokeTS_shockList,
                    'rewardPokeTimestamps': rewardPokeTS_shockList,
                    'shockTimestamps': shockTS_shockList
                    #    'pokeRate' : pokeRateList_shock,k
                    #    'entryRate' : entryRateList_shock,
                    #    'P/E ratio' : pokeEntryRatioList_shock

                    }
    
    

    if drugCheck(mouseInfoDict, 'virus'):
        df_training_dict['virus'] = virusList
        df_FR1_dict['virus'] = virusList_FR1
        df_shock_dict['virus'] = virusList_shock


    df_training = pd.DataFrame(df_training_dict)
    df_shock = pd.DataFrame(df_shock_dict)
    df_shock['date'] = pd.to_numeric(df_shock['date'], errors='coerce')
    df_shock = df_shock.sort_values('date').drop_duplicates(subset=['mouse'], keep='last')

    df_FR1 = pd.DataFrame(df_FR1_dict)

    


    if drugCheck(mouseInfoDict, 'virus'):
        def moveVirus(df):
            df = df[[col for col in df.columns if col != 'virus'][:df.columns.get_loc(
                'group')+1] + ['virus'] + [col for col in df.columns if col != 'virus'][df.columns.get_loc('group')+1:]]

            return df
        df_training = moveVirus(df_training) 
        df_FR1 = moveVirus(df_FR1)
        df_shock = moveVirus(df_shock)


    df_training.to_csv('RI30_RI60_data.csv')
    feather.write_feather(df_training, 'RI30_RI60_dataFrame.feather')
    feather.write_feather(df_shock, 'Shock_dataFrame.feather')
    df_shock.to_csv('Shock_Data.csv')
    df_FR1.to_csv('FR1_data.csv')
    
    # df_training = pd.read_csv('RI30_RI60_data.csv')
    # df_shock = pd.read_csv('Shock_Data.csv')
    groupBys = ['mouse', 'group', 'sesType', 'sex']
    if drugCheck(mouseInfoDict, 'virus'):
        groupBys.append('virus')
        # groupBys = ['mouse','drug','group','sesType']
    dfgrouped_training = df_training.groupby(groupBys, as_index=False).mean(numeric_only=True)
    dfgrouped_training.to_csv('RI60dataForCorrs.csv')

    mouselessGBs = [item for item in groupBys if item != 'mouse']
    dfgrouped1_training = dfgrouped_training.groupby(mouselessGBs, as_index=False).agg(
        {'pokeRate': list, 'entryRate': list, 'P/E ratio': list, 'efficiency': list, 'rewardRate': list})

    # TODO: make the labels of the columns the names of the mice, and make it not matter how many there are!
    dfgrouped1_training[['pokeRate{}'.format(i) for i in range(1, dfgrouped1_training['pokeRate'].str.len(
    ).max() + 1)]] = dfgrouped1_training['pokeRate'].apply(lambda x: pd.Series(x))
    dfgrouped1_training[['entryRate{}'.format(i) for i in range(1, dfgrouped1_training['entryRate'].str.len(
    ).max() + 1)]] = dfgrouped1_training['entryRate'].apply(lambda x: pd.Series(x))
    dfgrouped1_training[['P/E ratio{}'.format(i) for i in range(1, dfgrouped1_training['P/E ratio'].str.len(
    ).max() + 1)]] = dfgrouped1_training['P/E ratio'].apply(lambda x: pd.Series(x))
    dfgrouped1_training[['efficiency{}'.format(i) for i in range(1, dfgrouped1_training['efficiency'].str.len(
    ).max() + 1)]] = dfgrouped1_training['efficiency'].apply(lambda x: pd.Series(x))
    dfgrouped1_training[['rewardRate{}'.format(i) for i in range(1, dfgrouped1_training['rewardRate'].str.len(
    ).max() + 1)]] = dfgrouped1_training['rewardRate'].apply(lambda x: pd.Series(x))


    # drop the original 'pokeRate' column
    dfgrouped1_training.drop('pokeRate', axis=1, inplace=True)
    dfgrouped1_training.drop('entryRate', axis=1, inplace=True)
    dfgrouped1_training.drop('P/E ratio', axis=1, inplace=True)
    dfgrouped1_training.drop('efficiency', axis=1, inplace=True)
    dfgrouped1_training.drop('rewardRate', axis=1, inplace=True)

    dfgrouped1_training.to_csv('forPrismRI60.csv')

    # * Shock
    shockGBs = groupBys.copy()
    shockGBs.remove('sesType')
    shockGBs.append('dayOnType')
    mouselessGBsShock = mouselessGBs.copy()
    mouselessGBsShock.remove('sesType')
    mouselessGBsShock.append('dayOnType')
    df_shock['numShock'] = pd.to_numeric(df_shock['numShock'], errors='coerce')
    df_shock['numRew'] = pd.to_numeric(df_shock['numRew'], errors='coerce')
    dfgrouped_shock = df_shock.groupby(shockGBs, as_index=False).mean(numeric_only=True)

    dfgrouped1_shock = dfgrouped_shock.groupby(mouselessGBsShock, as_index=False).agg(
        {'numShock': list, 'numRew': list})  # , 'P/E ratio': list})

    dfgrouped1_shock[['numShock{}'.format(i) for i in range(1, dfgrouped1_shock['numShock'].str.len(
    ).max() + 1)]] = dfgrouped1_shock['numShock'].apply(lambda x: pd.Series(x))
    dfgrouped1_shock[['numRew{}'.format(i) for i in range(1, dfgrouped1_shock['numRew'].str.len(
    ).max() + 1)]] = dfgrouped1_shock['numRew'].apply(lambda x: pd.Series(x))

    # dfgrouped1_shock[['numShock1', 'numShock2', 'numShock3']] = dfgrouped1_shock['numShock'].apply(lambda x: pd.Series(x))
    # dfgrouped1_shock[['numRew1', 'numRew2', 'numRew3']] = dfgrouped1_shock['numRew'].apply(lambda x: pd.Series(x))
    # # dfgrouped1_shock[['PERatio1', 'PERatio2', 'PERatio3']] = dfgrouped1_shock['P/E ratio'].apply(lambda x: pd.Series(x))

    # drop the original 'pokeRate' column
    dfgrouped1_shock.drop('numShock', axis=1, inplace=True)
    dfgrouped1_shock.drop('numRew', axis=1, inplace=True)
    # dfgrouped1_shock.drop('P/E ratio', axis=1, inplace=True)

    dfgrouped1_shock.to_csv('forPrismShock.csv')

def shockHistogram(output_file_folder):
    os.chdir(output_file_folder)
     # Load the feather file
    df_shock = feather.read_feather('shock_dataFrame.feather')
    df_shock['dayOnType'] = df_shock['dayOnType'].astype(float).astype(int).astype(str)
    stressSessions = df_shock[(df_shock['group'] == 'stress') & (df_shock['dayOnType'] == '2')]
    naiveSessions = df_shock[(df_shock['group'] == 'naive') & (df_shock['dayOnType'] == '2')]

    #get last poke timestamp for each session and export it
    # Get last poke timestamp for each session
    last_poke_data = []
    for _, row in df_shock[df_shock['dayOnType'] == '2'].iterrows():
        timestamps = row['pokeTimestamps']
        if len(timestamps) > 0:
            last_poke = max(timestamps) / 60  # convert to minutes
            last_poke_data.append({
                'mouse': row['mouse'],
                'group': row['group'],
                'last_poke_min': last_poke
            })
    
    last_poke_df = pd.DataFrame(last_poke_data)
    last_poke_df.to_csv('shock_lastPoke.csv', index=False)


    # create histograms for individual mice with 5 minute bins (first centered at 2.5, last at 57.5), then get the slope of the line of best fit for each mouse
    mouseList_stress = stressSessions['mouse'].unique()
    mouseList_naive = naiveSessions['mouse'].unique()
    # Define bins centered at 2.5, 7.5, ..., 57.5 (5 minute bins)
    bin_edges = np.arange(0, 65, 5)  # 0, 5, 10, ..., 60
    bin_centers = np.arange(2.5, 60, 5)  # 2.5, 7.5, ..., 57.5
    
    slope_results = []
    
    for mouse in mouseList_stress:
        mouse_data = stressSessions[stressSessions['mouse'] == mouse]
        timestamps = np.concatenate(mouse_data['pokeTimestamps'].values) / 60  # convert to minutes
        hist_counts, _ = np.histogram(timestamps, bins=bin_edges)
        
        # Get slope between first bin and last non-zero bin
        if len(hist_counts) >= 2:
            # Find the last non-zero bin index
            non_zero_indices = np.where(hist_counts > 0)[0]
            if len(non_zero_indices) > 0 and hist_counts[0] > 0:
                last_idx = non_zero_indices[-1]
                raw_slope = (hist_counts[last_idx] - hist_counts[0]) / (bin_centers[last_idx] - bin_centers[0])
                normalized_slope = raw_slope / hist_counts[0]  # Normalize by first bin
                slope_results.append({'mouse': mouse, 'group': 'stress', 'slope': raw_slope,
                                      'normalized_slope': normalized_slope,
                                      'first_bin': hist_counts[0], 'last_bin': hist_counts[last_idx]})
    
    for mouse in mouseList_naive:
        mouse_data = naiveSessions[naiveSessions['mouse'] == mouse]
        timestamps = np.concatenate(mouse_data['pokeTimestamps'].values) / 60  # convert to minutes
        hist_counts, _ = np.histogram(timestamps, bins=bin_edges)
        
        # Get slope between first bin and last non-zero bin
        if len(hist_counts) >= 2:
            # Find the last non-zero bin index
            non_zero_indices = np.where(hist_counts > 0)[0]
            if len(non_zero_indices) > 0 and hist_counts[0] > 0:
                last_idx = non_zero_indices[-1]
                raw_slope = (hist_counts[last_idx] - hist_counts[0]) / (bin_centers[last_idx] - bin_centers[0])
                normalized_slope = raw_slope / hist_counts[0]  # Normalize by first bin
                slope_results.append({'mouse': mouse, 'group': 'naive', 'slope': raw_slope,
                                      'normalized_slope': normalized_slope,
                                      'first_bin': hist_counts[0], 'last_bin': hist_counts[last_idx]})

    slope_df = pd.DataFrame(slope_results)
    slope_df.to_csv('shock_slopes.csv', index=False)
    # print(slope_df)
    

    # create histograms for groups combined across all mice
    stressHisto = np.concatenate(stressSessions['pokeTimestamps'].values)
    naiveHisto = np.concatenate(naiveSessions['pokeTimestamps'].values)
    # Combine and plot both histograms
    stressHisto_min = stressHisto / 60
    naiveHisto_min = naiveHisto / 60

    # Pad shorter array with NaNs to match lengths
    max_len = max(len(stressHisto_min), len(naiveHisto_min))
    stress_col = np.pad(stressHisto_min, (0, max_len - len(stressHisto_min)), constant_values=np.nan)
    naive_col = np.pad(naiveHisto_min, (0, max_len - len(naiveHisto_min)), constant_values=np.nan)

    # Create DataFrame
    export_df = pd.DataFrame({
        'stress': stress_col,
        'naive': naive_col
    })

    # save new csv with number of sessions going into each condition
    session_counts = {
        'stress': np.shape(stressSessions)[0],
        'naive': np.shape(naiveSessions)[0]
    }
    with open('session_counts.txt', 'w') as f:
        for key, value in session_counts.items():
            f.write(f'{key}: {value}\n')

    export_df.to_csv('shock_poke_timestamps_by_group.csv', index=False)
    print(f'stress sessions: {np.shape(stressSessions)[0]}')
    print(f'naive sessions: {np.shape(naiveSessions)[0]}')
    # Set custom bin width (e.g., 1 minute bins)
    bin_width = .5  # in minutes
    min_edge = min(stressHisto_min.min(), naiveHisto_min.min())
    max_edge = max(stressHisto_min.max(), naiveHisto_min.max())
    bins = np.arange(min_edge, max_edge + bin_width, bin_width)

    # # Plot histograms
    # plt.hist(stressHisto_min, bins=bins, alpha=0.5, label='Stress', color='red', edgecolor='black')
    # plt.hist(naiveHisto_min, bins=bins, alpha=0.5, label='Naive', color='blue', edgecolor='black')

    # # Add titles and labels
    # plt.title("Histogram of Poke Timestamps (Minutes, Day 2)")
    # plt.xlabel("Time (minutes)")
    # plt.ylabel("Frequency")
    # plt.legend()
    # plt.show()

    a=1
