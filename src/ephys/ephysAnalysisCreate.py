#%%
import os
import h5py
import pandas as pd
import numpy as np
import matlab
import matlab.engine
import pprint
import scipy.signal
import matplotlib.pyplot as plt
from ephysAnalysisSetup import *
import pickle
import time
from _config import INTRINSIC_DIR, TOOLS
# Opto intensity ranges per cell, e.g.:
#   optoIntensityDict = {
#       'm005 s1c1': {15: range(1,4), 50: range(4,7), 100: range(7,10)},
#       ...
#   }
# Define this in a sibling file (opto_dicts.py) and edit there.
try:
    from opto_dicts import optoIntensityDict
except ImportError:
    optoIntensityDict = {}
    print('[warn] opto_dicts.optoIntensityDict not found; opto sweeps will have NaN intensity')
# %%
# #* create instances for all the mice
mouseDict = makeMice()
renew = False
# When True, skip any mouse already present in ephysMouseDict (loaded from the
# collective .pkl) instead of re-reading every _processed.pkl and re-running all
# analysis for it. Coarse but maximal speedup. Set False (or delete the mouse
# from the collective .pkl) if you've ADDED cells/files to a processed mouse.
skip_existing_mice = True

# When True, reuse a previously-computed cell from the collective .pkl whenever
# its source folder is unchanged (same .h5 files, sizes, and mtimes), instead of
# re-reading every _processed.pkl and re-running all analysis. Cells whose folder
# changed (new/edited/removed .h5) are reprocessed automatically, so you can add
# data without clearing the cache. NOTE: this keys off the DATA, not the analysis
# code — if you edit ephysAnalysisSetup.py, set renew=True (or delete the
# collective .pkl) once to force a full rebuild.
reuse_unchanged_cells = True

def getAllFoldersorFiles(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".h5"):
                file_paths.append(os.path.join(root, file)) 
    return file_paths

def stripSweep(sweepString):
    onlyNumStr = sweepString.split('_')[1]
    zerosRemoved = onlyNumStr.lstrip('0')
    sweepNum = int(zerosRemoved)
    return sweepNum


def folderFingerprint(file_paths):
    # Identity of a cell folder's raw data: each .h5 by name, size, and mtime.
    # Any added / removed / edited .h5 changes this, triggering a reprocess.
    fp = []
    for p in sorted(file_paths):
        try:
            st = os.stat(p)
            fp.append((os.path.basename(p), st.st_size, int(st.st_mtime)))
        except OSError:
            fp.append((os.path.basename(p), None, None))
    return tuple(fp)


#! going to need to loop through all days from an experiment
#* things to get here: mouse number and all associated info (sex, stress condition, etc)
# data_path = r"R:\Basic_Sciences\Phys\Lerner_Lab_tnl2633\Jacob\aCUS\_aCUS_August2023\Ephys\Raw Data Organized"
# data_path = r"C:\Users\jan7154\Documents\aCUS_analysis\ephys\_aCUS_August2023\Ephys\Raw Data Organized"
# data_path = r"D:\Data_analysis\Lerner_Lab\aCUS\Ephys\_aCUS_August2023\Ephys\Raw Data Organized"

# data_path = r'R:\Basic_Sciences\Phys\Lerner_Lab_tnl2633\Jacob\aCUS\ALL_INTRINSIC_EPHYS\raw_data_organized'
# analysis_path = r'R:\Basic_Sciences\Phys\Lerner_Lab_tnl2633\Jacob\aCUS\ALL_INTRINSIC_EPHYS\analysis'

# data_path = r'R:\Basic_Sciences\Phys\Lerner_Lab_tnl2633\Jacob\aCUS\ALL_oIPSC_EPHYS\raw_data_organized'
# analysis_path = r'R:\Basic_Sciences\Phys\Lerner_Lab_tnl2633\Jacob\aCUS\ALL_oIPSC_EPHYS\analysis'

# data_path = r'R:\Basic_Sciences\Phys\Lerner_Lab_tnl2633\Jacob\aCUS\ALL_M-CURRENT_EPHYS\raw_data_organized'
# analysis_path = r'R:\Basic_Sciences\Phys\Lerner_Lab_tnl2633\Jacob\aCUS\ALL_M-CURRENT_EPHYS\analysis'

data_path = r'R:\Basic_Sciences\Phys\Lerner_Lab_tnl2633\Jacob\aCUS\ALL_INTRINSIC_EPHYS\timecourse experiment\raw_data_organized'
# NOTE: retargeted from 'timecourse experiment\analysis' (a different dataset) to the canonical INTRINSIC_DIR
analysis_path = INTRINSIC_DIR


os.chdir(analysis_path)
if os.path.exists(os.path.join(analysis_path, 'aCUS_intrinsicData_collective.pkl')) == True:
    with open('aCUS_intrinsicData_collective.pkl', "rb") as file:
        ephysMouseDict = pickle.load(file)
else:
    ephysMouseDict = {}
    
mouse_folders = os.listdir(data_path)
mouse_folders1 = [item for item in mouse_folders if item.startswith('2')]

newMice = {}
# ephysMouseDict = {}

# Lazily start MATLAB only when a file actually needs converting. If every .h5
# already has a _processed.pkl, MATLAB is never started (saves the ~10-20s
# startup) and no per-file eng.cd round-trips happen.
eng = None
def get_engine():
    global eng
    if eng is None:
        print('starting MATLAB engine...')
        eng = matlab.engine.start_matlab()
        for ws_path in (TOOLS['matlab_wavesurfer'],):
            if os.path.exists(ws_path):
                eng.cd(ws_path)
                break
    return eng

# Map (mouseName, cellName) -> previously-computed EphysCell, for cell-level reuse.
cachedCells = {}
if reuse_unchanged_cells and renew is False:
    for _mname, _m in ephysMouseDict.items():
        for _c in getattr(_m, 'cells', []):
            cachedCells[(_mname, _c.name)] = _c

for mouseFolder in mouse_folders1:
    # mouseFolder = mouse_folders1[0]
        #* here's where we get the mouse ID and from that, sex and stress condition
    mouseID = mouseFolder.split(' ',1)[1]

    mouseInstance = mouseDict[mouseID]
    if mouseInstance.use is False:
        continue
    if skip_existing_mice and (mouseInstance.name in ephysMouseDict) and (renew is False):
        print(f'skipping {mouseID}, already in collective pkl')
        continue
    print(f'processing {mouseID}')
    mouseCond = mouseInstance.stressCon
    mouseSex = mouseInstance.sex

    mouseFolderPath = os.path.join(data_path,mouseFolder)
    cell_folders = os.listdir(mouseFolderPath)
    cell_folders1 = [item for item in cell_folders if item.startswith('m')]

    for cellIDFolder in cell_folders1:
    # cellIDFolder = cell_folders1[0]
        cellFolderPath = os.path.join(mouseFolderPath,cellIDFolder)
        cellID = cellIDFolder
        if ('-' in cellID) or ('bad' in cellID):
            print('skipping bead negative boi ' + cellID)
            continue
        if 'mz' in cellIDFolder:
            index = cellIDFolder.index('(mz')
            cellIDFolder = cellIDFolder[:index-1]


        # beadID = cellIDFolder[-1::]
        # if (beadID != '-') & (beadID != '+'):
        #     print('skipping ' + cellID)
        #     continue
        
        cell = EphysCell(cellID)
        #%%
        #! going to need to loop through all cells from a day
        #* things to get here: bead positive or negative

        # folder_path = r"D:\Data_analysis\Lerner_Lab\aCUS\Ephys\_aCUS_August2023\Ephys\Raw Data Organized\20230910 610-005\m005 s1c1 -"

        #! going to need to loop through all files in each cell  
        #* things to get from here: protocol being run
        file_paths = []
        for root, dirs, files in os.walk(cellFolderPath):
            for file in files:
                if file.endswith(".h5"):
                    file_paths.append(os.path.join(root, file))

        # Cell-level cache: if this exact folder was processed before and nothing
        # in it changed, reuse the stored cell and skip all reconversion/analysis.
        fingerprint = folderFingerprint(file_paths)
        cachedCell = cachedCells.get((mouseInstance.name, cellID))
        if cachedCell is not None \
                and getattr(cachedCell, '_sourceFingerprint', None) == fingerprint:
            print(f'reusing cached {cellID}, folder unchanged')
            mouseInstance.addCell(cachedCell)
            continue

        #%%

        for path in file_paths:
            targetDir = os.path.dirname(path)
            newName = os.path.basename(path).replace('.h5','_processed.pkl')
            processedPath = os.path.join(targetDir, newName)

            sweepNumbs = path.split('\\')[-1].split('_')[-1].split('.')[0].split('-')
            sweepNumbs = list(map(int, sweepNumbs))
            # if len(sweepNumbs) != 2:
            #     print(f'skipping {cellID} {sweepNumbs}, not Ih')
            #     continue
            # if sweepNumbs[1] - sweepNumbs[0] != 3:
            #     print(f'skipping {cellID} {sweepNumbs}, not Ih')
            #     continue

            if os.path.exists(processedPath) & (renew == False):
                print(f'won\'t convert {newName}, pickle already exists')
                with open(processedPath, 'rb') as file:
                    dataConverted = pickle.load(file)

            else:
            # startWS t= time.time()
            # print('starting WS')
                sweeps = path.split('\\')[-1]
                print(f'file {sweeps} parsing')
                dataConverted = get_engine().ws.loadDataFile(path)

                # endWS = time.time()
                # print ('WS took ', endWS-startWS, ' sec')

                dataConverted = dict(dataConverted)

                os.chdir(targetDir)
                with open(newName, 'wb') as file:
                    pickle.dump(dataConverted, file)

            protocol_full = dataConverted['header']['AbsoluteProtocolFileName'].rsplit('\\',1)[-1]

            stimuli_dict = dataConverted['header']['Stimulation']['StimulusLibrary']['Stimuli']

            #%%
            for key, value in stimuli_dict.items():
                if protocol_full not in protocolDict.keys():
                    print(f'protocol {protocol_full} not in protocolDict, skipping')
                    continue
                if 'Name' in value and value['Name'] == protocolDict[protocol_full]['stim']:
                # Found a match, store the 'Amplitude' and Duration values
                    if 'Delay' in value:
                        delay_value = value['Delay']
                    if 'EndTime' in value:
                        endTime_value = value['EndTime']
                    if 'Amplitude' in value:
                        amplitude_value = value['Amplitude']
                    if 'Duration' in value:
                        stimDuration = value['Duration']
            if (protocol_full == 'PA_Ch1_WC_VC_test pulse.cfg') | (protocol_full == 'VC test pulse JN.cfg'):
                testPulseParams = ProtocolParams(protocol_full, stimDuration=stimDuration, sampling_rate = 10000, amplitude_value = amplitude_value, delay=delay_value, end=endTime_value)
                cell.addInputResistance(dataConverted, testPulseParams)
                cell.addAccess_andCap(dataConverted, testPulseParams)                
                # cell.addCapacitance(dataConverted, testPulseParams)

            if protocol_full == 'PA_Ch1_CC excitability 20 pA steps to 200 pA.cfg':
                firingRateParams = ProtocolParams(protocol_full, threshold = 0, stimDuration=stimDuration, sampling_rate = 10000, amplitude_value = amplitude_value, delay=delay_value, end=endTime_value)
                cell.createCurrentDict(dataConverted, firingRateParams, 'FRCurrentDict')
                cell.addFiringRate(dataConverted, firingRateParams)
                # Fallback rheobase from the F-I sweeps (first current to spike in
                # the first 50 ms), used downstream only when no real 5 pA rheobase.
                cell.addRheobaseFromFI(dataConverted, firingRateParams)
                cell.createSpikeAnalysis(dataConverted, firingRateParams, 'FRCurrentDict', 'firing rate')
                # cell.phasePlotAnalysis(dataConverted)


            if protocol_full == 'PA_Ch1_CC excitability 5 pA steps to 100 pA.cfg':
                rheoParams = ProtocolParams(protocol_full, threshold = 0, stimDuration=stimDuration, sampling_rate = 10000, amplitude_value = amplitude_value, delay=delay_value, end=endTime_value)
                cell.createCurrentDict(dataConverted, rheoParams, 'rheoCurrentDict')
                cell.addRheobase(dataConverted, rheoParams)
                cell.createSpikeAnalysis(dataConverted, rheoParams, 'rheoCurrentDict', 'rheobase')


            if protocol_full == "PA_Ch1_CC spont firing for 30s.cfg":
                cell.addRMP(dataConverted)
                # rheoParams = ProtocolParams(protocol_full, threshold = 0, stimDuration=stimDuration, sampling_rate = 10000, amplitude_value = amplitude_value, delay=delay_value, end=endTime_value)
                # cell.createSpikeAnalysis(dataConverted, rheoParams, 0, 'resting')
            # Current-clamp voltage SAG (this CC protocol measures sag, not Ih).
            if protocol_full == "PA_Ch1_CC Ih -50 pA steps to -150 pA.cfg":
                sag_Params = ProtocolParams(protocol_full, threshold = 0, stimDuration=stimDuration, sampling_rate = 10000, amplitude_value = amplitude_value, delay=delay_value, end=endTime_value)
                cell.createCurrentDict(dataConverted, sag_Params, 'sagCurrentDict')
                cell.addSag(dataConverted, sag_Params)

            # --- Opto single 1ms green-light pulse ---
            # Per-sweep: 10ms VC test pulse at 0.1s, then 1ms light pulse at 2.0s.
            if protocol_full == 'JN_VC_green-light_1ms.cfg':
                io_ranges = optoIntensityDict.get(cellID, None)
                if io_ranges is None:
                    print(f'[warn] {cellID} not in optoIntensityDict; tagging NaN intensity')

                optoParams = ProtocolParams(
                    protocol_full,
                    sampling_rate=10000,
                    current_channel_index=0,
                    # Single light pulse at 2.0 s (hardcoded — protocol is fixed)
                    pulse_onsets_s=(2.0,),
                    # Access/cap: test pulse at 0.1s for 10ms
                    tp_delay_s=0.1,
                    tp_duration_s=0.01,
                    baseline_offset_s=0.05,
                    baseline_window_s=0.05,
                    amplitude_value=-5,
                    # Opto baseline window: pre-pulse-1, after test pulse settles
                    test_pulse_end_s=0.11,
                    test_pulse_recovery_s=0.2,
                )
                if io_ranges is not None:
                    optoParams.sweep_to_label = ('intensity', io_ranges)
                cell.addAccess_andCap(dataConverted, optoParams)
                cell.addOptoResponse(dataConverted, optoParams)

            # --- Opto PPR: 1ms green light, 100ms ISI, both pulses at full intensity ---
            # Per-sweep: 100ms VC test pulse at 1.0s; light pulses at 2.0s + 2.101s.
            if protocol_full == 'JN_ES_MZ_VC_PPR 1ms Green light 100ms ISI.cfg':
                optoParams = ProtocolParams(
                    protocol_full,
                    sampling_rate=10000,
                    current_channel_index=0,
                    # Two light pulses: 2.0 s + 2.101 s (100 ms ISI)
                    pulse_onsets_s=(2.0, 2.101),
                    tp_delay_s=1.0,
                    tp_duration_s=0.1,
                    baseline_offset_s=0.05,
                    baseline_window_s=0.05,
                    amplitude_value=-5,
                    test_pulse_end_s=1.1,
                    test_pulse_recovery_s=0.2,
                    # PPR is always at one (full) intensity per your protocol — constant tag
                    pulse_label=('intensity', 100),
                )
                cell.addAccess_andCap(dataConverted, optoParams)
                cell.addOptoResponse(dataConverted, optoParams)

            # --- Spontaneous IPSCs: 31s VC sweep, single test pulse at 0.5s ---
            # TTL monitor on AI ch index 1 ("Blue Light"). No opto analysis (no pulses).
            # Only access/cap from the test pulse. (Spontaneous-event method TBD.)
            if protocol_full == 'JN_ES_MZ_VC_spont_currents_for_31s.cfg':
                spontParams = ProtocolParams(
                    protocol_full,
                    sampling_rate=10000,
                    current_channel_index=0,
                    tp_delay_s=0.5,
                    tp_duration_s=0.1,
                    baseline_offset_s=0.05,
                    baseline_window_s=0.05,
                    amplitude_value=-5,
                )
                cell.addAccess_andCap(dataConverted, spontParams)

            # --- M-current: VC hyperpolarizing-step deactivation protocol ---
            # Hold -10 mV. Per sweep: 10ms -5mV test pulse at sweep start (t=0),
            # then 1s hyperpolarizing step at t=0.2s. Step voltages -70..-10 mV
            # across the 7 sweeps. Drug sweeps (XE-991 etc.) handled via the
            # standard drugSweeps mechanism downstream.
            # NOTE: addAccess_andCap is not called here — the test pulse sits at
            # t=0 with no pre-pulse baseline window. Monitor Rs from other protocols.
            if protocol_full == 'JN_Ch1_VC_mCurrent_hyperpolStep.cfg':
                mCurrentParams = ProtocolParams(
                    protocol_full,
                    sampling_rate=10000,
                    current_channel_index=0,
                    # Step voltage computed from sweep position in the file:
                    #   step_voltage = holding_mV + step_delta_mV*(pos - zero_step_position)
                    # 7 sweeps, hold -10 mV, 10 mV deltas, position 7 = no step:
                    #   pos 1..7 -> -70, -60, -50, -40, -30, -20, -10 mV
                    holding_mV=-10.0,
                    step_delta_mV=10.0,
                    zero_step_position=7,
                    step_onset_s=0.2,
                    step_duration_s=1.0,
                    E_K_mV=-97.0,        # 145 mM K in / 2.5 mM K out, 32 C
                )
                cell.addMCurrent(dataConverted, mCurrentParams)

            # --- Voltage-clamp Ih: hyperpolarizing step from -70 mV holding ---
            # 3 identical repeat sweeps (4 s, 10 kHz). Command channel is the
            # WaveSurfer delta vs the -70 mV amplifier holding: 0->+30 (depol,
            # 0.5-2.0s, -40 mV) -> -50 (hyperpol, 2.0-3.5s, -120 mV) -> 0. Ih is
            # the slow inward relaxation during the -120 mV step.
            if protocol_full == 'PA_Ch1_WC_VC_Ih test.cfg':
                vcIhParams = ProtocolParams(
                    protocol_full,
                    sampling_rate=10000,
                    current_channel_index=0,
                    holding_mV=-70.0,
                    step_onset_s=2.0,          # hyperpol step start
                    step_duration_s=1.5,       # 2.0 -> 3.5 s
                    step_delta_mV=-50.0,       # -> -120 mV actual
                    cap_skip_s=0.015,          # skip capacitive transient before fit
                    ss_window_s=0.10,          # last 100 ms = steady-state
                    tau_min_s=0.005, tau_max_s=2.0,
                    baseline_offset_s=0.1, baseline_window_s=0.3,  # holding baseline (0-0.5 s)
                    # --- auto QC thresholds (see EphysCell.addIh) ---
                    hold_max_pA=500.0,
                    baseline_drift_tol=100.0,
                    plateau_tol=50.0,
                )
                cell.addIh(dataConverted, vcIhParams)

        if hasattr(cell, 'spikeAnalysis'):
            cell.spikeAnalysis = pd.DataFrame(cell.spikeAnalysis)

        # Build xcorr templates and reclassify after all .h5 files for this cell
        if hasattr(cell, 'optoSweeps') and cell.optoSweeps is not None \
                and not cell.optoSweeps.empty:
            cell.buildOptoTemplates()

                # %%
        # Record the folder fingerprint so this cell can be reused next run.
        cell._sourceFingerprint = fingerprint
        mouseInstance.addCell(cell)
        print('!done with ' + cellIDFolder)
    newMice[mouseInstance.name] = mouseInstance
    print ('!!done with ' + mouseFolder)
if eng is not None:
    eng.quit()
print('done!')

ephysMouseDict.update(newMice)
os.chdir(analysis_path)
with open("aCUS_intrinsicData_collective.pkl", 'wb') as file:
    pickle.dump(ephysMouseDict, file)