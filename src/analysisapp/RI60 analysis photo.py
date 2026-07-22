# * as of 5/31/24, creating photodataframe is done in Jimmy, functions are run here
import os
from RI60_photo_analysis_FUNCTIONS import *
from _config import RI60_DIR
doExclusions = False
fileName = 'photoDataFrame_baseLineCorr.feather'




doShocks = False
doPSTHs = False
R_only = False

modelAUCs = False
doFastFFMPlots = True
doMerging = False
mergeType = 'rates' # 'comp' or 'rates'

addCumCount = False


#---------------------------------------------------------------
output_path = RI60_DIR
exclusionMice = [ ('578-1', 'TS'), ('648-L', 'DMS'), ('722-T', 'TS'), ('723-0', 'TS'), ('725-R', 'DMS')]


photoDF = loadPhotoDF(fileName)




if doExclusions == True:
    # photoDF = loadPhotoDF(fileName)
    print('Removing excluded mice from photoDF...')
    photoDF_remove = removeMouse(photoDF, exclusionMice)
    photoDF_remove.reset_index(drop=True).to_feather(fileName)
    print('Excluded mice removed and photoDF updated!')

if addCumCount == True:
    print('Adding cumulative poke count to photoDF...')
    # photoDF = loadPhotoDF(fileName)
    with open('behaviorData.pickle', 'rb') as file:
        behaviorData = pickle.load(file)

    photoDF = addCumulPokeCount(photoDF, behaviorData)
    photoDF.reset_index(drop=True).to_feather(fileName)
    print('Cumulative poke count added to photoDF and updated!')

if doPSTHs == True:
    print('Generating PSTH CSVs for each event...')
    # photoDF = loadPhotoDF(fileName)
    eventList = ['ReNP', 'UnNP', 'RePE']

    for event in eventList:

        generateCSVs(photoDF,event, 3, only_R=R_only)
    print('PSTH CSVs generated!')

if modelAUCs == True:
    print('Calculating model AUCs for each event...')
    # photoDF = loadPhotoDF(fileName)
    eventList = ['ReNP', 'UnNP', 'RePE']
    for event in eventList:
        modelAUC(photoDF, event)
    print('Model AUCs calculated and saved!')


if doShocks == True:
    print('Creating shock dictionary and adding shocks to photoDF...')
    shockDict = createShockDict(output_path)
    putShocksInPhotoDF(shockDict)

    print('Shocks added to photoDF!')

if doFastFFMPlots == True:
    print('Generating fast FMM plots...')
    plot_FMMs(output_path)
    print('Fast FMM plots generated!')

if doMerging == True:
    print(f'Merging photoDF with compDF based on {mergeType}...')
    mergePhotoAndComp(mergeType)
    print(f'Photo and comp merged based on {mergeType} and saved!')


print('All done!')