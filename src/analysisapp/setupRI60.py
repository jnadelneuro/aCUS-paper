import os
# import numpy as np
# import pandas as pd
# import pickle
import pyarrow.feather as feather
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
import panel as pn
import numpy as np
from RI60_analaysis_fx_PHOTO_JIMMYv2 import *


#import itertools
# nfrom GUI_for_mouse_info2 import *
# from RI60_analaysis_fx_v2_JIMMY import *
# from RI60_analaysis_fx_PHOTO_JIMMY import *




def get_RI60_layout(files_behavior, files_photometry, medPC_rawDataInput, h5_checkbox,tabs):
    # * RI60
    mark_down_1 = pn.pane.Markdown(
    """**Select Files for behavioral analysis**""", width=500)
    explain_path_structure = pn.pane.Markdown("""
                                   ***A Note on Path Structure:***<br>
                                   For behavior-only analysis, the folder you select should contain a folder called "rawMedData" that contains... 
                                   the raw med data. Everything else will be built out from there, placed in folders named by the app. 
                                   """, width=400)
    photoDFCreate_markdown = pn.pane.Markdown("""**Create Photometry Dataframe**<br>
                                        *WARNING:* data must be stored in (yourfolder)\processed""", width=250)

    photoDFCreate_button = pn.widgets.Button(name='create photo dataframe', button_type='success',
                                             width=250, align='start')

    
    # TODO: make checkboxes for:  summary data

    # transition_title = pn.pane.Markdown(
    #     '''**Switching Analysis**''', width=200)
    # transition_markdown = pn.pane.Markdown("""**Plot Switching Behavior**<br>
    #                                             *requires a 'transitionDataFrame' feather file. If you don't have one, run Create Key DataFrames first*<br>
    #                                             This function will quantify and plot switching behavior (post-transition) ** MAKE BETTER** """, width=400)
    # # transition_checkbox = pn.widgets.Checkbox(
    # #     name="Block switch analysis?", width=200)
    # transition_trials = pn.widgets.LiteralInput(
    #     name='# of trials:', value=0, type=int,  width=60)
    # transition_logfit = pn.widgets.Checkbox(
    #     name='log fit?', width=60)
    # transition_thirds = pn.widgets.Checkbox(name='thirds?', width=60)

    # WSLS_title = pn.pane.Markdown('''**Win-Stay Analysis**''')
    # WSLS_checkbox = pn.widgets.Checkbox(name='Win-Stay analysis?', width=200)
    # WSLS_plot = pn.widgets.Select(name='plot:',
    #                               value='None',
    #                               options=['None', 'rel. to all',
    #                                        'rel. to each'],
    #                               width=150)
    # regression_title = pn.pane.Markdown('''**Regression**''')
    # regression_markdown = pn.pane.Markdown('''**Plot regression coefficients**<br>
    #                                             *write about what this fx does!!!*''')
    # regression_checkboxes = pn.widgets.CheckBoxGroup(
    #     options=[('Choice'), ('Reward'), ('CxR')],
    #     inline=True, width=200)
    # regression_trialNumbs = pn.widgets.LiteralInput(
    #     name='# of trials:', value=0, type=int,  width=60)

    # latency_title = pn.pane.Markdown('''**Latency Analysis**''')
    # latency_markdown = pn.pane.Markdown(
    #     '''NOTE that this doesn't generate a plot, it just outputs a .csv file you can put in to GraphPad''')
    # latency_button = pn.widgets.Button(
    #     name='Create latency csv', button_type='warning')

    # # %%
    # # * photometry stuff
    # photoFunctions_markdown = pn.pane.Markdown("""**Run Functions**<br>
    #                                 Select functions to the right then click button below to run them""", width=250)

    # photoFunctions_button = pn.widgets.Button(name='Run Selected Functions', button_type='success',
    #                                         width=250, align='start')



    # photo_markdown = pn.pane.Markdown('''**Event Select**''')
    # # photo_checkboxes = pn.widgets.CheckBoxGroup(
    # #     options=[('Reward NP'), ('Unrewarded nosepoke'), ('Reward Retreival'), ('Unexpected reward (aligned to poke)'),
    # #              ('Unexpected omission (aligned to poke)'), ('Unexpected reward (aligned to retreival)')],
    # #     inline=False)

    # options_markdown = pn.pane.Markdown('''**Plotting Options**''')
    # loc_sel = pn.widgets.Select(name='recLoc:',
    #                             value='both',
    #                             options=['both', 'DMS', 'DLS', 'SNc'])
    # # movement_sel = pn.widgets.Checkbox(name='plot contra vs. ipsi?')
    # split_rows = pn.widgets.Select(name='factor to split rows of graph:',
    #                                value='None',
    #                                options=['None', 'sesType', 'recordingLoc', 'group', 'choiceMovement', 'outcomeMovement', 'surprise'])
    # split_cols = pn.widgets.Select(name='factor to split columns of graph:',
    #                                value='None',
    #                                options=['None', 'sesType', 'recordingLoc', 'group', 'choiceMovement', 'outcomeMovement', 'surprise'])

    # split_sel = pn.widgets.Select(name='groups to split (different lines on each plot):',
    #                               value='group',
    #                               options=['None', 'sesType', 'recordingLoc', 'group', 'choiceMovement', 'outcomeMovement', 'surprise'])

    # event_sel = pn.widgets.Select(name='events to plot',
    #                               value='None',
    #                               options=['None', 'rewarded NP', 'unrew NP', 'reward retrieval'])

    # days_to_exclude_sel = pn.widgets.LiteralInput(
    #     name='# of days to exclude per sesType:', value=0, type=int,  width=60)

    # options_selects = pn.Column(
    #     loc_sel, pn.Row(split_rows, split_cols), pn.Row(split_sel, days_to_exclude_sel))

    # photo_options_box = pn.WidgetBox(
    #     pn.Column(photo_markdown, event_sel), pn.Column(options_markdown, options_selects), width=650, width_policy='max')

    # photo_box = pn.Column(photo_options_box)

    # # individual_analysis_wd_2 = pn.Column('<br>',medPC_title,medPC_checkbox, keyDF_checkbox,WSLS_title, WSLS_checkbox, WSLS_plot,transition_title,transition_checkbox, transition_trials)

    

    # visualization_wd = pn.Row(visualize_zscore_or_dff, visualizeAverageResults, width=800)
    run_markdown = pn.pane.Markdown(
        """<br>**<span style="font-size:larger;">To run functions chosen on the main panel:</span>**""", width=250)
    runBeh_fx_button = pn.widgets.Button(name='Run Selected Functions', button_type='primary',
                                   width=500, sizing_mode="stretch_width", align='end')

    def checkSameLocation(arr, abspath):
        #abspath = []
        for i in range(len(arr)):
            abspath.append(os.path.dirname(arr[i]))
        abspath = np.asarray(abspath)
        abspath = np.unique(abspath)
        if len(abspath) > 1:
            raise Exception(
                'All the folders selected should be at the same location')

        return abspath
    def getAbsPath():
        arr_1, arr_2 = files_behavior.value, files_photometry.value
        #arr_1 = files_behavior.value
        if len(arr_1) == 0 and len(arr_2) == 0:
            raise Exception('No folder is selected for analysis')

        abspath = []
        if len(arr_1) > 0:
            abspath = checkSameLocation(arr_1, abspath)
        else:
            abspath = checkSameLocation(arr_2, abspath)

        abspath = np.unique(abspath)
        if len(abspath) > 1:
            raise Exception(
                'All the folders selected should be at the same location')
        return abspath
    def getInputParameters():
        abspath = getAbsPath()
        inputParameters = {
            "abspath": abspath[0],
            "folderNames": files_behavior.value,
            "photoFolderNames": files_photometry.value,
            "rawMedName": medPC_rawDataInput.value,
            'h5_check': h5_checkbox.value,
            # 'medExtractCheck': medPC_checkbox.value,
            # "behaviorParamSelect": tabs.active,
            # 'regressionCheck': regression_checkboxes.value,
            # 'regressionTrials': regression_trialNumbs.value,
            # # 'keyDFCheck' : keyDF_checkbox.value,
            # # 'transitionCheck': transition_checkbox.value,
            # 'trialsforTransCheck': transition_trials.value,
            # 'logFitCheck': transition_logfit.value,
            # 'thirdsCheck': transition_thirds.value,
            # # 'WSLSCheck' : WSLS_checkbox.value,
            # 'WSLSPlot': WSLS_plot.value,
            # # 'photoFunctions': photo_checkboxes.value,
            # 'photo_fig_specs': {
            #     'event_sel': event_sel.value,
            #     'loc_sel': loc_sel.value,
            #     # 'movement_sel': movement_sel.value,
            #     'split_rows': split_rows.value,
            #     'split_cols': split_cols.value,
            #     'split_sel': split_sel.value,
            #     'days_to_trim': days_to_exclude_sel.value}

        }
        return inputParameters

    # def onclickLatency(event=None):
    #     inputParameters = getInputParameters()
    #     blockLenAnalysis(inputParameters)

    def onclickPhotoDF(event=None):
        inputParameters = getInputParameters()
        createPhotoDF(inputParameters)

    # def onclickRunRI60(event=None):
    #     inputParameters = getInputParameters()
    #     if inputParameters['trialsforTransCheck'] > 0:
    #         create_analysis_dataframe(inputParameters)
    #     if inputParameters['WSLSPlot'] != 'None':
    #         WS_LS_plot(inputParameters)
    #     if any(regression_checkboxes.value) == True:
    #         runRegressionRI60(inputParameters)
    #     plt.show()

    # def onclickPhotoFx(event=None):
    #     inputParameters = getInputParameters()
    #     if inputParameters['photo_fig_specs']['event_sel'] == 'None':
    #         raise (Exception('No functions selected!! What the heck!'))
    #         return
    #     analyzePhotoDF(inputParameters)

    # photoDFCreate_button.on_click(onclickPhotoDF)
    # runBeh_fx_button.on_click(onclickRunRI60)
    # photoFunctions_button.on_click(onclickPhotoFx)
    # latency_button.on_click(onclickLatency)

    photoDFCreate_button.on_click(onclickPhotoDF)

    # right_top_box = pn.WidgetBox(
    #     explain_path_structure, transition_markdown, regression_markdown, width=450)

    
    # , psth_baseline_param))

    individual_analysis_wd_2 = pn.Column(run_markdown,runBeh_fx_button)
    psth_baseline_param = pn.Column()

    widget_RI60 = pn.Column(mark_down_1, files_behavior,
                            pn.Row(individual_analysis_wd_2, psth_baseline_param))
    
    photoandbehAnalysis = pn.Column(
            photoDFCreate_markdown, photoDFCreate_button)
    widget_RI60_photo = pn.Column(files_photometry,
                                  pn.Row(photoandbehAnalysis))

    # file_selector = pn.WidgetBox(files_1)
    behaviorOnly_RI60 = pn.Card(widget_RI60, title='Behavior-Only Analysis', width=1000)
    behaviorAndPhoto_RI60 = pn.Card(widget_RI60_photo, title='Photometry-Behavior Linking (select behavior and photo folders)', width=1000)

    return pn.Column(behaviorOnly_RI60, behaviorAndPhoto_RI60)