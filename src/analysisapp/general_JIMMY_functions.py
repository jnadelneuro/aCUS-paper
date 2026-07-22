import os
import numpy as np
import pandas as pd
import pickle
import pyarrow.feather as feather
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import scipy
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
import tkinter as tk
import json
# import itertools
import h5py
from readMEDPC_ASAP_Jan23 import *
from tkinter import filedialog
from tkinter import simpledialog

#! this is where you change the default path values now
def get_paths(defaultASAP=None, defaultRI60=None):
    
    
    root = tk.Tk()
    root.title("Enter Paths")

    paths = {}

    # Function to set the path based on user input
    def set_path(key, entry_widget):
        paths[key] = entry_widget.get()

    # Create Entry widgets for each path
    keys = ['behavior_ASAP', 'photometry_ASAP', 'behavior_RI60', 'photometry_RI60']
    entries = {}

    for key in keys:
        
        entry = tk.Entry(root, width=200)
        if ('ASAP' in key) & (defaultASAP != None):
            entry.insert(0, defaultASAP)
        elif ('RI60' in key) & (defaultRI60 != None):
            entry.insert(0, defaultRI60)
        else:
            entry.insert(0, '~')
        label = tk.Label(root, text=f"Enter {key.replace('_', ' ')} Path:")
        label.pack()
        entry.pack()
        entries[key] = entry

    # Button to submit the paths
    submit_button = tk.Button(root, text="Submit", command=lambda: root.quit() if all(
        entry.get() != '' for entry in entries.values()) else None)
    submit_button.pack()

    # Start the Tkinter main loop
    root.mainloop()

    # Collect paths after the Tkinter window is closed
    for key in keys:
        paths[key] = entries[key].get()
    root.destroy()

    return paths

class CheckboxDialog(simpledialog.Dialog):
    def __init__(self, parent, title, choices):
        self.choices = choices
        self.user_columns = []
        super().__init__(parent, title)

    def body(self, master):
        self.vars = [tk.IntVar() for _ in self.choices]
        for i, choice in enumerate(self.choices):
            tk.Checkbutton(master, text=choice, variable=self.vars[i]).pack(anchor="w")

        tk.Label(master, text="Enter custom columns (comma-separated):").pack(anchor="w")
        self.entry_var = tk.StringVar()
        entry = tk.Entry(master, textvariable=self.entry_var)
        entry.pack(anchor="w")

        return entry  # Focus on the entry widget

    def apply(self):
        self.user_columns = [col.strip() for col in self.entry_var.get().split(',')]
        selected_columns = [choice for i, choice in enumerate(self.choices) if self.vars[i].get()]
        self.result = selected_columns + self.user_columns

def getParamsForInfoDict(inputParameters):
    # Get existing columns
    common_columns = ["Mouse", "Sex", "Group", "Implant", "Signal Locations", "Expression Location", "Pellet"]

    # Create a custom checkbox dialog
    dialog = CheckboxDialog(None, "Select Columns", common_columns)
    user_columns = dialog.result

    # If user clicked Cancel or closed the dialog, return an empty list
    if user_columns is None:
        return []

    # If user selected columns, combine them with existing columns
    columns = user_columns

    return columns
# Example usage

def createMouseDict(inputParameters):
    if inputParameters['behaviorParamSelect'] == 0: # if it's ASAP
        folderSelected = inputParameters['folderNames'][0]
        output_file_folder = os.path.join(folderSelected, 'output_datafiles')
        if not os.path.exists(output_file_folder):
            raise Exception('There is no output_datafiles folder in the selected folder!')

        os.chdir(output_file_folder)

        root = tk.Tk()
        try:
            with open("mouse_info.json", "r") as f:
                data = json.load(f)
            columnsOG = list(list(data.items())[0][1].keys())
            lowercase_columns = [column.lower() for column in columnsOG]


            if 'mouse' not in lowercase_columns:
                columns = ['mouse'] + columnsOG
            else:
                columns = columnsOG


            data_entries = [[] for _ in range(len(columns))]
            column_labels = []

            for mouse, values in data.items():
                # columns = list(values.keys())
                
                for i, col in enumerate(columns):
                    if ('mouse' not in lowercase_columns) & (col == 'mouse'):
                        entry = tk.Entry(root, width=20)
                        entry.insert(0, mouse)
                        entry.grid(row=len(data_entries[i]) + 3, column=i)
                        data_entries[i].append(entry)
                        label = tk.Label(root, text=col)
                        label.grid(row=1, column=i)
                        column_labels.append(label)
                    else:
                        entry = tk.Entry(root, width=20)
                        entry.insert(0, values.get(col, ""))
                        entry.grid(row=len(data_entries[i]) + 3, column=i)
                        data_entries[i].append(entry)
                        label = tk.Label(root, text=col)
                        label.grid(row=1, column=i)
                        column_labels.append(label)

        except FileNotFoundError:
            # Explanation label
            explanation_label = tk.Label(root, text="Fill in the table below with the data.")
            explanation_label.grid(row=0, column=0, columnspan=8)

            columns = getParamsForInfoDict(inputParameters)  # Get columns from user input

            # Create labels for each column dynamically
            column_labels = []
            for i, col in enumerate(columns):
                label = tk.Label(root, text=col)
                label.grid(row=1, column=i)
                column_labels.append(label)

            # Empty lists to store data
            data_entries = [[] for _ in range(len(columns))]

        # Load data from existing JSON file, if it exists
        

        def add_row():
            # Create an entry widget for each column and add it to the table
            for i in range(len(columns)):
                entry = tk.Entry(root, width=20)
                entry.grid(row=len(data_entries[i]) + 3, column=i)
                data_entries[i].append(entry)

        def delete_row():
            # Remove the last row from the table and the data lists
            for entries in data_entries:
                if len(entries) > 0:
                    entries[-1].grid_forget()
                    entries.pop()

        def save_data():
            # Save the data as a JSON file
            data = {}
            for i in range(len(data_entries[0])):
                row_data = {}
                for j, col in enumerate(columns):
                    row_data[col] = data_entries[j][i].get()
                data[data_entries[0][i].get()] = row_data

            with open("mouse_info.json", "w") as f:
                json.dump(data, f, indent=4)

        # Buttons for adding rows, deleting rows, editing columns, adding columns, deleting columns, and saving data
        add_button = tk.Button(root, text="Add Row", command=add_row)
        add_button.grid(row=2, column=len(columns))

        delete_button = tk.Button(root, text="Delete Row", command=delete_row)
        delete_button.grid(row=2, column=len(columns) + 1)

        save_button = tk.Button(root, text="Save as JSON", command=save_data)
        save_button.grid(row=2, column=len(columns) + 5)

        root.mainloop()


    # elif  inputParameters['behaviorParamSelect'] == 1: # if it's RI60
    #     folderSelected = inputParameters['folderNames'][0]
    #     #! check if there's an output_datafiles folder
    #     output_file_folder = os.path.join(folderSelected, 'output_datafiles')
    #     if not os.path.exists(output_file_folder):
    #         raise Exception(
    #             'There is no output_datafiles folder in the selected folder!')

    #     os.chdir(output_file_folder)

    #     # Create a new tkinter window
    #     root = tk.Tk()

    #     # Create a label for explanation
    #     explanation_label = tk.Label(root, text="Fill in the table below with the Mouse, Sex, Group (aCUS or control), drug condition (h or y), sigLoc.\nClick the Add Row button to add a new row to the table.\nClick the Delete Row button to delete the last row from the table.\nClick the Save as JSON button to save the data as a .json file.")
    #     explanation_label.grid(row=0, column=0, columnspan=6)

    #     # Create a table with labels for each column
    #     mouse_label = tk.Label(root, text="Mouse")
    #     mouse_label.grid(row=1, column=0)
    #     sex_label = tk.Label(root, text="Sex")
    #     sex_label.grid(row=1, column=1)
    #     group_label = tk.Label(root, text="Group")
    #     group_label.grid(row=1, column=2)
    #     implant_label = tk.Label(root, text="Drug")
    #     implant_label.grid(row=1, column=3)
    #     sigLoc_label = tk.Label(root, text="Signal Locations")
    #     sigLoc_label.grid(row=1, column=4)
    #     expressionGroup_label = tk.Label(root, text="Expression Location")
    #     expressionGroup_label.grid(row=1, column=5)
    #     pellet_label = tk.Label(root, text="pellet")
    #     pellet_label.grid(row=1, column=6)
        

    #     # Create empty lists to store the data
    #     mouse_data = []
    #     sex_data = []
    #     group_data = []
    #     drug_data = []
    #     sigLoc_data = []
    #     # expGrp_data = []

    #     # Load data from existing JSON file, if it exists
    #     try:
    #         with open("mouse_info.json", "r") as f:
    #             data = json.load(f)
    #             for mouse, values in data.items():
    #                 mouse_entry = tk.Entry(root, width=20)
    #                 mouse_entry.insert(0, mouse)
    #                 mouse_entry.grid(row=len(mouse_data)+3, column=0)
    #                 mouse_data.append(mouse_entry)

    #                 sex_entry = tk.Entry(root, width=20)
    #                 sex_entry.insert(0, values["sex"])
    #                 sex_entry.grid(row=len(sex_data)+3, column=1)
    #                 sex_data.append(sex_entry)

    #                 group_entry = tk.Entry(root, width=20)
    #                 group_entry.insert(0, values["group"])
    #                 group_entry.grid(row=len(group_data)+3, column=2)
    #                 group_data.append(group_entry)

    #                 drug_entry = tk.Entry(root, width=20)
    #                 drug_entry.insert(0, values["drug"])
    #                 drug_entry.grid(row=len(drug_data)+3, column=3)
    #                 drug_data.append(drug_entry)

    #                 sigLoc_entry = tk.Entry(root, width=20)
    #                 sigLoc_entry.insert(0, values["SigLocs"])
    #                 sigLoc_entry.grid(row=len(sigLoc_data)+3, column=4)
    #                 sigLoc_data.append(sigLoc_entry)

    #                 # expGrp_entry = tk.Entry(root, width=20)
    #                 # expGrp_entry.insert(0, values["expGrp"])
    #                 # expGrp_entry.grid(row=len(expGrp_data)+3, column=5)
    #                 # expGrp_data.append(expGrp_entry)
    #     except FileNotFoundError:
    #         pass

    #     def add_row():
    #         # Create an entry widget for each column and add it to the table
    #         mouse_entry = tk.Entry(root, width=20)
    #         mouse_entry.grid(row=len(mouse_data)+3, column=0)
    #         mouse_data.append(mouse_entry)

    #         sex_entry = tk.Entry(root, width=20)
    #         sex_entry.grid(row=len(sex_data)+3, column=1)
    #         sex_data.append(sex_entry)

    #         group_entry = tk.Entry(root, width=20)
    #         group_entry.grid(row=len(group_data)+3, column=2)
    #         group_data.append(group_entry)

    #         drug_entry = tk.Entry(root, width=20)
    #         drug_entry.grid(row=len(drug_data)+3, column=3)
    #         drug_data.append(drug_entry)

    #         sigLoc_entry = tk.Entry(root, width=20)
    #         sigLoc_entry.grid(row=len(sigLoc_data)+3, column=4)
    #         sigLoc_data.append(sigLoc_entry)

    #         # expGrp_entry = tk.Entry(root, width=20)
    #         # expGrp_entry.grid(row=len(expGrp_data)+3, column=4)
    #         # expGrp_data.append(expGrp_entry)

    #     def delete_row():
    #         # Remove the last row from the table and the data lists
    #         if len(mouse_data) > 0:
    #             mouse_data[-1].grid_forget()
    #             mouse_data.pop()

    #             sex_data[-1].grid_forget()
    #             sex_data.pop()

    #             group_data[-1].grid_forget()
    #             group_data.pop()

    #             drug_data[-1].grid_forget()
    #             drug_data.pop()

    #             sigLoc_data[-1].grid_forget()
    #             sigLoc_data.pop()

    #             # expGrp_data[-1].grid_forget()
    #             # expGrp_data.pop()

    #     def save_data():
    #         # Save the data as a JSON file
    #         data = {}
    #         for i in range(len(mouse_data)):
    #             mouse = mouse_data[i].get()
    #             sex = sex_data[i].get()
    #             group = group_data[i].get()
    #             drug = drug_data[i].get()
    #             # expGrp = expGrp_data[i].get()
    #             if ',' in sigLoc_data[i].get():
    #                 sigLoc = sigLoc_data[i].get()
    #                 sigLoc = sigLoc.split(", ")
    #             else:
    #                 sigLoc = [sigLoc_data[i].get()]
    #             data[mouse] = {"sex": sex, "group": group,
    #                         "drug": drug, 'SigLocs': sigLoc}
    #         with open("mouse_info.json", "w") as f:
    #             json.dump(data, f, indent=4)

    #     # Create buttons for adding rows, deleting rows, and saving data
    #     add_button = tk.Button(root, text="Add Row", command=add_row)
    #     add_button.grid(row=2, column=0)

    #     delete_button = tk.Button(root, text="Delete Row", command=delete_row)
    #     delete_button.grid(row=2, column=1)

    #     save_button = tk.Button(root, text="Save as JSON", command=save_data)
    #     save_button.grid(row=2, column=2)

    #     # Run the tkinter event loop
    #     root.mainloop()

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



def make_dir(filepath):
    op = os.path.join(filepath, 'inputParameters')
    if not os.path.exists(op):
        os.mkdir(op)
    return op

