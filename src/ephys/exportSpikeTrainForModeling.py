import pandas as pd
import json
import numpy as np
import os
from _config import INTRINSIC_DIR

os.chdir(INTRINSIC_DIR)

# 1. Load the raw experimental data
# Make sure the filename matches your actual CSV file!
df = pd.read_csv('_clusters_spike train properties.csv')

# 2. Filter out rows where the cluster is missing
df = df.dropna(subset=['cluster_y'])

# Ensure current and cluster are integers for clean dictionary keys
df['current injected'] = df['current injected'].astype(int)
df['cluster_y'] = df['cluster_y'].astype(int)

# 3. Group by Cluster and Current Injection, then calculate the Mean
# We use nanmean to ignore missing values (e.g. if a cell didn't fire enough spikes to have an ISI_CV)
grouped = df.groupby(['cluster_y', 'current injected']).agg({
    'adapt': lambda x: np.nanmean(x),
    'latency': lambda x: np.nanmean(x),
    'isi_cv': lambda x: np.nanmean(x),
    'mean_isi': lambda x: np.nanmean(x)
}).reset_index()

# 4. Restructure the data into a nested dictionary
# Format: { cluster_id: { feature_name: { current_inj: mean_value } } }
target_dict = {1: {}, 2: {}, 3: {}}

for cluster in [1, 2, 3]:
    cluster_data = grouped[grouped['cluster_y'] == cluster]
    
    # Create empty dictionaries for this cluster
    target_dict[cluster] = {
        'adapt': {},
        'latency': {},
        'isi_cv': {},
        'mean_isi': {}
    }
    
    for _, row in cluster_data.iterrows():
        amp = int(row['current injected'])
        
        # Only add to dictionary if the value is not NaN
        if not pd.isna(row['adapt']):
            target_dict[cluster]['adapt'][amp] = float(row['adapt'])
            
        if not pd.isna(row['latency']):
            target_dict[cluster]['latency'][amp] = float(row['latency'])
            
        if not pd.isna(row['isi_cv']):
            target_dict[cluster]['isi_cv'][amp] = float(row['isi_cv'])

        if not pd.isna(row['mean_isi']):
            target_dict[cluster]['mean_isi'][amp] = float(row['mean_isi'])

# 5. Save the clean data out to a JSON file
output_file = 'biological_targets.json'
with open(output_file, 'w') as f:
    json.dump(target_dict, f, indent=4)

print(f"Successfully processed data and saved targets to {output_file}!")