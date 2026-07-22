# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
# from ephysAnalysisAnalyze import excludeCells
import pickle
import os
import copy
from sklearn.cluster import KMeans
from statsmodels.stats.multitest import multipletests
from _config import INTRINSIC_DIR


def excludeCells(mouseList, excludeThreshold):
    newList = {}
    for key, mouse in mouseList.items():
        # if mouse.use == False:
        #     print(f'skipping {mouse.name} because no use')
        #     continue
        newMouse = copy.deepcopy(mouse)
        newMouse.cells = []

        for cell in mouse.cells:
            if hasattr(cell, 'accessResistance'):
                accessResistanceDict = cell.accessResistance
                accessResistance = np.mean(list(accessResistanceDict.values()))
                if accessResistance > excludeThreshold:
                    print('excluding ' + cell.name)
                    continue
                else:
                    newMouse.addCell(cell)
            else:
                newMouse.addCell(cell)
        newList[key] = newMouse

    return newList


analysis_path = INTRINSIC_DIR


os.chdir(analysis_path)
# os.chdir(r'C:\Users\jan7154\Documents\GitHub\EphysAnalysis')
with open('aCUS_intrinsicData_collective.pkl', "rb") as file:
    ephysMouseDict = pickle.load(file)

ephysMouseDict = excludeCells(ephysMouseDict, 20)
fileDict = {'inputRes': 'input_resistance.csv',
            'firing rate': 'baseline Firing Rate.csv',
            'rheobase current': 'baseline rheobase.csv',
            'resting pot': 'baseline RMP.csv',
            'trainProps' : 'spike train properties.csv',
            'capacitance' : 'capacitance.csv',
            'spikeProps' : 'single spike properties.csv',
            'sag' : 'sag.csv',
            'ih' : 'Ih.csv',  }


# ---------------------------------------------------------------------------
# Cluster hierarchical mixed-effects modeling
# ---------------------------------------------------------------------------
# Runs the *exact same* hierarchical mixed-effects process used elsewhere
# (ephysAnalysisAnalyze.py: runModel / runCurrentModel / runModel_allcells),
# substituting the `cluster` dimension for `projID`. Two analyses per feature:
#   1. cluster x group  -- one model across all clusters with the group factor
#                          and its interaction (mirrors runModel_allcells for
#                          scalar features, adds cluster to runCurrentModel for
#                          per-current features).
#   2. per-cluster group -- within each cluster, the effect of group only
#                          (mirrors runModel / runCurrentModel).
# Summaries are written to the cluster_models/ folder.
# ---------------------------------------------------------------------------

CLUSTER_MODELS_DIR = os.path.join(analysis_path, 'cluster_models')
MAD_THRESHOLD = 3.5  # modified z-score cutoff for remove_outliers_robust

# Value-column -> model-canonical variable name (formula-safe identifiers),
# matching the variable names used in ephysAnalysisAnalyze.py. Columns not
# listed here are used verbatim (already valid identifiers).
VALUE_TO_VAR = {'inputRes': 'IR',
                'rheobase current': 'rheobase',
                'resting pot': 'RMP',
                'firing rate': 'firingRate'}

# Per-feature spec, keyed by the fileDict prop name. 'kind' selects the
# modeling process; 'current' names the per-current column to canonicalize;
# 'outliers'/'pre_dropna' mirror the per-feature behavior of run_models;
# 'min_current' mirrors the trainProps current floor used elsewhere.
CLUSTER_FEATURE_SPEC = {
    'inputRes':         {'cols': ['inputRes'],          'kind': 'scalar',  'outliers': True},
    'rheobase current': {'cols': ['rheobase current'],  'kind': 'scalar',  'outliers': True, 'pre_dropna': True},
    'resting pot':      {'cols': ['resting pot'],       'kind': 'scalar',  'outliers': True},
    'capacitance':      {'cols': ['capacitance'],       'kind': 'scalar',  'outliers': True},
    'spikeProps':       {'cols': ['threshold_v', 'upstroke', 'peak_v', 'trough_v',
                                  'upstroke_v', 'downstroke', 'downstroke_v', 'width'],
                         'kind': 'scalar',  'outliers': False},
    'firing rate':      {'cols': ['firing rate'],       'kind': 'current', 'current': 'current'},
    'sag':              {'cols': ['sag'],               'kind': 'current', 'current': 'current'},
    'ih':               {'cols': ['Ih'],                'kind': 'scalar',  'outliers': True},
    'trainProps':       {'cols': ['adapt', 'latency', 'isi_cv', 'mean_isi',
                                  'median_isi', 'first_isi'],
                         'kind': 'current', 'current': 'current injected', 'min_current': 40},
}


def remove_outliers_robust(df, value_col, group_col, ID_col, threshold=MAD_THRESHOLD):
    """MAD-based per-group outlier removal. Copied verbatim from
    ephysAnalysisAnalyze.py so the cluster models clean data identically."""
    removed_outliers = []

    def detect_outliers(group):
        if len(group) < 3:
            return group
        median = np.median(group[value_col])
        abs_deviation = np.abs(group[value_col] - median)
        mad = np.median(abs_deviation)
        modified_z_score = 0.6745 * abs_deviation / (mad if mad != 0 else 1)
        outlier_mask = modified_z_score > threshold
        removed_outliers.append(group[outlier_mask][[group_col, value_col, ID_col]])
        return group[~outlier_mask]

    cleaned_df = df.groupby(group_col, group_keys=False).apply(detect_outliers)
    outliers_df = pd.concat(removed_outliers, ignore_index=True) if removed_outliers else pd.DataFrame()
    return cleaned_df, outliers_df


def _sig_stars(p_value):
    if p_value < 0.0001:
        return '****'
    elif p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'


def _fit_scalar_group(modelDF, variable):
    """Per-cluster effect of group: `variable ~ C(Stress)`. Mirrors runModel."""
    modelDF = modelDF.dropna(subset=[variable])
    if modelDF.empty:
        return None
    formula = f"{variable} ~ C(Stress)"
    vc = {"cellID": "0 + cellID:mouseID"}
    model = sm.MixedLM.from_formula(
        formula, data=modelDF, vc_formula=vc, groups=modelDF["mouseID"])
    try:
        return model.fit()
    except Exception as e:
        print(f"  [{variable}] per-cluster group model failed: {e}")
        return None


def _attach_joint_ftests(result, tests):
    """Attach joint F-tests to a fitted result. `tests` maps an attribute name
    to a predicate over exog term names. Mirrors the joint-test pattern in
    runCurrentModel."""
    names = result.model.exog_names
    nparam = len(result.params)

    def joint(pred):
        idx = [i for i, nm in enumerate(names) if pred(nm)]
        if not idx:
            return None
        C = np.zeros((len(idx), nparam))
        for r, c in enumerate(idx):
            C[r, c] = 1
        return result.f_test(C)

    for attr, pred in tests.items():
        setattr(result, attr, joint(pred))
    return result


def _fit_current_group(modelDF, variable):
    """Per-cluster effect of group for per-current features:
    `variable ~ C(Stress, Sum) * C(currentInjected, Sum)`. Mirrors
    runCurrentModel."""
    modelDF = modelDF.dropna(subset=[variable]).copy()
    if modelDF.empty:
        return None
    formula = f"{variable} ~ C(Stress, Sum) * C(currentInjected, Sum)"
    vc = {"cellID": "0 + C(cellID)"}
    model = sm.MixedLM.from_formula(
        formula, data=modelDF, vc_formula=vc, groups=modelDF["mouseID"])
    try:
        result = model.fit()
    except Exception as e:
        print(f"  [{variable}] per-cluster current model failed: {e}")
        return None
    return _attach_joint_ftests(result, {
        '_stress_main':  lambda nm: 'Stress' in nm and 'currentInjected' not in nm,
        '_current_main': lambda nm: 'currentInjected' in nm and 'Stress' not in nm,
        '_interaction':  lambda nm: 'Stress' in nm and 'currentInjected' in nm,
    })


def _fit_current_clusterXgroup(modelDF, variable):
    """cluster x group for per-current features:
    `variable ~ C(Stress, Sum) * C(cluster) * C(currentInjected, Sum)`.
    Extends runCurrentModel with the cluster dimension."""
    modelDF = modelDF.dropna(subset=[variable]).copy()
    if modelDF.empty:
        return None
    formula = f"{variable} ~ C(Stress, Sum) * C(cluster) * C(currentInjected, Sum)"
    vc = {"cellID": "0 + C(cellID)"}
    model = sm.MixedLM.from_formula(
        formula, data=modelDF, vc_formula=vc, groups=modelDF["mouseID"])
    try:
        result = model.fit()
    except Exception as e:
        print(f"  [{variable}] cluster x group current model failed: {e}")
        return None
    return _attach_joint_ftests(result, {
        '_cluster_main':    lambda nm: 'cluster' in nm and 'Stress' not in nm and 'currentInjected' not in nm,
        '_stress_main':     lambda nm: 'Stress' in nm and 'cluster' not in nm and 'currentInjected' not in nm,
        '_current_main':    lambda nm: 'currentInjected' in nm and 'Stress' not in nm and 'cluster' not in nm,
        '_stress_x_cluster': lambda nm: 'Stress' in nm and 'cluster' in nm and 'currentInjected' not in nm,
    })


def _fit_scalar_clusterXgroup(modelDF, variable):
    """cluster x group for scalar features: `variable ~ C(Stress) * C(cluster)`
    plus Sidak-corrected post-hoc pairwise cluster comparisons. Mirrors
    runModel_allcells, with cluster substituted for projID and the pairwise
    contrasts generated dynamically for whatever clusters are present."""
    modelDF = modelDF.dropna(subset=[variable])
    if modelDF.empty:
        print(f"DataFrame empty for '{variable}'. Skipping cluster x group model.")
        return None

    formula = f"{variable} ~ C(Stress) * C(cluster)"
    vc = {"cellID": "0 + cellID:mouseID"}
    model = sm.MixedLM.from_formula(
        formula, data=modelDF, vc_formula=vc, groups=modelDF["mouseID"])
    try:
        result = model.fit()
    except Exception as e:
        print(f"Model for '{variable}' failed to converge. Error: {e}")
        return None

    fe_names = result.model.exog_names
    coef_idx = {name: i for i, name in enumerate(fe_names)}
    num_coef = len(fe_names)

    # For each non-reference cluster level, build the "level vs reference"
    # contrast, averaged over the two group levels (=> 0.5 weight on the
    # group:cluster interaction term), exactly as runModel_allcells does.
    level_vec = {}
    for nm in fe_names:
        if nm.startswith('C(cluster)[T.') and ':' not in nm:
            level = nm[len('C(cluster)[T.'):-1]
            v = np.zeros(num_coef)
            v[coef_idx[nm]] = 1.0
            for nm2 in fe_names:
                if ':' in nm2 and f'C(cluster)[T.{level}]' in nm2 and 'Stress' in nm2:
                    v[coef_idx[nm2]] = 0.5
            level_vec[level] = v

    levels = sorted(modelDF['cluster'].unique(), key=lambda x: int(x))
    ref = next((lv for lv in levels if lv not in level_vec), None)

    contrasts = {}
    if ref is not None:
        for level, v in level_vec.items():
            contrasts[f'cluster {level} vs. {ref}'] = v
    nonref = list(level_vec.keys())
    for i in range(len(nonref)):
        for j in range(i + 1, len(nonref)):
            a, b = nonref[i], nonref[j]
            contrasts[f'cluster {a} vs. {b}'] = level_vec[a] - level_vec[b]

    model_summary_str = result.summary().as_text()

    if contrasts:
        test_results = {nm: result.t_test(vec.reshape(1, -1))
                        for nm, vec in contrasts.items()}
        p_uncorrected = [r.pvalue for r in test_results.values()]
        t_values = [r.tvalue for r in test_results.values()]
        _, pvals_corrected, _, _ = multipletests(
            np.ravel(p_uncorrected), alpha=0.05, method='sidak')
        significance = [_sig_stars(p) for p in pvals_corrected]
        posthoc_results = pd.DataFrame({
            'Comparison': list(test_results.keys()),
            'T-value': np.ravel(t_values),
            'P-value': np.ravel(p_uncorrected),
            'Adjusted P-value (Sidak)': pvals_corrected,
            'Significance': significance,
        })
        posthoc_summary_str = posthoc_results.to_string(
            index=False,
            formatters={
                'T-value': '{:,.4f}'.format,
                'P-value': '{:,.6f}'.format,
                'Adjusted P-value (Sidak)': '{:,.6f}'.format,
            })
    else:
        posthoc_summary_str = "(only one cluster present; no pairwise comparisons)"

    return (
        f"--- Hierarchical Mixed-Effects Model Summary ---\n"
        f"Variable: {variable}\n"
        f"{model_summary_str}\n\n\n"
        f"--- Post-Hoc Pairwise Comparisons for Main Effect of cluster ---\n"
        f"(Sidak correction for multiple comparisons)\n"
        f"{posthoc_summary_str}\n"
    )


def _write_current_result(result, path, header):
    """Write a fitted per-current result and its attached joint F-tests.
    Tags absent from the result (None) are skipped."""
    os.makedirs(CLUSTER_MODELS_DIR, exist_ok=True)
    with open(path, 'w') as f:
        f.write(header + '\n')
        if result is not None and hasattr(result, 'summary'):
            f.write(result.summary().as_text())
            for tag, attr in [('main effect of cluster',  '_cluster_main'),
                              ('main effect of Stress',   '_stress_main'),
                              ('main effect of current',  '_current_main'),
                              ('Stress x cluster',        '_stress_x_cluster'),
                              ('Stress x current',        '_interaction')]:
                t = getattr(result, attr, None)
                if t is not None:
                    f.write(f"\n\n{tag}:\n{t}")
        else:
            f.write("model did not converge / no data\n")


def runClusterModels(prop, prop_df_clusters):
    """For one feature's clustered DataFrame, run and save (1) the cluster x
    group model and (2) the per-cluster effect-of-group models, mirroring the
    hierarchical mixed-effects process used elsewhere. Called wherever the
    `_clusters_` CSVs are generated."""
    spec = CLUSTER_FEATURE_SPEC.get(prop)
    if spec is None:
        return
    os.makedirs(CLUSTER_MODELS_DIR, exist_ok=True)

    df = prop_df_clusters.copy()

    # Resolve the cluster column. A merge onto a frame that already carried a
    # 'cluster' column (e.g. spike train properties) produces cluster_x (old)
    # and cluster_y (the freshly merged, authoritative one).
    ccol = ('cluster' if 'cluster' in df.columns
            else 'cluster_y' if 'cluster_y' in df.columns else None)
    if ccol is None:
        print(f'[cluster models] {prop}: no cluster column, skipping')
        return
    if ccol != 'cluster':
        df = df.rename(columns={ccol: 'cluster'})

    df = df.dropna(subset=['cluster'])
    if df.empty:
        print(f'[cluster models] {prop}: no clustered rows, skipping')
        return
    df['cluster'] = df['cluster'].astype(float).astype(int).astype(str)

    df = df.rename(columns={'cell name': 'cellID', 'mouse': 'mouseID',
                            'stress': 'Stress', 'sex': 'Sex', 'proj': 'projID'})
    if 'sweep' in df.columns:
        df = df.drop(columns=['sweep'])

    # Base stats models run on baseline data only (matches run_models).
    if 'drug state' in df.columns:
        df = df[df['drug state'] == 'baseline']

    kind = spec['kind']
    if kind == 'current':
        df = df.rename(columns={spec['current']: 'currentInjected'})
        if spec.get('min_current') is not None:
            df = df[df['currentInjected'] >= spec['min_current']]

    clusters = sorted(df['cluster'].unique(), key=lambda x: int(x))

    for col in spec['cols']:
        var = VALUE_TO_VAR.get(col, col)
        work = df.rename(columns={col: var}) if var != col else df.copy()

        # ---- (1) cluster x group ----
        xpath = os.path.join(CLUSTER_MODELS_DIR, f'clusterXgroup_{var}_summary.txt')
        if kind == 'scalar':
            out = _fit_scalar_clusterXgroup(work, var)
            with open(xpath, 'w') as f:
                f.write(out if out is not None else f'cluster x group model failed for {var}\n')
        else:
            res = _fit_current_clusterXgroup(work, var)
            _write_current_result(res, xpath, f'cluster x group (with current) for {var}')

        # ---- (2) per-cluster effect of group ----
        for clust in clusters:
            sub = work[work['cluster'] == clust]
            cpath = os.path.join(
                CLUSTER_MODELS_DIR, f'cluster_{clust}_{var}_group_summary.txt')
            if kind == 'scalar':
                if spec.get('pre_dropna'):
                    sub = sub.dropna(subset=[var])
                outliers = pd.DataFrame()
                if spec.get('outliers'):
                    sub, outliers = remove_outliers_robust(sub, var, 'Stress', 'cellID')
                res = _fit_scalar_group(sub, var)
                with open(cpath, 'w') as f:
                    f.write(f'cluster {clust}: effect of group for {var}\n')
                    if res is not None and hasattr(res, 'summary'):
                        f.write(res.summary().as_text())
                    else:
                        f.write('model did not converge / no data\n')
                    if spec.get('outliers'):
                        f.write(f"\n\nRemoved outliers (MAD, threshold={MAD_THRESHOLD}):\n")
                        f.write("None.\n" if outliers.empty
                                else outliers.to_string(index=False) + "\n")
            else:
                res = _fit_current_group(sub, var)
                _write_current_result(
                    res, cpath, f'cluster {clust}: effect of group (with current) for {var}')

    print(f'[cluster models] wrote {prop} models -> {CLUSTER_MODELS_DIR}')


def prepForClustering(ephysMouseDict):
    df_all = pd.DataFrame()

    for name, mouse in ephysMouseDict.items():
        for cell in ephysMouseDict[name].cells:
            if hasattr(cell, 'firingRateData') == False:
                continue
            # get cell info into dict
            cellInfo = {'cell name': cell.name,
                        'mouse': mouse.name,
                        'sex': mouse.sex,
                        'stress': mouse.stressCon,
                        'proj': mouse.proj,
                        'RMP': np.mean(list(cell.RMPData.values())) if hasattr(cell, 'RMPData') else np.nan,
                        'IR': np.mean(list(cell.inputResistance.values())) if hasattr(cell, 'inputResistance') else np.nan
                        }
            df_cellInfo = pd.DataFrame(cellInfo, index=[0])

            # get firing rates into their own column
            FRData = cell.firingRateData.copy()
            if 'sweep' in FRData.keys():
                del FRData['sweep']

            df_FR = pd.DataFrame(FRData)
            df_FR = df_FR.groupby('current amp').mean().reset_index()
            df_FR1 = df_FR.set_index('current amp').T
            df_FR1.columns = [f"FR_{col}" for col in df_FR1.columns]
            df_FR1 = df_FR1.drop(['FR_0'], axis=1)

            # now get the adaptation and latency but don't care about current so just do means!
            spikeTrainData = cell.spikeAnalysis.copy()
            adaptation = np.mean(
                spikeTrainData[spikeTrainData['protocol'] == 'firing rate']['adapt'])
            latency = np.mean(
                spikeTrainData[spikeTrainData['protocol'] == 'firing rate']['latency'])
            isi_cv = np.mean(
                spikeTrainData[spikeTrainData['protocol'] == 'firing rate']['isi_cv'])
            mean_isi = np.mean(
                spikeTrainData[spikeTrainData['protocol'] == 'firing rate']['mean_isi'])

            trainDict = {'adaptation': adaptation,
                         'latency': latency,
                         'isi_cv': isi_cv,
                         'mean_isi': mean_isi}

            df_train = pd.DataFrame(trainDict, index=[0])

            df_cellInfo_final = pd.concat([df_cellInfo.reset_index(
                drop=True), df_train.reset_index(drop=True), df_FR1.reset_index(drop=True)], axis=1)

            df_all = pd.concat([df_all, df_cellInfo_final], axis=0)
    proj = 'SNL'
    df = df_all[df_all['proj'] == proj]
    df = df.drop(['proj'], axis=1)
    df = df.dropna()

    df.to_csv('clusterPreppedBoi.csv')


# %%
# * come back after clustering in R
def addToCSVs(fileDict):
    dataframes = {}
    clustered_cells = pd.read_csv('clustered_cells.csv')
    for prop, fileName in fileDict.items():
        prop_df = pd.read_csv(fileName)
        prop_df = prop_df[prop_df['proj'] == 'SNL']
        prop_df_clusters = pd.merge(
            prop_df,
            clustered_cells[['cell.name', 'cluster']],
            left_on='cell name',
            right_on='cell.name',
            how='left'
        ).drop(columns=['cell.name'])
        prop_df_clusters.to_csv(f'_clusters_{fileName}')

        # Per-cell means across all current injections (mean adaptation,
        # latency, isi_cv, mean_isi, ...) with the cluster label attached.
        if prop == 'trainProps':
            id_cols = ['cell name', 'mouse', 'sex', 'stress', 'proj']
            mean_cols = ['adapt', 'latency', 'isi_cv', 'mean_isi',
                         'median_isi', 'first_isi']
            train_means = (
                prop_df_clusters
                .groupby(id_cols, as_index=False)
                .agg({**{c: 'mean' for c in mean_cols}, 'cluster': 'first'}))
            train_means.to_csv(
                '_clusters_spike train means per cell.csv', index=False)

        # Also run the hierarchical mixed-effects models (cluster x group, then
        # per-cluster effect of group) the same way as elsewhere, saving to the
        # cluster_models/ folder.
        runClusterModels(prop, prop_df_clusters)

    # Add cluster as attribute to cell objects
    for _, row in clustered_cells.iterrows():
        cell_name = row['cell.name']
        cluster = row['cluster']
        for mouse in ephysMouseDict.values():
            for cell in mouse.cells:
                if cell.name == cell_name:
                    cell.cluster = cluster
                    break
    
    # Save updated ephysMouseDict
    with open('aCUS_intrinsicData_collective.pkl', 'wb') as file:
        pickle.dump(ephysMouseDict, file)

    print('Updated ephysMouseDict saved to aCUS_intrinsicData_collective.pkl')


def modelClusters(fileDict):
    # now run the model
    clustered_cells = pd.read_csv('clustered_cells.csv')

    for prop, fileName in fileDict.items():
        prop_df = pd.read_csv(fileName)
        prop_df = prop_df[prop_df['proj'] == 'SNL']
        prop_df_clusters = pd.merge(
            prop_df,
            clustered_cells[['cell.name', 'cluster']],
            left_on='cell name',
            right_on='cell.name',
            how='left'
        ).drop(columns=['cell.name'])
        prop_df_clusters = prop_df_clusters.dropna(subset=['cluster'])

        prop_df_clusters['cluster'] = prop_df_clusters['cluster'].astype(str)

        if prop == 'firing rate':
            for clust in range(1, 4):
                # continue
                
                renamer = {'firing rate' : 'firingRate',
                        'cell name' : 'cellID'}
                
                        
                
                formula = "firingRate ~ C(stress) + current + C(stress):current"
                vc = {"cellID": "0 + cellID:mouse"}
                modelDF = prop_df_clusters.dropna(subset=[prop])
                
                modelDF = modelDF.rename(
                    columns=renamer)
                # Fit the mixed-effects model with random effects (mice and repeated measure)
                modelDF = modelDF[modelDF['cluster'].astype(float) == clust]
                model = sm.MixedLM.from_formula(
                    formula, data=modelDF, vc_formula=vc, groups=modelDF["mouse"])
                result = model.fit()
                with open(f'firingRate_cluster_{clust}_summary.txt', 'w') as f:
                    f.write(f'for firingRate cluster {clust}\n')
                    f.write(str(result.summary()))

        elif prop == 'spikeProps' or prop == 'trainProps':
            columns = prop_df_clusters.columns
            cols_to_exclude = ['Unnamed: 0', 'cell name', 'mouse', 'stress', 'cluster', 'sex', 'proj', 'current injected']
            cols_to_include = [col for col in columns if col not in cols_to_exclude]

            for col in cols_to_include:
                a=1
            
                modelDF = prop_df_clusters.dropna(subset=[col, 'cluster'])
                modelDF = modelDF.rename(columns={'cell name': 'cellID'})
                formula = f"{col} ~ C(stress) + C(cluster) + C(stress):C(cluster)"
                vc = {"cellID": "0 + cellID:mouse"}
                model = sm.MixedLM.from_formula(
                    formula, data=modelDF, vc_formula=vc, groups=modelDF["mouse"])

                result = model.fit()
                print('for ' + col)
                print(result.summary())

        else:
            continue
            renamer = {'cell name' : 'cellID'}
            modelDF = prop_df_clusters.dropna(subset=[prop, 'cluster'])
            modelDF = modelDF.rename(
                    columns=renamer)
            if 'rheobase current' in modelDF.columns:
                renamer = {'rheobase current' : 'rheobase'}
                modelDF = modelDF.rename(columns=renamer)
                prop = 'rheobase'
            if 'resting pot' in modelDF.columns:
                renamer = {'resting pot' : 'RMP'}
                modelDF = modelDF.rename(columns=renamer)
                prop = 'RMP'

            formula = f"{prop} ~ C(stress)  + C(cluster) + C(stress) * C(cluster)"
            vc = {"cellID": "0 + cellID:mouse"}
            model = sm.MixedLM.from_formula(
                formula, data=modelDF, vc_formula=vc, groups=modelDF["mouse"])
            result = model.fit()
            print('for ' + prop)
            print(result.summary())

        a = 1
# %%
# prepForClustering(ephysMouseDict)
addToCSVs(fileDict)
# modelClusters(fileDict)