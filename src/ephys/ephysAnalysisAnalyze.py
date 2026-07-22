# %%
import pickle
from unittest import result
import pandas as pd
import statsmodels.api as sm
import os
import numpy as np
import copy
from scipy.interpolate import interp1d
from scipy.stats import t as student_t
import matplotlib.pyplot as plt
from ephysAnalysisSetup import *
import warnings
from skbio.stats.distance import DistanceMatrix, permanova
from skbio.stats.ordination import pcoa
from sklearn.metrics import pairwise_distances 
import itertools
from statsmodels.stats.multitest import multipletests
from getPhasePlotv3 import *
from _config import INTRINSIC_DIR

# analysis_path = r'R:\Basic_Sciences\Phys\Lerner_Lab_tnl2633\Jacob\aCUS\ALL_INTRINSIC_EPHYS\analysis'
# NOTE: retargeted from 'timecourse experiment\analysis' (a different dataset) to the canonical INTRINSIC_DIR
analysis_path = INTRINSIC_DIR
# analysis_path = r'R:\Basic_Sciences\Phys\Lerner_Lab_tnl2633\Jacob\aCUS\ALL_oIPSC_EPHYS\analysis'
# analysis_path = r'R:\Basic_Sciences\Phys\Lerner_Lab_tnl2633\Jacob\aCUS\ALL_M-CURRENT_EPHYS\analysis'

    

os.chdir(analysis_path)
# os.chdir(r'C:\Users\jan7154\Documents\GitHub\EphysAnalysis')

# All statistical-model summary .txt files are written here (created on demand).
MODELS_DIR = os.path.join(analysis_path, 'models')

with open('aCUS_intrinsicData_collective.pkl', "rb") as file:
    ephysMouseDict = pickle.load(file)



# a=1
# os.chdir(r'C:\Users\jacob\OneDrive\Documents\GitHub\EphysAnalysis')
def addClusters(mouseList):
    clusterDF = pd.read_csv("clustered_cells.csv", usecols=["mouse", "cell.name", "cluster"])
    newList = {}
    for key, mouse in mouseList.items():
        # if mouse.use == False:
        #     print(f'skipping {mouse.name} because no use')
        #     continue
        newMouse = copy.deepcopy(mouse)
        newMouse.cells = []

        for cell in mouse.cells:
            try:
                cell.cluster = int(clusterDF[clusterDF['cell.name'] == cell.name]['cluster'])
            except:
                cell.cluster = np.nan
            newMouse.addCell(cell)
        newList[key] = newMouse

    return newList


MAD_THRESHOLD = 3.5  # modified z-score cutoff for remove_outliers_robust
ROUT_Q = 0.001  # ROUT false discovery rate (0.1%), matches the rest of the paper


def remove_outliers_robust(df, value_col, group_col, ID_col, threshold=MAD_THRESHOLD):
    """
    Removes outliers from a DataFrame using Median Absolute Deviation (MAD),
    which is robust to non-normally distributed data.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with multiple columns.
        value_col (str): Column used for outlier detection.
        group_col (str): Column used to group data.
        extra_col (str): Column whose values should be returned with removed outliers.
        threshold (float): MAD threshold for outlier detection (default: 3.5).
    
    Returns:
        tuple: (cleaned_df, outliers_df)
            - cleaned_df: Original DataFrame with outliers removed.
            - outliers_df: DataFrame containing removed outliers and the specified extra column.
    """
    removed_outliers = []  # List to store removed outliers

    def detect_outliers(group):
        """Detects and removes outliers using MAD for each group separately."""
        if len(group) < 3:  # Skip small groups
            return group
        
        median = np.median(group[value_col])
        abs_deviation = np.abs(group[value_col] - median)
        mad = np.median(abs_deviation)  # Median Absolute Deviation

        # Compute modified Z-score based on MAD
        modified_z_score = 0.6745 * abs_deviation / (mad if mad != 0 else 1)

        # Identify outliers
        outlier_mask = modified_z_score > threshold
        
        # Store removed outliers
        removed_outliers.append(group[outlier_mask][[group_col, value_col, ID_col]])
        
        # Return only non-outliers (keep full DataFrame structure)
        return group[~outlier_mask]

    # Apply function to each group separately
    cleaned_df = df.groupby(group_col, group_keys=False).apply(detect_outliers)
    
    # Combine all removed outliers into a single DataFrame
    outliers_df = pd.concat(removed_outliers, ignore_index=True) if removed_outliers else pd.DataFrame()

    return cleaned_df, outliers_df


def detect_outliers_ROUT(df, value_col, group_col, ID_col, Q=ROUT_Q):
    """Prism-style ROUT outlier removal (Motulsky & Brown, BMC Bioinformatics 2006),
    applied per group. Returns (cleaned_df, outliers_df) with the same interface as
    remove_outliers_robust, so it can be swapped in later by changing the call site.

    Method (column-data case, no regression X):
        1. robust center C = median of the group
        2. residuals R_i = x_i - C
        3. RSDR (robust SD of residuals) = percentile_68.27(|R_i|) * N/(N-K), K=1
        4. t_i = |R_i| / RSDR; two-tailed P from Student-t with df = N-K
        5. flag outliers via Benjamini-Hochberg FDR at Q (here 0.1%)

    Caveat: this reproduces the ROUT *FDR outlier test* faithfully, but Prism's internal
    robust fit (Lorentzian-likelihood regression) is not fully published; using the median
    as the center makes results very close to Prism but an occasional borderline point may
    differ. For exact Prism parity, run the same column in Prism.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        value_col (str): Column used for outlier detection.
        group_col (str): Column used to group data.
        ID_col (str): Identifier column returned with removed outliers.
        Q (float): Maximum false discovery rate (default ROUT_Q = 0.001 = 0.1%).

    Returns:
        tuple: (cleaned_df, outliers_df)
    """
    removed_outliers = []  # List to store removed outliers

    def detect_outliers(group):
        """Detects and removes ROUT outliers for each group separately."""
        x = group[value_col].astype(float)
        n = len(x)
        if n < 3:  # FDR is meaningless at tiny n; skip (matches remove_outliers_robust)
            return group

        center = np.median(x)
        abs_resid = np.abs(x - center)

        K = 1  # one parameter (the center) for column data
        rsdr = np.percentile(abs_resid, 68.27) * n / (n - K)
        if rsdr == 0:  # degenerate spread -> nothing to flag
            return group

        t_vals = abs_resid / rsdr
        p_vals = 2 * (1 - student_t.cdf(t_vals, df=n - K))

        # Benjamini-Hochberg FDR: points rejected at level Q are outliers.
        outlier_mask, *_ = multipletests(p_vals, alpha=Q, method='fdr_bh')
        outlier_mask = pd.Series(outlier_mask, index=group.index)

        # Store removed outliers
        removed_outliers.append(group[outlier_mask][[group_col, value_col, ID_col]])

        # Return only non-outliers (keep full DataFrame structure)
        return group[~outlier_mask]

    # Apply function to each group separately
    cleaned_df = df.groupby(group_col, group_keys=False).apply(detect_outliers)

    # Combine all removed outliers into a single DataFrame
    outliers_df = pd.concat(removed_outliers, ignore_index=True) if removed_outliers else pd.DataFrame()

    return cleaned_df, outliers_df

def runModel_allcells(modelDF, variable):
    """
    Fits a hierarchical mixed-effects model and performs post-hoc pairwise
    comparisons for the main effect of 'projID'. The results are combined
    into a single, publication-ready string.

    This version includes formatted p-values (no scientific notation) and a
    significance star column in the post-hoc table.

    Args:
        modelDF (pd.DataFrame): The input DataFrame containing the data.
        variable (str): The name of the dependent variable column.

    Returns:
        str: A formatted string containing both the main model summary and the
             post-hoc pairwise comparison results.
             Returns None if the model fails to converge or the data is empty.
    """
    # --- Helper function for significance stars ---
    def get_significance_stars(p_value):
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

    modelDF = modelDF.dropna(subset=[variable])
    if modelDF.empty:
        print(f"DataFrame is empty after dropping NaNs for variable '{variable}'. Skipping model.")
        return None
        
    formula = f"{variable} ~ C(Stress) * C(projID)"
    vc = {"cellID": "0 + cellID:mouseID"}
    model = sm.MixedLM.from_formula(formula, data=modelDF, vc_formula=vc, groups=modelDF["mouseID"])

    try:
        result = model.fit()
    except Exception as e:
        print(f"Model for '{variable}' failed to converge. Error: {e}")
        return None

    # --- Create Contrast Matrices for Post-Hoc Comparisons ---
    fe_names = result.model.exog_names
    coef_map = {name: i for i, name in enumerate(fe_names)}
    num_coef = len(fe_names)

    contrasts = {}
    try:
        # Define contrasts for main effects
        c1 = np.zeros(num_coef); c1[coef_map['C(projID)[T.PAG]']] = 1; c1[coef_map['C(Stress)[T.stress]:C(projID)[T.PAG]']] = 0.5; contrasts['PAG vs. SNL'] = c1
        c2 = np.zeros(num_coef); c2[coef_map['C(projID)[T.BNST]']] = 1; c2[coef_map['C(Stress)[T.stress]:C(projID)[T.BNST]']] = 0.5; contrasts['BNST vs. SNL'] = c2
        c3 = np.zeros(num_coef); c3[coef_map['C(projID)[T.PAG]']] = 1; c3[coef_map['C(Stress)[T.stress]:C(projID)[T.PAG]']] = 0.5; c3[coef_map['C(projID)[T.BNST]']] = -1; c3[coef_map['C(Stress)[T.stress]:C(projID)[T.BNST]']] = -0.5; contrasts['PAG vs. BNST'] = c3
    except KeyError as e:
        print(f"ERROR: A coefficient name was not found in the model: {e}"); print(f"Available coefficients are: {fe_names}"); return None

    # --- Run t-tests and collect results ---
    test_results = {}
    for name, contrast_vector in contrasts.items():
        contrast_matrix = contrast_vector.reshape(1, -1)
        test_results[name] = result.t_test(contrast_matrix)

    p_values_uncorrected = [res.pvalue for res in test_results.values()]
    t_values = [res.tvalue for res in test_results.values()]

    # --- Adjust P-values and get significance stars ---
    reject, pvals_corrected, _, _ = multipletests(np.ravel(p_values_uncorrected), alpha=0.05, method='sidak')
    significance = [get_significance_stars(p) for p in pvals_corrected]

    # --- Format Results into a DataFrame ---
    posthoc_results = pd.DataFrame({
        'Comparison': list(test_results.keys()),
        'T-value': np.ravel(t_values),
        'P-value': np.ravel(p_values_uncorrected),
        'Adjusted P-value (Sidak)': pvals_corrected,
        'Significance': significance
    })

    # --- Combine all results into a single string for easy saving ---
    model_summary_str = result.summary().as_text()
    
    # Format the DataFrame to string with custom float formatting
    posthoc_summary_str = posthoc_results.to_string(
        index=False,
        formatters={
            'T-value': '{:,.4f}'.format,
            'P-value': '{:,.6f}'.format,
            'Adjusted P-value (Sidak)': '{:,.6f}'.format
        }
    )

    combined_output = (
        f"--- Hierarchical Mixed-Effects Model Summary ---\n"
        f"Variable: {variable}\n"
        f"{model_summary_str}\n\n\n"
        f"--- Post-Hoc Pairwise Comparisons for Main Effect of projID ---\n"
        f"(Šidák correction for multiple comparisons)\n"
        f"{posthoc_summary_str}\n"
    )

    return combined_output

def runModel(modelDF, variable):
    # modelDF_beadOnly = modelDF[modelDF['beadID'] == '+']
    # * and now for the model!
    # formula = "firingRate ~ C(projID) + C(Stress) + currentInjected + C(projID):C(Stress) + C(projID):currentInjected + C(Stress):currentInjected + C(projID):C(Stress):currentInjected"
    modelDF = modelDF.dropna(subset=[variable])
    formula = f"{variable} ~ C(Stress)"
    vc = {"cellID": "0 + cellID:mouseID"}

    # Fit the mixed-effects model with random effects (mice and repeated measure)
    model = sm.MixedLM.from_formula(formula, data=modelDF, vc_formula=vc, groups=modelDF["mouseID"])
    # model = sm.MixedLM.from_formula(formula, data=modelDF, groups=modelDF["mouseID"])

    try:
        result = model.fit()
    except:
        print('model no work')
        result = []

    return result


def runCurrentModel(modelDF, variable):
    modelDF = modelDF.dropna(subset=[variable]).copy()
    formula = f"{variable} ~ C(Stress, Sum) * C(currentInjected, Sum)"
    vc = {"cellID": "0 + C(cellID)"}

    model = sm.MixedLM.from_formula(
        formula, data=modelDF, vc_formula=vc, groups=modelDF["mouseID"])
    try:
        result = model.fit()
    except Exception as e:
        print(f"  [{variable}] model failed: {e}")
        return None

    names = result.model.exog_names
    stress_idx   = [i for i, nm in enumerate(names)
                    if 'Stress' in nm and 'currentInjected' not in nm]   # main
    current_idx  = [i for i, nm in enumerate(names)
                    if 'currentInjected' in nm and 'Stress' not in nm]   # main
    interact_idx = [i for i, nm in enumerate(names)
                    if 'Stress' in nm and 'currentInjected' in nm]       # interaction

    def joint(idx):
        if not idx:
            return None
        C = np.zeros((len(idx), len(result.params)))
        for r, c in enumerate(idx):
            C[r, c] = 1
        return result.f_test(C)

    result._stress_main  = joint(stress_idx)
    result._current_main = joint(current_idx)
    result._interaction  = joint(interact_idx)
    return result


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


# ---------------------------------------------------------------------------
# Shared extraction / modeling helpers
# ---------------------------------------------------------------------------
#
# Every per-feature pipeline does the same thing:
#   walk mouse->cell->cell.<attr>, flatten to long-form rows with metadata,
#   group to cell-level (or cell-and-current level), write a CSV,
#   rename to model-canonical names, run per-projection models + a total model.
#
# The helpers below collapse the repetition while keeping the deliberate
# per-feature differences (filter_positive for IR, skip_ts_in_loop and
# pre_outlier_dropna for rheobase, etc.) as explicit arguments.
# ---------------------------------------------------------------------------

# Maps long-form column names to model-canonical names. Applied uniformly;
# missing keys are silently ignored by pandas .rename().
BASE_RENAMER = {'cell name': 'cellID',
                'current':   'currentInjected',
                'sex':       'Sex',
                'proj':      'projID',
                'mouse':     'mouseID',
                'stress':    'Stress'}


def _drug_state(sweep, mouse):
    """Return 'drug' if this sweep falls in mouse.drugSweeps, else 'baseline'.
    All mice currently have empty drugSweeps, so this returns 'baseline'
    until the drug experiment is run."""
    if hasattr(mouse, 'drugSweeps') and mouse.drugSweeps:
        if any(sweep in r for r in mouse.drugSweeps):
            return 'drug'
    return 'baseline'


def _row_metadata(cell, mouse, drug_split):
    """Common metadata fields appended to every long-form row."""
    md = {'cell name': cell.name,
          'mouse':     mouse.name,
          'sex':       mouse.sex,
          'stress':    mouse.stressCon,
          'proj':      mouse.proj}
    if drug_split:
        md['drug type'] = getattr(mouse, 'drugType', None)
    return md


def _attach_age(df, mouseList):
    """Add an 'age' column (mapped by mouse name) to a finished per-cell/per-mouse
    DataFrame, but only when at least one mouse in the dataset carries an age.

    Age is recorded for just a subset of mice (the timecourse dataset lives in a
    separate location from the age-less mice), so datasets that never recorded it
    don't gain an empty column, while mixed datasets get NaN for the mice that
    lack one. Mapping by mouse name *after* grouping -- rather than adding age as
    a groupby key -- is deliberate: a dropna=True groupby would otherwise drop
    every cell whose mouse has no age. Inserted right after the 'mouse' column.
    """
    if df is None or getattr(df, 'empty', True) or 'mouse' not in df.columns:
        return df
    ages = {mouse.name: mouse.age for mouse in mouseList.values()}
    if not any(a is not None for a in ages.values()):
        return df
    age_series = df['mouse'].map(ages)
    if 'age' in df.columns:
        df['age'] = age_series
    else:
        df.insert(df.columns.get_loc('mouse') + 1, 'age', age_series)
    return df


def extract_scalar(mouseList, attr, value_col, csv_name,
                   drug_split=False, filter_positive=False):
    """
    Walk mouse->cell, pull cell.<attr> (a {sweep: value} dict), flatten to
    long-form rows, group to cell-level mean, write CSV. Returns the
    cell-level grouped DataFrame ready to feed into run_models.

    Parameters
    ----------
    attr : str
        Cell attribute name holding a {sweep: value} dict.
    value_col : str
        Column name to use for the measured value in the long-form DF.
    csv_name : str
        Filename for the per-cell CSV.
    drug_split : bool
        If True, tag each row with drug state and include drug columns in
        the groupby keys (so baseline/drug rows stay separated).
    filter_positive : bool
        If True, drop rows where value_col <= 0 after groupby (used for IR).
    """
    rows = []
    for mouse in mouseList.values():
        for cell in mouse.cells:
            if hasattr(cell, 'bead') and cell.bead != '+':
                continue
            if not hasattr(cell, attr):
                print(f"{cell.name} has no {attr} data")
                continue
            data = getattr(cell, attr)
            for sweep, value in data.items():
                row = {value_col: value, 'sweep': sweep}
                row.update(_row_metadata(cell, mouse, drug_split))
                if drug_split:
                    row['drug state'] = _drug_state(sweep, mouse)
                rows.append(row)

    df = pd.DataFrame(rows)

    group_cols = ['cell name', 'proj', 'stress', 'mouse', 'sex']
    if drug_split:
        group_cols += ['drug type', 'drug state']

    # dropna=False only when drug_split: drug type is NaN for control mice
    # and we don't want them dropped. For non-drug features, default dropna
    # behavior matches the original code.
    grouped = df.groupby(group_cols, dropna=not drug_split).mean(
        numeric_only=True).reset_index()

    if filter_positive:
        grouped = grouped[grouped[value_col] > 0]

    _attach_age(grouped, mouseList)
    grouped.to_csv(csv_name)
    return grouped


def extract_per_current(mouseList, attr, value_col, csv_name,
                        drug_split=False):
    """
    Walk mouse->cell, pull cell.<attr> (a dict-of-lists with keys
    'current amp', value_col, 'sweep'), flatten to long-form rows
    (one per current per sweep), group to cell-and-current level mean,
    write CSV. Returns grouped DataFrame.
    """
    rows = []
    for mouse in mouseList.values():
        for cell in mouse.cells:
            if hasattr(cell, 'bead') and cell.bead != '+':
                continue
            if not hasattr(cell, attr):
                print(f"{cell.name} has no {attr} data")
                continue
            data = getattr(cell, attr)
            currents = data['current amp']
            values = data[value_col]
            sweeps = data['sweep']
            for current, value, sweep in zip(currents, values, sweeps):
                row = {'current': current, value_col: value, 'sweep': sweep}
                row.update(_row_metadata(cell, mouse, drug_split))
                if drug_split:
                    row['drug state'] = _drug_state(sweep, mouse)
                rows.append(row)

    df = pd.DataFrame(rows)

    group_cols = ['cell name', 'current', 'proj', 'stress', 'mouse']
    if drug_split:
        group_cols += ['drug type', 'drug state']

    # dropna=False only when drug_split (see extract_scalar comment)
    grouped = df.groupby(group_cols, dropna=not drug_split).mean(
        numeric_only=True).reset_index()

    _attach_age(grouped, mouseList)
    grouped.to_csv(csv_name)
    return grouped


def _age_tag(age):
    """Short, filesystem-safe string for an age value, used in model filenames.
    Whole-number floats (42.0) render as '42'; invalid path chars are dashed."""
    if isinstance(age, float) and float(age).is_integer():
        age = int(age)
    s = str(age)
    for ch in r'\/:*?"<>| ':
        s = s.replace(ch, '-')
    return s


def _age_strata(modelDF):
    """Yield (filename_tag, subset) pairs of DataFrames to fit models on.

    Always yields the pooled data first (tag '' -> unchanged filenames). Then,
    when an 'age' column with real values is present (only the timecourse
    dataset carries age), yields one subset per distinct age (tag '_age<value>')
    so a separate model set is written for each age. Cells with no recorded age
    appear only in the pooled set, never in an age-specific stratum.
    """
    yield '', modelDF
    if 'age' not in modelDF.columns:
        return
    for age in sorted(modelDF['age'].dropna().unique()):
        yield f'_age{_age_tag(age)}', modelDF[modelDF['age'] == age]


def run_models(df, value_col, variable, label,
               model_fn=None, do_outliers=True,
               pre_outlier_dropna=False,
               skip_ts_in_loop=False,
               filter_positive_total=False,
               require_all_three=False):
    """
    Per-projection models + total all-cells model. Writes summary text files.

    Runs the full model set once on the pooled data (filenames unchanged), and
    -- when df carries an 'age' column with real values (the timecourse dataset)
    -- once more per distinct age, tagged '_age<value>' in the filenames. Cells
    with no recorded age contribute only to the pooled models.

    Parameters
    ----------
    df : DataFrame
        Cell-level grouped DataFrame from extract_scalar / extract_per_current.
    value_col : str
        Name of the value column in df (e.g. 'accessRes', 'rheobase current').
    variable : str
        Model-canonical name (e.g. 'AR', 'rheobase'). Used in formula and
        for outlier detection. df gets renamed so that value_col -> variable.
    label : str
        Used in the output filenames: model_<label>_summary_<projType>.txt
        and model_<label>_summary_total.txt. Usually equal to variable but
        not always (e.g. sag's variable='sag' with label='voltageSag').
    model_fn : callable
        Either runModel (scalar) or runCurrentModel (per-current). Default
        runModel.
    do_outliers : bool
        If True, run remove_outliers_robust before fitting per-proj models.
        Disabled for FR and sag (per-current features).
    pre_outlier_dropna : bool
        If True, drop NaN rows on `variable` before outlier removal. Needed
        for rheobase, where some cells legitimately have no rheobase value.
    skip_ts_in_loop : bool
        If True, skip the runModel call when projType == 'TS' inside the
        per-projection loop. Outlier detection still runs for TS.
        (Used for rheobase, since TS-projecting cells fire spontaneously
        and don't have a rheobase.)
    filter_positive_total : bool
        If True, also filter `variable > 0` before the total model
        (a redundant safety filter that exists in the original IR code).
    require_all_three : bool
        If True, skip total model when fewer than 3 projections present.
    """
    if model_fn is None:
        model_fn = runModel

    # Build a per-feature renamer: BASE_RENAMER plus the value-col rename.
    renamer = dict(BASE_RENAMER)
    renamer[value_col] = variable

    modelDF = df.rename(columns=renamer)
    if 'sweep' in modelDF.columns:
        modelDF = modelDF.drop(columns=['sweep'])

    # If drug_split was used in extraction, only the baseline state goes into
    # the base stats model (drug analysis is run separately).
    if 'drug state' in modelDF.columns:
        modelDF = modelDF[modelDF['drug state'] == 'baseline']

    def _fit_stratum(stratum, out_label):
        """Per-projection models + total model for one data subset, writing files
        tagged with out_label. Called once for the pooled data and once per age."""
        # ---- Per-projection models ----
        for projType in stratum['projID'].unique():
            modelDF_proj = stratum[stratum['projID'] == projType]

            if pre_outlier_dropna:
                modelDF_proj = modelDF_proj.dropna(subset=[variable])

            outliers = pd.DataFrame()
            if do_outliers:
                modelDF_proj, outliers = remove_outliers_robust(
                    modelDF_proj, variable, 'Stress', 'cellID')
                if not outliers.empty:
                    print(f'outliers detected for {projType} ({out_label})!')
                    print(outliers)

            if skip_ts_in_loop and projType == 'TS':
                continue

            result = model_fn(modelDF_proj, variable)
            os.makedirs(MODELS_DIR, exist_ok=True)
            with open(os.path.join(MODELS_DIR, f"model_{out_label}_summary_{projType}.txt"), "w") as f:
                if result is not None and hasattr(result, 'summary'):
                    f.write(result.summary().as_text())
                    for tag, attr in [('main effect of Stress',  '_stress_main'),
                                      ('main effect of current', '_current_main'),
                                      ('Stress x current',       '_interaction')]:
                        t = getattr(result, attr, None)
                        if t is not None:
                            f.write(f"\n\n{tag}:\n{t}")
                # Record the outliers removed before fitting this projection's model.
                f.write(f"\n\nRemoved outliers (MAD, threshold={MAD_THRESHOLD}):\n")
                if outliers.empty:
                    f.write("None.\n")
                else:
                    f.write(outliers.to_string(index=False) + "\n")

        # ---- Total all-cells model ----
        total_df = stratum
        if filter_positive_total:
            total_df = total_df[total_df[variable] > 0]

        total_df = total_df[total_df['projID'] != 'TS'].copy()

        if require_all_three and total_df['projID'].nunique() < 3:
            print(f'not all proj types present, skipping total model ({out_label})')
            return
        # runModel_allcells builds SNL/PAG/BNST contrasts, so it needs all three.
        # In an age stratum a projection may be missing -- skip rather than crash.
        if not {'SNL', 'PAG', 'BNST'}.issubset(set(total_df['projID'].unique())):
            print(f'total model needs SNL/PAG/BNST; skipping total model ({out_label})')
            return

        total_df['projID'] = total_df['projID'].astype('category')
        total_df['projID'] = total_df['projID'].cat.reorder_categories(
            ['SNL', 'PAG', 'BNST'])

        result = runModel_allcells(total_df, variable)
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(os.path.join(MODELS_DIR, f"model_{out_label}_summary_total.txt"), "w") as f:
            if result is not None:
                f.write(result)

    # Pooled across all ages (unchanged filenames), then one model set per age
    # when the timecourse dataset's 'age' column is present.
    for age_tag, stratum in _age_strata(modelDF):
        _fit_stratum(stratum, f"{label}{age_tag}")


# ---------------------------------------------------------------------------
# GraphPad Prism-ready exports.
#
# Purely additive: builds wide, copy-paste-ready Excel tables from the SAME
# long-form DataFrames that feed the long CSVs, so the numbers match by
# construction. Tables are separated by projection type (one sheet each) and,
# within a projection, by stress condition (naive vs stress datasets).
#   - scalar measures   -> Prism "Column table"  (columns: naive / stress)
#   - current-response  -> Prism "Grouped table" (X = current down the side,
#                          one replicate sub-column per cell, naive then stress)
# One .xlsx per analysis function in PRISM_DIR; each sheet is one Prism table.
# ---------------------------------------------------------------------------

PRISM_DIR   = os.path.join(analysis_path, 'prism')
PROJ_ORDER  = ['SNL', 'PAG', 'BNST', 'TS']   # sheet order; any extras appended
GROUP_ORDER = ['naive', 'stress']            # dataset (column) order


def _prism_baseline(df):
    """Keep only baseline drug state if that column exists (mirrors models)."""
    if 'drug state' in df.columns:
        return df[df['drug state'] == 'baseline']
    return df


def _ordered_unique(values, preferred):
    """Values present, `preferred` ones first (in order), then any extras."""
    present = list(pd.unique(pd.Series(values).dropna()))
    ordered = [v for v in preferred if v in present]
    ordered += [v for v in present if v not in preferred]
    return ordered


def _prism_column_block(sub, value_col, group_col='stress', id_col='cell name'):
    """
    Column table: one column per group (naive/stress), each a NaN-padded list
    of the per-cell values. Returns a DataFrame, or None if no data.
    """
    series = {}
    for g in _ordered_unique(sub[group_col], GROUP_ORDER):
        vals = (sub[sub[group_col] == g]
                .sort_values(id_col)[value_col]
                .dropna()
                .reset_index(drop=True))
        if not vals.empty:
            series[g] = vals
    if not series:
        return None
    # pd.concat aligns by position, padding shorter columns with NaN.
    return pd.concat(series, axis=1)


def _prism_grouped_block(sub, value_col, x_col,
                         group_col='stress', id_col='cell name'):
    """
    Grouped table: X down the side, one replicate sub-column per cell, grouped
    naive-then-stress. Returned as a plain DataFrame whose column headers are
    the group labels (blank over the X column) and whose first row holds the
    x-axis label + cell IDs. Writing with index=False then yields
    row 1 = group labels, row 2 = x-label + cell IDs, then the data rows.
    Plain single-level columns sidestep pandas' unsupported "MultiIndex
    columns + index=False" to_excel path. Returns None if no data.
    """
    sub = sub.dropna(subset=[value_col])
    if sub.empty:
        return None

    x_vals = sorted(sub[x_col].dropna().unique())
    group_labels = ['']            # header row 1: blank over the X column
    cell_labels = [x_col]          # header row 2: the X-axis label
    columns_data = [list(x_vals)]  # the X column values

    for g in _ordered_unique(sub[group_col], GROUP_ORDER):
        g_sub = sub[sub[group_col] == g]
        for cell in sorted(g_sub[id_col].dropna().unique()):
            # one value per x (already grouped upstream); mean is a safe no-op
            lookup = g_sub[g_sub[id_col] == cell].groupby(x_col)[value_col].mean()
            group_labels.append(g)
            cell_labels.append(cell)
            columns_data.append([lookup.get(x, np.nan) for x in x_vals])

    if len(group_labels) == 1:        # only the X column, no cells
        return None

    rows = [cell_labels]              # first body row = x-label + cell IDs
    rows += [[col[r] for col in columns_data] for r in range(len(x_vals))]
    return pd.DataFrame(rows, columns=group_labels)


def _pick_excel_engine():
    """First available pandas Excel writer engine, or None if none installed."""
    for eng in ('openpyxl', 'xlsxwriter'):
        try:
            __import__(eng)
            return eng
        except ImportError:
            continue
    return None


def _write_prism_workbook(blocks, out_name):
    """Write {sheet_name: block_df} to PRISM_DIR/<out_name>.xlsx, one sheet per
    block. Header rows follow each block's column structure (1 row for column
    tables, 2 for grouped). Skips the file entirely if there is nothing to write.
    If no Excel engine (openpyxl/xlsxwriter) is installed, warns and skips rather
    than letting the analysis pipeline crash."""
    blocks = {s: b for s, b in blocks.items() if b is not None and not b.empty}
    if not blocks:
        print(f'[prism] no data for {out_name}, skipping workbook')
        return
    engine = _pick_excel_engine()
    if engine is None:
        print(f'[prism] no Excel engine (openpyxl/xlsxwriter) installed; skipping '
              f'{out_name}.xlsx. Run `pip install openpyxl` to enable Prism export.')
        return
    os.makedirs(PRISM_DIR, exist_ok=True)
    path = os.path.join(PRISM_DIR, f'{out_name}.xlsx')
    with pd.ExcelWriter(path, engine=engine) as writer:
        for sheet, block in blocks.items():
            block.to_excel(writer, sheet_name=sheet[:31], index=False)
    print(f'[prism] wrote {path} ({len(blocks)} sheet(s), engine={engine})')


def export_prism(df, value_col, out_name, kind, x_col=None):
    """One workbook, one sheet per projection. kind in {'column', 'grouped'}.
    Never raises: any failure is reported and the analysis pipeline continues."""
    try:
        df = _prism_baseline(df)
        blocks = {}
        for proj in _ordered_unique(df['proj'], PROJ_ORDER):
            sub = df[df['proj'] == proj]
            if kind == 'column':
                block = _prism_column_block(sub, value_col)
            else:
                block = _prism_grouped_block(sub, value_col, x_col)
            if block is not None:
                blocks[proj] = block
        _write_prism_workbook(blocks, out_name)
    except Exception as e:
        print(f'[prism] WARNING: export for {out_name} failed ({e}); continuing.')


def export_prism_multi(df, value_cols, out_name, kind, x_col=None):
    """One workbook; one sheet per projection x metric ('<proj>_<metric>').
    Never raises: any failure is reported and the analysis pipeline continues."""
    try:
        df = _prism_baseline(df)
        blocks = {}
        for proj in _ordered_unique(df['proj'], PROJ_ORDER):
            sub = df[df['proj'] == proj]
            for metric in value_cols:
                if metric not in sub.columns:
                    continue
                if kind == 'column':
                    block = _prism_column_block(sub, metric)
                else:
                    block = _prism_grouped_block(sub, metric, x_col)
                if block is not None:
                    blocks[f'{proj}_{metric}'] = block
        _write_prism_workbook(blocks, out_name)
    except Exception as e:
        print(f'[prism] WARNING: export for {out_name} failed ({e}); continuing.')


# ---------------------------------------------------------------------------
# Per-feature pipelines. Each is now ~3 lines: extract -> run_models.
# Per-feature differences (which features filter positive values, which skip
# TS in the per-projection loop, etc.) are explicit arguments.
# ---------------------------------------------------------------------------

def getAccessResistance(mouseList):
    df = extract_scalar(mouseList, attr='accessResistance',
                        value_col='accessRes',
                        csv_name='access_resistance.csv')
    run_models(df, value_col='accessRes', variable='AR', label='AR')
    export_prism(df, 'accessRes', 'access_resistance', 'column')


def getCapacitance(mouseList):
    df = extract_scalar(mouseList, attr='capacitance',
                        value_col='capacitance',
                        csv_name='capacitance.csv')
    run_models(df, value_col='capacitance', variable='capacitance',
               label='capacitance')
    export_prism(df, 'capacitance', 'capacitance', 'column')


def getInputResistance(mouseList):
    df = extract_scalar(mouseList, attr='inputResistance',
                        value_col='inputRes',
                        csv_name='input_resistance.csv',
                        drug_split=True,
                        filter_positive=True)
    run_models(df, value_col='inputRes', variable='IR', label='IR',
               filter_positive_total=True,
               require_all_three=True)
    export_prism(df, 'inputRes', 'input_resistance', 'column')


def getRMPData(mouseList):
    df = extract_scalar(mouseList, attr='RMPData',
                        value_col='resting pot',
                        csv_name='baseline RMP.csv',
                        drug_split=True)
    run_models(df, value_col='resting pot', variable='RMP', label='RMP')
    export_prism(df, 'resting pot', 'RMP', 'column')


def _resolve_rheobase_with_FI_fallback(mouseList):
    """Final per-cell rheobase, in memory (the pickle on disk is untouched):

      - if the cell fired on >=1 real 5 pA step, use the 5 pA value(s)
        (source '5pA'); censored reps that never fired are dropped from the mean.
      - otherwise -- no 5 pA recording at all, OR the 5 pA protocol was run but
        the cell never fired (previously censored to max+step) -- use the F-I
        50 ms fallback (EphysCell.addRheobaseFromFI), source 'FI_50ms'.
      - if neither is available, leave it empty (source 'none').

    A "real" 5 pA value is one equal to an actual tested current in
    rheoCurrentDict; a censored max+step placeholder is NOT real, so those
    cells fall back to the F-I value (per the chosen scope). Tags
    cell.rheobaseSource. Requires cells to carry fiRheobase, which is populated
    by ephysAnalysisCreate (re-run create, or backfill, to compute it).
    """
    n5 = nfi = nnone = 0
    for mouse in mouseList.values():
        for cell in mouse.cells:
            rb = getattr(cell, 'rheobase', {}) or {}
            ladder = set(c for c in (getattr(cell, 'rheoCurrentDict', {}) or {}).values()
                         if c is not None)
            real = {sw: v for sw, v in rb.items()
                    if v is not None and not (isinstance(v, float) and np.isnan(v))
                    and v in ladder}
            if real:                                   # fired on a real 5 pA step
                cell.rheobase = real                   # drop censored/never-fired reps
                cell.rheobaseSource = '5pA'
                n5 += 1
                continue
            fi = getattr(cell, 'fiRheobase', {}) or {}
            if fi:
                cell.rheobase = dict(fi)
                cell.rheobaseSource = 'FI_50ms'
                nfi += 1
            else:
                cell.rheobase = {}
                cell.rheobaseSource = 'none'
                nnone += 1
    print(f'[rheobase] resolved: {n5} from 5 pA, {nfi} from F-I 50 ms fallback, '
          f'{nnone} with neither (need F-I backfill, or genuinely no recording)')


def getRheobase(mouseList):
    _resolve_rheobase_with_FI_fallback(mouseList)
    df = extract_scalar(mouseList, attr='rheobase',
                        value_col='rheobase current',
                        csv_name='baseline rheobase.csv')
    # Provenance: tag each cell's value with the protocol it came from.
    src = {(m.name, c.name): getattr(c, 'rheobaseSource', 'none')
           for m in mouseList.values() for c in m.cells}
    df['rheobase source'] = [src.get((mo, cn), 'none')
                             for mo, cn in zip(df['mouse'], df['cell name'])]
    # Exclude non-physical rheobase values (<= 0) everywhere: the saved CSV, the
    # models, and the Prism export.
    n_bad = int((df['rheobase current'] <= 0).sum())
    if n_bad:
        print(f'[rheobase] excluding {n_bad} cell(s) with rheobase <= 0')
    df = df[df['rheobase current'] > 0]
    df.to_csv('baseline rheobase.csv')
    run_models(df, value_col='rheobase current', variable='rheobase',
               label='rheobase',
               pre_outlier_dropna=True,
               skip_ts_in_loop=True,
               require_all_three=True)
    export_prism(df, 'rheobase current', 'rheobase', 'column')


def getFiringRateData(mouseList):
    df = extract_per_current(mouseList, attr='firingRateData',
                             value_col='firing rate',
                             csv_name='baseline Firing Rate.csv',
                             drug_split=True)
    run_models(df, value_col='firing rate', variable='firingRate',
               label='firingRate',
               model_fn=runCurrentModel,
               do_outliers=False,
               require_all_three=True)
    export_prism(df, 'firing rate', 'firing_rate', 'grouped', x_col='current')


def getTrainProps(mouseList):
    spikeProp_DF = pd.DataFrame()
        
    def appendTrainData(existingDF, spikePropDF, cell, mouse):
        cellInfo = {
            'cell name': cell.name,
            'mouse': mouse.name,
            'sex': mouse.sex,
            'stress': mouse.stressCon,
            'proj': mouse.proj,
            'drug type': getattr(mouse, 'drugType', None)  # CHANGED TO MOUSE
        }

        # Create a DataFrame with repeated cellInfo for each row in spikePropDF
        cellInfoDF = pd.DataFrame([cellInfo] * len(spikePropDF))

        # Concatenate the cellInfoDF with spikePropDF along columns
        combinedDF = pd.concat([cellInfoDF, spikePropDF.reset_index(drop=True)], axis=1)

        # If an existing DataFrame is provided, concatenate with it
        if existingDF is not None:
            combinedDF = pd.concat([existingDF, combinedDF], ignore_index=True)

        return combinedDF

    def get_drug_state(sweep, mouse):  # CHANGED TO MOUSE
        if hasattr(mouse, 'drugSweeps') and mouse.drugSweeps:  # CHANGED TO MOUSE
            if any(sweep in r for r in mouse.drugSweeps):      # CHANGED TO MOUSE
                return 'drug'
        return 'baseline'
        
    for mouse in mouseList.values():
        for cell in mouse.cells:
            if hasattr(cell, 'spikeAnalysis') == False:
                continue

            spikes = cell.spikeAnalysis[(cell.spikeAnalysis['protocol'] == 'firing rate') & (
                ~cell.spikeAnalysis['threshold_v'].isna())]
            if spikes.empty:
                continue
                
            trainData = pd.DataFrame()
            columns_for_extraction = ['adapt', 'latency', 'isi_cv', 'mean_isi', 'median_isi', 'first_isi']
            columns_to_keep = ['current injected', 'sweep']

            for col in spikes.columns:
                if col in columns_for_extraction:
                    trainData[col] = spikes[col].apply(lambda x: x)
                elif col in columns_to_keep:
                    trainData[col] = spikes[col]
                else:
                    pass
            
            # Tag the drug state before taking the mean!
            trainData['drug state'] = trainData['sweep'].apply(lambda s: get_drug_state(s, mouse))  # CHANGED TO MOUSE
            
            meanTrain = trainData.groupby(['current injected', 'drug state']).mean(numeric_only=True).reset_index()
            spikeProp_DF = appendTrainData(spikeProp_DF, meanTrain, cell, mouse)

    spikeProp_DF['latency'] = (spikeProp_DF['latency'] - 0.5) * 1000
    spikeProp_DF['latency'] = spikeProp_DF['latency'].apply(lambda x: np.nan if x < 0 else x)
    for col in columns_for_extraction:
        spikeProp_DF[col] = spikeProp_DF[col].replace(0, np.nan)
    _attach_age(spikeProp_DF, mouseList)
    spikeProp_DF.to_csv('spike train properties.csv')

    export_prism_multi(spikeProp_DF,
                       ['adapt', 'latency', 'isi_cv', 'mean_isi',
                        'median_isi', 'first_isi'],
                       'spike_train_properties', 'grouped',
                       x_col='current injected')
    
    renamer = {'mouse' : 'mouseID',
               'stress' : 'Stress',
               'cell name' : 'cellID',
               'current injected' : 'currentInjected',
               'proj' : 'projID'}
    modelDF = spikeProp_DF.rename(columns=renamer)
    modelDF = modelDF[modelDF['currentInjected'] >= 40]
    
    # Only run base stats models on baseline data
    modelDF_baseline = modelDF[modelDF['drug state'] == 'baseline']

    # Pooled, then one model set per age when present (see _age_strata).
    for age_tag, stratum in _age_strata(modelDF_baseline):
        for projType in stratum['projID'].unique():
            modelDF_proj = stratum[stratum['projID'] == projType]

            for col in columns_for_extraction:
                result = runCurrentModel(modelDF_proj, col)
                os.makedirs(MODELS_DIR, exist_ok=True)
                with open(os.path.join(MODELS_DIR, f"model_trainProps_{col}_summary_{projType}{age_tag}.txt"), "w") as f:
                    if result is not None and hasattr(result, 'summary'):
                        f.write(result.summary().as_text())
                        for tag, attr in [('main effect of Stress',  '_stress_main'),
                                          ('main effect of current', '_current_main'),
                                          ('Stress x current',       '_interaction')]:
                            t = getattr(result, attr, None)
                            if t is not None:
                                f.write(f"\n\n{tag}:\n{t}")
        # print(f"for {col}")
        # print(result.summary())

def getPhasePlot_RawSweeps(mouseList, whichSpikes, whichCurrents='all'):
    phasePlotDF = pd.DataFrame()
    spikeDF = pd.DataFrame()
    groupBys = ['stress', 'proj', 'cluster']
    V_GB = ['proj', 'cluster']
    def sem(x):
        return x.std() / np.sqrt(len(x))

    def appendPhaseData(existingDF, spikePropDF, cell, mouse):
        cellInfo = {
            'cell name': cell.name,
            'mouse': mouse.name,
            'sex': mouse.sex,
            'stress': mouse.stressCon,
            'proj': mouse.proj,
            'cluster' : cell.cluster
        }

        # Create a DataFrame with repeated cellInfo for each row in spikePropDF
        cellInfoDF = pd.DataFrame([cellInfo] * len(spikePropDF))

        # Concatenate the cellInfoDF with spikePropDF along columns
        combinedDF = pd.concat([cellInfoDF, spikePropDF.reset_index(drop=True)], axis=1)

        # If an existing DataFrame is provided, concatenate with it
        if existingDF is not None:
            combinedDF = pd.concat([existingDF, combinedDF], ignore_index=True)

        return combinedDF
            
    for mouse in mouseList.values():
        for cell in mouse.cells:
            if mouse.proj == 'SNL' or mouse.proj == 'TS':
                if hasattr(cell, 'spikeAnalysis') == False:
                    continue
                
                phaseDF_cell = pd.DataFrame(columns=['sweep','current','dV_dt', 'V', 'spikeNum'])
                spikeDF_cell = pd.DataFrame(columns=['sweep','current','spikeV', 'spikeNum'])
                sweeps_wSpikes = cell.spikeAnalysis[(cell.spikeAnalysis['protocol'] == 'firing rate') & (
                    ~cell.spikeAnalysis['phasePlotData'].isna())]
                for index, sweep in sweeps_wSpikes.iterrows():
                    if whichCurrents != 'all':
                        if int(sweep['current injected']) not in whichCurrents:
                            continue
                    spikes = sweep['rawSpikes']
                    spikeDF_train = pd.DataFrame({'spikeV': spikes})
                    spikeDF_train['spikeNum'] = range(1, len(spikeDF_train) + 1)
                    spikeDF_train['current'] = sweep['current injected']
                    spikeDF_train['sweep'] = sweep['sweep']
                    
                    phasePlots = sweep['phasePlotData']
                    phaseDF_train = pd.DataFrame(phasePlots, columns=['dV_dt', 'V'])
                    phaseDF_train['spikeNum'] = range(1, len(phaseDF_train) + 1)
                    phaseDF_train['current'] = sweep['current injected']
                    phaseDF_train['sweep'] = sweep['sweep']
                    
                    if whichSpikes == 'all':
                        pass
                    if whichSpikes == 'first':
                        phaseDF_train = phaseDF_train.iloc[[0]]
                        spikeDF_train = spikeDF_train.iloc[[0]]
                    if whichSpikes == 'mid':
                        midSpike = round(len(phaseDF_train)/2)
                        phaseDF_train = phaseDF_train.iloc[[midSpike]]
                        spikeDF_train = spikeDF_train.iloc[[midSpike]]


                    phaseDF_cell = pd.concat([phaseDF_cell, phaseDF_train], ignore_index=True)     
                    spikeDF_cell = pd.concat([spikeDF_cell, spikeDF_train], ignore_index=True)

                phaseDF_cell_dV_dt = phaseDF_cell['dV_dt'].tolist()
                phaseDF_cell_dV_dt = [arr for arr in phaseDF_cell_dV_dt if isinstance(arr, (list, np.ndarray)) and len(arr) == 100]

                phaseDF_cell_V = phaseDF_cell['V'].tolist()
                phaseDF_cell_V =  [arr for arr in phaseDF_cell_V if isinstance(arr, (list, np.ndarray)) and len(arr) == 100]

                spikeDF_cell_V = spikeDF_cell['spikeV'].tolist()
                spikeDF_cell_V = [arr for arr in spikeDF_cell_V if isinstance(arr, (list, np.ndarray)) and len(arr) == 6000]
                avg_spike_V = np.mean(spikeDF_cell_V, axis=0)

                # if whichSpikes == 'all':
                avg_dV_dt = np.mean(phaseDF_cell_dV_dt, axis=0)
                avg_V = np.mean(phaseDF_cell_V, axis=0)
                    
                avg_dV_dt = avg_dV_dt/1000
                avg_df_cell = pd.DataFrame([[avg_dV_dt, avg_V]], columns=['dV_dt', 'V'])
                phasePlotDF = appendPhaseData(phasePlotDF, avg_df_cell, cell, mouse)

                avg_df_spike_cell = pd.DataFrame([[avg_spike_V]], columns=['spikeV'])
                spikeDF = appendPhaseData(spikeDF, avg_df_spike_cell, cell, mouse)

    phasePlotDF = phasePlotDF[phasePlotDF['dV_dt'].apply(lambda x: isinstance(x, (list, np.ndarray)) and len(x) == 100 and not np.isnan(x).any())].copy()
    spikeDF = spikeDF[spikeDF['spikeV'].apply(lambda x: isinstance(x, (list, np.ndarray)) and len(x) == 6000 and not np.isnan(x).any())].copy()

    # ===== INTERPOLATION =====
    # Create a standard voltage grid based on the overall V range in phasePlotDF
    all_V = np.concatenate(phasePlotDF['V'].apply(np.array).tolist())
    v_min = np.min(all_V)
    v_max = np.max(all_V)
    n_points = 100  # Resolution for interpolation; adjust as needed
    v_grid = np.linspace(v_min, v_max, n_points)

    interpolated_curves = []
    stress_labels = []

# Add checks before interpolation and handle extreme values in your curves

    for idx, row in phasePlotDF.iterrows():
        V_arr = np.array(row['V'])
        dV_dt_arr = np.array(row['dV_dt'])
        
        # If V is not strictly monotonic, sort V and dV/dt by increasing voltage
        if not (np.all(np.diff(V_arr) > 0) or np.all(np.diff(V_arr) < 0)):
            sort_idx = np.argsort(V_arr)
            V_arr = V_arr[sort_idx]
            dV_dt_arr = dV_dt_arr[sort_idx]
        
        # Check for and handle inf or NaN values in dV_dt_arr
        if np.any(np.isinf(dV_dt_arr)) or np.any(np.isnan(dV_dt_arr)):
            print(f"Warning: Found inf or NaN values in dV_dt for row {idx}")
            # Replace inf values with large but finite values
            dV_dt_arr = np.nan_to_num(dV_dt_arr, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Build a linear interpolation function (extrapolating if necessary)
        try:
            f_interp = interp1d(V_arr, dV_dt_arr, kind='linear', bounds_error=False, fill_value='extrapolate')
            dV_dt_interp = f_interp(v_grid)
            
            # Check for inf values in interpolated data
            if np.any(np.isinf(dV_dt_interp)) or np.any(np.isnan(dV_dt_interp)):
                print(f"Warning: Interpolation produced inf/NaN values for row {idx}")
                # Clip extreme values
                dV_dt_interp = np.nan_to_num(dV_dt_interp, nan=0.0, posinf=1e6, neginf=-1e6)
                
            interpolated_curves.append(dV_dt_interp)
            stress_labels.append(row['stress'])
        except Exception as e:
            print(f"Error in interpolation for row {idx}: {str(e)}")
            continue

    interpolated_array = np.vstack(interpolated_curves)

    # Calculate pairwise Euclidean distances between the interpolated curves
    dist_matrix = pairwise_distances(interpolated_array, metric='euclidean')

    # Force symmetry by averaging with transpose
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    # Force diagonal to be exactly zero
    np.fill_diagonal(dist_matrix, 0)

    # Create IDs for distance matrix
    ids = [str(i) for i in range(len(stress_labels))]

    # Create DistanceMatrix for statistical tests
    dm = DistanceMatrix(dist_matrix, ids=ids)

    # Convert stress labels to metadata for statistical analysis
    metadata = pd.DataFrame({
    'stress': stress_labels,
    'cluster': phasePlotDF['cluster'].values if 'cluster' in phasePlotDF.columns else np.nan
        }, index=ids)

    # Filter out rows where cluster is NaN
    # Filter out rows where cluster is NaN
        # Filter out rows where cluster is NaN
    valid_cluster_mask = ~metadata['cluster'].isna()
    if sum(valid_cluster_mask) > 0:
        # Create subset distance matrix and metadata for samples with valid clusters
        # Ensure the array is C-contiguous by using np.ascontiguousarray
        subset_matrix = np.ascontiguousarray(dist_matrix[valid_cluster_mask][:, valid_cluster_mask])
        cluster_dm = DistanceMatrix(subset_matrix, 
                                   ids=list(np.array(ids)[valid_cluster_mask]))
        cluster_metadata = metadata.loc[valid_cluster_mask]
        
        # Run PERMANOVA test for stress effect on all data
        permanova_stress_result = permanova(dm, metadata, column='stress', permutations=999)
        
        # Run PERMANOVA test for cluster effect on filtered data
        permanova_cluster_result = permanova(cluster_dm, cluster_metadata, column='cluster', permutations=999)
        
        # For interaction, we need to create a combined factor column
        # because the older scikit-bio version doesn't support formula-based tests
        cluster_metadata['stress_cluster'] = cluster_metadata['stress'] + '_' + cluster_metadata['cluster'].astype(str)
        permanova_interaction_result = permanova(cluster_dm, cluster_metadata, column='stress_cluster', permutations=999)
        
        # Run separate tests for each cluster to see stress effects within clusters
        unique_clusters = cluster_metadata['cluster'].unique()
        cluster_stress_results = {}
        for cluster_id in unique_clusters:
            cluster_mask = cluster_metadata['cluster'] == cluster_id
            if sum(cluster_mask) > 1:  # Need at least 2 samples for distance comparison
                cluster_specific_dm = DistanceMatrix(
                    np.ascontiguousarray(subset_matrix[cluster_mask][:, cluster_mask]),
                    ids=list(np.array(cluster_metadata.index)[cluster_mask])
                )
                cluster_specific_metadata = cluster_metadata.loc[cluster_mask]
                try:
                    result = permanova(cluster_specific_dm, cluster_specific_metadata, 
                                      column='stress', permutations=999)
                    cluster_stress_results[f'cluster_{cluster_id}'] = result
                except Exception as e:
                    print(f"Error running PERMANOVA for cluster {cluster_id}: {str(e)}")
        
        print(f"\nPERMANOVA results for {whichSpikes} spikes at {whichCurrents} pA:")
        print("Stress effect:")
        print(permanova_stress_result)
        print("\nCluster effect:")
        print(permanova_cluster_result)
        print("\nInteraction effect (stress*cluster):")
        print(permanova_interaction_result)
        for cluster_id, result in cluster_stress_results.items():
            print(f"\nStress effect within {cluster_id}:")
            print(result)
        
        # Save results to file with all models
        with open(f'stats_PP_{whichSpikes}Spikes_{whichCurrents}_pA.txt', 'w') as f:
            f.write(f"PERMANOVA results for phase plots ({whichSpikes} spikes, {whichCurrents} pA):\n")
            f.write("Stress effect:\n")
            f.write(str(permanova_stress_result) + "\n\n")
            f.write("Cluster effect:\n")
            f.write(str(permanova_cluster_result) + "\n\n")
            f.write("Interaction effect (stress*cluster):\n")
            f.write(str(permanova_interaction_result) + "\n\n")
            
            for cluster_id, result in cluster_stress_results.items():
                f.write(f"Stress effect within {cluster_id}:\n")
                f.write(str(result) + "\n\n")
    plt.figure(figsize=(10, 6))
    # After your regular phase plot, create cluster-specific subplots
    if sum(valid_cluster_mask) > 0 and 'cluster' in phasePlotDF.columns:
        # Get unique clusters
        unique_clusters = sorted(phasePlotDF['cluster'].dropna().unique())
        
        # Create a subplot figure with one plot per cluster
        n_clusters = len(unique_clusters)
        fig, axs = plt.subplots(1, n_clusters, figsize=(5*n_clusters, 5), sharey=True)
        
        # If there's only one cluster, axs needs to be in a list format
        if n_clusters == 1:
            axs = [axs]
        
        # For each cluster, create a separate subplot
        for i, cluster_id in enumerate(unique_clusters):
            # Get data for this cluster only
            cluster_data = phasePlotDF[phasePlotDF['cluster'] == cluster_id]
            
            # Plot each stress condition within this cluster
            for stress_condition, color, style in [('stress', 'red', '--'), ('naive', 'black', '-')]:
                stress_subset = cluster_data[cluster_data['stress'] == stress_condition]
                
                # Skip if no data for this condition in this cluster
                if len(stress_subset) == 0:
                    continue
                
                # Get mean values
                mean_V = np.mean(np.vstack(stress_subset['V'].values), axis=0)
                mean_dV_dt = np.mean(np.vstack(stress_subset['dV_dt'].values), axis=0)
                # mean_dV_dt = mean_dV_dt / 1000 # Convert to V/s
                
                # Calculate SEM
                sem_dV_dt = np.std(np.vstack(stress_subset['dV_dt'].values), axis=0) / np.sqrt(len(stress_subset))
                
                # Plot mean line with label that includes n
                label = f"{stress_condition} (n={len(stress_subset)})"
                axs[i].plot(mean_V, mean_dV_dt*100, label=label, 
                         color=color, linewidth=2.0, linestyle=style)
                
                # Add shaded SEM area
                axs[i].fill_between(mean_V, 
                                 (mean_dV_dt-sem_dV_dt)*100, 
                                 (mean_dV_dt+sem_dV_dt)*100, 
                                 color=color, alpha=0.2)
            
            # Set subplot title and other formatting
            axs[i].set_title(f'Cluster {int(cluster_id)}', fontsize=14)
            axs[i].grid(True, linestyle='--', alpha=0.7)
            axs[i].legend(loc='best', fontsize=10)
            
            # Only set xlabel for bottom row plots
            axs[i].set_xlabel('Membrane Potential (mV)')
        
        # Set ylabel only for leftmost plot
        axs[0].set_ylabel('dV/dt (V/s)')
        
        plt.suptitle(f'Phase Plots by Cluster - {whichSpikes} Spikes, {whichCurrents} pA', fontsize=16)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f'PhasePlot_Clusters_{whichSpikes}Spikes_{whichCurrents}_pA.png', dpi=300)
        plt.close()

        # Create a second plot with cluster data and stats results as annotations
        fig, axs = plt.subplots(1, n_clusters, figsize=(5*n_clusters, 5), sharey=True)
        # Plot average phase plots for each stress condition, ignoring clusters

    dV_dt_df = phasePlotDF.drop(columns=['dV_dt', 'V']).join(pd.DataFrame(phasePlotDF['dV_dt'].to_list(), columns=[f'dV_dt_{i}' for i in range(100)]))
    V_df = phasePlotDF.drop(columns=['dV_dt', 'V']).join(pd.DataFrame(phasePlotDF['V'].to_list(), columns=[f'V_{i}' for i in range(100)]))
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        dV_df_mean = dV_dt_df.groupby(groupBys).mean().reset_index()
        dV_df_SEM = dV_dt_df.groupby(groupBys).agg(
            lambda x: np.std(x.tolist(), axis=0) / np.sqrt(len(x))).reset_index()

        V_mean = V_df.groupby(V_GB).mean().reset_index()
        V_SEM = V_df.groupby(V_GB).agg(
            lambda x: np.std(x.tolist(), axis=0) / np.sqrt(len(x))).reset_index()

    os.chdir(INTRINSIC_DIR)
    # spikes dataframe stuff now get them meanz
    spike_df = spikeDF.drop(columns=['spikeV']).join(pd.DataFrame(spikeDF['spikeV'].to_list(), columns=[f'spikeV_{i}' for i in range(6000)]))
    spike_df.to_csv(f'_clusters_rawSpike_{whichSpikes}Spikes_{whichCurrents}_pA.csv', index=False)


    dV_df_mean.to_csv(f'_clusters_PP_dV_dt_mean_{whichSpikes}Spikes_{whichCurrents}_pA.csv', index=False)
    dV_df_SEM.to_csv(f'_clusters_PP_dV_dt_SEM_{whichSpikes}Spikes_{whichCurrents}_pA.csv', index=False)
    V_mean.to_csv(f'_clusters_PP_V_mean_{whichSpikes}Spikes_{whichCurrents}_pA.csv', index=False)
    V_SEM.to_csv(f'_clusters_PP_V_SEM_{whichSpikes}Spikes_{whichCurrents}_pA.csv', index=False)


    #* do it again without clusters
    dV_dt_df = phasePlotDF.drop(columns=['dV_dt', 'V']).join(pd.DataFrame(phasePlotDF['dV_dt'].to_list(), columns=[f'dV_dt_{i}' for i in range(100)]))
    V_df = phasePlotDF.drop(columns=['dV_dt', 'V']).join(pd.DataFrame(phasePlotDF['V'].to_list(), columns=[f'V_{i}' for i in range(100)]))

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        dV_df_mean = dV_dt_df.groupby(['proj', 'stress']).mean().reset_index()
        dV_df_SEM = dV_dt_df.groupby(['proj', 'stress']).agg(
            lambda x: np.std(x.tolist(), axis=0) / np.sqrt(len(x))).reset_index()

        V_mean = V_df.groupby('proj').mean().reset_index()
        V_SEM = V_df.groupby('proj').agg(
            lambda x: np.std(x.tolist(), axis=0) / np.sqrt(len(x))).reset_index()

    os.chdir(os.path.join(INTRINSIC_DIR, 'phasePlots'))
    dV_df_mean.to_csv(f'PP_dV_dt_mean_{whichSpikes}Spikes_{whichCurrents}_pA.csv', index=False)
    dV_df_SEM.to_csv(f'PP_dV_dt_SEM_{whichSpikes}Spikes_{whichCurrents}_pA.csv', index=False)
    V_mean.to_csv(f'PP_V_mean_{whichSpikes}Spikes_{whichCurrents}_pA.csv', index=False)
    V_SEM.to_csv(f'PP_V_SEM_{whichSpikes}Spikes_{whichCurrents}_pA.csv', index=False)

    interp_dv_dt = pd.DataFrame(interpolated_array, 
                               columns=[f'v_{i}' for i in range(n_points)])
    
    # Add identifying columns to interp_dv_dt
    interp_dv_dt = pd.concat([
        phasePlotDF[['cell name', 'mouse', 'sex', 'stress', 'proj', 'cluster']].reset_index(drop=True),
        interp_dv_dt
    ], axis=1)
    
    # Create a DataFrame with just the voltage grid (X-axis points)
    v_grid_df = pd.DataFrame({'voltage': v_grid})
    
    # Save the interpolated data
    # os.chdir(r'R:\Basic_Sciences\Phys\Lerner_Lab_tnl2633\Jacob\aCUS\ALL_INTRINSIC_EPHYS\analysis\phasePlots')
    _attach_age(interp_dv_dt, mouseList)
    interp_dv_dt.to_csv(f'PP_interpolated_dVdt_{whichSpikes}Spikes_{whichCurrents}_pA.csv', index=False)
    # v_grid_df.to_csv(f'PP_voltage_grid_{whichSpikes}Spikes_{whichCurrents}_pA.csv', index=False)

    os.chdir(INTRINSIC_DIR)

        # Create DataFrames for saving the interpolated data



def getSpikeProps(mouseList):
    spikePropList = []
    
    def appendFRData(spikePropList, spikePropDict, cell, mouse):
        cellInfo = {'cell name': cell.name,
                    'mouse': mouse.name,
                    'sex': mouse.sex,
                    'stress': mouse.stressCon,
                    'proj': mouse.proj,
                    'drug type': getattr(mouse, 'drugType', None)}  # CHANGED TO MOUSE
        combinedDict = {**cellInfo, **spikePropDict}
        spikePropList.append(combinedDict)

    def get_drug_state(sweep, mouse):  # CHANGED TO MOUSE
        if hasattr(mouse, 'drugSweeps') and mouse.drugSweeps:  # CHANGED TO MOUSE
            if any(sweep in r for r in mouse.drugSweeps):      # CHANGED TO MOUSE
                return 'drug'
        return 'baseline'
    
    for mouse in mouseList.values():
        for cell in mouse.cells:
            if hasattr(cell, 'spikeAnalysis') == False:
                continue

            if hasattr(cell, 'rheobase'):
                rheoSpikes = cell.spikeAnalysis[(cell.spikeAnalysis['protocol'] == 'rheobase') & (
                    ~cell.spikeAnalysis['threshold_v'].isna())]
            else:
                rheoSpikes = cell.spikeAnalysis[(cell.spikeAnalysis['protocol'] == 'firing rate') & (
                    ~cell.spikeAnalysis['threshold_v'].isna())]
            
            if rheoSpikes.empty:
                continue
                
            firstSpike = pd.DataFrame()
            columns_for_extraction = ['threshold_v', 'upstroke', 'peak_v',
                                    'trough_v', 'upstroke_v', 'downstroke', 'downstroke_v', 'width']
            columns_to_keep = ['current injected', 'sweep']

            for col in rheoSpikes.columns:
                if col in columns_for_extraction:
                    firstSpike[col] = rheoSpikes[col].apply(lambda x: x[0])
                elif col in columns_to_keep:
                    firstSpike[col] = rheoSpikes[col]
                else:
                    pass
            
            # Tag the drug state before taking the mean
            firstSpike['drug state'] = firstSpike['sweep'].apply(lambda s: get_drug_state(s, mouse))  # CHANGED TO MOUSE
            
            # Calculate mean separately for baseline and drug sweeps
            for state in firstSpike['drug state'].unique():
                state_df = firstSpike[firstSpike['drug state'] == state]
                meanSpike = dict(state_df.mean(numeric_only=True))
                meanSpike['drug state'] = state
                appendFRData(spikePropList, meanSpike, cell, mouse)

    spikeProp_DF = pd.DataFrame(spikePropList)
    if spikeProp_DF.empty:
        return
        
    spikeProp_DF['width'] = spikeProp_DF['width'].apply(
        lambda x: x*1000)  # convert to ms
    for col in columns_for_extraction:
        spikeProp_DF[col] = spikeProp_DF[col].replace(0, np.nan)

    _attach_age(spikeProp_DF, mouseList)
    spikeProp_DF.to_csv('single spike properties.csv')

    export_prism_multi(spikeProp_DF,
                       ['threshold_v', 'upstroke', 'peak_v', 'trough_v',
                        'upstroke_v', 'downstroke', 'downstroke_v', 'width'],
                       'single_spike_properties', 'column')
    
    renamer = {'mouse' : 'mouseID',
               'stress' : 'Stress',
               'cell name' : 'cellID',
               'current injected' : 'currentInjected',
               'proj' : 'projID'}
    modelDF = spikeProp_DF.rename(columns=renamer)
    
    # Only run base stats models on baseline data
    modelDF_baseline = modelDF[modelDF['drug state'] == 'baseline']

    # Pooled, then one model set per age when present (see _age_strata).
    for age_tag, stratum in _age_strata(modelDF_baseline):
        for projType in stratum['projID'].unique():
            modelDF_proj = stratum[stratum['projID'] == projType]
            for col in columns_for_extraction:
                result = runModel(modelDF_proj, col)
                os.makedirs(MODELS_DIR, exist_ok=True)
                with open(os.path.join(MODELS_DIR, f"model_spikeProps_{col}_summary_{projType}{age_tag}.txt"), "w") as f:
                    if result is not None and hasattr(result, 'summary'):
                        f.write(result.summary().as_text())

                    
def plotRandomTrains(mouseList, trainsPerCluster=10):
    trainDF = pd.DataFrame()
    def sem(x):
        return x.std() / np.sqrt(len(x))

    def appendPhaseData(existingDF, spikePropDF, cell, mouse):
        if hasattr(cell, 'cluster') == False:
            return None
        cellInfo = {
            'cell name': cell.name,
            'mouse': mouse.name,
            'sex': mouse.sex,
            'stress': mouse.stressCon,
            'proj': mouse.proj,
            'cluster' : cell.cluster
        }

        # Create a DataFrame with repeated cellInfo for each row in spikePropDF
        cellInfoDF = pd.DataFrame([cellInfo] * len(spikePropDF))

        # Concatenate the cellInfoDF with spikePropDF along columns
        combinedDF = pd.concat([cellInfoDF, spikePropDF.reset_index(drop=True)], axis=1)

        # If an existing DataFrame is provided, concatenate with it
        if existingDF is not None:
            combinedDF = pd.concat([existingDF, combinedDF], ignore_index=True)

        return combinedDF
            
    for mouse in mouseList.values():
        for cell in mouse.cells:
            if mouse.proj == 'SNL':
                if hasattr(cell, 'spikeAnalysis') == False:
                    continue
                
                trainDF_cell = pd.DataFrame(columns=['sweep','current'])
                sweeps_wSpikes = cell.spikeAnalysis[(cell.spikeAnalysis['protocol'] == 'firing rate') & (
                    ~cell.spikeAnalysis['phasePlotData'].isna())]
                for index, sweep in sweeps_wSpikes.iterrows():
                    train = sweep['rawTrain']
                    trainDF_train = pd.DataFrame({'trainV': [np.transpose(train)]})
                    trainDF_train['spikeNum'] = range(1, len(trainDF_train) + 1)
                    trainDF_train['current'] = sweep['current injected']
                    trainDF_train['sweep'] = sweep['sweep']
                    
                    trainDF_cell = pd.concat([trainDF_cell, trainDF_train], ignore_index=True)


                trainDF = appendPhaseData(trainDF, trainDF_cell, cell, mouse)

    for clust in range(1, 4):
        trainDF_clust = trainDF[trainDF['cluster'] == clust]
        trainDF_clust = trainDF_clust[trainDF_clust['stress'] == 'naive']
        if trainDF_clust.empty:
            continue

        # sample up to trainsPerCluster trains
        random_rows = trainDF_clust.sample(
            n=min(trainsPerCluster, len(trainDF_clust))
        ).reset_index(drop=True)
        n_plots = len(random_rows)
        _attach_age(random_rows, mouseList)
        random_rows.to_csv(f'cluster_{clust}_randomTrains.csv', index=False)
        # make a vertical stack of subplots
        fig, axes = plt.subplots(
            n_plots, 1,
            figsize=(8, 3*n_plots),
            sharex=True, sharey=True
        )
        # if there’s only one subplot, wrap it in a list for consistent indexing
        if n_plots == 1:
            axes = [axes]

        # plot each train in its own axes
        for ax, (_, row) in zip(axes, random_rows.iterrows()):
            ax.plot(np.linspace(0, 2000, len(row['trainV'])), row['trainV'])
            ax.set_xlim(0, 2000)  # limit x-axis to 6000 ms
            ax.set_title(
                f"Group: {row['stress']}  |  "
                f"Cell: {row['cell name']}  |  "
                f"Sweep: {row['sweep']}  |  "
                f"Current: {row['current']} | "
                f"Cluster: {row['cluster']}",
            )
            ax.set_ylabel('Voltage (mV)')
            ax.grid(True)

        # only set the bottom x-label once
        axes[-1].set_xlabel('Time (ms)')

        # overall title

        fig.suptitle(f'Random Spike Trains for Cluster {clust}', y=1.02)
        plt.tight_layout()
        plt.show()
        a=1

def getSagData(mouseList):
    # Current-clamp voltage sag (per hyperpolarizing current step).
    df = extract_per_current(mouseList, attr='sagData',
                             value_col='sag',
                             csv_name='sag.csv',
                             drug_split=True)
    run_models(df, value_col='sag', variable='sag', label='voltageSag',
               model_fn=runCurrentModel,
               do_outliers=False,
               require_all_three=True)
    export_prism(df, 'sag', 'sag', 'grouped', x_col='current')


def getIhData(mouseList):
    # Voltage-clamp Ih (real Ih): one scalar per cell, the mean fit-based Ih
    # amplitude across the identical repeat sweeps (bad sweeps are NaN'd in
    # addIh and dropped by the groupby mean). The window-based metric is
    # stored as cell.IhWin and can be exported the same way for comparison.
    df = extract_scalar(mouseList, attr='Ih',
                        value_col='Ih',
                        csv_name='Ih.csv',
                        drug_split=True)
    run_models(df, value_col='Ih', variable='Ih', label='Ih',
               model_fn=runModel)
    export_prism(df, 'Ih', 'Ih', 'column')


def _plot_opto_cell(cell, mouse, out_dir, n_random=6, sr=10000):
    """
    Per-cell plot of randomly selected sweep windows colored by xcorr classification,
    plus the template overlay and a correlation score histogram.
    One subplot column per intensity. Saves PDFs to out_dir.
    """
    if not hasattr(cell, 'optoSweeps') or cell.optoSweeps is None \
            or cell.optoSweeps.empty:
        return
    if not hasattr(cell, 'optoRawWindows') or not cell.optoRawWindows:
        return

    df1         = cell.optoSweeps[(cell.optoSweeps['pulse_num'] == 1) & (cell.optoSweeps['n_pulses_in_sweep'] == 1)].copy()
    if 'intensity' not in df1.columns:
        df1['intensity'] = 'all'
    intensities = sorted(df1['intensity'].dropna().unique())
    if not intensities:
        return

    pre_samps = int(round(0.020 * sr))
    n_intens  = len(intensities)

    fig, axes = plt.subplots(1, n_intens, figsize=(4 * n_intens, 4), sharey=True)
    if n_intens == 1:
        axes = [axes]

    for ax, intens in zip(axes, intensities):
        sub        = df1[df1['intensity'] == intens]
        sweep_nums = sub['sweep'].values
        np.random.seed(42)
        selected = np.random.choice(sweep_nums,
                                    size=min(n_random, len(sweep_nums)),
                                    replace=False)

        for sw in selected:
            if sw not in cell.optoRawWindows:
                continue
            win  = cell.optoRawWindows[sw]
            t_ms = (np.arange(len(win)) - pre_samps) / sr * 1000

            row = sub[sub['sweep'] == sw]
            if row.empty or 'is_responder_xcorr' not in row.columns:
                color, alpha, lw = 'gray', 0.4, 0.8
            else:
                is_resp = row['is_responder_xcorr'].values[0]
                if is_resp is True:
                    color, alpha, lw = '#2196F3', 0.6, 1.0   # blue = success
                elif is_resp is False:
                    color, alpha, lw = '#BDBDBD', 0.4, 0.7   # gray = failure
                else:
                    color, alpha, lw = '#FF9800', 0.4, 0.7   # orange = unclassified

            ax.plot(t_ms, win, color=color, alpha=alpha, lw=lw)

        # Template overlay
        if hasattr(cell, 'templates') and cell.templates.get(intens) is not None:
            tmpl   = cell.templates[intens]
            t_tmpl = (np.arange(len(tmpl)) - pre_samps) / sr * 1000
            src    = getattr(cell, 'template_source', {}).get(intens, '?')
            ax.plot(t_tmpl, tmpl, color='black', lw=2.0,
                    label=f'template (src={src}%)')
            ax.legend(fontsize=7)

        ax.axvline(0, color='red', lw=0.8, ls='--')
        ax.set_title(f'{intens}%', fontsize=10)
        ax.set_xlabel('Time re pulse (ms)', fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel('Current - baseline (pA)', fontsize=9)

    cond = mouse.stressCon
    fig.suptitle(f'{cell.name}  |  {cond}  |  blue=success  gray=failure', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{cell.name}_sweeps_classified.pdf'),
                bbox_inches='tight')
    plt.close(fig)

    # Correlation score histogram
    if 'corr_score' in cell.optoSweeps.columns:
        scores = cell.optoSweeps[cell.optoSweeps['pulse_num'] == 1]['corr_score'].dropna()
        if len(scores) > 2:
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            ax2.hist(scores.values, bins=15, color='steelblue', edgecolor='white')
            if hasattr(cell, 'xcorr_threshold') and np.isfinite(cell.xcorr_threshold):
                ax2.axvline(cell.xcorr_threshold, color='red', lw=1.5, ls='--',
                            label=f'thresh={cell.xcorr_threshold:.2f}')
                ax2.legend(fontsize=8)
            ax2.set_xlabel('Correlation score (r)', fontsize=9)
            ax2.set_ylabel('Count', fontsize=9)
            ax2.set_title(f'{cell.name}  |  xcorr distribution', fontsize=9)
            fig2.tight_layout()
            fig2.savefig(os.path.join(out_dir, f'{cell.name}_xcorr_dist.pdf'),
                         bbox_inches='tight')
            plt.close(fig2)


def getOptoData(mouseList,
                sweeps_csv_single='opto_sweeps_single.csv',
                sweeps_csv_ppr='opto_sweeps_ppr.csv',
                grouped_csv_single='opto_grouped_single.csv',
                grouped_csv_ppr='opto_grouped_ppr.csv'):
    """
    Pull per-sweep and per-pulse-grouped opto response data off all cells that
    have run cell.addOptoResponse. Tags each row with mouse metadata, splits
    by protocol (single-pulse vs PPR via n_pulses_in_sweep), writes four CSVs,
    produces per-cell classified-sweep plots, and returns
    (sweeps_single, sweeps_ppr, grouped_single, grouped_ppr).
    """
    sweeps_rows  = []
    grouped_rows = []

    for mouse in mouseList.values():
        for cell in mouse.cells:
            meta = {
                'cell name': cell.name,
                'mouse':     mouse.name,
                'sex':       mouse.sex,
                'stress':    mouse.stressCon,
                'proj':      mouse.proj,
                'drug type': getattr(mouse, 'drugType', None),
            }

            if hasattr(cell, 'optoSweeps') and cell.optoSweeps is not None \
                    and not cell.optoSweeps.empty:
                df = cell.optoSweeps.copy()
                for k, v in meta.items():
                    df[k] = v
                sweeps_rows.append(df)

            if hasattr(cell, 'optoGrouped') and cell.optoGrouped is not None \
                    and not cell.optoGrouped.empty:
                df = cell.optoGrouped.copy()
                for k, v in meta.items():
                    df[k] = v
                grouped_rows.append(df)

    sweeps_df  = pd.concat(sweeps_rows,  ignore_index=True) if sweeps_rows  else pd.DataFrame()
    grouped_df = pd.concat(grouped_rows, ignore_index=True) if grouped_rows else pd.DataFrame()
    _attach_age(sweeps_df,  mouseList)
    _attach_age(grouped_df, mouseList)

    # Split by protocol via n_pulses_in_sweep
    if not sweeps_df.empty:
        unexpected = sweeps_df[~sweeps_df['n_pulses_in_sweep'].isin([1, 2])]
        if not unexpected.empty:
            n_uniq = unexpected[['cell name', 'sweep']].drop_duplicates().shape[0]
            print(f'[warn] {n_uniq} sweep(s) had n_pulses_in_sweep not in {{1,2}}; '
                  f'omitting from CSVs')
        sweeps_single = sweeps_df[sweeps_df['n_pulses_in_sweep'] == 1].copy()
        sweeps_ppr    = sweeps_df[sweeps_df['n_pulses_in_sweep'] == 2].copy()
        if not sweeps_single.empty:
            sweeps_single.to_csv(sweeps_csv_single, index=False)
        if not sweeps_ppr.empty:
            sweeps_ppr.to_csv(sweeps_csv_ppr, index=False)
    else:
        sweeps_single = sweeps_ppr = pd.DataFrame()

    if not grouped_df.empty:
        grouped_single = grouped_df[grouped_df['n_pulses_in_sweep'] == 1].copy()
        grouped_ppr    = grouped_df[grouped_df['n_pulses_in_sweep'] == 2].copy()
        if not grouped_single.empty:
            grouped_single.to_csv(grouped_csv_single, index=False)
        if not grouped_ppr.empty:
            grouped_ppr.to_csv(grouped_csv_ppr, index=False)
    else:
        grouped_single = grouped_ppr = pd.DataFrame()

    # ── Per-cell classified sweep plots ───────────────────────────────────────
    plot_dir = os.path.join(analysis_path, 'opto_sweep_plots')
    os.makedirs(plot_dir, exist_ok=True)
    for mouse in mouseList.values():
        for cell in mouse.cells:
            _plot_opto_cell(cell, mouse, plot_dir)

    return sweeps_single, sweeps_ppr, grouped_single, grouped_ppr
    """
    Pull per-sweep and per-pulse-grouped opto response data off all cells that
    have run cell.addOptoResponse. Tags each row with mouse metadata, splits
    by protocol (single-pulse vs PPR via n_pulses_in_sweep), writes four CSVs,
    returns (sweeps_single, sweeps_ppr, grouped_single, grouped_ppr).

    Splitting key: n_pulses_in_sweep == 1 → single, == 2 → PPR. Sweeps with
    other pulse counts (shouldn't happen with hardcoded onsets) are skipped
    with a warning.

    Per-sweep CSVs: one row per (sweep, pulse). Includes peak, latency, rise,
        decay tau, charge, is_responder, plus PPR_2vs1 on the pulse-2 rows of
        paired-pulse sweeps. Use for sweep-level stats or failure-inclusive
        means.

    Grouped CSVs: responder-only means per (cell, pulse_num, n_pulses_in_sweep
        [, intensity]), plus n_sweeps_used / n_sweeps_total counts so you can
        recover failure-inclusive means downstream. Use for cell-level summaries.

    No modeling tail here — opto stats are protocol-specific.
    """
    sweeps_rows = []
    grouped_rows = []

    for mouse in mouseList.values():
        for cell in mouse.cells:
            meta = {
                'cell name': cell.name,
                'mouse':     mouse.name,
                'sex':       mouse.sex,
                'stress':    mouse.stressCon,
                'proj':      mouse.proj,
                'drug type': getattr(mouse, 'drugType', None),
            }

            if hasattr(cell, 'optoSweeps') and cell.optoSweeps is not None \
                    and not cell.optoSweeps.empty:
                df = cell.optoSweeps.copy()
                for k, v in meta.items():
                    df[k] = v
                sweeps_rows.append(df)

            if hasattr(cell, 'optoGrouped') and cell.optoGrouped is not None \
                    and not cell.optoGrouped.empty:
                df = cell.optoGrouped.copy()
                for k, v in meta.items():
                    df[k] = v
                grouped_rows.append(df)

    sweeps_df = pd.concat(sweeps_rows, ignore_index=True) if sweeps_rows else pd.DataFrame()
    grouped_df = pd.concat(grouped_rows, ignore_index=True) if grouped_rows else pd.DataFrame()

    # Split by protocol via n_pulses_in_sweep
    if not sweeps_df.empty:
        unexpected = sweeps_df[~sweeps_df['n_pulses_in_sweep'].isin([1, 2])]
        if not unexpected.empty:
            n_uniq = unexpected[['cell name','sweep']].drop_duplicates().shape[0]
            print(f'[warn] {n_uniq} sweep(s) had n_pulses_in_sweep not in {{1,2}}; '
                  f'omitting from CSVs')
        sweeps_single = sweeps_df[sweeps_df['n_pulses_in_sweep'] == 1].copy()
        sweeps_ppr    = sweeps_df[sweeps_df['n_pulses_in_sweep'] == 2].copy()
        if not sweeps_single.empty:
            sweeps_single.to_csv(sweeps_csv_single, index=False)
        if not sweeps_ppr.empty:
            sweeps_ppr.to_csv(sweeps_csv_ppr, index=False)
    else:
        sweeps_single = sweeps_ppr = pd.DataFrame()

    if not grouped_df.empty:
        grouped_single = grouped_df[grouped_df['n_pulses_in_sweep'] == 1].copy()
        grouped_ppr    = grouped_df[grouped_df['n_pulses_in_sweep'] == 2].copy()
        if not grouped_single.empty:
            grouped_single.to_csv(grouped_csv_single, index=False)
        if not grouped_ppr.empty:
            grouped_ppr.to_csv(grouped_csv_ppr, index=False)
    else:
        grouped_single = grouped_ppr = pd.DataFrame()

def getMCurrentData(mouseList,
                    sweeps_csv='mCurrent_sweeps.csv',
                    grouped_csv='mCurrent_grouped.csv'):
    """
    Pull per-sweep M-current data off all cells. Tags each row with mouse metadata
    and drug state, then groups them by (cell, drug state, step voltage).
    """
    sweeps_rows = []

    for mouse in mouseList.values():
        for cell in mouse.cells:
            meta = {
                'cell name': cell.name,
                'mouse':     mouse.name,
                'sex':       mouse.sex,
                'stress':    mouse.stressCon,
                'proj':      mouse.proj,
                'drug type': getattr(mouse, 'drugType', None),
            }

            if hasattr(cell, 'mCurrentSweeps') and cell.mCurrentSweeps is not None \
                    and not cell.mCurrentSweeps.empty:
                df = cell.mCurrentSweeps.copy()
                # Tag drug state per sweep via the standard drugSweeps mechanism
                df['drug state'] = df['sweep'].apply(lambda s: _drug_state(s, mouse))
                
                # Add all mouse-level metadata
                for k, v in meta.items():
                    df[k] = v
                sweeps_rows.append(df)

    sweeps_df = pd.concat(sweeps_rows, ignore_index=True) if sweeps_rows else pd.DataFrame()
    _attach_age(sweeps_df, mouseList)

    if not sweeps_df.empty:
        sweeps_df.to_csv(sweeps_csv, index=False)
        
        # Now that drug state and metadata are tagged, calculate the grouped DF here
        metric_cols = ['holding_current_pA', 'inst_current_pA', 'ss_current_pA', 
                       'relaxation_pA', 'deact_tau_ms', 'tail_amp_pA', 'tail_tau_ms', 
                       'conductance_nS', 'cell_capacitance_pF', 'relaxation_pA_per_pF',
                       'holding_current_pA_per_pF', 'inst_current_pA_per_pF',
                       'ss_current_pA_per_pF', 'tail_amp_pA_per_pF']
        
        # Round the voltage to the nearest mV to prevent fragmentation
        sweeps_df['step_voltage_mV_rounded'] = sweeps_df['step_voltage_mV'].round(0)
        
        group_keys = ['cell name', 'mouse', 'sex', 'stress', 'proj', 
                      'drug type', 'drug state', 'step_voltage_mV_rounded']
        
        # Group to get means
        grouped_df = sweeps_df.groupby(group_keys, dropna=False)[metric_cols].mean().reset_index()
        
        # Group to get n_sweeps
        n_sweeps = sweeps_df.groupby(group_keys, dropna=False).size().reset_index(name='n_sweeps')
        grouped_df = grouped_df.merge(n_sweeps, on=group_keys, how='outer')

        _attach_age(grouped_df, mouseList)
        grouped_df.to_csv(grouped_csv, index=False)
    else:
        grouped_df = pd.DataFrame()

    return sweeps_df, grouped_df
    

# %%
ephysMouseDict = excludeCells(ephysMouseDict, 20)
# getInputResistance(ephysMouseDict)
# getOptoData(ephysMouseDict)
# getCapacitance(ephysMouseDict)
getFiringRateData(ephysMouseDict)
# getAccessResistance(ephysMouseDict)
# getTrainProps(ephysMouseDict)
# getSagData(ephysMouseDict)
# getIhData(ephysMouseDict)
# getRMPData(ephysMouseDict)
# getRheobase(ephysMouseDict)
# getSpikeProps(ephysMouseDict)

# getMCurrentData(ephysMouseDict)
# a = ['mid', 'all', 'first']
# a = ['first',  'all']
# b = ['all', [100]]
# for kind in a:
#     for curr in b:
#         getPhasePlot_RawSweeps(ephysMouseDict, kind, curr)

spike_df, cell_df, pca, results = getPhasePlotV3(
    ephysMouseDict,
    qc_n_sd=1.5,
    output_dir=analysis_path,
    do_pca=False,
)