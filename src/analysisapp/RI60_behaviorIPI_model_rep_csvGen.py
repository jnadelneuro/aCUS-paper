"""
Dump Prism-ready CSVs for the schematic using pooled data + global mean pi.
Run from the behaviormodels folder.
"""
import pandas as pd, numpy as np
from scipy.stats import lognorm
import os
from _config import RI60_DIR

os.chdir(os.path.join(RI60_DIR, "behaviormodels"))

FEATHER_RI = os.path.join(RI60_DIR, "RI30_RI60_dataFrame.feather")

cp = pd.read_csv('component_params.csv')
beh = pd.read_feather(FEATHER_RI)

mu = cp['mu_log'].values
sigma = cp['sigma_log'].values
K = len(mu)

sess = pd.read_csv('session_weights.csv')
weights = sess['n_irts'].values
pi_vec = np.average(
    sess[[f'pi_{k}' for k in range(K)]].values,
    weights=weights,
    axis=0,
)
print(f"IRT-weighted mean pi: {pi_vec.round(3)}")

# ---- Pool all RI60 IRTs (days 1-14) ----
all_irts = []
for _, row in beh[beh['sesType'] == 'RI60'].iterrows():
    try:
        day = int(float(row['dayOnType']))
    except:
        continue
    if day < 1 or day > 14:
        continue
    ts = np.sort(np.array(row['allPokeTimestamps']))
    if len(ts) < 30:
        continue
    irts = np.diff(ts)
    irts = irts[irts > 0.001]
    all_irts.append(irts)
all_irts = np.concatenate(all_irts)
print(f"Pooled {len(all_irts):,} IRTs")

# ---- Histogram (log-spaced bins) ----
bins = np.logspace(np.log10(0.05), np.log10(max(all_irts.max() * 1.1, 1)), 70)
counts, edges = np.histogram(all_irts, bins=bins)
widths = np.diff(edges)
centers = 0.5 * (edges[:-1] + edges[1:])
density = counts / (counts.sum() * widths)

# ---- Fit curves on smooth grid ----
xg = np.logspace(np.log10(0.05), np.log10(300), 500)
total_fit = np.zeros_like(xg)
comp_curves = {}
for k in range(K):
    y = pi_vec[k] * lognorm.pdf(xg, s=sigma[k], scale=np.exp(mu[k]))
    comp_curves[k] = y
    total_fit += y

# ---- Prism CSV 1: histogram ----
pd.DataFrame({
    'IRT_s': centers,
    'density': density,
}).to_csv('schematic_histogram.csv', index=False)

# ---- Prism CSV 2: fit curves ----
fit_df = pd.DataFrame({'IRT_s': xg, 'total_fit': total_fit})
for k in range(K):
    fit_df[f'c{k}_{np.exp(mu[k]):.2f}s'] = comp_curves[k]
fit_df.to_csv('schematic_fit_curves.csv', index=False)

# ---- Prism CSV 3: pi bar ----
pd.DataFrame({
    f'c{k}_{np.exp(mu[k]):.2f}s': [float(pi_vec[k])] for k in range(K)
}).to_csv('schematic_pi_bar.csv', index=False)

print("Wrote schematic_histogram.csv, schematic_fit_curves.csv, schematic_pi_bar.csv")

"""
Heatmap CSV for schematic: one row per session, five pi columns,
sorted by group then c4 weight. Run from behaviormodels folder.
"""
sess = pd.read_csv('session_weights.csv')
K = 5

# Sort: group (naive first), then c4 descending within group
sess['group_order'] = sess['group'].map({'naive': 0, 'stress': 1})
sess = sess.sort_values(['group_order', 'pi_4'], ascending=[True, False]).reset_index(drop=True)

# Build the heatmap CSV
heatmap = pd.DataFrame({
    'row':   range(len(sess)),
    'group': sess['group'].values,
    'mouse': sess['mouse'].values,
    'day':   sess['day'].values,
})
for k in range(K):
    heatmap[f'pi_{k}'] = sess[f'pi_{k}'].values

heatmap.to_csv('schematic_heatmap.csv', index=False)
print(f"Wrote schematic_heatmap.csv ({len(heatmap)} rows)")
print(f"Rows 0-{(sess['group']=='naive').sum()-1}: naive (sorted by pi_4 descending)")
print(f"Rows {(sess['group']=='naive').sum()}-{len(sess)-1}: stress (sorted by pi_4 descending)")

"""
Fit-curve CSVs for K=2..4 overlaid on the pooled IRT histogram.
Run from the k_select folder (or adjust paths).
"""
import pandas as pd, numpy as np
from scipy.stats import lognorm
import os

os.chdir(os.path.join(RI60_DIR, "behaviormodels", "k_select"))

comp = pd.read_csv('K_sweep_components.csv')
sess = pd.read_csv('K_sweep_session_pis.csv')

xg = np.logspace(np.log10(0.05), np.log10(300), 500)

for K in [2, 3, 4]:
    ck = comp[comp['K'] == K].sort_values('component').reset_index(drop=True)
    sk = sess[sess['K'] == K]

    # IRT-weighted mean pi
    pi_cols = [f'pi_{k}' for k in range(K)]
    pi_vec = np.average(sk[pi_cols].values, weights=sk['n_irts'].values, axis=0)

    mu = ck['mu_log'].values
    sigma = ck['sigma_log'].values

    total_fit = np.zeros_like(xg)
    fit_df = pd.DataFrame({'IRT_s': xg})
    for k in range(K):
        y = pi_vec[k] * lognorm.pdf(xg, s=sigma[k], scale=np.exp(mu[k]))
        fit_df[f'c{k}_{np.exp(mu[k]):.2f}s'] = y
        total_fit += y
    fit_df['total_fit'] = total_fit

    fit_df.to_csv(f'fit_curves_K{K}.csv', index=False)
    print(f"K={K}: pi={pi_vec.round(3)}, wrote fit_curves_K{K}.csv")