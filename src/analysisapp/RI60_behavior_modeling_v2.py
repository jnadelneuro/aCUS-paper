"""
RI60 microstructure pipeline. K=5, days 1..MAX_DAY_INCLUSIVE.
One file, VSCode play button. No shock analysis.

Stages (skip if all outputs exist; force via FORCE_REDO; forcing cascades):
  1. fit_main      -> hier_mix_results.pkl, component_params.csv, session_weights.csv
  2. regressions   -> group_effects.csv, interaction_disengagement.csv,
                      stats_group_effects.txt, stats_interaction.txt,
                      stats_time_share.txt
  3. fit_quality   -> fit_quality.csv
  4. trajectories  -> per_mouse_trajectories.csv
  5. plot_data     -> prism_density_data.csv, prism_density_fit.csv,
                      prism_weights_by_group.csv, prism_disengagement_by_day.csv,
                      prism_time_share_by_group.csv,
                      density_plot_loglog.png, density_plot_linear.png,
                      weights_plot.png, disengagement_by_day_plot.png,
                      time_share_plot.png
  6. time_to_RePE  -> photoDF_R_with_weights_timeToRePE.feather
                      (adds per-ReNP retrieval latency from raw RI30_RI60 arrays)

Log: hier_mix.log
"""
import numpy as np
import pandas as pd
from itertools import groupby as igroupby
import statsmodels.api as sm_api
import pickle, time, sys, os, warnings
from scipy.special import logsumexp
from scipy.stats import lognorm, ttest_ind
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from _config import RI60_DIR

# ===========================================================
# Configuration
# ===========================================================
os.chdir(os.path.join(RI60_DIR, "behaviormodels"))
FEATHER_RI = os.path.join(RI60_DIR, "RI30_RI60_dataFrame.feather")
FEATHER_PHOTO_IN  = os.path.join(RI60_DIR, "photoDF_R_with_rates.feather")
FEATHER_PHOTO_OUT = os.path.join(RI60_DIR, "photoDF_R_with_weights.feather")
FEATHER_PHOTO_TTR = os.path.join(RI60_DIR, "photoDF_R_with_weights_timeToRePE.feather")
# Second photometry frame (same events, different trace column 'photoTrace');
# bout labels are written here too for RI60_Photo_bootstrapCIs.py.
FEATHER_PHOTODF   = os.path.join(RI60_DIR, "photoDataFrame.feather")
# Per-mouse shock counts (one row per mouse on the shock-induction day).
FEATHER_SHOCK     = os.path.join(RI60_DIR, "Shock_dataFrame.feather")


K_MAIN = 5
MAX_DAY_INCLUSIVE = 14
LATE_TRAINING_LAST_N = 5

# ---- Force re-runs. Empty = run only missing. ----
# Valid: 'fit_main','regressions','fit_quality','trajectories','plot_data', photo_merge, time_to_RePE
FORCE_REDO = ['fit_main', 'reward_aligned', 'regressions', 'fit_quality', 'trajectories', 'plot_data', 'photo_merge', 'bout_assign', 'time_to_RePE']

LOG_2PI = np.float32(np.log(2*np.pi))
COL_NAIVE  = '#3b6db5'
COL_STRESS = '#c83737'

# ===========================================================
# Logger
# ===========================================================
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s); st.flush()
    def flush(self):
        for st in self.streams: st.flush()

# ===========================================================
# EM core
# ===========================================================
def _em_step(all_x, all_j, ns_j, mu, sigma, pi, J, K):
    log_sigma = np.log(sigma)
    z = (all_x[:,None] - mu[None,:]) / sigma[None,:]
    log_comp = -0.5*LOG_2PI - log_sigma[None,:] - 0.5*z*z
    log_pi_pp = np.log(pi[all_j] + 1e-30)
    log_joint = log_pi_pp + log_comp
    log_marg = logsumexp(log_joint, axis=1)
    total_ll = float(log_marg.sum())
    r = np.exp(log_joint - log_marg[:,None])
    Nk = r.sum(0) + 1e-12
    mu_new = np.einsum('nk,n->k', r, all_x) / Nk
    diff = all_x[:,None] - mu_new[None,:]
    var = np.einsum('nk,nk->k', r, diff*diff) / Nk
    sigma_new = np.sqrt(var + 1e-6)
    pi_new = np.empty((J, K), dtype=np.float32)
    for k in range(K):
        pi_new[:,k] = np.bincount(all_j, weights=r[:,k], minlength=J)
    pi_new /= ns_j[:,None]
    np.clip(pi_new, 1e-6, 1.0, out=pi_new)
    pi_new /= pi_new.sum(1, keepdims=True)
    return mu_new.astype(np.float32), sigma_new.astype(np.float32), pi_new, total_ll

def _run_em(all_x, all_j, ns_j, mu, sigma, pi, J, K, max_iter=300, tol=1e-7):
    prev = -np.inf
    for it in range(max_iter):
        mu, sigma, pi, ll = _em_step(all_x, all_j, ns_j, mu, sigma, pi, J, K)
        if abs(ll-prev) < tol*(abs(prev)+1): break
        prev = ll
    return mu, sigma, pi, ll, it+1

def fit_hier_ln_mixture(irts_list, K, n_restarts=10, restart_iter=80,
                         polish_iter=400, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    xs=[]; js=[]
    for j, irts in enumerate(irts_list):
        xs.append(np.log(irts).astype(np.float32))
        js.append(np.full(len(irts), j, dtype=np.int32))
    all_x = np.concatenate(xs); all_j = np.concatenate(js)
    J = len(irts_list); N = len(all_x)
    ns_j = np.array([len(x) for x in irts_list], dtype=np.float32)
    best_init = None; best_ll_w = -np.inf
    for rs in range(n_restarts):
        qs = np.sort(rng.uniform(0.05, 0.95, size=K))
        mu0 = np.quantile(all_x, qs).astype(np.float32)
        sigma0 = (np.float32(0.6) + rng.uniform(-0.1, 0.1, K).astype(np.float32))
        pi0 = np.full((J, K), 1.0/K, dtype=np.float32)
        mu, sigma, pi, ll, nit = _run_em(all_x, all_j, ns_j, mu0, sigma0, pi0,
                                         J, K, max_iter=restart_iter, tol=1e-5)
        if verbose:
            print(f"  [K={K}] restart {rs}: it={nit}  ll={ll:.1f}", flush=True)
        if ll > best_ll_w:
            best_ll_w = ll
            best_init = (mu.copy(), sigma.copy(), pi.copy())
    mu, sigma, pi = best_init
    mu, sigma, pi, ll, nit = _run_em(all_x, all_j, ns_j, mu, sigma, pi,
                                     J, K, max_iter=polish_iter, tol=1e-7)
    if verbose:
        print(f"  [K={K}] polish: it={nit}  ll={ll:.1f}", flush=True)
    n_params = 2*K + J*(K-1)
    fit = dict(mu=mu, sigma=sigma, pi=pi, log_lik=ll,
               bic=n_params*np.log(N) - 2*ll, aic=2*n_params - 2*ll,
               K=K, J=J, N=N, n_params=n_params)
    order = np.argsort(fit['mu'])
    fit['mu']    = fit['mu'][order]
    fit['sigma'] = fit['sigma'][order]
    fit['pi']    = fit['pi'][:, order]
    return fit

# ===========================================================
# Data loading
# ===========================================================
def load_ri60():
    ri = pd.read_feather(FEATHER_RI)
    d = ri[ri['sesType']=='RI60'].copy().reset_index(drop=True)
    n_total = len(d)
    irts_list=[]; meta=[]
    n_dropped_day = 0
    for _, row in d.iterrows():
        try: day = int(float(row['dayOnType']))
        except: day = -1
        if MAX_DAY_INCLUSIVE is not None and (day < 1 or day > MAX_DAY_INCLUSIVE):
            n_dropped_day += 1
            continue
        ts = np.sort(np.array(row['allPokeTimestamps']))
        if len(ts) < 30: continue
        irts = np.diff(ts); irts = irts[irts > 0.001]
        if len(irts) < 100: continue
        irts_list.append(irts)
        meta.append(dict(
            session_idx=len(irts_list)-1,
            mouse=row['mouse'], group=row['group'], sex=row['sex'],
            day=day, date=row['date'], n_irts=len(irts),
        ))
    if MAX_DAY_INCLUSIVE is not None:
        print(f"Day filter: keeping dayOnType in [1,{MAX_DAY_INCLUSIVE}]; "
              f"dropped {n_dropped_day} sessions outside window "
              f"(of {n_total} total RI60 sessions)", flush=True)
    return irts_list, pd.DataFrame(meta)

# ===========================================================
# Per-session fit quality
# ===========================================================
def fit_quality_per_session(irts_list, meta_df, fit, outpath='fit_quality.csv'):
    K = fit['K']
    rows = []
    for sidx, irts in enumerate(irts_list):
        x = np.sort(irts)
        F_emp = np.arange(1, len(x)+1) / len(x)
        F_fit = np.zeros_like(x, dtype=float)
        pdf_mix = np.zeros_like(x, dtype=float)
        for k in range(K):
            pk = float(fit['pi'][sidx, k])
            s  = float(fit['sigma'][k])
            sc = np.exp(float(fit['mu'][k]))
            F_fit   += pk * lognorm.cdf(x, s=s, scale=sc)
            pdf_mix += pk * lognorm.pdf(x, s=s, scale=sc)
        ks = float(np.max(np.abs(F_emp - F_fit)))
        mean_ll = float(np.mean(np.log(pdf_mix + 1e-30)))
        row = meta_df.iloc[sidx].to_dict()
        row.update(dict(ks_distance=ks, mean_loglik_per_poke=mean_ll, n_irts=len(x)))
        for k in range(K):
            row[f'pi_{k}'] = float(fit['pi'][sidx, k])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(outpath, index=False)
    print(f"Wrote {outpath}  (n={len(df)})", flush=True)
    print("KS distance by group:", flush=True)
    print(df.groupby('group')['ks_distance']
          .describe()[['mean','50%','75%','max']].round(4))
    return df

# ===========================================================
# Regressions
# ===========================================================
def logit(p, eps=1e-3):
    p = np.clip(p, eps, 1-eps)
    return np.log(p/(1-p))

def fit_lmm_stable(formula, data, groups):
    methods = ['powell', 'lbfgs', 'bfgs']
    last_err = None
    for method in methods:
        try:
            m = smf.mixedlm(formula, data, groups=groups)
            r = m.fit(reml=True, method=method)
            if np.all(np.isfinite(r.bse.values)) and r.bse.max() < 1e3:
                return r, method
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All optimizers failed; last err: {last_err}")

def run_regressions(sess_df, K, label=''):
    sess_df = sess_df.copy()
    sess_df['group'] = pd.Categorical(sess_df['group'], categories=['naive','stress'])
    rows = []
    for k in range(K):
        col = f'pi_{k}'
        sess_df[f'logit_{col}'] = logit(sess_df[col].values)
        try:
            r, method = fit_lmm_stable(
                f'logit_{col} ~ C(group) + day',
                sess_df, groups=sess_df['mouse'])
            coef = [n for n in r.params.index if 'group' in n][0]
            beta = float(r.params[coef]); se = float(r.bse[coef])
            pval = float(r.pvalues[coef])
            day_b = float(r.params.get('day', np.nan))
            day_p = float(r.pvalues.get('day', np.nan))
            med_day = sess_df['day'].median()
            inter = float(r.params['Intercept'])
            mu_n = inter + day_b * med_day
            mu_s = inter + day_b * med_day + beta
            pi_n = 1/(1+np.exp(-mu_n)); pi_s = 1/(1+np.exp(-mu_s))
            rows.append(dict(
                K_label=label, component=k,
                group_beta_logit=beta, group_se=se, group_p=pval,
                day_beta=day_b, day_p=day_p,
                pi_naive_at_median_day=pi_n,
                pi_stress_at_median_day=pi_s,
                method_used=method,
            ))
        except Exception as e:
            rows.append(dict(K_label=label, component=k, group_beta_logit=np.nan,
                             error=str(e)))
    return pd.DataFrame(rows)

def run_interaction_test(sess_df, comp_idx, label=''):
    sess_df = sess_df.copy()
    sess_df['group'] = pd.Categorical(sess_df['group'], categories=['naive','stress'])
    col = f'pi_{comp_idx}'
    sess_df[f'logit_{col}'] = logit(sess_df[col].values)
    r, method = fit_lmm_stable(
        f'logit_{col} ~ C(group) * day',
        sess_df, groups=sess_df['mouse'])
    print(f"\n--- Group x Day interaction on component {comp_idx} ({label}) ---", flush=True)
    print(f"  optimizer: {method}", flush=True)
    print(r.summary().tables[1], flush=True)
    out = []
    for name in r.params.index:
        out.append(dict(
            term=name, beta=float(r.params[name]),
            se=float(r.bse[name]), p=float(r.pvalues[name]),
            method_used=method,
        ))
    return pd.DataFrame(out), r.summary().as_text(), method

# ===========================================================
# Per-mouse trajectories
# ===========================================================
def per_mouse_trajectories(sess_df, K, last_n=LATE_TRAINING_LAST_N):
    rows = []
    for mouse, gdf in sess_df.groupby('mouse'):
        gdf = gdf.sort_values('day')
        late = gdf.tail(last_n)
        out = dict(
            mouse=mouse, group=gdf['group'].iloc[0], sex=gdf['sex'].iloc[0],
            n_sessions=len(gdf), n_late=len(late),
            late_n_irts_mean=float(late['n_irts'].mean()),
            mean_late_day=float(late['day'].mean()),
        )
        for k in range(K):
            col = f'pi_{k}'
            out[f'late_mean_{col}'] = float(late[col].mean())
            x = gdf['day'].values.astype(float)
            y = logit(gdf[col].values)
            if len(x) >= 3 and np.std(x) > 0:
                slope, _ = np.polyfit(x, y, 1)
                out[f'slope_logit_{col}'] = float(slope)
            else:
                out[f'slope_logit_{col}'] = np.nan
        rows.append(out)
    return pd.DataFrame(rows)

# ===========================================================
# Stage plumbing
# ===========================================================
STAGE_OUTPUTS = {
    'fit_main':     ['hier_mix_results.pkl','component_params.csv','session_weights.csv'],
    'regressions':  ['group_effects.csv','interaction_disengagement.csv',
                     'stats_group_effects.txt','stats_interaction.txt',
                     'stats_time_share.txt',
                     'stats_between_bout_composition.csv',
                     'stats_between_bout_composition.txt'],
    'fit_quality':  ['fit_quality.csv'],
    'trajectories': ['per_mouse_trajectories.csv'],
    'plot_data':    ['prism_density_data.csv','prism_density_fit.csv',
                     'prism_weights_by_group.csv',
                     'prism_disengagement_by_day.csv',
                     'prism_time_share_by_group.csv',
                     'prism_engaged_pause_frac_by_day.csv',
                     'prism_engaged_pause_frac_total.csv',
                     'density_plot_loglog.png','density_plot_linear.png',
                     'weights_plot.png','disengagement_by_day_plot.png',
                     'time_share_plot.png'],
    'reward_aligned': ['component_reward_aligned.png',
                   'prism_RA_bout_length.csv', 'prism_RA_interbout_interval.csv',
                   'prism_RA_frac_disengaged.csv', 'prism_RA_frac_within_bout.csv',
                   'prism_RA_med_within_bout.csv', 'prism_RA_med_disengaged.csv',
                   'prism_RA_all_component_fractions.csv',
                   'prism_RA_between_bout_stats.csv',
                   'prism_RA_bout_length_by_group.csv',
                   'prism_RA_interbout_by_group.csv',
                   'prism_RA_frac_disengaged_by_group.csv',
                   'prism_RA_n_sessions.csv', 'prism_RA_session_bout_length.csv',
                    'prism_RA_session_interbout.csv',
                    'prism_RA_mouse_bout_length.csv',
                    'prism_RA_mouse_interbout.csv', 'stats_reward_aligned_lmm.csv'],
    'photo_merge':  [FEATHER_PHOTO_OUT],
    'bout_assign':  ['psth_bout_status_TS_naive.png',
                     'psth_bout_status_TS_stress.png',
                     'prism_psth_bout_status_TS_naive.csv',
                     'prism_psth_bout_status_TS_stress.csv'],
    'time_to_RePE': [FEATHER_PHOTO_TTR],
}

DOWNSTREAM = {
    'fit_main':     ['regressions','fit_quality','trajectories','plot_data','photo_merge'],
    'regressions':  ['plot_data'],
    'fit_quality':  [],
    'trajectories': [],
    'plot_data':    [],
    'photo_merge':  ['bout_assign', 'time_to_RePE'],
    'bout_assign':  [],
    'reward_aligned': [],
    'time_to_RePE': [],
}

def stage_needs_run(stage, force_set):
    if stage in force_set: return True
    return any(not os.path.exists(p) for p in STAGE_OUTPUTS[stage])

def expand_force(force_list):
    out = set(force_list)
    changed = True
    while changed:
        changed = False
        for s in list(out):
            for d in DOWNSTREAM.get(s, []):
                if d not in out:
                    out.add(d); changed = True
    return out

# ===========================================================
# Stage: fit_main
# ===========================================================
def stage_fit_main(state):
    print(f"\n=== Stage: fit_main (K={K_MAIN}) ===", flush=True)
    if 'irts_list' not in state:
        state['irts_list'], state['meta_df'] = load_ri60()
    irts_list = state['irts_list']; meta_df = state['meta_df']
    print(f"Sessions: {len(irts_list)}  mice: {meta_df['mouse'].nunique()}  "
          f"total IRTs: {sum(len(x) for x in irts_list):,}", flush=True)
    print(f"Per group: {meta_df.groupby('group').size().to_dict()}", flush=True)
    print(f"Day range: {meta_df['day'].min()}..{meta_df['day'].max()}", flush=True)
    t0 = time.time()
    fit = fit_hier_ln_mixture(irts_list, K=K_MAIN, n_restarts=10,
                              restart_iter=80, polish_iter=400, seed=K_MAIN*101)
    print(f"K={K_MAIN}: BIC={fit['bic']:.1f}  ll={fit['log_lik']:.1f}  "
          f"({time.time()-t0:.1f}s)", flush=True)
    for k in range(K_MAIN):
        print(f"  c{k}: mean_IRT={np.exp(fit['mu'][k]):7.3f}s  "
              f"sigma_log={fit['sigma'][k]:.3f}  "
              f"mean_pi={fit['pi'][:,k].mean():.3f}", flush=True)

    sess_df = meta_df.copy()
    for k in range(K_MAIN):
        sess_df[f'pi_{k}'] = fit['pi'][:,k]
    sess_df.to_csv('session_weights.csv', index=False)

    comp_df = pd.DataFrame(dict(
        component=np.arange(K_MAIN),
        mu_log=fit['mu'], mean_IRT_s=np.exp(fit['mu']),
        sigma_log=fit['sigma'], mean_pi=fit['pi'].mean(0),
    ))
    comp_df.to_csv('component_params.csv', index=False)

    bundle = dict(K_main=K_MAIN, fit=fit, meta_df=meta_df, sess_df=sess_df,
                  comp_df=comp_df, max_day_inclusive=MAX_DAY_INCLUSIVE)
    with open('hier_mix_results.pkl','wb') as f:
        pickle.dump(bundle, f)
    state['fit_main'] = fit
    state['sess_df']  = sess_df
    state['comp_df']  = comp_df
    state['bundle']   = bundle

def _ensure_fit_loaded(state):
    if 'fit_main' in state and 'sess_df' in state: return
    with open('hier_mix_results.pkl','rb') as f:
        bundle = pickle.load(f)
    state['bundle']  = bundle
    state['fit_main']= bundle['fit']
    state['sess_df'] = bundle['sess_df']
    state['meta_df'] = bundle['meta_df']
    state['comp_df'] = bundle['comp_df']

# ===========================================================
# Stage: regressions
# ===========================================================
def stage_regressions(state):
    print("\n=== Stage: regressions ===", flush=True)
    _ensure_fit_loaded(state)
    fit = state['fit_main']; sess_df = state['sess_df']
    K = fit['K']

    # --- Test 1: per-component group + day LMM ---
    print(f"\n--- Mixed-effects regressions at K={K} ---", flush=True)
    reg_df = run_regressions(sess_df, K, label=f'K={K}')
    reg_df['comp_mean_IRT_s'] = [float(np.exp(fit['mu'][k])) for k in reg_df['component']]
    reg_df.to_csv('group_effects.csv', index=False)
    print(reg_df.round(4).to_string(index=False))

    with open('stats_group_effects.txt','w',encoding='utf-8') as f:
        f.write("TEST 1: Per-component group effect on poke-share weights\n")
        f.write("="*72 + "\n")
        f.write("Model: logit(pi_k) ~ C(group) + day + (1|mouse)\n")
        f.write("Fitter: statsmodels MixedLM, REML, optimizer fallback powell/lbfgs/bfgs\n")
        f.write("Reference group: naive\n\n")
        f.write(f"Day filter: dayOnType in [1, {MAX_DAY_INCLUSIVE}]\n")
        f.write(f"Sessions: {len(sess_df)}  Mice: {sess_df['mouse'].nunique()}  "
                f"Per group: {sess_df.groupby('group').size().to_dict()}\n\n")
        f.write("Component parameters (global mu, sigma, mean pi):\n")
        f.write(state['comp_df'].round(4).to_string(index=False) + "\n\n")
        f.write("Regression results:\n")
        f.write(reg_df.round(4).to_string(index=False) + "\n\n")
        sig = reg_df[reg_df['group_p']<0.05]
        if len(sig):
            f.write("Components with p<0.05 group effect: "
                    + ", ".join(f"c{int(c)}" for c in sig['component']) + "\n")
        else:
            f.write("No components significant at p<0.05\n")

    # --- Test 2: Group x day interaction on disengagement ---
    diseng_idx = int(np.argmax(fit['mu']))
    inter_df, inter_summary, inter_method = run_interaction_test(
        sess_df, diseng_idx, label=f'K={K} c{diseng_idx}')
    inter_df.to_csv('interaction_disengagement.csv', index=False)

    with open('stats_interaction.txt','w',encoding='utf-8') as f:
        f.write(f"TEST 2: Group x day interaction on disengagement (c{diseng_idx})\n")
        f.write("="*72 + "\n")
        f.write(f"Component c{diseng_idx}: mean_IRT = "
                f"{np.exp(fit['mu'][diseng_idx]):.2f}s, "
                f"sigma_log = {fit['sigma'][diseng_idx]:.3f}\n\n")
        f.write("Model: logit(pi_cDISENG) ~ C(group) * day + (1|mouse)\n")
        f.write(f"Optimizer used: {inter_method}\n")
        f.write("Reference group: naive\n\n")
        f.write(inter_summary + "\n\n")
        f.write("Term interpretation:\n")
        f.write("  Intercept        = logit(pi_cDISENG) for naive at day=0\n")
        f.write("  C(group)[stress] = additive shift for stress at day=0\n")
        f.write("  day              = per-day slope for naive\n")
        f.write("  group:day        = ADDITIONAL per-day slope for stress\n")
        f.write("                     (negative = stress disengages less over time)\n")

    # --- Test 3: Per-mouse time-share Welch t-test ---
    e_irt = np.exp(np.array(fit['mu']) + np.array(fit['sigma'])**2 / 2)
    pi_mat = sess_df[[f'pi_{k}' for k in range(K)]].values
    time_unnorm = pi_mat * e_irt[None,:]
    time_share = time_unnorm / time_unnorm.sum(1, keepdims=True)
    for k in range(K):
        sess_df[f'tshare_{k}'] = time_share[:,k]
    state['sess_df'] = sess_df

    mouse_ts = sess_df.groupby(['mouse','group'])[
        [f'tshare_{k}' for k in range(K)]].mean().reset_index()
    ts_rows = []
    for k in range(K):
        n = mouse_ts[mouse_ts['group']=='naive'][f'tshare_{k}']
        s = mouse_ts[mouse_ts['group']=='stress'][f'tshare_{k}']
        t = ttest_ind(n, s, equal_var=False)
        ts_rows.append(dict(
            component=k,
            comp_mean_IRT_s=float(np.exp(fit['mu'][k])),
            n_mice_naive=len(n), n_mice_stress=len(s),
            mean_tshare_naive=float(n.mean()),
            sem_tshare_naive=float(n.sem()),
            mean_tshare_stress=float(s.mean()),
            sem_tshare_stress=float(s.sem()),
            diff_stress_minus_naive=float(s.mean()-n.mean()),
            welch_t=float(t.statistic),
            welch_p=float(t.pvalue),
        ))
    ts_stats = pd.DataFrame(ts_rows)

    with open('stats_time_share.txt','w',encoding='utf-8') as f:
        f.write("TEST 3: Per-mouse time-share comparison (cross-check of Test 1)\n")
        f.write("="*72 + "\n")
        f.write("Metric: tshare_k = pi_k * E[IRT|k] / sum_j(pi_j * E[IRT|j])\n")
        f.write("        where E[IRT|k] = exp(mu_k + sigma_k^2 / 2) is the lognormal mean.\n")
        f.write("Aggregation: average tshare within mouse, then Welch (unequal var)\n")
        f.write("             t-test between groups.\n\n")
        f.write("NOTE: This is NOT the primary test. Test 1 (poke-share LMM) is\n")
        f.write("      the formal hypothesis test because it uses every session as\n")
        f.write("      an observation. The time-share metric is a biological\n")
        f.write("      interpretability aid: it translates the poke-share result\n")
        f.write("      into the fraction of session wall-clock time spent in each\n")
        f.write("      behavioral mode. The two metrics answer the same question\n")
        f.write("      with different sensitivity and should agree in direction.\n\n")
        f.write(ts_stats.round(4).to_string(index=False) + "\n")

    ts_stats.to_csv('time_share_stats.csv', index=False)

    # --- Test 4: Between-bout composition: c3/(c3+c4) ---
    diseng_idx = int(np.argmax(fit['mu']))
    bb_idx = diseng_idx - 1  # c3
    sess_df['frac_engaged_pause'] = (
        sess_df[f'pi_{bb_idx}'] /
        (sess_df[f'pi_{bb_idx}'] + sess_df[f'pi_{diseng_idx}'] + 1e-10)
    )
    sess_df['logit_fep'] = logit(sess_df['frac_engaged_pause'].values)
    r_fep, method_fep = fit_lmm_stable(
        'logit_fep ~ C(group) * day', sess_df, groups=sess_df['mouse'])
    print(f"\n--- Between-bout composition: c3/(c3+c4) ---", flush=True)
    print(r_fep.summary().tables[1], flush=True)

    fep_rows = []
    for name in r_fep.params.index:
        fep_rows.append(dict(
            term=name, beta=float(r_fep.params[name]),
            se=float(r_fep.bse[name]), p=float(r_fep.pvalues[name]),
            method_used=method_fep,
        ))
    pd.DataFrame(fep_rows).to_csv('stats_between_bout_composition.csv', index=False)

    with open('stats_between_bout_composition.txt', 'w', encoding='utf-8') as f:
        f.write("TEST 4: Between-bout composition c3/(c3+c4)\n")
        f.write("=" * 72 + "\n")
        f.write("Metric: fraction of non-bout time in 'engaged pause' (c3)\n")
        f.write("        vs 'full disengagement' (c4).\n")
        f.write(f"  c{bb_idx} mean_IRT = {np.exp(fit['mu'][bb_idx]):.2f}s (between-bout)\n")
        f.write(f"  c{diseng_idx} mean_IRT = {np.exp(fit['mu'][diseng_idx]):.2f}s (disengaged)\n\n")
        f.write("Model: logit(c3/(c3+c4)) ~ C(group) * day + (1|mouse)\n")
        f.write(f"Optimizer: {method_fep}\n\n")
        f.write(r_fep.summary().as_text() + "\n")
    state['sess_df'] = sess_df

    print("Wrote group_effects.csv, interaction_disengagement.csv, "
          "stats_group_effects.txt, stats_interaction.txt, stats_time_share.txt, "
          "stats_between_bout_composition.csv/txt",
          flush=True)

def stage_fit_quality(state):
    print("\n=== Stage: fit_quality ===", flush=True)
    _ensure_fit_loaded(state)
    if 'irts_list' not in state:
        state['irts_list'], _ = load_ri60()
    fit_quality_per_session(state['irts_list'], state['meta_df'], state['fit_main'])

def stage_trajectories(state):
    print("\n=== Stage: trajectories ===", flush=True)
    if 'sess_df' not in state:
        state['sess_df'] = pd.read_csv('session_weights.csv')
    K = sum(c.startswith('pi_') and c[3:].isdigit() for c in state['sess_df'].columns)
    traj = per_mouse_trajectories(state['sess_df'], K)
    traj.to_csv('per_mouse_trajectories.csv', index=False)
    print(f"Wrote per_mouse_trajectories.csv ({len(traj)} mice)", flush=True)
    print("\nLate-training mean pi by group:")
    print(traj.groupby('group')[[f'late_mean_pi_{k}' for k in range(K)]]
          .mean().round(4).to_string())
    print("\nPer-mouse slope of logit(pi) vs day, by group:")
    print(traj.groupby('group')[[f'slope_logit_pi_{k}' for k in range(K)]]
          .mean().round(4).to_string())
    state['traj'] = traj

# ===========================================================
# Stage: plot_data
# ===========================================================
def _mixture_pdf(x, mu, sigma, pi_vec):
    y = np.zeros_like(x, dtype=float)
    for k in range(len(mu)):
        y += pi_vec[k] * lognorm.pdf(x, s=float(sigma[k]), scale=np.exp(float(mu[k])))
    return y

def stage_plot_data(state):
    print("\n=== Stage: plot_data ===", flush=True)
    _ensure_fit_loaded(state)
    if 'irts_list' not in state:
        state['irts_list'], _ = load_ri60()
    fit = state['fit_main']
    sess_df = state['sess_df']
    meta_df = state['meta_df']
    irts_list = state['irts_list']
    K = fit['K']
    mu = np.array(fit['mu'], dtype=float)
    sigma = np.array(fit['sigma'], dtype=float)
    mean_irt = np.exp(mu)
    diseng_idx = int(np.argmax(mu))

    # ----- Density data + fit, all / naive / stress -----
    all_irts = np.concatenate(irts_list)
    bins = np.logspace(np.log10(0.05), np.log10(max(all_irts.max()*1.1, 1)), 80)
    centers = 0.5*(bins[:-1]+bins[1:])
    xg = np.logspace(np.log10(0.05), np.log10(300), 500)

    def _hist(group):
        if group is None:
            data = all_irts
        else:
            idx = meta_df[meta_df['group']==group]['session_idx'].values
            data = np.concatenate([irts_list[i] for i in idx])
        counts, edges = np.histogram(data, bins=bins)
        widths = np.diff(edges)
        return counts / (counts.sum() * widths), len(data)

    dens_all,    n_all    = _hist(None)
    dens_naive,  n_naive  = _hist('naive')
    dens_stress, n_stress = _hist('stress')

    pi_all    = sess_df[[f'pi_{k}' for k in range(K)]].mean().values
    pi_naive  = sess_df[sess_df['group']=='naive'][[f'pi_{k}' for k in range(K)]].mean().values
    pi_stress = sess_df[sess_df['group']=='stress'][[f'pi_{k}' for k in range(K)]].mean().values

    fit_all    = _mixture_pdf(xg, mu, sigma, pi_all)
    fit_naive  = _mixture_pdf(xg, mu, sigma, pi_naive)
    fit_stress = _mixture_pdf(xg, mu, sigma, pi_stress)

    # Two CSVs: one for data (histogram centers), one for fit (continuous grid)
    prism_density_data = pd.DataFrame({
        'IRT_s': centers,
        'density_all':    dens_all,
        'density_naive':  dens_naive,
        'density_stress': dens_stress,
    })
    prism_density_data.to_csv('prism_density_data.csv', index=False)

    prism_density_fit = pd.DataFrame({
        'IRT_s': xg,
        'fit_all':    fit_all,
        'fit_naive':  fit_naive,
        'fit_stress': fit_stress,
    })
    for k in range(K):
        prism_density_fit[f'comp{k}_{mean_irt[k]:.2f}s_all'] = (
            pi_all[k] * lognorm.pdf(xg, s=sigma[k], scale=np.exp(mu[k])))
        prism_density_fit[f'comp{k}_{mean_irt[k]:.2f}s_naive'] = (
            pi_naive[k] * lognorm.pdf(xg, s=sigma[k], scale=np.exp(mu[k])))
        prism_density_fit[f'comp{k}_{mean_irt[k]:.2f}s_stress'] = (
            pi_stress[k] * lognorm.pdf(xg, s=sigma[k], scale=np.exp(mu[k])))
    prism_density_fit.to_csv('prism_density_fit.csv', index=False)
    print("Wrote prism_density_data.csv, prism_density_fit.csv", flush=True)

    # ----- Density plots: log-log and linear -----
    def _draw_density(logscale, fname, title):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=logscale)
        panels = [
            ('all',    dens_all,    pi_all,    'k',        n_all),
            ('naive',  dens_naive,  pi_naive,  COL_NAIVE,  n_naive),
            ('stress', dens_stress, pi_stress, COL_STRESS, n_stress),
        ]
        for ax, (gname, dens, pi_vec, color, n) in zip(axes, panels):
            ax.step(centers, dens, where='mid', color='grey', alpha=0.8, label='data')
            ax.fill_between(centers, dens, step='mid', alpha=0.25, color='grey')
            total = _mixture_pdf(xg, mu, sigma, pi_vec)
            ax.plot(xg, total, color=color, lw=2, label='fit')
            for k in range(K):
                y = pi_vec[k] * lognorm.pdf(xg, s=sigma[k], scale=np.exp(mu[k]))
                ax.plot(xg, y, '--', color=color, lw=0.8, alpha=0.5)
            if logscale:
                ax.set_xscale('log'); ax.set_yscale('log')
                ax.set_ylim(1e-5, 2)
            else:
                ax.set_xlim(0, 30)
            ax.set_xlabel('IRT (s)')
            ax.set_title(f'{gname} (n IRTs = {n:,})')
            ax.legend(fontsize=8, loc='lower left' if logscale else 'upper right')
        axes[0].set_ylabel('density' + (' (log)' if logscale else ''))
        fig.suptitle(title, y=1.02)
        plt.tight_layout()
        plt.savefig(fname, dpi=140, bbox_inches='tight')
        plt.close()

    _draw_density(True,  'density_plot_loglog.png',
                  f'IRT density + K={K} mixture fit (log-log)')
    _draw_density(False, 'density_plot_linear.png',
                  f'IRT density + K={K} mixture fit (linear x, 0-30s)')
    print("Wrote density_plot_loglog.png, density_plot_linear.png", flush=True)

    # ----- Weights by group: individual mouse data for Prism -----
    # Rows = components, Columns = one per mouse (naive first, then stress)
    # Values = per-mouse mean pi_k across sessions
    ge = pd.read_csv('group_effects.csv')
    mouse_means = sess_df.groupby(['mouse','group'])[[f'pi_{k}' for k in range(K)]].mean()
    mouse_means = mouse_means.reset_index()
    mouse_group = mouse_means.set_index('mouse')['group']
    naive_mice = sorted(mouse_group[mouse_group=='naive'].index)
    stress_mice = sorted(mouse_group[mouse_group=='stress'].index)

    COMPONENT_LABELS = {
        0: f'motor ({mean_irt[0]:.2f}s)',
        1: f'fast check ({mean_irt[1]:.2f}s)',
        2: f'within-bout ({mean_irt[2]:.2f}s)',
        3: f'between-bout ({mean_irt[3]:.2f}s)',
        4: f'disengaged ({mean_irt[4]:.2f}s)',
    }

    prism_w = pd.DataFrame({'component': [COMPONENT_LABELS[k] for k in range(K)]})
    for m in naive_mice:
        vals = mouse_means[mouse_means['mouse']==m][[f'pi_{k}' for k in range(K)]].values[0]
        prism_w[f'naive_{m}'] = vals
    for m in stress_mice:
        vals = mouse_means[mouse_means['mouse']==m][[f'pi_{k}' for k in range(K)]].values[0]
        prism_w[f'stress_{m}'] = vals
    prism_w.to_csv('prism_weights_by_group.csv', index=False)

    # Compute summary stats for the plot from mouse-level means
    n_means = mouse_means[mouse_means['group']=='naive'][[f'pi_{k}' for k in range(K)]].mean().values
    s_means = mouse_means[mouse_means['group']=='stress'][[f'pi_{k}' for k in range(K)]].mean().values
    n_sems  = mouse_means[mouse_means['group']=='naive'][[f'pi_{k}' for k in range(K)]].sem().values
    s_sems  = mouse_means[mouse_means['group']=='stress'][[f'pi_{k}' for k in range(K)]].sem().values

    fig, ax = plt.subplots(figsize=(8,5))
    xpos = np.arange(K); w = 0.38
    ax.bar(xpos - w/2, n_means, w, yerr=n_sems,
           color=COL_NAIVE, label='naive', capsize=3)
    ax.bar(xpos + w/2, s_means, w, yerr=s_sems,
           color=COL_STRESS, label='stress', capsize=3)
    for k in range(K):
        p = ge[ge['component']==k]['group_p'].values[0]
        if p < 0.05:
            ymax = max(n_means[k]+n_sems[k], s_means[k]+s_sems[k])
            ax.text(k, ymax+0.015, '*', ha='center', fontsize=18, fontweight='bold')
            ax.text(k, ymax+0.04, f'p={p:.3f}', ha='center', fontsize=8)
    COMPONENT_LABELS = {
        0: 'motor',
        1: 'fast check',
        2: 'within-bout',
        3: 'between-bout',
        4: 'disengaged',
    }
    ax.set_xticks(xpos)
    ax.set_xticklabels([f'{COMPONENT_LABELS[k]}\n({mean_irt[k]:.2f}s)'
                        for k in range(K)])
    ax.set_ylabel('mean pi (fraction of pokes)')
    ax.set_title(f'RI60 component weights by group (K={K}, days 1-{MAX_DAY_INCLUSIVE})')
    ax.legend()
    plt.tight_layout()
    plt.savefig('weights_plot.png', dpi=140, bbox_inches='tight')
    plt.close()
    print("Wrote prism_weights_by_group.csv, weights_plot.png", flush=True)

    # Individual mouse trajectories, wide format for Prism grouped table:
    # Rows = days 1-14
    # Columns = one per mouse, grouped naive first then stress
    # Values = pi_4 for that mouse on that day (NaN if missing)
    diseng_col = f'pi_{diseng_idx}'
    pivot = sess_df.pivot_table(index='day', columns='mouse',
                                values=diseng_col, aggfunc='first')
    
    # Order columns: naive mice first, then stress
    mouse_group = sess_df.drop_duplicates('mouse').set_index('mouse')['group']
    naive_mice = sorted(mouse_group[mouse_group=='naive'].index)
    stress_mice = sorted(mouse_group[mouse_group=='stress'].index)
    
    # Rename columns to include group prefix
    naive_cols = {m: f'naive_{m}' for m in naive_mice}
    stress_cols = {m: f'stress_{m}' for m in stress_mice}
    
    pivot = pivot[naive_mice + stress_mice]
    pivot = pivot.rename(columns={**naive_cols, **stress_cols})
    pivot.to_csv('prism_disengagement_by_day.csv')

    # ----- Between-bout composition by day: c3/(c3+c4) -----
    if 'frac_engaged_pause' not in sess_df.columns:
        bb_idx = diseng_idx - 1
        sess_df['frac_engaged_pause'] = (
            sess_df[f'pi_{bb_idx}'] /
            (sess_df[f'pi_{bb_idx}'] + sess_df[f'pi_{diseng_idx}'] + 1e-10)
        )
    pivot_fep = sess_df.pivot_table(index='day', columns='mouse',
                                     values='frac_engaged_pause', aggfunc='first')
    fep_mouse_group = sess_df.drop_duplicates('mouse').set_index('mouse')['group']
    fep_naive = sorted(fep_mouse_group[fep_mouse_group=='naive'].index)
    fep_stress = sorted(fep_mouse_group[fep_mouse_group=='stress'].index)
    pivot_fep = pivot_fep[fep_naive + fep_stress]
    pivot_fep = pivot_fep.rename(columns={
        **{m: f'naive_{m}' for m in fep_naive},
        **{m: f'stress_{m}' for m in fep_stress}})
    pivot_fep.to_csv('prism_engaged_pause_frac_by_day.csv')
    print("Wrote prism_engaged_pause_frac_by_day.csv", flush=True)

    # ----- Between-bout composition: total (per-mouse mean) -----
    mouse_fep = sess_df.groupby(['mouse','group'])['frac_engaged_pause'].mean().reset_index()
    naive_vals = mouse_fep[mouse_fep['group']=='naive']['frac_engaged_pause'].values
    stress_vals = mouse_fep[mouse_fep['group']=='stress']['frac_engaged_pause'].values
    max_len = max(len(naive_vals), len(stress_vals))
    fep_out = pd.DataFrame({
        'naive': np.pad(naive_vals, (0, max_len - len(naive_vals)),
                        constant_values=np.nan),
        'stress': np.pad(stress_vals, (0, max_len - len(stress_vals)),
                         constant_values=np.nan),
    })
    fep_out.to_csv('prism_engaged_pause_frac_total.csv', index=False)
    print("Wrote prism_engaged_pause_frac_total.csv", flush=True)


    inter_df = pd.read_csv('interaction_disengagement.csv')
    inter_row = inter_df[inter_df['term']=='C(group)[T.stress]:day']
    inter_b = float(inter_row['beta'].values[0]) if len(inter_row) else np.nan
    inter_p = float(inter_row['p'].values[0]) if len(inter_row) else np.nan


    g = sess_df.groupby(['group','day'])[diseng_col].agg(['mean','sem','count']).reset_index()
    fig, ax = plt.subplots(figsize=(8,5))
    for grp, col in [('naive', COL_NAIVE), ('stress', COL_STRESS)]:
        sub = g[g['group']==grp]
        ax.errorbar(sub['day'], sub['mean'], yerr=sub['sem'],
                    color=col, marker='o', label=grp, ms=5, lw=1.5, capsize=2)
    ax.set_xlabel('training day')
    ax.set_ylabel(f'mean pi_{diseng_idx} (disengagement, '
                  f'{mean_irt[diseng_idx]:.1f}s)')
    ax.set_title(f'Disengagement weight x day\n'
                 f'Group x day: beta={inter_b:.3f}, p={inter_p:.4f} (LMM logit)')
    ax.set_xticks(range(1, MAX_DAY_INCLUSIVE+1))
    ax.legend()
    plt.tight_layout()
    plt.savefig('disengagement_by_day_plot.png', dpi=140, bbox_inches='tight')
    plt.close()
    print("Wrote prism_disengagement_by_day.csv, disengagement_by_day_plot.png",
          flush=True)

    # ----- Time share by group -----
    e_irt = np.exp(mu + sigma**2/2)
    if f'tshare_0' not in sess_df.columns:
        pi_mat = sess_df[[f'pi_{k}' for k in range(K)]].values
        time_unnorm = pi_mat * e_irt[None,:]
        time_share = time_unnorm / time_unnorm.sum(1, keepdims=True)
        for k in range(K):
            sess_df[f'tshare_{k}'] = time_share[:,k]
    # ----- Time share by group: individual mouse data for Prism -----
    # Same structure as weights: rows = components, cols = mice
    ts_stats = pd.read_csv('time_share_stats.csv')
    mouse_ts_means = sess_df.groupby(['mouse','group'])[[f'tshare_{k}' for k in range(K)]].mean()
    mouse_ts_means = mouse_ts_means.reset_index()
    ts_mouse_group = mouse_ts_means.set_index('mouse')['group']
    ts_naive = sorted(ts_mouse_group[ts_mouse_group=='naive'].index)
    ts_stress = sorted(ts_mouse_group[ts_mouse_group=='stress'].index)

    prism_ts = pd.DataFrame({'component': [COMPONENT_LABELS[k] for k in range(K)]})
    for m in ts_naive:
        vals = mouse_ts_means[mouse_ts_means['mouse']==m][[f'tshare_{k}' for k in range(K)]].values[0]
        prism_ts[f'naive_{m}'] = vals
    for m in ts_stress:
        vals = mouse_ts_means[mouse_ts_means['mouse']==m][[f'tshare_{k}' for k in range(K)]].values[0]
        prism_ts[f'stress_{m}'] = vals
    prism_ts.to_csv('prism_time_share_by_group.csv', index=False)

    # fig, ax = plt.subplots(figsize=(8,5))
    # xpos = np.arange(K); w = 0.38
    # ax.bar(xpos - w/2, prism_ts['mean_tshare_naive'].values, w,
    #        yerr=prism_ts['sem_tshare_naive'].values,
    #        color=COL_NAIVE, label='naive', capsize=3)
    # ax.bar(xpos + w/2, prism_ts['mean_tshare_stress'].values, w,
    #        yerr=prism_ts['sem_tshare_stress'].values,
    #        color=COL_STRESS, label='stress', capsize=3)
    # for k in range(K):
    #     p = prism_ts['welch_p'].iloc[k]
    #     if p < 0.05:
    #         ymax = max(prism_ts['mean_tshare_naive'].iloc[k]+prism_ts['sem_tshare_naive'].iloc[k],
    #                    prism_ts['mean_tshare_stress'].iloc[k]+prism_ts['sem_tshare_stress'].iloc[k])
    #         ax.text(k, ymax+0.02, '*', ha='center', fontsize=18, fontweight='bold')
    #         ax.text(k, ymax+0.06, f'p={p:.3f}', ha='center', fontsize=8)
    # ax.set_xticks(xpos)
    # ax.set_xticklabels([f'c{k}\n{mean_irt[k]:.2f}s' for k in range(K)])
    # ax.set_ylabel('mean time share (fraction of session-seconds)')
    # ax.set_title(f'RI60 time share by group (per-mouse Welch t, days 1-{MAX_DAY_INCLUSIVE})')
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig('time_share_plot.png', dpi=140, bbox_inches='tight')
    # plt.close()
    # print("Wrote prism_time_share_by_group.csv, time_share_plot.png", flush=True)

# ===========================================================
# Stage: reward_aligned  (paste after stage_plot_data)
# ===========================================================
COMP_GROUPS = {0: 'within_bout', 1: 'within_bout', 2: 'within_bout',
               3: 'between_bout', 4: 'disengaged'}
COMP_GROUP_ORDER = ['within_bout', 'between_bout', 'disengaged']


def _detect_bouts(cgroups, irts):
    """Run-length encode component-group labels to find bouts and gaps.
    Returns (bout_lengths_in_pokes, bout_durations_s, gap_durations_s)."""
    runs = []
    for label, items in igroupby(zip(cgroups, irts), key=lambda x: x[0]):
        durs = [x[1] for x in items]
        runs.append((label, len(durs), sum(durs)))
    bout_idx = [i for i, (l, _, _) in enumerate(runs) if l == 'within_bout']
    bout_lengths   = [runs[i][1] + 1 for i in bout_idx]  # pokes = irts + 1
    bout_durations = [runs[i][2]     for i in bout_idx]
    gap_durations  = []
    for bi in range(len(bout_idx) - 1):
        gap_durations.append(
            sum(runs[j][2] for j in range(bout_idx[bi] + 1, bout_idx[bi + 1])))
    return bout_lengths, bout_durations, gap_durations


def stage_reward_aligned(state):
    print("\n=== Stage: reward_aligned ===", flush=True)
    _ensure_fit_loaded(state)
    fit = state['fit_main']; sess_df = state['sess_df']
    K = fit['K']
    mus  = np.array(fit['mu'],    dtype=float)
    sigs = np.array(fit['sigma'], dtype=float)
    mean_irt = np.exp(mus)

    ri = pd.read_feather(FEATHER_RI)
    ri60 = ri[ri['sesType'] == 'RI60'].copy().reset_index(drop=True)
    ri60['sid'] = ri60['mouse'] + '|' + ri60['date'].astype(str)
    ri60['n_rew'] = ri60['rewPokeTimestamps'].apply(len)

    sess_df = sess_df.copy()
    sess_df['sid'] = sess_df['mouse'] + '|' + sess_df['date'].astype(str)
    pi_cols = [f'pi_{k}' for k in range(K)]
    ri60 = ri60[ri60['sid'].isin(set(sess_df['sid']))].reset_index(drop=True)
    ri60 = ri60.merge(sess_df[['sid'] + pi_cols], on='sid', how='inner')

    # drop aborted
    def _t_end(r):
        arrs = [np.asarray(r[k], dtype=float)
                for k in ['allPokeTimestamps', 'allEntryTS', 'rewPokeTimestamps']]
        arrs = [a for a in arrs if len(a)]
        return max(a.max() for a in arrs) if arrs else 0
    ri60['t_end_min'] = ri60.apply(_t_end, axis=1) / 60
    ri60 = ri60[~((ri60['t_end_min'] < 30) & (ri60['n_rew'] < 50))].reset_index(drop=True)
    print(f"  Sessions: {len(ri60)}  "
          f"(naive {(ri60.group=='naive').sum()}, stress {(ri60.group=='stress').sum()})",
          flush=True)

    # ── assign each IRT to component, tag with IRI index ───────────────
    irt_records = []
    for _, row in ri60.iterrows():
        pokes   = np.sort(np.asarray(row['allPokeTimestamps'], dtype=float))
        rewards = np.sort(np.asarray(row['rewPokeTimestamps'], dtype=float))
        if len(pokes) < 2 or len(rewards) == 0: continue
        irts = np.diff(pokes); end = pokes[1:]
        keep = (irts > 0) & (end <= 3600)
        irts = irts[keep]; end = end[keep]
        if len(irts) == 0: continue

        iri = np.searchsorted(rewards, end, side='right')
        pis = np.array([row[f'pi_{k}'] for k in range(K)])
        pis = np.clip(pis, 1e-10, None); pis /= pis.sum()
        log_irt = np.log(irts)
        log_lik = (-0.5 * ((log_irt[:, None] - mus[None, :]) / sigs[None, :]) ** 2
                   - np.log(sigs[None, :]))
        log_post = log_lik + np.log(pis[None, :])
        log_post -= log_post.max(axis=1, keepdims=True)
        post = np.exp(log_post); post /= post.sum(axis=1, keepdims=True)
        assign = post.argmax(axis=1)
        for i in range(len(irts)):
            irt_records.append((row['sid'], row['group'],
                                iri[i], assign[i], COMP_GROUPS[assign[i]], irts[i]))

    irt_df = pd.DataFrame(irt_records,
                          columns=['sid','group','iri_idx','component','cgroup','irt'])
    irt_df = irt_df[irt_df['iri_idx'].between(1, 50)].copy()
    print(f"  Total IRTs assigned: {len(irt_df):,}", flush=True)

    # ── per (session, IRI) statistics ──────────────────────────────────
    def _iri_stats(g):
        cgs = g['cgroup'].values; ir = g['irt'].values
        bl, _, gd = _detect_bouts(cgs, ir)
        out = {'n_irts': len(g),
               'mean_bout_length':   np.mean(bl) if bl else np.nan,
               'mean_interbout_dur': np.mean(gd) if gd else np.nan}
        total_dur = ir.sum()
        for cg in COMP_GROUP_ORDER:
            mask = (cgs == cg)
            out[f'frac_{cg}'] = ir[mask].sum() / total_dur if total_dur > 0 else np.nan
            sub = ir[mask]
            out[f'med_{cg}'] = float(np.median(sub)) if len(sub) else np.nan
        return pd.Series(out)
    
    sb = irt_df.groupby(['sid','group','iri_idx']).apply(
        _iri_stats).reset_index()
    print(f"  IRI-level rows: {len(sb):,}", flush=True)

    METRICS = [
        ('mean_bout_length',   'Mean bout length',           'Pokes'),
        ('mean_interbout_dur', 'Mean inter-bout interval',   'Seconds'),
        ('frac_disengaged',    'Fraction disengaged (C4)',    'Fraction'),
        ('frac_within_bout',   'Fraction within-bout (C0-2)','Fraction'),
        ('frac_between_bout',  'Fraction between-bout (C3)', 'Fraction'),
        ('med_within_bout',    'Median within-bout IRT',     'Seconds'),
        ('med_between_bout',   'Median between-bout IRT',    'Seconds'),
        ('med_disengaged',     'Median disengaged pause',    'Seconds'),
        ('rate',               'Per-IRI poke rate',          'Pokes/s'),
    ]

    # mouse-level means first
    sb['mouse'] = sb['sid'].str.split('|').str[0]

    # ── IRI duration and poke rate ─────────────────────────────────────
    iri_dur_records = []
    for _, row in ri60.iterrows():
        rewards = np.sort(np.asarray(row['rewPokeTimestamps'], dtype=float))
        pokes   = np.sort(np.asarray(row['allPokeTimestamps'], dtype=float))
        if len(rewards) == 0 or len(pokes) < 2: continue
        boundaries = np.concatenate([[0.0], rewards])
        for k in range(1, len(boundaries)):
            t0, t1 = boundaries[k-1], boundaries[k]
            dur = t1 - t0
            n_pk = int(((pokes > t0) & (pokes <= t1)).sum())
            if dur > 0:
                iri_dur_records.append((row['sid'], k, dur, n_pk / dur))
    iri_dur_df = pd.DataFrame(iri_dur_records,
                              columns=['sid', 'iri_idx', 'iri_dur', 'rate'])
    iri_dur_df = iri_dur_df[iri_dur_df['iri_idx'].between(1, 50)]
    sb = sb.merge(iri_dur_df[['sid', 'iri_idx', 'rate']],
                  on=['sid', 'iri_idx'], how='left')


    # ── Session-level and mouse-level bout summaries ───────────────
    sess_bout = sb.groupby(['sid','mouse','group']).agg(
        mean_bout_length   = ('mean_bout_length','mean'),
        mean_interbout_dur = ('mean_interbout_dur','mean'),
    ).reset_index()

    # one row per mouse
    mouse_bout = sess_bout.groupby(['mouse','group']).agg(
        mean_bout_length   = ('mean_bout_length','mean'),
        mean_interbout_dur = ('mean_interbout_dur','mean'),
    ).reset_index()

    # Prism CSVs: wide format, one column per mouse
    for metric, fname in [('mean_bout_length',   'prism_RA_session_bout_length.csv'),
                        ('mean_interbout_dur', 'prism_RA_session_interbout.csv')]:
        naive_mice  = sorted(mouse_bout[mouse_bout.group=='naive']['mouse'])
        stress_mice = sorted(mouse_bout[mouse_bout.group=='stress']['mouse'])
        # session-level: rows = sessions, cols = naive then stress
        # (ragged across mice, so pivot)
        piv = sess_bout.pivot_table(index=sess_bout.groupby('mouse').cumcount(),
                                    columns='mouse', values=metric)
        piv = piv[[m for m in naive_mice + stress_mice if m in piv.columns]]
        piv.columns = [f"{'naive' if m in naive_mice else 'stress'}_{m}"
                    for m in piv.columns]
        piv.to_csv(fname, index=False)
        print(f"    Wrote {fname}", flush=True)

    # mouse-level: one row per component, columns = mice (same layout as weights)
    for metric, fname in [('mean_bout_length',   'prism_RA_mouse_bout_length.csv'),
                        ('mean_interbout_dur', 'prism_RA_mouse_interbout.csv')]:
        naive_mice  = sorted(mouse_bout[mouse_bout.group=='naive']['mouse'])
        stress_mice = sorted(mouse_bout[mouse_bout.group=='stress']['mouse'])
        out = pd.DataFrame({'mouse': naive_mice + stress_mice,
                            'group': ['naive']*len(naive_mice) + ['stress']*len(stress_mice)})
        out[metric] = [float(mouse_bout[mouse_bout.mouse==m][metric].values[0])
                    for m in out['mouse']]
        out.to_csv(fname, index=False)
        print(f"    Wrote {fname}", flush=True)

    
    mouse_avg = sb.groupby(['mouse','group','iri_idx']).agg(
        **{c: (c,'mean') for c,_,_ in METRICS},
    ).reset_index()

    pop = mouse_avg.groupby('iri_idx').agg(
        **{f'{c}_m': (c,'mean') for c,_,_ in METRICS},
        **{f'{c}_se':(c,'sem')  for c,_,_ in METRICS},
        n=('mouse','count'),
    ).reset_index()

    grp = mouse_avg[mouse_avg['iri_idx']<=40].groupby(['group','iri_idx']).agg(
        **{f'{c}_m': (c,'mean') for c,_,_ in METRICS},
        **{f'{c}_se':(c,'sem')  for c,_,_ in METRICS},
        n=('mouse','count'),
    ).reset_index()

    # ── within-session paired slopes ───────────────────────────────────
    from scipy.stats import t as tdist
    print("\n  Within-session paired slopes (per reward):", flush=True)
    for col, lbl, _ in METRICS:
        slopes = []
        for sid, g in sb.groupby('sid'):
            d = g[g['iri_idx'] <= 40][['iri_idx', col]].dropna()            
            if len(d) < 15: continue
            x = d['iri_idx'].values; y = d[col].values
            if x.std() == 0: continue
            slopes.append(np.polyfit(x, y, 1)[0])
        slopes = np.array(slopes)
        if len(slopes) < 5: continue
        se = slopes.std(ddof=1) / np.sqrt(len(slopes))
        t = slopes.mean() / se
        p = 2 * (1 - tdist.cdf(abs(t), df=len(slopes)-1))
        print(f'    {lbl:35s} slope={slopes.mean():+.5f}  '
              f'SE={se:.5f}  n={len(slopes)}  t={t:+.2f}  p={p:.2e}', flush=True)

    # ── LMM: group × reward interaction with random intercept per mouse ──
    print("\n  LMM group x reward (iri 1..40, random intercept per mouse):", flush=True)
    sub40 = sb[sb['iri_idx'] <= 40].copy()
    sub40['group_cat'] = pd.Categorical(sub40['group'], categories=['naive','stress'])

    # Rows we don't report: model intercept and the random-effect variance.
    LMM_SKIP_TERMS = {'Intercept', 'Group Var'}

    def _sig_stars(p):
        if p is None or not np.isfinite(p): return ''
        if p < 1e-4: return '****'
        if p < 1e-3: return '***'
        if p < 1e-2: return '**'
        if p < 0.05: return '*'
        return 'ns'

    lmm_rows = []
    for col, lbl, _ in METRICS:
        d = sub40[['iri_idx','group_cat','mouse',col]].dropna()
        if len(d) < 20: continue
        try:
            r, method = fit_lmm_stable(
                f'{col} ~ C(group_cat) * iri_idx', d, groups=d['mouse'])
            for t in r.params.index:
                if t in LMM_SKIP_TERMS: continue
                lmm_rows.append(dict(
                    metric=lbl, term=t, beta=float(r.params[t]),
                    se=float(r.bse[t]), p=float(r.pvalues[t]),
                    method=method))
                print(f'    {lbl:35s} {t}: beta={r.params[t]:+.5f}  '
                      f'p={r.pvalues[t]:.2e}', flush=True)
        except Exception as e:
            lmm_rows.append(dict(metric=lbl, term='FAILED', beta=np.nan,
                                 se=np.nan, p=np.nan, method=str(e)))
            print(f'    {lbl:35s} LMM failed: {e}', flush=True)
    lmm_df = pd.DataFrame(lmm_rows)
    lmm_df['sig'] = lmm_df['p'].apply(_sig_stars)
    lmm_df.to_csv('stats_reward_aligned_lmm.csv', index=False)
    print(f"    Wrote stats_reward_aligned_lmm.csv", flush=True)
    
    # ── group x reward interaction ─────────────────────────────────────
    print("\n  Group x reward-number interaction (iri 1..40):", flush=True)
    sub40 = sb[sb['iri_idx']<=40].copy()
    sub40['stress'] = (sub40['group']=='stress').astype(int)
    sub40['inter']  = sub40['iri_idx'] * sub40['stress']
    for col, lbl, _ in METRICS:
        d = sub40[['iri_idx','stress','inter',col,'sid']].dropna()
        if len(d) < 20: continue
        m = sm_api.OLS(d[col], sm_api.add_constant(d[['iri_idx','stress','inter']])).fit(
            cov_type='cluster', cov_kwds={'groups': d['sid']})
        print(f'    {lbl:35s} interaction={m.params["inter"]:+.5f}  '
              f'p={m.pvalues["inter"]:.2e}', flush=True)

    def _sp_individual(df_in, col, fname):
            piv = df_in.pivot_table(index='iri_idx', columns='mouse', values=col)
            mg = df_in.drop_duplicates('mouse').set_index('mouse')['group']
            naive  = sorted(m for m in piv.columns if mg[m]=='naive')
            stress = sorted(m for m in piv.columns if mg[m]=='stress')
            piv = piv[naive + stress]
            piv.columns = [f"{'naive' if mg[m]=='naive' else 'stress'}_{m}" for m in piv.columns]
            piv.index.name = 'Reward_Number'
            piv.to_csv(fname)
            print(f"    Wrote {fname}", flush=True)

    print("\n  Saving Prism CSVs...", flush=True)
    for col, fname in [
        ('mean_bout_length',   'prism_RA_bout_length.csv'),
        ('mean_interbout_dur', 'prism_RA_interbout_interval.csv'),
        ('frac_disengaged',    'prism_RA_frac_disengaged.csv'),
        ('frac_within_bout',   'prism_RA_frac_within_bout.csv'),
        ('med_within_bout',    'prism_RA_med_within_bout.csv'),
        ('med_disengaged',     'prism_RA_med_disengaged.csv'),
        ('frac_between_bout',  'prism_RA_frac_between_bout.csv'),
        ('med_between_bout',   'prism_RA_med_between_bout.csv'),
    ]:
        _sp_individual(mouse_avg, col, fname)
    pop[['iri_idx','n']].to_csv('prism_RA_n_sessions.csv', index=False)
    print(f"    Wrote prism_RA_n_sessions.csv", flush=True)

    # ── By-group summaries: mean ± SEM per group per reward number ─────
    def _by_group_summary(col, fname):
        piv_m = grp.pivot_table(index='iri_idx', columns='group', values=f'{col}_m')
        piv_s = grp.pivot_table(index='iri_idx', columns='group', values=f'{col}_se')
        out = pd.DataFrame({'Reward_Number': piv_m.index})
        for g in ['naive', 'stress']:
            if g in piv_m.columns:
                out[f'{g}_mean'] = piv_m[g].values
                out[f'{g}_sem']  = piv_s[g].values
        out.to_csv(fname, index=False)
        print(f"    Wrote {fname}", flush=True)

    _by_group_summary('mean_bout_length',   'prism_RA_bout_length_by_group.csv')
    _by_group_summary('mean_interbout_dur', 'prism_RA_interbout_by_group.csv')
    _by_group_summary('frac_disengaged',    'prism_RA_frac_disengaged_by_group.csv')

    # ── All-component fractions (population, by reward number) ─────────
    all_frac = pop[['iri_idx',
                    'frac_within_bout_m',  'frac_within_bout_se',
                    'frac_between_bout_m', 'frac_between_bout_se',
                    'frac_disengaged_m',   'frac_disengaged_se',
                    'n']].copy()
    all_frac.columns = ['Reward_Number',
                        'within_bout_mean',  'within_bout_sem',
                        'between_bout_mean', 'between_bout_sem',
                        'disengaged_mean',   'disengaged_sem',
                        'n_mice']
    all_frac.to_csv('prism_RA_all_component_fractions.csv', index=False)
    print(f"    Wrote prism_RA_all_component_fractions.csv", flush=True)

    # ── Between-bout statistics (frac and median, population) ──────────
    between_bb = pop[['iri_idx',
                      'frac_between_bout_m', 'frac_between_bout_se',
                      'med_between_bout_m',  'med_between_bout_se',
                      'n']].copy()
    between_bb.columns = ['Reward_Number',
                          'frac_mean', 'frac_sem',
                          'med_mean',  'med_sem',
                          'n_mice']
    between_bb.to_csv('prism_RA_between_bout_stats.csv', index=False)
    print(f"    Wrote prism_RA_between_bout_stats.csv", flush=True)

    # ── Plots ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    def _pp(ax, mc, sc, title, ylabel, ylim=None, hline=None):
        x=pop['iri_idx']; m=pop[mc]; e=pop[sc]
        ax.fill_between(x, m-e, m+e, alpha=0.3, color='C0')
        ax.plot(x, m, 'o-', color='C0', lw=2, ms=4)
        ax.set_xlabel('Reward number'); ax.set_ylabel(ylabel); ax.set_title(title)
        if ylim: ax.set_ylim(ylim)
        if hline is not None: ax.axhline(hline, ls='--', color='gray', lw=0.7)
        ax2 = ax.twinx()
        ax2.bar(x, pop['n'], alpha=0.12, color='gray')
        ax2.set_ylabel('n sessions', color='gray', fontsize=8)
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=8)
        ax2.set_ylim(0, pop['n'].max()*4)

    def _pg(ax, mc, sc, title, ylabel, ylim=None):
        for g, col in [('naive',COL_NAIVE),('stress',COL_STRESS)]:
            d = grp[grp.group==g]
            ax.fill_between(d['iri_idx'], d[mc]-d[sc], d[mc]+d[sc],
                            alpha=0.25, color=col)
            ax.plot(d['iri_idx'], d[mc], 'o-', color=col, lw=2, ms=4, label=g)
        ax.set_xlabel('Reward number'); ax.set_ylabel(ylabel); ax.set_title(title)
        if ylim: ax.set_ylim(ylim)
        ax.legend(fontsize=8)

    _pp(axes[0,0], 'mean_bout_length_m','mean_bout_length_se',
        'A. Mean bout length','Pokes')
    _pp(axes[0,1], 'mean_interbout_dur_m','mean_interbout_dur_se',
        'B. Mean inter-bout interval','Seconds')
    _pp(axes[0,2], 'frac_disengaged_m','frac_disengaged_se',
        'C. Fraction disengaged (C4)','Fraction')
    _pp(axes[1,0], 'frac_within_bout_m','frac_within_bout_se',
        'D. Fraction within-bout (C0+C1+C2)','Fraction')
    _pp(axes[1,1], 'med_within_bout_m','med_within_bout_se',
        'E. Median within-bout IRT','Seconds', ylim=(0,1.5))
    _pp(axes[1,2], 'med_disengaged_m','med_disengaged_se',
        'F. Median disengaged pause','Seconds')
    _pg(axes[2,0], 'mean_bout_length_m','mean_bout_length_se',
        'G. Bout length by group','Pokes')
    _pg(axes[2,1], 'mean_interbout_dur_m','mean_interbout_dur_se',
        'H. Inter-bout interval by group','Seconds')
    _pg(axes[2,2], 'frac_disengaged_m','frac_disengaged_se',
        'I. Disengaged fraction by group','Fraction')

    plt.tight_layout()
    plt.savefig('component_reward_aligned.png', dpi=140, bbox_inches='tight')
    plt.close()
    print("  Wrote component_reward_aligned.png", flush=True)


def stage_photo_merge(state):
    """Merge per-session K=5 weights onto per-event photometry dataframe.

    Photometry is per-event (many rows per session). Session weights are
    one row per session. Merge is many-to-one on (mouse, sesType, dayOnType),
    validated so that any duplicates on the right side raise rather than
    silently cartesian-exploding (the bug in the old `comp` branch).
    """
    print("\n=== Stage: photo_merge ===", flush=True)

    # ---- Load both sides ----
    if 'sess_df' not in state:
        state['sess_df'] = pd.read_csv('session_weights.csv')
    sess_df = state['sess_df']

    if not os.path.exists(FEATHER_PHOTO_IN):
        raise FileNotFoundError(f"photometry feather not found: {FEATHER_PHOTO_IN}")
    photo = pd.read_feather(FEATHER_PHOTO_IN)
    print(f"Loaded photo feather: {len(photo):,} rows, "
        f"{photo['mouse'].nunique()} mice", flush=True)

    # Mouse IDs in photo use '_' where behavior uses '.' for some mice
    # (e.g. photo '578_1-1' == behavior '578.1-1'). Build a normalized merge
    # key on a temp column so we don't touch the original 'mouse' column.
    photo['_mouse_key'] = photo['mouse'].str.replace('_', '.', regex=False)
    n_renamed = (photo['_mouse_key'] != photo['mouse']).sum()
    renamed_ids = photo.loc[photo['_mouse_key'] != photo['mouse'], 'mouse'].unique()
    if n_renamed:
        print(f"  normalized {n_renamed} rows across {len(renamed_ids)} mouse IDs "
            f"for merge key: {sorted(renamed_ids)}", flush=True)
    print(f"  sesType breakdown: {photo['sesType'].value_counts().to_dict()}", flush=True)

    # ---- Build the right-hand table: one row per (mouse, sesType, dayOnType) ----
    # session_weights.csv is all RI60 (fit_main only fits RI60).
    # We need to add a sesType column so the merge key matches.
    pi_cols = [c for c in sess_df.columns if c.startswith('pi_') and c[3:].isdigit()]
    K = len(pi_cols)
    keep = ['mouse', 'day', 'n_irts'] + pi_cols
    weights = sess_df[keep].copy()
    weights = weights.rename(columns={'day': 'dayOnType'})
    weights['sesType'] = 'RI60'
    # Match dtype: the photo feather is int32 on dayOnType
    weights['dayOnType'] = weights['dayOnType'].astype('int32')

    # ---- QC before merge ----
    # Photo feather must also be int on dayOnType
    if photo['dayOnType'].dtype != weights['dayOnType'].dtype:
        photo = photo.copy()
        photo['dayOnType'] = photo['dayOnType'].astype('int32')

    # Check for duplicates on the right-hand merge key.
    # If fit_main ran on day-filtered data, there should be at most ONE row
    # per (mouse, sesType, dayOnType). If not, something upstream is wrong.
    dup_mask = weights.duplicated(subset=['mouse','sesType','dayOnType'], keep=False)
    if dup_mask.any():
        dup_rows = weights[dup_mask].sort_values(['mouse','dayOnType'])
        print(f"\n!!! WARNING: {dup_mask.sum()} duplicate rows in weights table "
              f"for (mouse, sesType, dayOnType):", flush=True)
        print(dup_rows.to_string(index=False), flush=True)
        print("  This usually means dayOnType labeling is non-unique upstream "
              "(e.g. same day number on two different dates for one mouse).\n"
              "  Collapsing to .first() deterministically — real fix is upstream.",
              flush=True)
        weights = (weights
                   .groupby(['mouse','sesType','dayOnType'], as_index=False)
                   .first())

    # Final uniqueness assertion
    assert not weights.duplicated(subset=['mouse','sesType','dayOnType']).any(), \
        "weights table still non-unique on merge key after collapse"
    print(f"Weights table: {len(weights)} unique (mouse, sesType, dayOnType) rows "
          f"for RI60, K={K}", flush=True)

    # ---- Merge with explicit many-to-one validation ----
    # Use _mouse_key (underscore-normalized) as the merge key so photo mice
    # like '578_1-1' match weights mice like '578.1-1'. The output's 'mouse'
    # column stays as the original photo name ('578_1-1') because we only
    # touched the temp column, not 'mouse'.
    weights['_mouse_key'] = weights['mouse']
    weights = weights.drop(columns=['mouse'])  # avoid mouse_x/mouse_y suffixing

    n_before = len(photo)
    merged = photo.merge(
        weights,
        on=['_mouse_key','sesType','dayOnType'],
        how='left',
        validate='many_to_one',
    )
    n_after = len(merged)
    assert n_before == n_after, \
        f"Row count changed during merge: {n_before} -> {n_after} (should be equal)"
    print(f"Merge OK: {n_before:,} -> {n_after:,} rows (unchanged, as expected)",
        flush=True)
        
    
    # ---- Post-merge QC: report coverage ----
    # RI60 rows that got weight columns populated vs NaN
    ri60_mask = merged['sesType'] == 'RI60'
    ri60_total = int(ri60_mask.sum())
    ri60_with_weights = int(ri60_mask.sum() - merged.loc[ri60_mask, pi_cols[0]].isna().sum())
    ri60_without = ri60_total - ri60_with_weights
    print(f"\nRI60 coverage:", flush=True)
    print(f"  RI60 photometry rows total:     {ri60_total:,}", flush=True)
    print(f"  with K={K} weights populated:   {ri60_with_weights:,} "
          f"({ri60_with_weights/ri60_total*100:.1f}%)", flush=True)
    print(f"  without weights (NaN):          {ri60_without:,}", flush=True)

    if ri60_without > 0:
        # Which (mouse, day) combinations are present in photo but missing from weights?
        photo_keys = set(zip(
            merged.loc[ri60_mask & merged[pi_cols[0]].isna(), 'mouse'],
            merged.loc[ri60_mask & merged[pi_cols[0]].isna(), 'dayOnType'],
        ))
        print(f"  unique (mouse, dayOnType) RI60 pairs missing weights: "
              f"{len(photo_keys)}", flush=True)
        # Print up to 15 of them so you can see what got dropped
        sample = sorted(photo_keys)[:15]
        print("  examples:", sample, flush=True)
        print(f"  NOTE: these are RI60 sessions that exist in photometry but\n"
              f"        were NOT fit by fit_main. Most likely causes:\n"
              f"        (a) dayOnType outside [1, {MAX_DAY_INCLUSIVE}] filter\n"
              f"        (b) session had <30 pokes and was dropped during fit\n"
              f"        (c) mouse has photometry but no RI60 behavior in feather",
              flush=True)

    # Non-RI60 rows (RI30, FR1) will have NaN weights by design
    non_ri60 = merged[~ri60_mask]
    if len(non_ri60) > 0:
        print(f"\nNon-RI60 rows (will have NaN weight columns):", flush=True)
        print(f"  {non_ri60['sesType'].value_counts().to_dict()}", flush=True)

    # ---- Column hygiene: old code had group_x/group_y collisions ----
    # Here we only pulled pi cols + n_irts from weights, so no collisions.
    # But assert in case future edits add columns.
    collision_cols = [c for c in merged.columns if c.endswith('_x') or c.endswith('_y')]
    if collision_cols:
        print(f"\n!!! WARNING: column suffix collisions detected: {collision_cols}",
              flush=True)

    # ---- Add numShocks (day-2 shock count) per mouse ----
    # Source: Shock_dataFrame.feather. We HARD-FILTER to dayOnType == 2 here so
    # numShocks is always the day-2 shock total, independent of whatever else
    # may live in that feather. It is a per-mouse covariate: the same day-2
    # total is broadcast to ALL of a mouse's rows/sessions (NOT per-session).
    # Mouse IDs are matched on the same underscore->dot normalization used for
    # the weight merge (photo '578_1-1' == shock '578.1-1').
    if os.path.exists(FEATHER_SHOCK):
        shock = pd.read_feather(FEATHER_SHOCK)
        shock['dayOnType'] = shock['dayOnType'].astype(float).astype(int)
        shock_d2 = shock[shock['dayOnType'] == 2].copy()
        shock_d2['_shock_key'] = shock_d2['mouse'].str.replace('_', '.', regex=False)
        # One row per mouse (defensive; should already be unique on day 2).
        nshock = (shock_d2[['_shock_key', 'numShock']]
                  .groupby('_shock_key', as_index=False)
                  .first()
                  .rename(columns={'numShock': 'numShocks'}))
        assert not nshock.duplicated('_shock_key').any(), \
            "Shock_dataFrame has >1 day-2 row per mouse after collapse"

        merged['_shock_key'] = merged['mouse'].str.replace('_', '.', regex=False)
        n_before_shock = len(merged)
        merged = merged.merge(nshock, on='_shock_key', how='left',
                              validate='many_to_one')
        merged = merged.drop(columns=['_shock_key'])
        assert len(merged) == n_before_shock, \
            f"numShocks merge changed row count {n_before_shock} -> {len(merged)}"

        n_rows_cov = int(merged['numShocks'].notna().sum())
        n_mice_cov = merged.loc[merged['numShocks'].notna(), 'mouse'].nunique()
        n_mice_tot = merged['mouse'].nunique()
        print(f"\nnumShocks (dayOnType==2): {n_mice_cov}/{n_mice_tot} mice matched, "
              f"{n_rows_cov:,}/{len(merged):,} rows populated", flush=True)
        missing_mice = sorted(merged.loc[merged['numShocks'].isna(), 'mouse'].unique())
        if missing_mice:
            print(f"  mice with NO day-2 shock entry (numShocks=NaN): "
                  f"{missing_mice}", flush=True)
    else:
        print(f"\n[warn] {FEATHER_SHOCK} not found; numShocks column NOT added",
              flush=True)

    # ---- Save ----
    merged.to_feather(FEATHER_PHOTO_OUT)
    print(f"\nWrote {FEATHER_PHOTO_OUT}  "
          f"({len(merged):,} rows, {len(merged.columns)} cols)", flush=True)

# ===========================================================
# Stage: bout_assign
# ===========================================================
# Per-poke bout label. c0-c2 = within-bout; c3 = between-bout (its own class,
# excluded from the within_bout/disengaged PSTH + bootstrap comparisons);
# c4 = disengaged. Any non-within-bout gap still ends the current bout (the
# depth walk below resets on anything != 'within_bout').
POKE_STATUS = {0: 'within_bout', 1: 'within_bout', 2: 'within_bout',
               3: 'between_bout', 4: 'disengaged'}
PSTH_STATUSES = ['within_bout', 'disengaged']
PSTH_STATUS_COL = {'within_bout': '#2ca02c',
                   'disengaged':  '#7f3fbf'}
PSTH_EVENTS = ['ReNP', 'UnNP']
PSTH_RECLOC = 'TS'   # PSTHs use ONLY this recording site (do NOT pool TS + DMS)


def _label_pokes_on_photo(photo, ri, pi_lookup, mus, sigs, K, tag=''):
    """Add poke_component / bout_status / poke_within_bout to a photo frame.

    `photo` only needs columns: mouse (or _mouse_key), date, timestamp, sesType.
    Labels are location-independent (behavioral), so this works for any photo
    frame keyed by the same (mouse, date, timestamp) scheme. Returns the frame
    with the three columns added/refreshed (idempotent).
    """
    photo = photo.drop(columns=[c for c in
            ['poke_component', 'bout_status', 'poke_within_bout']
            if c in photo.columns])
    mk = (photo['_mouse_key'] if '_mouse_key' in photo.columns
          else photo['mouse'].str.replace('_', '.', regex=False))

    n = len(photo)
    comp_out   = np.full(n, np.nan)
    depth_out  = np.full(n, np.nan)
    status_out = np.array([None] * n, dtype=object)

    TS_TOL = 0.01
    ts_all = photo['timestamp'].to_numpy(dtype=float)
    mk_all = mk.to_numpy()
    dt_all = photo['date'].to_numpy()
    st_all = photo['sesType'].to_numpy()

    n_sess = 0; n_match = 0; n_unmatched = 0; n_no_pi = 0
    ev_col = photo['event'].to_numpy() if 'event' in photo.columns else None
    ev_match = {}
    for _, srow in ri.iterrows():
        key = (srow['mouse'], srow['date'])
        if key not in pi_lookup:
            n_no_pi += 1
            continue
        pis = pi_lookup[key]
        pokes = np.asarray(srow['allPokeTimestamps'], dtype=float)
        u = np.unique(pokes[np.isfinite(pokes)])   # sorted, strictly increasing
        if u.size < 2:
            continue
        irts = np.diff(u)

        # --- posterior component assignment for each gap ---
        pis_c = np.clip(pis, 1e-10, None); pis_c /= pis_c.sum()
        log_irt = np.log(irts)
        log_lik = (-0.5 * ((log_irt[:, None] - mus[None, :]) / sigs[None, :]) ** 2
                   - np.log(sigs[None, :]))
        log_post = log_lik + np.log(pis_c[None, :])
        assign = log_post.argmax(axis=1)

        # --- per-poke walk: label each poke by its INCOMING gap ---
        m = u.size
        p_comp   = np.full(m, np.nan)
        p_depth  = np.ones(m, dtype=float)
        p_status = np.array([None] * m, dtype=object)
        for j in range(1, m):
            c = int(assign[j - 1])
            st = POKE_STATUS[c]
            p_comp[j] = c
            p_status[j] = st
            p_depth[j] = p_depth[j - 1] + 1 if st == 'within_bout' else 1

        # --- match photo rows for this session by nearest timestamp ---
        sel = np.where((mk_all == key[0]) & (dt_all == key[1])
                       & (st_all == 'RI60'))[0]
        if sel.size == 0:
            continue
        n_sess += 1
        for ridx in sel:
            t = ts_all[ridx]
            pos = np.searchsorted(u, t)
            best = -1; bd = TS_TOL
            for cand in (pos - 1, pos):
                if 0 <= cand < m and abs(u[cand] - t) <= bd:
                    bd = abs(u[cand] - t); best = cand
            if best >= 0:
                comp_out[ridx]   = p_comp[best]
                depth_out[ridx]  = p_depth[best]
                status_out[ridx] = p_status[best]
                n_match += 1
                if ev_col is not None:
                    ev_match[ev_col[ridx]] = ev_match.get(ev_col[ridx], 0) + 1
            else:
                n_unmatched += 1

    photo['poke_component']   = comp_out
    photo['poke_within_bout'] = depth_out
    photo['bout_status']      = status_out

    pre = f"  [{tag}] " if tag else "  "
    print(f"{pre}Sessions matched: {n_sess}  (no pi weights: {n_no_pi})", flush=True)
    print(f"{pre}RI60 rows labeled: {n_match:,}   unmatched (no poke within tol): "
          f"{n_unmatched:,}", flush=True)
    if ev_match:
        print(f"{pre}Labeled rows by event: {ev_match}", flush=True)
    lab = photo.loc[photo['bout_status'].notna()]
    print(f"{pre}Bout-status distribution: "
          f"{lab['bout_status'].value_counts().to_dict()}", flush=True)
    return photo


def stage_bout_assign(state):
    """Assign each poke a mixture component + bout status, write onto photoDF.

    Process (identical assignment math to stage_reward_aligned):
      For every RI60 session, IRTs = diff(sorted unique allPokeTimestamps).
      Each IRT (the GAP between two pokes) is assigned to the K=5 component
      with the highest posterior under that session's pi weights and the
      global (mu, sigma). Components map to bout status via POKE_STATUS.

    A POKE is then labeled by its INCOMING gap (the IRT that ended at it):
      - bout_status      = status of the incoming gap
      - poke_within_bout = depth in the current bout. A within-bout incoming
                           gap increments depth (continuing a bout); a
                           between/disengaged incoming gap starts a new bout
                           at depth 1. The session's first poke has no incoming
                           gap -> depth 1, status NaN.

    These per-poke labels are matched back onto photoDF rows by nearest
    timestamp (<= TS_TOL) within (mouse, date) and three columns are added to
    photoDF_R_with_weights.feather:  poke_component, bout_status,
    poke_within_bout.  PE events (port entries, not pokes) stay NaN.

    Finally draws ReNP/UnNP PSTHs split by bout status, one figure per group.
    """
    print("\n=== Stage: bout_assign ===", flush=True)
    _ensure_fit_loaded(state)
    fit = state['fit_main']; sess_df = state['sess_df']
    K = fit['K']
    mus  = np.array(fit['mu'],    dtype=float)
    sigs = np.array(fit['sigma'], dtype=float)
    if 'date' not in sess_df.columns:
        raise KeyError("sess_df has no 'date' column; cannot key per-session "
                       "pi weights to raw sessions.")

    # ---- per-session pi lookup keyed by (mouse, date) ----
    pi_cols = [f'pi_{k}' for k in range(K)]
    pi_lookup = {(r['mouse'], r['date']):
                 np.array([r[c] for c in pi_cols], dtype=float)
                 for _, r in sess_df.iterrows()}
    print(f"Loaded {len(pi_lookup)} session pi-weight vectors (K={K})", flush=True)

    ri = pd.read_feather(FEATHER_RI)
    ri = ri[ri['sesType'] == 'RI60'].copy()

    # ---- label photoDF_R_with_weights (trimTrace frame; used for PSTHs) ----
    if not os.path.exists(FEATHER_PHOTO_OUT):
        raise FileNotFoundError(
            f"need photo_merge output first: {FEATHER_PHOTO_OUT}")
    photo = pd.read_feather(FEATHER_PHOTO_OUT)
    print(f"Loaded {FEATHER_PHOTO_OUT}: {len(photo):,} rows", flush=True)
    photo = _label_pokes_on_photo(photo, ri, pi_lookup, mus, sigs, K,
                                  tag='photoDF_R_with_weights')
    photo.to_feather(FEATHER_PHOTO_OUT)
    print(f"  Wrote {FEATHER_PHOTO_OUT} (+3 cols)", flush=True)

    # ---- label photoDataFrame.feather (photoTrace frame; for bootstrapCIs) ----
    if os.path.exists(FEATHER_PHOTODF):
        pdf = pd.read_feather(FEATHER_PHOTODF)
        print(f"Loaded {FEATHER_PHOTODF}: {len(pdf):,} rows", flush=True)
        pdf = _label_pokes_on_photo(pdf, ri, pi_lookup, mus, sigs, K,
                                    tag='photoDataFrame')
        pdf.to_feather(FEATHER_PHOTODF)
        print(f"  Wrote {FEATHER_PHOTODF} (+3 cols)", flush=True)
    else:
        print(f"  SKIP photoDataFrame labeling (not found: {FEATHER_PHOTODF})",
              flush=True)

    # ---- PSTHs: ReNP/UnNP split by bout status, one figure per group ----
    canon = photo.loc[photo['sesType'] == 'RI60', 'trimTime']
    time_axis = np.asarray(canon.iloc[0], dtype=float)
    T = len(time_axis)

    def _mouse_mean_traces(sub):
        """Mouse-level mean traces -> (n_mice, T) stacked array.

        A small fraction of traces carry scattered NaNs (artifact-trimmed
        samples), so reduce with nanmean to avoid wiping whole timepoints.
        """
        out = []
        for _, gg in sub.groupby('mouse'):
            arrs = [np.asarray(a, dtype=float) for a in gg['trimTrace']
                    if a is not None and len(a) == T]
            if arrs:
                out.append(np.nanmean(np.vstack(arrs), axis=0))
        return np.vstack(out) if out else np.empty((0, T))

    for grp in ['naive', 'stress']:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        csv = {'time_s': time_axis}
        for ax, ev in zip(axes, PSTH_EVENTS):
            for st in PSTH_STATUSES:
                sub = photo[(photo['sesType'] == 'RI60')
                            & (photo['recordingLoc'] == PSTH_RECLOC)
                            & (photo['group'] == grp)
                            & (photo['event'] == ev)
                            & (photo['bout_status'] == st)]
                M = _mouse_mean_traces(sub)
                if len(M) == 0:
                    csv[f'{ev}_{st}_mean'] = np.full(T, np.nan)
                    csv[f'{ev}_{st}_sem']  = np.full(T, np.nan)
                    csv[f'{ev}_{st}_nmice'] = np.full(T, 0)
                    continue
                mean = np.nanmean(M, axis=0)
                nvalid = np.isfinite(M).sum(0)
                sem = (np.nanstd(M, axis=0, ddof=1) / np.sqrt(np.maximum(nvalid, 1))
                       if len(M) > 1 else np.zeros(T))
                ax.plot(time_axis, mean, color=PSTH_STATUS_COL[st], lw=1.8,
                        label=f'{st} (n={len(M)} mice)')
                ax.fill_between(time_axis, mean - sem, mean + sem,
                                color=PSTH_STATUS_COL[st], alpha=0.25)
                csv[f'{ev}_{st}_mean'] = mean
                csv[f'{ev}_{st}_sem']  = sem
                csv[f'{ev}_{st}_nmice'] = np.full(T, len(M))
            ax.axvline(0, ls='--', color='k', lw=0.7)
            ax.set_title(f'{ev}')
            ax.set_xlabel('time from poke (s)')
            ax.legend(fontsize=7, loc='upper right')
        axes[0].set_ylabel('signal (trimTrace)')
        fig.suptitle(f'{grp} | {PSTH_RECLOC} | event PSTH split by bout status '
                     f'(mouse-level mean ± SEM)', y=1.02)
        plt.tight_layout()
        plt.savefig(f'psth_bout_status_{PSTH_RECLOC}_{grp}.png',
                    dpi=140, bbox_inches='tight')
        plt.close()
        pd.DataFrame(csv).to_csv(
            f'prism_psth_bout_status_{PSTH_RECLOC}_{grp}.csv', index=False)
        print(f"  Wrote psth_bout_status_{PSTH_RECLOC}_{grp}.png, "
              f"prism_psth_bout_status_{PSTH_RECLOC}_{grp}.csv", flush=True)


# ===========================================================
# Stage: time_to_RePE
# ===========================================================
def stage_time_to_RePE(state):
    """Add 'timeTo_RePE' column to photoDF for ReNP events.

    Latency = (first reward-port-entry after this ReNP) - (ReNP timestamp).
    Computed from RI30_RI60 raw session arrays (rewPokeTimestamps,
    rewardEntryTS), NOT from photoDF event ordering, because photoDF events
    may have been removed during artifact processing.

    For each rewarded poke i, the retrieval is the first entry in rewardEntryTS
    within [poke_ts[i], poke_ts[i+1]). NaN if the next reward was triggered
    before this one was retrieved (genuine no-retrieval case).
    """
    print("\n=== Stage: time_to_RePE ===", flush=True)

    if not os.path.exists(FEATHER_PHOTO_OUT):
        raise FileNotFoundError(
            f"need photo_merge output first: {FEATHER_PHOTO_OUT}")
    photo = pd.read_feather(FEATHER_PHOTO_OUT)
    sess  = pd.read_feather(FEATHER_RI)
    print(f"Loaded photo: {len(photo):,} rows", flush=True)
    print(f"Loaded sess:  {len(sess):,} sessions "
          f"({sess['sesType'].value_counts().to_dict()})", flush=True)

    # Normalize mouse naming for matching (photoDF '578_1-1' vs sess '578.1-1').
    # Don't overwrite the real 'mouse' column.
    photo['_mouse_match'] = photo['mouse'].astype(str).str.replace('_', '.', regex=False)
    sess['_mouse_match']  = sess['mouse'].astype(str).str.replace('_', '.', regex=False)

    photo['timeTo_RePE'] = np.nan

    TS_TOL = 0.01  # timestamp match tolerance (s)
    n_matched, n_no_retr = 0, 0
    n_renp_total = int((photo['event'] == 'ReNP').sum())

    for _, srow in sess.iterrows():
        mouse, date = srow['_mouse_match'], srow['date']
        poke_ts  = np.asarray(srow['rewPokeTimestamps'], dtype=float)
        entry_ts = np.asarray(srow['rewardEntryTS'],     dtype=float)
        if poke_ts.size == 0 or entry_ts.size == 0:
            continue

        # Per-poke retrieval latency
        next_poke = np.concatenate([poke_ts[1:], [np.inf]])
        latencies = np.full(poke_ts.shape, np.nan)
        for i, (p, np_p) in enumerate(zip(poke_ts, next_poke)):
            cand = entry_ts[(entry_ts >= p) & (entry_ts < np_p)]
            if cand.size > 0:
                latencies[i] = cand[0] - p

        # Map back to photoDF ReNP rows for this session
        mask = ((photo['_mouse_match'] == mouse)
                & (photo['date']  == date)
                & (photo['event'] == 'ReNP'))
        idxs   = photo.index[mask].to_numpy()
        sub_ts = photo.loc[mask, 'timestamp'].to_numpy(dtype=float)

        for didx, ts in zip(idxs, sub_ts):
            hits = np.where(np.abs(poke_ts - ts) <= TS_TOL)[0]
            if hits.size > 0:
                lat = latencies[hits[0]]
                photo.at[didx, 'timeTo_RePE'] = lat
                if np.isnan(lat):
                    n_no_retr += 1
                else:
                    n_matched += 1

    photo = photo.drop(columns=['_mouse_match'])

    # ---- Report ----
    n_no_session_or_ts = n_renp_total - n_matched - n_no_retr
    print(f"\nReNP coverage:", flush=True)
    print(f"  total ReNPs:                              {n_renp_total:,}", flush=True)
    print(f"  assigned latency:                         {n_matched:,} "
          f"({100*n_matched/n_renp_total:.1f}%)", flush=True)
    print(f"  NaN, no retrieval before next reward:     {n_no_retr:,} "
          f"({100*n_no_retr/n_renp_total:.1f}%)", flush=True)
    print(f"  NaN, no matching session/timestamp:       {n_no_session_or_ts:,} "
          f"({100*n_no_session_or_ts/n_renp_total:.1f}%)", flush=True)

    lat = photo.loc[photo['event']=='ReNP','timeTo_RePE'].dropna().values
    if len(lat):
        print(f"\nLatency distribution (s) over n={len(lat):,}:", flush=True)
        print(f"  median={np.median(lat):.2f}  mean={lat.mean():.2f}  "
              f"max={lat.max():.1f}", flush=True)
        print(f"  pct  25/50/75/90/99: "
              f"{np.percentile(lat,25):.2f} / {np.percentile(lat,50):.2f} / "
              f"{np.percentile(lat,75):.2f} / {np.percentile(lat,90):.2f} / "
              f"{np.percentile(lat,99):.2f}", flush=True)

    photo.to_feather(FEATHER_PHOTO_TTR)
    print(f"\nWrote {FEATHER_PHOTO_TTR}  "
          f"({len(photo):,} rows, {len(photo.columns)} cols)", flush=True)

# ===========================================================
# Main
# ===========================================================
STAGE_FUNCS = [
    ('fit_main',     stage_fit_main),
    ('regressions',  stage_regressions),
    ('fit_quality',  stage_fit_quality),
    ('trajectories', stage_trajectories),
    ('plot_data',    stage_plot_data),
    ('reward_aligned', stage_reward_aligned),
    ('photo_merge',  stage_photo_merge),
    ('bout_assign',  stage_bout_assign),
    ('time_to_RePE', stage_time_to_RePE),
]

def main():
    log_f = open('hier_mix.log','w',encoding='utf-8')
    try: sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError: pass
    sys.stdout = _Tee(sys.__stdout__, log_f)

    t0 = time.time()
    print(f"Day filter: MAX_DAY_INCLUSIVE = {MAX_DAY_INCLUSIVE}", flush=True)
    force_set = expand_force(FORCE_REDO)
    if force_set:
        print(f"FORCE_REDO expanded to: {sorted(force_set)}", flush=True)

    state = {}
    print("Plan:", flush=True)
    for name, _ in STAGE_FUNCS:
        will = stage_needs_run(name, force_set)
        print(f"  {'RUN ' if will else 'skip'} {name}", flush=True)

    for name, fn in STAGE_FUNCS:
        if stage_needs_run(name, force_set):
            fn(state)
        else:
            print(f"\n--- Skipping {name} (outputs exist) ---", flush=True)

    print(f"\nTotal runtime: {time.time()-t0:.1f}s", flush=True)

if __name__ == '__main__':
    main()