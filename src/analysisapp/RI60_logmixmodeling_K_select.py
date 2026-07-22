"""
K sweep with elbow diagnostics for the hierarchical lognormal mixture.

For each K in 2..8:
  - Runs 5 random-init EM restarts (each to ~convergence) + polishes the winner
  - Records the converged mu vector of every restart for stability analysis
  - Computes per-session KS distance to empirical CDF for the polished best fit
  - Saves global components and session weights

Three diagnostics for picking K:
  1. delta_bic_per_param  -- when this falls toward ~1, BIC is no longer
                             rewarding new components meaningfully
  2. ks_median  -- when this stops dropping, you've captured the structure
                   that matters for fit quality
  3. mu_std_avg, n_reliable_modes  -- when components stop being reproducible
                                       across random restarts, the model is
                                       no longer identifiable

Outputs:
  K_sweep_summary.csv     one row per K with all diagnostics + deltas
  K_sweep_components.csv  one row per (K, component): global mu, sigma, mean pi
  K_sweep_session_pis.csv one row per (K, session): per-session weights + KS
  K_sweep_fits.pkl        full fit objects, in case you want to replot
  K_sweep.log             everything that printed to console
"""
import numpy as np
import pandas as pd
import pickle, time, sys, os, warnings
from scipy.special import logsumexp
from scipy.stats import lognorm
warnings.filterwarnings('ignore')

# ---- Edit these for your machine ----
from _config import RI60_DIR
os.chdir(os.path.join(RI60_DIR, "behaviormodels", "k_select"))
FEATHER = os.path.join(RI60_DIR, "RI30_RI60_dataFrame.feather")

LOG_2PI = np.float32(np.log(2*np.pi))

# ----------------------------------------------------------
# EM core
# ----------------------------------------------------------
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

def _run_em(all_x, all_j, ns_j, mu, sigma, pi, J, K, max_iter=250, tol=1e-6):
    prev = -np.inf
    for it in range(max_iter):
        mu, sigma, pi, ll = _em_step(all_x, all_j, ns_j, mu, sigma, pi, J, K)
        if abs(ll-prev) < tol*(abs(prev)+1): break
        prev = ll
    return mu, sigma, pi, ll, it+1

# ----------------------------------------------------------
# Data loading (same as your main script)
# ----------------------------------------------------------
def load_ri60():
    ri = pd.read_feather(FEATHER)
    d = ri[ri['sesType']=='RI60'].copy().reset_index(drop=True)
    irts_list=[]; meta=[]
    for _, row in d.iterrows():
        ts = np.sort(np.array(row['allPokeTimestamps']))
        if len(ts) < 30: continue
        irts = np.diff(ts); irts = irts[irts > 0.001]
        if len(irts) < 30: continue
        try: day = int(float(row['dayOnType']))
        except: day = -1
        irts_list.append(irts)
        meta.append(dict(
            session_idx=len(irts_list)-1,
            mouse=row['mouse'], group=row['group'], sex=row['sex'],
            day=day, date=row['date'], n_irts=len(irts),
        ))
    return irts_list, pd.DataFrame(meta)

# ----------------------------------------------------------
# Fit K with multiple restarts, track each restart for stability
# ----------------------------------------------------------
def fit_K_with_restarts(all_x, all_j, ns_j, J, N, K,
                         n_restarts=5, restart_iter=80, polish_iter=300,
                         seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    restart_records = []
    best_ll = -np.inf; best_state = None

    for rs in range(n_restarts):
        qs = np.sort(rng.uniform(0.05, 0.95, size=K))
        mu0 = np.quantile(all_x, qs).astype(np.float32)
        sigma0 = (np.float32(0.6) + rng.uniform(-0.1, 0.1, K).astype(np.float32))
        pi0 = np.full((J, K), 1.0/K, dtype=np.float32)
        mu, sigma, pi, ll, nit = _run_em(
            all_x, all_j, ns_j, mu0, sigma0, pi0, J, K,
            max_iter=restart_iter, tol=1e-5)
        order = np.argsort(mu)
        restart_records.append(dict(
            restart=rs,
            mu=np.array(mu[order]),
            sigma=np.array(sigma[order]),
            ll=ll, nit=nit,
        ))
        if verbose:
            mus_str = ", ".join(f"{np.exp(m):.2f}s" for m in mu[order])
            print(f"  [K={K}] restart {rs}: it={nit}  ll={ll:.1f}  "
                  f"centers=[{mus_str}]", flush=True)
        if ll > best_ll:
            best_ll = ll
            best_state = (mu.copy(), sigma.copy(), pi.copy())

    # Polish the winner to tighter tol
    mu, sigma, pi = best_state
    mu, sigma, pi, ll, nit = _run_em(
        all_x, all_j, ns_j, mu, sigma, pi, J, K,
        max_iter=polish_iter, tol=1e-7)
    if verbose:
        print(f"  [K={K}] polish: it={nit}  ll={ll:.1f}", flush=True)

    n_params = 2*K + J*(K-1)
    fit = dict(mu=mu, sigma=sigma, pi=pi, log_lik=ll,
               bic=n_params*np.log(N) - 2*ll,
               aic=2*n_params - 2*ll,
               K=K, J=J, N=N, n_params=n_params)
    order = np.argsort(fit['mu'])
    fit['mu']    = fit['mu'][order]
    fit['sigma'] = fit['sigma'][order]
    fit['pi']    = fit['pi'][:, order]
    return fit, restart_records

# ----------------------------------------------------------
# Stability metrics
# ----------------------------------------------------------
def stability_metrics(restart_records):
    """
    Across restarts, how reproducible are the converged components?

    Components are sorted by mu within each restart, so component k of restart i
    is matched to component k of restart j by their position in that ordering.

    Returns:
      ll_best         - best log-likelihood across restarts
      ll_range        - max - min log-lik (small = restarts agree on the optimum)
      mu_std_avg      - mean over components of std(mu_k) across restarts (in log-time units)
      mu_std_max      - max  over components of std(mu_k) across restarts
      sigma_std_avg   - same for sigma
      n_reliable_modes - count of components with mu_std_k < 0.10 (reliable across restarts)
    """
    mus  = np.array([r['mu']    for r in restart_records])  # (n_restarts, K)
    sigs = np.array([r['sigma'] for r in restart_records])
    lls  = np.array([r['ll']    for r in restart_records])

    mu_std  = mus.std(axis=0)
    sig_std = sigs.std(axis=0)
    return dict(
        ll_best=float(lls.max()),
        ll_std=float(lls.std()),
        ll_range=float(lls.max() - lls.min()),
        mu_std_avg=float(mu_std.mean()),
        mu_std_max=float(mu_std.max()),
        sigma_std_avg=float(sig_std.mean()),
        n_reliable_modes=int((mu_std < 0.10).sum()),
    )

# ----------------------------------------------------------
# Per-session KS distance to empirical CDF
# ----------------------------------------------------------
def per_session_ks(irts_list, fit):
    K = fit['K']
    out = np.empty(len(irts_list))
    for sidx, irts in enumerate(irts_list):
        x = np.sort(irts)
        F_emp = np.arange(1, len(x)+1) / len(x)
        F_fit = np.zeros_like(x, dtype=float)
        for k in range(K):
            pk = float(fit['pi'][sidx, k])
            s  = float(fit['sigma'][k])
            sc = np.exp(float(fit['mu'][k]))
            F_fit += pk * lognorm.cdf(x, s=s, scale=sc)
        out[sidx] = float(np.max(np.abs(F_emp - F_fit)))
    return out

# ----------------------------------------------------------
# Tee logger (utf-8 safe)
# ----------------------------------------------------------
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s); st.flush()
    def flush(self):
        for st in self.streams: st.flush()

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    log_f = open('K_sweep.log', 'w', encoding='utf-8')
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
    sys.stdout = _Tee(sys.__stdout__, log_f)

    t0 = time.time()
    print("Loading data...", flush=True)
    irts_list, meta_df = load_ri60()
    print(f"Sessions: {len(irts_list)}  mice: {meta_df['mouse'].nunique()}  "
          f"total IRTs: {sum(len(x) for x in irts_list):,}")
    print(f"Per group: {meta_df.groupby('group').size().to_dict()}\n")

    # Build flat arrays once
    xs=[]; js=[]
    for j, irts in enumerate(irts_list):
        xs.append(np.log(irts).astype(np.float32))
        js.append(np.full(len(irts), j, dtype=np.int32))
    all_x = np.concatenate(xs); all_j = np.concatenate(js)
    J = len(irts_list); N = len(all_x)
    ns_j = np.array([len(x) for x in irts_list], dtype=np.float32)

    K_range = list(range(2, 9))   # K = 2..8
    summary_rows = []
    component_rows = []
    sess_pi_rows = []
    fits = {}

    for K in K_range:
        tK = time.time()
        print(f"\n=== K = {K} ===", flush=True)
        fit, restarts = fit_K_with_restarts(
            all_x, all_j, ns_j, J, N, K,
            n_restarts=50, restart_iter=300, polish_iter=400,
            seed=K*101, verbose=True,
        )
        fits[K] = fit

        stab = stability_metrics(restarts)
        ks_arr = per_session_ks(irts_list, fit)
        ks_med  = float(np.median(ks_arr))
        ks_p75  = float(np.percentile(ks_arr, 75))
        ks_p95  = float(np.percentile(ks_arr, 95))
        ks_max  = float(np.max(ks_arr))

        is_naive  = meta_df['group'].values == 'naive'
        is_stress = meta_df['group'].values == 'stress'
        ks_naive_med  = float(np.median(ks_arr[is_naive]))
        ks_stress_med = float(np.median(ks_arr[is_stress]))

        dt = time.time() - tK
        print(f"  -> BIC={fit['bic']:.1f}  best_ll={fit['log_lik']:.1f}  "
              f"KS_med={ks_med:.4f}  KS_p95={ks_p95:.4f}  ({dt:.1f}s)", flush=True)
        print(f"     stability: ll_range={stab['ll_range']:.1f}  "
              f"mu_std_avg={stab['mu_std_avg']:.3f}  "
              f"mu_std_max={stab['mu_std_max']:.3f}  "
              f"reliable_modes={stab['n_reliable_modes']}/{K}", flush=True)
        for k in range(K):
            print(f"     c{k}: mean_IRT={np.exp(fit['mu'][k]):7.3f}s  "
                  f"sigma_log={fit['sigma'][k]:.3f}  "
                  f"mean_pi={fit['pi'][:,k].mean():.3f}", flush=True)

        summary_rows.append(dict(
            K=K,
            log_lik=fit['log_lik'],
            bic=fit['bic'],
            aic=fit['aic'],
            n_params=fit['n_params'],
            ks_median=ks_med, ks_p75=ks_p75, ks_p95=ks_p95, ks_max=ks_max,
            ks_naive_median=ks_naive_med, ks_stress_median=ks_stress_med,
            **stab,
            wallclock_s=dt,
        ))

        for k in range(K):
            component_rows.append(dict(
                K=K, component=k,
                mu_log=float(fit['mu'][k]),
                mean_IRT_s=float(np.exp(fit['mu'][k])),
                sigma_log=float(fit['sigma'][k]),
                mean_pi=float(fit['pi'][:,k].mean()),
            ))
        for sidx in range(J):
            row = dict(K=K, session_idx=sidx,
                       mouse=meta_df.iloc[sidx]['mouse'],
                       group=meta_df.iloc[sidx]['group'],
                       day=int(meta_df.iloc[sidx]['day']),
                       n_irts=int(meta_df.iloc[sidx]['n_irts']),
                       ks_distance=float(ks_arr[sidx]))
            for k in range(K):
                row[f'pi_{k}'] = float(fit['pi'][sidx, k])
            sess_pi_rows.append(row)

    # ----- Build and save summary -----
    summary = pd.DataFrame(summary_rows).sort_values('K').reset_index(drop=True)
    summary['delta_bic']            = summary['bic'].diff()
    summary['delta_log_lik']        = summary['log_lik'].diff()
    summary['delta_n_params']       = summary['n_params'].diff()
    summary['delta_bic_per_param']  = summary['delta_bic'] / summary['delta_n_params']
    summary['delta_ks_median']      = summary['ks_median'].diff()

    print("\n========== SUMMARY ==========\n")
    cols = ['K','log_lik','bic','delta_bic','delta_bic_per_param',
            'ks_median','ks_p95','mu_std_avg','mu_std_max','n_reliable_modes']
    print(summary[cols].round(4).to_string(index=False))

    print("\n========== HOW TO READ ==========")
    print("BIC will probably keep dropping at this N -- that is expected.")
    print("delta_bic_per_param: when this falls toward ~1, BIC is no longer")
    print("  meaningfully rewarding new components.")
    print("ks_median: when this stops dropping (or drops by < 0.001 per K), you")
    print("  have captured the structure that matters for fit quality.")
    print("mu_std_avg, mu_std_max: log-time std of fitted component centers across")
    print("  the 5 random restarts. Small = restarts converge to the same answer.")
    print("  Above ~0.2 means at least one component is no longer reliably found,")
    print("  i.e. the model is overspecified.")
    print("n_reliable_modes: number of components whose mu_std < 0.10. Should equal")
    print("  K. When it falls below K, your data cannot reliably support that many")
    print("  components and you should stop one K earlier.")
    print()
    print("ELBOW = the largest K such that:")
    print("  (a) ks_median is still meaningfully decreasing")
    print("  (b) n_reliable_modes is still equal to K")
    print("  (c) delta_bic_per_param is still well above 1")
    print("If all three say 'stop' at the same K, that K is the answer. If they")
    print("disagree, prefer the smaller K (the more falsifiable model).")

    summary.to_csv('K_sweep_summary.csv', index=False)
    pd.DataFrame(component_rows).to_csv('K_sweep_components.csv', index=False)
    pd.DataFrame(sess_pi_rows).to_csv('K_sweep_session_pis.csv', index=False)
    with open('K_sweep_fits.pkl', 'wb') as f:
        pickle.dump(dict(summary=summary.to_dict(orient='records'),
                         meta_df=meta_df, fits=fits), f)

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
    print("Files: K_sweep_summary.csv, K_sweep_components.csv, "
          "K_sweep_session_pis.csv, K_sweep_fits.pkl, K_sweep.log")

if __name__ == '__main__':
    main()