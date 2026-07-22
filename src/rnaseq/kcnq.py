#!/usr/bin/env python3
"""
Kcnq2/3/5 expression across putative CeA projection groups.

Projection groups defined by marker genes from Wang & Sternson (eLife 2023)
Figure 6 logistic regression: Drd1+/Sema3c- → SN, Dlk1+ → BNST, Crh+ → PAG.

Requires elife_poa_e84262_Supplementary_file_1.xlsx in data/GSE213828/

To choose which steps run: edit the RUN list near the top of this file,
then just click Run in VSCode. e.g. set RUN = ['plots'] to regenerate
only the plots and skip the slow GLM. The (cached) data load + group
definitions always run first.
(Power-user option: passing steps on the command line overrides RUN,
e.g. python kcnq.py plots export — but you never have to.)

Outputs (under OUT_DIR):
  kcnq_by_projection_group.png
  kcnq_proportions_by_projection.png
  Kcnq{2,3,5}_for_prism.csv
  kcnq_grouped_proj_x_gene_for_prism.csv   rows=projection, cols=gene, cells=replicates
  kcnq3_over_q3q5_ratio_for_prism.csv
  kcnq_longformat.csv
  kcnq_stats.txt   Kruskal-Wallis + Dunn post-hoc + heterodimer (non-parametric)
  glm_negbinomial/   kcnq_glm_negbinomial.txt + _coefs.csv  (standard NB, MLE dispersion)
  glm_zeroinflated/  kcnq_glm_zeroinflated.txt + _coefs.csv (zero-inflated NB + NB-vs-ZINB)
"""

import os, sys, contextlib, warnings
from types import SimpleNamespace
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from scipy.stats import kruskal, chi2
from _config import RNASEQ_DIR
warnings.filterwarnings('ignore')

# Tables/help text use non-ASCII glyphs (─, →); keep them printable on cp1252 consoles.
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

os.chdir(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = RNASEQ_DIR
os.makedirs(OUT_DIR, exist_ok=True)
GLM_NB_DIR = os.path.join(OUT_DIR, 'glm_negbinomial')    # standard negative-binomial GLM
GLM_ZI_DIR = os.path.join(OUT_DIR, 'glm_zeroinflated')   # zero-inflated negative-binomial

# ============================================================
# WHICH STEPS TO RUN  —  edit this, then click Run in VSCode.
# Comment out (#) the ones you don't want to re-run.
# ============================================================
RUN = [
    'stats',     # Kruskal-Wallis + Dunn post-hoc + heterodimer tables
    'glm',       # standard negative-binomial GLM   -> glm_negbinomial/
    'glm_zinb',  # zero-inflated negative-binomial  -> glm_zeroinflated/
    'plots',     # the two PNGs
    'export',    # the Prism CSVs
]


def sig(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'


class _Tee:
    """Write to several streams at once (console + a .txt log)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


@contextlib.contextmanager
def tee_stdout(path):
    """Mirror everything printed inside the block to `path` (UTF-8)."""
    with open(path, 'w', encoding='utf-8') as f:
        old = sys.stdout
        sys.stdout = _Tee(old, f)
        try:
            yield
        finally:
            sys.stdout = old


def dunn_posthoc(groups_vals, labels):
    """Dunn's post-hoc (Benjamini-Hochberg adjusted) — the correct follow-up to a
    significant Kruskal-Wallis. groups_vals: dict label->1D array.
    Returns [(g1, g2, p_adj), ...] for each unordered pair."""
    import scikit_posthocs as sp   # pip install scikit-posthocs
    arrs = [np.asarray(groups_vals[g]) for g in labels]
    m = sp.posthoc_dunn(arrs, p_adjust='fdr_bh')   # k x k DataFrame (1-indexed)
    out = []
    for i, g1 in enumerate(labels):
        for j in range(i + 1, len(labels)):
            out.append((g1, labels[j], float(m.iloc[i, j])))
    return out


def coef_table(res):
    """Coefficient table (coef/SE/z/p/95% CI) — works for GLM and discrete results."""
    ci = np.asarray(res.conf_int())
    return pd.DataFrame({'coef': np.asarray(res.params), 'std_err': np.asarray(res.bse),
                         'z': np.asarray(res.tvalues), 'p': np.asarray(res.pvalues),
                         'ci_low': ci[:, 0], 'ci_high': ci[:, 1]},
                        index=getattr(res.params, 'index', None))


def add_sig_bars(ax, pairs, color='k'):
    """Draw stacked significance brackets above the data.
    pairs: [(x_i, x_j, p_adj), ...]; only p_adj < 0.05 are drawn, labelled with sig()."""
    pairs = [(i, j, p) for i, j, p in pairs if p < 0.05]
    if not pairs:
        return
    ytop = ax.get_ylim()[1]
    step = ytop * 0.09
    tick = step * 0.25
    pairs = sorted(pairs, key=lambda t: abs(t[1] - t[0]))   # shortest spans drawn lowest
    y = ytop * 1.01
    for xi, xj, p in pairs:
        ax.plot([xi, xi, xj, xj], [y, y + tick, y + tick, y], lw=1.1, c=color, clip_on=False)
        ax.text((xi + xj) / 2.0, y + tick, sig(p), ha='center', va='bottom', fontsize=11, color=color)
        y += step
    ax.set_ylim(top=y)


# ============================================================
# PREP: LOAD / QC / DEFINE GROUPS / DERIVED TABLES (always runs)
# ============================================================
def prep():
    # ---- load / download / QC ----
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'GSE213828')
    os.makedirs(data_dir, exist_ok=True)
    h5ad_path = os.path.join(data_dir, 'cea_scrna_v2.h5ad')

    if os.path.exists(h5ad_path):
        adata = sc.read_h5ad(h5ad_path)
    else:
        import urllib.request
        counts_file = os.path.join(data_dir, 'GSE213828_merged_counts.txt.gz')
        info_file = os.path.join(data_dir, 'GSE213828_merged_sample_info.txt.gz')
        base_http = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE213nnn/GSE213828/suppl/"
        for fname, fpath in [('GSE213828_merged_counts.txt.gz', counts_file),
                             ('GSE213828_merged_sample_info.txt.gz', info_file)]:
            if not os.path.exists(fpath):
                print(f"  Downloading {fname}...")
                urllib.request.urlretrieve(base_http + fname, fpath)

        df_counts = pd.read_csv(counts_file, sep='\t', compression='gzip')
        df_counts = df_counts.set_index('gene_name').drop(columns=['gene_id'])
        df_counts = df_counts[~df_counts.index.duplicated(keep='first')]
        adata = sc.AnnData(df_counts.T)

        df_info = pd.read_csv(info_file, sep='\t', index_col=0, compression='gzip')
        for col in df_info.columns:
            adata.obs[col] = df_info.reindex(adata.obs_names)[col]

        # QC
        adata = adata[adata.obs['sample'] == 'CEA'].copy()
        adata.obs['pct_ercc'] = pd.to_numeric(adata.obs['pct_ercc'], errors='coerce')
        adata.obs['gene_det'] = pd.to_numeric(adata.obs['gene_det'], errors='coerce')
        adata = adata[(adata.obs['pct_ercc'] <= 25) & (adata.obs['gene_det'] >= 2000)].copy()
        sc.pp.filter_genes(adata, min_cells=3)
        adata.raw = adata
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat')
        sc.pp.pca(adata, n_comps=30)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        adata.write_h5ad(h5ad_path)
        print(f"  Saved: {h5ad_path}")

    adata_raw = adata.raw.to_adata() if adata.raw is not None else adata
    print(f"  Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

    def get_expr(gene):
        v = adata_raw[:, gene].X
        if hasattr(v, 'toarray'):
            v = v.toarray()
        return np.array(v).flatten()

    # ---- define projection groups ----
    drd1, sema3c, dlk1, crh = get_expr('Drd1'), get_expr('Sema3c'), get_expr('Dlk1'), get_expr('Crh')

    sn_mask = (drd1 > 0) & (sema3c == 0)
    bnst_mask = dlk1 > 0
    pag_mask = crh > 0

    # Priority: PAG > BNST > SN (last assignment wins)
    proj_group = np.full(adata_raw.shape[0], 'other', dtype=object)
    proj_group[sn_mask] = 'SN (Drd1+/Sema3c-)'
    proj_group[bnst_mask] = 'BNST (Dlk1+)'
    proj_group[pag_mask] = 'PAG (Crh+)'

    GROUPS = ['SN (Drd1+/Sema3c-)', 'BNST (Dlk1+)', 'PAG (Crh+)']
    ALL_GROUPS = GROUPS + ['other']
    KCNQ = ['Kcnq2', 'Kcnq3', 'Kcnq5']

    print("=" * 70)
    print("PUTATIVE PROJECTION GROUPS")
    print("=" * 70)
    for g in ALL_GROUPS:
        print(f"  {g:<22} {np.sum(proj_group == g):>5}")
    print(f"  {'Total':<22} {len(proj_group):>5}")

    print(f"\n  Overlaps (before priority):")
    print(f"    SN & BNST: {np.sum(sn_mask & bnst_mask)}")
    print(f"    SN & PAG:  {np.sum(sn_mask & pag_mask)}")
    print(f"    BNST & PAG: {np.sum(bnst_mask & pag_mask)}")

    # ---- derived tables shared by downstream steps ----
    group_data = {}
    for grp in ALL_GROUPS:
        mask = proj_group == grp
        group_data[grp] = {gene: get_expr(gene)[mask] for gene in KCNQ}

    q2, q3, q5 = get_expr('Kcnq2'), get_expr('Kcnq3'), get_expr('Kcnq5')

    # long-format counts (for GLM + export)
    rows = []
    for grp in GROUPS:
        mask = proj_group == grp
        for gene in KCNQ:
            for v in get_expr(gene)[mask]:
                rows.append({'projection': grp, 'gene': gene, 'count': int(v)})
    df_long = pd.DataFrame(rows)

    # Kcnq3/(Kcnq3+Kcnq5) ratio per group (for heterodimer stats + export)
    ratio_data = {}
    for grp in GROUPS:
        mask = proj_group == grp
        q3v, q5v = q3[mask], q5[mask]
        has_either = (q3v > 0) | (q5v > 0)
        if has_either.sum() > 0:
            ratio_data[grp] = q3v[has_either] / (q3v[has_either] + q5v[has_either])

    return SimpleNamespace(
        adata=adata, adata_raw=adata_raw, get_expr=get_expr, proj_group=proj_group,
        GROUPS=GROUPS, ALL_GROUPS=ALL_GROUPS, KCNQ=KCNQ,
        group_data=group_data, q2=q2, q3=q3, q5=q5, df_long=df_long, ratio_data=ratio_data,
    )


# ============================================================
# STATS: descriptive + nonparametric tests + heterodimer
# ============================================================
def run_stats(ctx):
    GROUPS, KCNQ = ctx.GROUPS, ctx.KCNQ   # 'other' is intentionally excluded everywhere here
    group_data, proj_group = ctx.group_data, ctx.proj_group
    q2, q3, q5, ratio_data = ctx.q2, ctx.q3, ctx.q5, ctx.ratio_data

    # ---- KCNQ expression by group ----
    print(f"\n{'=' * 70}")
    print("KCNQ EXPRESSION BY PROJECTION GROUP")
    print(f"{'=' * 70}")

    print(f"\n  {'Group':<22} {'n':>5}  {'Kcnq2':>14}  {'Kcnq3':>14}  {'Kcnq5':>14}")
    print(f"  {'-' * 75}")
    for grp in GROUPS:
        n = len(group_data[grp]['Kcnq2'])
        parts = [f"  {grp:<22} {n:>5}"]
        for gene in KCNQ:
            vals = group_data[grp][gene]
            parts.append(f"  {np.mean(vals):>6.1f}({100 * np.mean(vals > 0):>3.0f}%)")
        print(''.join(parts))

    print(f"\n  {'Group':<22} {'Q2:Q3':>8} {'Q2:Q5':>8} {'Q3:Q5':>8}")
    print(f"  {'-' * 50}")
    for grp in GROUPS:
        m = {g: np.mean(group_data[grp][g]) for g in KCNQ}
        r23 = m['Kcnq2'] / m['Kcnq3'] if m['Kcnq3'] > 0 else np.inf
        r25 = m['Kcnq2'] / m['Kcnq5'] if m['Kcnq5'] > 0 else np.inf
        r35 = m['Kcnq3'] / m['Kcnq5'] if m['Kcnq5'] > 0 else np.inf
        print(f"  {grp:<22} {r23:>7.1f}x {r25:>7.1f}x {r35:>7.1f}x")

    # ---- per-gene statistics ----
    print(f"\n{'=' * 70}")
    print("PER-GENE STATISTICS (Kruskal-Wallis + Dunn post-hoc)")
    print(f"{'=' * 70}")
    print("  (counts are non-Gaussian -> rank-based omnibus + Dunn, BH-adjusted)")

    for gene in KCNQ:
        data = [group_data[g][gene] for g in GROUPS]
        h, p = kruskal(*data)
        print(f"\n  {gene}")
        print(f"  {'─' * 50}")
        for grp in GROUPS:
            vals = group_data[grp][gene]
            med = np.median(vals)
            q1, q3p = np.percentile(vals, [25, 75])
            print(f"    {grp:<22} n={len(vals):<4}  median={med:>5.1f}  IQR=[{q1:.0f}, {q3p:.0f}]")
        print(f"\n    Kruskal-Wallis: H={h:.2f}, p={p:.4f} {sig(p)}")
        if p < 0.05:
            print(f"    Post-hoc Dunn (Benjamini-Hochberg adjusted):")
            for g1, g2, padj in dunn_posthoc({g: group_data[g][gene] for g in GROUPS}, GROUPS):
                print(f"      {g1} vs {g2}  p_adj={padj:.4f} {sig(padj)}")
        else:
            print(f"    Omnibus n.s. (p>=0.05) -> Dunn post-hoc not performed.")

    # ---- heterodimer co-expression ----
    print(f"\n{'=' * 70}")
    print("HETERODIMER CO-EXPRESSION BY PROJECTION GROUP")
    print(f"{'=' * 70}")

    print(f"\n  {'Group':<22} {'n':>4}  {'Q2+Q3+':>7} {'Q2+Q5+':>7} {'Q3+Q5+':>7} {'Q2only':>7} {'allQ-':>7}")
    print(f"  {'-' * 68}")
    for grp in GROUPS:
        mask = proj_group == grp
        q2p, q3p, q5p = q2[mask] > 0, q3[mask] > 0, q5[mask] > 0
        print(f"  {grp:<22} {mask.sum():>4}  "
              f"{100 * np.mean(q2p & q3p):>5.1f}%  "
              f"{100 * np.mean(q2p & q5p):>5.1f}%  "
              f"{100 * np.mean(q3p & q5p):>5.1f}%  "
              f"{100 * np.mean(q2p & ~q3p & ~q5p):>5.1f}%  "
              f"{100 * np.mean(~q2p & ~q3p & ~q5p):>5.1f}%")

    # Q3/(Q3+Q5) ratio
    print(f"\n  Kcnq3/(Kcnq3+Kcnq5) ratio (cells with Q3>0 or Q5>0 only)")
    print(f"  Higher = Q2/3 dominant, Lower = Q2/5 dominant")
    print(f"\n  {'Group':<22} {'n':>4}  {'median':>7} {'IQR':>14}")
    print(f"  {'-' * 52}")
    for grp in GROUPS:
        if grp in ratio_data:
            ratio = ratio_data[grp]
            med = np.median(ratio)
            q1r, q3r = np.percentile(ratio, [25, 75])
            print(f"  {grp:<22} {len(ratio):>4}  {med:>6.3f}  [{q1r:.3f}, {q3r:.3f}]")

    present = [g for g in GROUPS if g in ratio_data]
    if len(present) >= 2:
        h, p = kruskal(*[ratio_data[g] for g in present])
        print(f"\n  Kruskal-Wallis on Q3/(Q3+Q5): H={h:.2f}, p={p:.4f} {sig(p)}")
        if p < 0.05:
            print(f"  Post-hoc Dunn (Benjamini-Hochberg adjusted):")
            for g1, g2, padj in dunn_posthoc({g: ratio_data[g] for g in present}, present):
                print(f"    {g1} vs {g2}  p_adj={padj:.4f} {sig(padj)}")
        else:
            print(f"  Omnibus n.s. (p>=0.05) -> Dunn post-hoc not performed.")


# ============================================================
# GLM: negative binomial count ~ projection * gene
# ============================================================
def run_glm(ctx):
    GROUPS = ctx.GROUPS
    df_long = ctx.df_long
    os.makedirs(GLM_NB_DIR, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("STANDARD NEGATIVE BINOMIAL GLM: count ~ projection * gene")
    print(f"{'=' * 70}")

    # Estimate the NB dispersion (alpha) by MLE; statsmodels' GLM otherwise silently
    # fixes alpha=1.0, which is arbitrary and distorts the SEs / p-values. We reuse
    # this single estimate across all nested models so the deviance LRTs stay valid.
    nb_mle = smf.negativebinomial('count ~ C(projection) * C(gene)', data=df_long).fit(disp=0)
    alpha_hat = float(nb_mle.params['alpha'])
    print(f"  NB dispersion alpha (MLE) = {alpha_hat:.4f}  "
          f"[converged={nb_mle.mle_retvals.get('converged')}]  (default 1.0 is arbitrary)\n")

    def nb():
        return sm.families.NegativeBinomial(alpha=alpha_hat)

    model = smf.glm('count ~ C(projection) * C(gene)', data=df_long, family=nb()).fit()
    print(model.summary2().tables[1].to_string())
    coef_table(model).to_csv(os.path.join(GLM_NB_DIR, 'kcnq_glm_negbinomial_coefs.csv'))

    # Deviance tests
    print(f"\n  Deviance tests:")
    models = {
        'Projection':        'count ~ C(gene)',
        'Gene':              'count ~ C(projection)',
        'Projection x Gene': 'count ~ C(projection) + C(gene)',
    }
    for name, formula in models.items():
        m_reduced = smf.glm(formula, data=df_long, family=nb()).fit()
        dev = m_reduced.deviance - model.deviance
        dof = m_reduced.df_resid - model.df_resid
        p = 1 - chi2.cdf(dev, dof)
        print(f"    {name:<20} deviance={dev:>8.1f}  df={dof}  p={p:.4f} {sig(p)}")

    # Post-hoc pairwise projection contrasts
    print(f"\n  Post-hoc pairwise projection contrasts:")
    pairs = []
    for ref in GROUPS:
        # build the releveled frame without mutating the shared df_long (it gets exported)
        releveled = df_long.assign(proj_relevel=pd.Categorical(
            df_long['projection'], categories=[ref] + [g for g in GROUPS if g != ref]))
        m = smf.glm('count ~ C(proj_relevel) + C(gene)', data=releveled, family=nb()).fit()
        for param in m.params.index:
            if 'proj_relevel' in param:
                other = param.split('[T.')[1].rstrip(']')
                pair = tuple(sorted([ref, other]))
                if pair not in [p[0] for p in pairs]:
                    coef = m.params[param]
                    pval = m.pvalues[param]
                    fold = np.exp(coef)
                    pairs.append((pair, coef, fold, pval))

    n_comp = len(pairs)
    print(f"\n    {'Comparison':<55} {'coef':>6} {'fold':>6} {'p_raw':>10} {'p_bonf':>10} {'sig':>4}")
    print(f"    {'-' * 95}")
    for pair, coef, fold, pval in sorted(pairs, key=lambda x: x[3]):
        p_bonf = min(pval * n_comp, 1.0)
        print(f"    {pair[0]:<25} vs {pair[1]:<25} {coef:>+5.3f} {fold:>5.2f}x {pval:>10.4f} {p_bonf:>10.4f} {sig(p_bonf):>4}")


# ============================================================
# ZERO-INFLATED NB GLM: separate "structural off" from "low when on"
# ============================================================
def run_glm_zinb(ctx):
    df_long = ctx.df_long
    os.makedirs(GLM_ZI_DIR, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("ZERO-INFLATED NEGATIVE BINOMIAL GLM")
    print(f"{'=' * 70}")
    print("  count part:     count ~ C(projection) * C(gene)")
    print("  inflation part: P(structural zero) ~ C(gene)")
    print("  (Kcnq3/Kcnq5 are mostly zeros; this separates 'off' from 'low when")
    print("   on' and checks whether the projection effects survive.)\n")

    y, Xc = patsy.dmatrices('count ~ C(projection) * C(gene)', df_long, return_type='dataframe')
    _, Xi = patsy.dmatrices('count ~ C(gene)', df_long, return_type='dataframe')
    zinb = ZeroInflatedNegativeBinomialP(y, Xc, exog_infl=Xi, inflation='logit').fit(maxiter=300, disp=0)

    print(f"  converged = {zinb.mle_retvals.get('converged')}   "
          f"(rows prefixed 'inflate_' = excess-zero / logit part)\n")
    coefs = coef_table(zinb)
    print(coefs.to_string())
    coefs.to_csv(os.path.join(GLM_ZI_DIR, 'kcnq_glm_zeroinflated_coefs.csv'))

    # Is zero-inflation actually warranted vs the standard NB?
    nb = smf.negativebinomial('count ~ C(projection) * C(gene)', data=df_long).fit(disp=0)
    lr = 2 * (zinb.llf - nb.llf)
    df_infl = Xi.shape[1]
    p_lr = chi2.sf(lr, df_infl)
    print(f"\n  Is zero-inflation warranted (vs standard NB)?")
    print(f"    NB    : llf={nb.llf:>10.1f}  AIC={nb.aic:>10.1f}")
    print(f"    ZINB  : llf={zinb.llf:>10.1f}  AIC={zinb.aic:>10.1f}")
    print(f"    dAIC (ZINB - NB) = {zinb.aic - nb.aic:+.1f}   (negative => ZINB preferred)")
    print(f"    LR test for excess zeros: LR={lr:.2f}, df={df_infl}, p={p_lr:.4g} {sig(p_lr)}")
    print(f"    (inflation params tested at a boundary -> LR is conservative; read with dAIC)")


# ============================================================
# PLOTS
# ============================================================
def run_plots(ctx):
    GROUPS, KCNQ = ctx.GROUPS, ctx.KCNQ
    group_data = ctx.group_data   # per-cell counts per (group, gene); 'other' excluded
    short = {g: g.replace(' (', '\n(') for g in GROUPS}
    box_colors = {'Kcnq2': '#2196F3', 'Kcnq3': '#FF9800', 'Kcnq5': '#4CAF50'}

    # Fig 1: per-gene box-and-whisker by projection + Dunn (BH) significance bars.
    # Whiskers = 1.5*IQR; fliers hidden for readability (Dunn uses all cells).
    fig, axes = plt.subplots(3, 1, figsize=(7, 10))
    for ax, gene in zip(axes, KCNQ):
        data = [group_data[g][gene] for g in GROUPS]
        bp = ax.boxplot(data, positions=range(len(GROUPS)), widths=0.6, showfliers=False,
                        patch_artist=True, medianprops=dict(color='black', lw=1.5))
        for patch in bp['boxes']:
            patch.set_facecolor(box_colors[gene]); patch.set_alpha(0.65)
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([short[g] for g in GROUPS])
        ax.set_ylabel('transcript counts')
        _, p = kruskal(*data)
        ax.set_title(f"{gene}   (Kruskal-Wallis p={p:.3g})", fontweight='bold')
        if p < 0.05:   # bars only for a significant omnibus, from the Dunn post-hoc
            pos = {g: i for i, g in enumerate(GROUPS)}
            bars = [(pos[a], pos[b], padj) for a, b, padj in
                    dunn_posthoc({g: group_data[g][gene] for g in GROUPS}, GROUPS)]
            add_sig_bars(ax, bars)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'kcnq_by_projection_group.png'), dpi=200, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    means = {gene: [np.mean(group_data[g][gene]) for g in GROUPS] for gene in KCNQ}
    totals = np.array([sum(means[g][i] for g in KCNQ) for i in range(len(GROUPS))])
    totals[totals == 0] = 1
    bottom = np.zeros(len(GROUPS))
    colors = {'Kcnq2': '#2196F3', 'Kcnq3': '#FF9800', 'Kcnq5': '#4CAF50'}
    for gene in KCNQ:
        fracs = np.array(means[gene]) / totals
        ax.bar(range(len(GROUPS)), fracs, bottom=bottom, label=gene, color=colors[gene])
        bottom += fracs
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels(['SN\n(Drd1+/Sema3c-)', 'BNST\n(Dlk1+)', 'PAG\n(Crh+)'])
    ax.set_ylabel('Fraction of total KCNQ expression')
    ax.set_title('KCNQ subunit composition by putative projection target', fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'kcnq_proportions_by_projection.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {os.path.join(OUT_DIR, 'kcnq_by_projection_group.png')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'kcnq_proportions_by_projection.png')}")


# ============================================================
# EXPORTS FOR PRISM
# ============================================================
def run_export(ctx):
    GROUPS, KCNQ = ctx.GROUPS, ctx.KCNQ   # 'other' excluded from all exports
    get_expr, proj_group = ctx.get_expr, ctx.proj_group
    df_long, ratio_data = ctx.df_long, ctx.ratio_data

    # Per-gene Kcnq counts
    for gene in KCNQ:
        cols = {}
        max_n = 0
        for grp in GROUPS:
            vals = get_expr(gene)[proj_group == grp].astype(float)
            cols[grp] = vals
            max_n = max(max_n, len(vals))
        df_out = pd.DataFrame({grp: np.pad(cols[grp], (0, max_n - len(cols[grp])),
                               constant_values=np.nan) for grp in GROUPS})
        out_path = os.path.join(OUT_DIR, f'{gene}_for_prism.csv')
        df_out.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

    # Grouped table for Prism: rows = putative projection, columns = gene,
    # replicates = individual cells. Projections have different cell counts, so
    # each gene block is padded with blanks out to the largest projection.
    max_cells = max(int((proj_group == grp).sum()) for grp in GROUPS)
    grouped = []
    for grp in GROUPS:
        row = {'projection': grp}
        for gene in KCNQ:
            vals = get_expr(gene)[proj_group == grp].astype(float)
            vals = np.pad(vals, (0, max_cells - len(vals)), constant_values=np.nan)
            for i in range(max_cells):
                row[f'{gene}_cell{i + 1}'] = vals[i]
        grouped.append(row)
    col_order = ['projection'] + [f'{gene}_cell{i + 1}' for gene in KCNQ for i in range(max_cells)]
    grouped_path = os.path.join(OUT_DIR, 'kcnq_grouped_proj_x_gene_for_prism.csv')
    pd.DataFrame(grouped)[col_order].to_csv(grouped_path, index=False)
    print(f"Saved: {grouped_path}")

    # Heterodimer ratio
    cols = {}
    max_n = 0
    for grp in GROUPS:
        cols[grp] = ratio_data[grp].astype(float)
        max_n = max(max_n, len(cols[grp]))
    df_ratio = pd.DataFrame({grp: np.pad(cols[grp], (0, max_n - len(cols[grp])),
                             constant_values=np.nan) for grp in GROUPS})
    ratio_path = os.path.join(OUT_DIR, 'kcnq3_over_q3q5_ratio_for_prism.csv')
    df_ratio.to_csv(ratio_path, index=False)
    print(f"Saved: {ratio_path}")

    # Long-format
    long_path = os.path.join(OUT_DIR, 'kcnq_longformat.csv')
    df_long.to_csv(long_path, index=False)
    print(f"Saved: {long_path}")


# ============================================================
# DISPATCH
# ============================================================
STEPS = {
    'stats': run_stats,
    'glm': run_glm,
    'glm_zinb': run_glm_zinb,
    'plots': run_plots,
    'export': run_export,
}

# Steps whose console output is also saved to a .txt (full path -> own folder).
TXT = {
    'stats':    os.path.join(OUT_DIR, 'kcnq_stats.txt'),
    'glm':      os.path.join(GLM_NB_DIR, 'kcnq_glm_negbinomial.txt'),
    'glm_zinb': os.path.join(GLM_ZI_DIR, 'kcnq_glm_zeroinflated.txt'),
}


def main():
    # Clicking Run in VSCode passes no args -> use the RUN list at the top.
    # (Passing step names on the command line overrides RUN, if you ever do.)
    steps = sys.argv[1:] or RUN
    bad = [s for s in steps if s not in STEPS]
    if bad:
        sys.exit(f"unknown step(s) {bad}; choose from {', '.join(STEPS)}")

    ctx = prep()
    for name in STEPS:                # fixed order regardless of how listed
        if name not in steps:
            continue
        if name in TXT:               # also save stats/glm console output to a .txt
            path = TXT[name]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with tee_stdout(path):
                STEPS[name](ctx)
            print(f"Saved: {path}")
        else:
            STEPS[name](ctx)


if __name__ == '__main__':
    main()
