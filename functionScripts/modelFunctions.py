"""
modelFunctions.py
=================
Time × Stressor modelling for cFos lightsheet data.

Design notes
------------
* Each animal is sacrificed at one timepoint → purely between-subjects design.
* The dataset is **unbalanced** (FS has no 21D cohort), so classic Type-I SS
  ANOVA gives incorrect results.  We therefore use:

  Primary  : OLS with Type-III (marginal) Sums-of-Squares via sequential
             model comparison — equivalent to what statsmodels/lme4 produce
             for a fixed-effects-only model with random intercepts collapsed
             (appropriate here because animals are not repeated).
  Fallback : Type-I SS ANOVA (scipy.stats.f_oneway) used per-factor when the
             OLS design matrix is rank-deficient (e.g. a region has only a
             single non-NaN group).

Outputs
-------
* Per-region DataFrame with p-values + partial η² for Stressor, Time, Interaction
* FDR-corrected q-values (Benjamini-Hochberg)
* Ranked region table
* Three plot types: ranked bar chart, interaction heatmap, condition-means lines
"""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scipy.stats as stats
from itertools import product
from collections import defaultdict

# ─────────────────────────────────────────────
# 1.  DATA RESHAPING
# ─────────────────────────────────────────────

TIME_ORDER     = ['Acute', '7D', '14D', '21D']
STRESSOR_ORDER = ['Ctrl', 'FS', 'FSW', 'RS', 'TS']


def melt_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape wide-format CSV (regions × condition-columns) to long format.

    Returns DataFrame with columns:
        Region, Stressor, Time, Animal, Value
    """
    id_vars  = ['Region.Name', 'Region ID', 'acronym']
    val_cols = [c for c in df.columns if c.endswith('_normalized')]

    long = df.melt(id_vars=id_vars, value_vars=val_cols,
                   var_name='col_name', value_name='Value')

    pattern = re.compile(r'^([A-Za-z]+)_(Acute|\d+D)_(F\d+)_normalized$')
    parsed  = long['col_name'].str.extract(pattern)
    parsed.columns = ['Stressor', 'Time', 'Animal']

    long = pd.concat([long[id_vars], parsed, long[['Value']]], axis=1)
    long = long.rename(columns={'Region.Name': 'Region', 'Region ID': 'RegionID'})

    # Drop rows where parsing failed or value is NaN
    long = long.dropna(subset=['Stressor', 'Time', 'Animal', 'Value'])
    long['Value'] = pd.to_numeric(long['Value'], errors='coerce')
    long = long.dropna(subset=['Value'])

    # Ordered categoricals for cleaner plots
    long['Time']     = pd.Categorical(long['Time'],     categories=TIME_ORDER,     ordered=True)
    long['Stressor'] = pd.Categorical(long['Stressor'], categories=STRESSOR_ORDER, ordered=True)

    return long.reset_index(drop=True)


# ─────────────────────────────────────────────
# 2.  STATISTICAL ENGINE
# ─────────────────────────────────────────────

def _effects_dummies(series: pd.Series, categories: list) -> np.ndarray:
    """
    Sum-to-zero (effects) coding for a categorical variable.
    Returns array of shape (n, k-1).  Last category is reference (coded −1).
    """
    n, k   = len(series), len(categories)
    X      = np.zeros((n, k - 1))
    codes  = {cat: i for i, cat in enumerate(categories)}
    ref    = k - 1
    for row, val in enumerate(series):
        c = codes.get(val, None)
        if c is None:
            continue
        if c < ref:
            X[row, c] = 1.0
        else:
            X[row, :] = -1.0
    return X


def _ols_rss(X: np.ndarray, y: np.ndarray):
    """Return (RSS, df_residual) for OLS fit of y ~ X."""
    coef, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    if len(res) == 0:
        fitted = X @ coef
        rss    = float(np.sum((y - fitted) ** 2))
    else:
        rss = float(res[0])
    df_res = max(len(y) - rank, 1)
    return rss, df_res, rank


def _type3_ftest(y: np.ndarray,
                 X_full: np.ndarray,
                 X_reduced: np.ndarray,
                 df_term: int):
    """
    Type-III F-test:  compare full model vs reduced model (term removed).
    Returns (F, p, partial_eta_sq).
    """
    rss_full, df_res, rank_full = _ols_rss(X_full, y)
    rss_red,  _,     _          = _ols_rss(X_reduced, y)

    ss_term = max(rss_red - rss_full, 0.0)
    ms_term = ss_term / max(df_term, 1)
    ms_res  = rss_full / max(df_res, 1)

    if ms_res < 1e-300:
        return np.nan, np.nan, np.nan

    F = ms_term / ms_res
    p = float(stats.f.sf(F, df_term, df_res))

    # Partial η²  =  SS_effect / (SS_effect + SS_residual)
    denom = ss_term + rss_full
    eta2p = ss_term / denom if denom > 0 else np.nan

    return F, p, eta2p


def _anova_fallback(data: pd.DataFrame,
                    factor: str) -> tuple:
    """
    Simple one-way ANOVA (scipy) as fallback.
    Returns (F, p, partial_eta_sq).
    """
    groups = [g['Value'].values for _, g in data.groupby(factor) if len(g) >= 2]
    if len(groups) < 2:
        return np.nan, np.nan, np.nan
    F, p = stats.f_oneway(*groups)
    # η² approximation
    all_vals = np.concatenate(groups)
    grand    = all_vals.mean()
    ss_total = np.sum((all_vals - grand) ** 2)
    ss_bet   = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    eta2p    = ss_bet / ss_total if ss_total > 0 else np.nan
    return float(F), float(p), float(eta2p)


def _model_one_region(sub: pd.DataFrame,
                      stressor_cats: list,
                      time_cats: list) -> dict:
    """
    Fit OLS Type-III model for a single region subset.
    Falls back to simple ANOVA if OLS is rank-deficient.

    Returns dict with keys:
        method, n,
        F_stressor, p_stressor, eta2p_stressor,
        F_time,     p_time,     eta2p_time,
        F_interact, p_interact, eta2p_interact
    """
    y = sub['Value'].values.astype(float)
    n = len(y)

    stressor_present = sorted(sub['Stressor'].dropna().unique())
    time_present     = sorted(sub['Time'].dropna().unique())

    # Need ≥ 2 stressors and ≥ 2 timepoints for a two-way test
    if len(stressor_present) < 2 or len(time_present) < 2 or n < 6:
        return dict(method='insufficient_data', n=n,
                    F_stressor=np.nan, p_stressor=np.nan, eta2p_stressor=np.nan,
                    F_time=np.nan,     p_time=np.nan,     eta2p_time=np.nan,
                    F_interact=np.nan, p_interact=np.nan, eta2p_interact=np.nan)

    # Build design matrices with sum-to-zero coding
    X_s  = _effects_dummies(sub['Stressor'], stressor_cats)
    X_t  = _effects_dummies(sub['Time'],     time_cats)

    # Interaction columns
    X_st = np.column_stack([X_s[:, i] * X_t[:, j]
                             for i, j in product(range(X_s.shape[1]),
                                                  range(X_t.shape[1]))])
    intercept = np.ones((n, 1))

    df_s  = X_s.shape[1]
    df_t  = X_t.shape[1]
    df_st = X_st.shape[1]

    X_full     = np.hstack([intercept, X_s, X_t, X_st])
    X_no_s     = np.hstack([intercept,       X_t, X_st])
    X_no_t     = np.hstack([intercept, X_s,        X_st])
    X_no_st    = np.hstack([intercept, X_s, X_t        ])

    # Check rank of full model — fall back if near-singular
    rank_full = np.linalg.matrix_rank(X_full)
    if rank_full < X_full.shape[1] * 0.7:
        # Fallback: separate one-way ANOVAs
        Fs, ps, e2s = _anova_fallback(sub, 'Stressor')
        Ft, pt, e2t = _anova_fallback(sub, 'Time')
        return dict(method='anova_fallback', n=n,
                    F_stressor=Fs, p_stressor=ps, eta2p_stressor=e2s,
                    F_time=Ft,     p_time=pt,     eta2p_time=e2t,
                    F_interact=np.nan, p_interact=np.nan, eta2p_interact=np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Fs,  ps,  e2s  = _type3_ftest(y, X_full, X_no_s,  df_s)
        Ft,  pt,  e2t  = _type3_ftest(y, X_full, X_no_t,  df_t)
        Fst, pst, e2st = _type3_ftest(y, X_full, X_no_st, df_st)

    return dict(method='ols_type3', n=n,
                F_stressor=Fs,  p_stressor=ps,  eta2p_stressor=e2s,
                F_time=Ft,      p_time=pt,       eta2p_time=e2t,
                F_interact=Fst, p_interact=pst,  eta2p_interact=e2st)


# ─────────────────────────────────────────────
# 3.  RUN ACROSS ALL REGIONS
# ─────────────────────────────────────────────

def run_models(long_df: pd.DataFrame,
               verbose: bool = True) -> pd.DataFrame:
    """
    Run Time × Stressor model for every brain region.

    Parameters
    ----------
    long_df : output of melt_to_long()
    verbose : print progress

    Returns
    -------
    DataFrame indexed by Region with per-factor stats + FDR q-values.
    """
    stressor_cats = [s for s in STRESSOR_ORDER if s in long_df['Stressor'].unique()]
    time_cats     = [t for t in TIME_ORDER     if t in long_df['Time'].unique()]

    regions = long_df['Region'].unique()
    records = []

    for i, region in enumerate(regions):
        if verbose and i % 100 == 0:
            print(f'  Modelling region {i+1}/{len(regions)} …')

        sub    = long_df[long_df['Region'] == region].copy()
        result = _model_one_region(sub, stressor_cats, time_cats)
        result['Region'] = region

        # Grab acronym
        acro = sub['acronym'].dropna().iloc[0] if not sub['acronym'].dropna().empty else region
        result['acronym'] = acro
        records.append(result)

    results = pd.DataFrame(records).set_index('Region')

    # ── FDR correction (Benjamini–Hochberg) ──────────────────────────────────
    for factor, col in [('stressor', 'p_stressor'),
                        ('time',     'p_time'),
                        ('interact', 'p_interact')]:
        p_vals = results[col].dropna().values
        if len(p_vals) == 0:
            results[f'q_{factor}'] = np.nan
            continue
        q_vals = _bh_correction(results[col].values)
        results[f'q_{factor}'] = q_vals

    # Sort by interaction effect size (descending)
    results = results.sort_values('eta2p_interact', ascending=False)
    return results


def _bh_correction(p_array: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR correction. NaNs are preserved."""
    q = np.full_like(p_array, np.nan, dtype=float)
    valid_idx = np.where(~np.isnan(p_array))[0]
    p_valid   = p_array[valid_idx]
    m         = len(p_valid)
    if m == 0:
        return q

    order     = np.argsort(p_valid)
    p_sorted  = p_valid[order]
    q_sorted  = np.minimum(1, p_sorted * m / (np.arange(1, m + 1)))

    # Enforce monotonicity (step-down)
    for i in range(m - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])

    q_valid             = np.empty(m)
    q_valid[order]      = q_sorted
    q[valid_idx]        = q_valid
    return q


# ─────────────────────────────────────────────
# 4.  RANKED TABLE HELPER
# ─────────────────────────────────────────────

def get_ranked_table(results: pd.DataFrame,
                     effect: str = 'interact',
                     top_n: int = 30,
                     q_thresh: float = 0.05) -> pd.DataFrame:
    """
    Return top_n regions ranked by partial η² for the chosen effect.

    Parameters
    ----------
    effect   : 'interact' | 'stressor' | 'time'
    top_n    : number of regions to return
    q_thresh : FDR threshold to flag significance
    """
    eta_col = f'eta2p_{effect}'
    q_col   = f'q_{effect}'
    p_col   = f'p_{effect}'

    out = results[[eta_col, p_col, q_col, 'acronym', 'method', 'n']].copy()
    out = out.dropna(subset=[eta_col]).sort_values(eta_col, ascending=False)
    out[f'sig (q<{q_thresh})'] = out[q_col] < q_thresh
    return out.head(top_n)


# ─────────────────────────────────────────────
# 5.  VISUALISATIONS
# ─────────────────────────────────────────────

def plot_ranked_regions(results: pd.DataFrame,
                        effect: str = 'interact',
                        top_n: int = 25,
                        q_thresh: float = 0.05,
                        save_path: str = None) -> plt.Figure:
    """
    Horizontal bar chart — top brain regions ranked by partial η² for the
    chosen effect, coloured by FDR significance.
    """
    ranked = get_ranked_table(results, effect=effect, top_n=top_n, q_thresh=q_thresh)
    ranked = ranked.iloc[::-1]   # flip so largest is on top

    eta_col = f'eta2p_{effect}'
    q_col   = f'q_{effect}'
    sig     = ranked[q_col] < q_thresh

    colours = np.where(sig, '#E05C5C', '#A8BCCC')
    labels  = ranked.index.tolist()
    acros   = ranked['acronym'].tolist()
    display = [f'{r}  ({a})' if str(a) != str(r) else r
               for r, a in zip(labels, acros)]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.35)), dpi=150)
    bars = ax.barh(range(len(ranked)), ranked[eta_col].values,
                   color=colours, edgecolor='white', linewidth=0.4, height=0.7)

    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels(display, fontsize=7)
    ax.set_xlabel('Partial η²', fontsize=9)

    effect_label = {'interact': 'Stressor × Time Interaction',
                    'stressor': 'Stressor Main Effect',
                    'time':     'Time Main Effect'}.get(effect, effect)
    ax.set_title(f'Top {top_n} Regions — {effect_label}', fontsize=11, pad=10)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#E05C5C', label=f'FDR q < {q_thresh}'),
                       Patch(facecolor='#A8BCCC', label='Not significant')]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right')

    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def plot_interaction_heatmap(results: pd.DataFrame,
                             top_n: int = 25,
                             metric: str = 'eta2p_interact',
                             q_thresh: float = 0.05,
                             save_path: str = None) -> plt.Figure:
    """
    Heatmap of three effect metrics (Stressor, Time, Interaction) for the
    top_n regions ranked by interaction partial η².
    """
    cols   = ['eta2p_stressor', 'eta2p_time', 'eta2p_interact']
    q_cols = ['q_stressor',     'q_time',     'q_interact']
    labels = ['Stressor', 'Time', 'Stressor × Time']

    top = results.dropna(subset=['eta2p_interact']).head(top_n)
    mat = top[cols].values.astype(float)

    # Build significance asterisk array
    sigs = np.full(mat.shape, '', dtype=object)
    for j, qc in enumerate(q_cols):
        if qc in top.columns:
            sigs[:, j] = np.where(top[qc].values < q_thresh, '*', '')

    fig, ax = plt.subplots(figsize=(5, max(6, top_n * 0.22)), dpi=150)
    im = ax.imshow(mat, aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=np.nanpercentile(mat, 98))

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top.index.tolist(), fontsize=6)
    ax.set_title(f'Partial η² — top {top_n} regions\n(* = FDR q < {q_thresh})',
                 fontsize=10, pad=8)

    # Asterisks
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if sigs[i, j]:
                ax.text(j, i, sigs[i, j], ha='center', va='center',
                        fontsize=7, color='black', fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.02)
    cbar.set_label('Partial η²', fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def plot_condition_means(long_df: pd.DataFrame,
                         results: pd.DataFrame,
                         top_n: int = 24,
                         effect: str = 'interact',
                         q_thresh: float = 0.05,
                         save_path: str = None) -> plt.Figure:
    """
    Line plots of condition means (Time on x-axis, one line per Stressor) for
    the top_n regions ranked by partial η² for the chosen effect.

    Shaded bands = ±1 SEM.  Only significant regions (FDR q < q_thresh) are
    included; falls back to top_n by effect size if none pass FDR.
    """
    q_col   = f'q_{effect}'
    eta_col = f'eta2p_{effect}'

    ranked = results.dropna(subset=[eta_col]).sort_values(eta_col, ascending=False)
    sig    = ranked[ranked[q_col] < q_thresh] if q_col in ranked.columns else pd.DataFrame()
    pool   = sig.head(top_n) if len(sig) >= 3 else ranked.head(top_n)

    regions   = pool.index.tolist()
    n_regions = len(regions)

    ncols = min(4, n_regions)
    nrows = int(np.ceil(n_regions / ncols))

    stressor_colors = {
        'Ctrl': '#555555',
        'FS':   '#3A86FF',
        'FSW':  '#FF006E',
        'RS':   '#FB5607',
        'TS':   '#8338EC',
    }
    time_order = [t for t in TIME_ORDER if t in long_df['Time'].cat.categories]

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 3.0),
                             dpi=130, squeeze=False)

    for idx, region in enumerate(regions):
        ax  = axes[idx // ncols][idx % ncols]
        sub = long_df[long_df['Region'] == region]
        acro = pool.loc[region, 'acronym']
        eta  = pool.loc[region, eta_col]

        for stressor, grp in sub.groupby('Stressor', observed=True):
            means = grp.groupby('Time', observed=True)['Value'].mean().reindex(time_order)
            sems  = grp.groupby('Time', observed=True)['Value'].sem().reindex(time_order)
            x     = range(len(time_order))
            color = stressor_colors.get(str(stressor), 'grey')
            ax.plot(x, means.values, marker='o', markersize=4,
                    label=str(stressor), color=color, linewidth=1.5)
            ax.fill_between(x,
                            (means - sems).values,
                            (means + sems).values,
                            alpha=0.15, color=color)

        ax.set_xticks(range(len(time_order)))
        ax.set_xticklabels(time_order, fontsize=14, rotation=30)
        ax.set_title(f'{acro}\nη²={eta:.3f}', fontsize=14, pad=4)
        ax.set_ylabel('Norm. cFos', fontsize=14)
        ax.tick_params(labelsize=14)
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # Remove empty panels
    for idx in range(n_regions, nrows * ncols):
        fig.delaxes(axes[idx // ncols][idx % ncols])

    # Shared legend
    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=5,
               fontsize=20, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    effect_label = {'interact': 'Stressor × Time Interaction',
                    'stressor': 'Stressor', 'time': 'Time'}.get(effect, effect)
    fig.suptitle(f'Condition Means — Top {n_regions} Regions\n({effect_label})',
                 fontsize=20, y=1.01)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


# ─────────────────────────────────────────────
# 6.  CONVENIENCE WRAPPER
# ─────────────────────────────────────────────

def run_full_pipeline(csv_path: str,
                      out_dir: str = '.',
                      top_n_bar: int = 25,
                      top_n_heatmap: int = 25,
                      top_n_lines: int = 24,
                      q_thresh: float = 0.05,
                      verbose: bool = True) -> dict:
    """
    End-to-end pipeline:
        1. Load CSV
        2. Melt to long format
        3. Run models for every region
        4. Save results CSV
        5. Produce and save all three plot types

    Returns dict with keys: 'results', 'long_df', 'figures'.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    if verbose:
        print('Loading data …')
    df      = pd.read_csv(csv_path, sep=';')
    long_df = melt_to_long(df)

    if verbose:
        print(f'Long-format rows: {len(long_df):,}  |  '
              f'Regions: {long_df["Region"].nunique()}  |  '
              f'Stressors: {sorted(long_df["Stressor"].dropna().unique())}  |  '
              f'Timepoints: {sorted(long_df["Time"].dropna().unique())}')
        print('Running models …')

    results = run_models(long_df, verbose=verbose)

    # Save results table
    csv_out = os.path.join(out_dir, 'model_results.csv')
    results.to_csv(csv_out)
    if verbose:
        print(f'Results saved → {csv_out}')

    # ── Plots ────────────────────────────────────────────────────────────────
    figures = {}

    if verbose:
        print('Generating plots …')

    for effect in ['interact', 'stressor', 'time']:
        label = effect.replace('interact', 'interaction')

        fig_bar = plot_ranked_regions(
            results, effect=effect, top_n=top_n_bar, q_thresh=q_thresh,
            save_path=os.path.join(out_dir, f'ranked_{label}.svg'))
        figures[f'ranked_{label}'] = fig_bar

    fig_heat = plot_interaction_heatmap(
        results, top_n=top_n_heatmap, q_thresh=q_thresh,
        save_path=os.path.join(out_dir, 'heatmap_effects.svg'))
    figures['heatmap'] = fig_heat

    fig_lines = plot_condition_means(
        long_df, results, top_n=top_n_lines, effect='interact',
        q_thresh=q_thresh,
        save_path=os.path.join(out_dir, 'condition_means.svg'))
    figures['condition_means'] = fig_lines

    if verbose:
        # Summary stats
        n_sig = (results['q_interact'] < q_thresh).sum()
        print(f'\n── Summary ──────────────────────────────────────')
        print(f'  Regions tested          : {len(results)}')
        print(f'  Sig. interaction (FDR)  : {n_sig}')
        print(f'  Top 5 by interaction η² :')
        top5 = results[['acronym', 'eta2p_interact', 'q_interact']].head(5)
        print(top5.to_string())

    return {'results': results, 'long_df': long_df, 'figures': figures}
