"""
anova_pipeline.py  –  Full end-to-end pipeline (ANOVA version)
===============================================================
Replaces the LMM in lmm_pipeline.py with a correct model for your design:

    Each animal is sacrificed at ONE timepoint (between-subjects).
    F1–F5 labels repeat across conditions but are DIFFERENT animals.
    → No random effect is warranted.  OLS Type-III ANOVA is correct.

Stage 1 : Load & reshape data  (wide CSV → long format)
Stage 2 : OLS Type-III ANOVA per brain region  +  Kruskal-Wallis (optional)
Stage 3 : Extract effect sizes (partial η², F-stats, p-values, FDR q-values)
Stage 4 : Build enriched sample × feature matrices
Stage 5 : Classifier  (LogReg / SVM via sklearn)
Stage 5b: SHAP importance
Stage 5c: Top-k sweep
Stage 6 : GNN  (PyTorch Geometric – optional)

Kruskal-Wallis note
-------------------
KW tests whether ANY group differs (non-parametric, no normality assumption).
It replaces the F-test for main effects but has no standard equivalent for
the interaction term.  This pipeline offers:
    anova_method = "ols"    → OLS Type-III for all three effects
    anova_method = "kw"     → KW for stressor & timepoint main effects,
                              OLS for the interaction term
    anova_method = "both"   → run both; store all stats; use OLS for ranking

Requirements:
    pip install statsmodels scikit-learn matplotlib seaborn tqdm scipy
    pip install torch torch-geometric          # for Stage 6
    pip install shap                           # for Stage 5b

Usage:
    python anova_pipeline.py
    or: from anova_pipeline import run_pipeline; run_pipeline()
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import re
import warnings
from itertools import product as iproduct

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from sklearn.preprocessing import LabelEncoder, RobustScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest, f_classif

# ── GNN (optional) ───────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F_torch
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GATConv, global_mean_pool
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    print("[INFO] PyTorch Geometric not installed – Stage 6 (GNN) skipped.")

# ── SHAP (optional) ──────────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[INFO] shap not installed – SHAP plots skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # ── data ─────────────────────────────────────────────────────────────────
    csv_path        = "merged_wide_with_acro.csv",
    csv_sep         = ";",
    min_obs         = 6,        # skip regions with fewer total observations
    log_transform   = True,     # log10-transform density before modelling

    # ── ANOVA ────────────────────────────────────────────────────────────────
    anova_method    = "both",   # "ols" | "kw" | "both"
    anova_cache     = "anova_results_cache.pkl",   # None to always refit
    anova_rank_by   = "stressor",  # which effect to rank regions by for selection
                                   # "stressor" | "time" | "inter"

    # ── classifier ───────────────────────────────────────────────────────────
    clf_label       = "stressor",      # "stressor" | "timepoint"
    clf_top_k       = 50,              # regions kept by ANOVA q-value filter
    clf_model       = "LogReg",        # "LogReg" | "SVM"
    clf_n_splits    = 100,
    clf_test_size   = 0.25,
    clf_rand_seed   = 42,
    clf_scale       = True,

    # ── GNN ──────────────────────────────────────────────────────────────────
    gnn_label       = "stressor",   # "stressor" | "timepoint"
    gnn_edge_corr   = 0.7,
    gnn_hidden      = 64,
    gnn_heads       = 4,
    gnn_epochs      = 200,
    gnn_lr          = 1e-3,
    gnn_test_size   = 0.2,

    # ── k-sweep ──────────────────────────────────────────────────────────────
    k_sweep_enabled = True,
    k_sweep_values  = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500],

    # ── output ───────────────────────────────────────────────────────────────
    out_dir         = "anova_pipeline_output",
    fdr_alpha       = 0.05,
)

STRESSOR_ORDER  = ["Ctrl", "FS", "FSW", "RS", "TS"]
TIMEPOINT_ORDER = ["Acute", "7D", "14D", "21D"]


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 – Load & reshape
# ═════════════════════════════════════════════════════════════════════════════
def load_and_melt(cfg: dict) -> pd.DataFrame:
    """Read wide CSV → tidy long format."""
    print("\n" + "═" * 60)
    print("STAGE 1 – Loading and reshaping data")
    print("═" * 60)

    df = pd.read_csv(cfg["csv_path"], sep=cfg["csv_sep"])
    print(f"  Wide CSV shape : {df.shape}")

    df_long = df.melt(id_vars=["Region.Name"],
                      var_name="sample_col",
                      value_name="density")

    pat = re.compile(r"(Ctrl|FS(?:W)?|RS|TS)_(7D|14D|21D|Acute)_(F\d+)_normalized")
    parsed = df_long["sample_col"].str.extract(pat)
    parsed.columns = ["stressor", "timepoint", "animal"]
    df_long = pd.concat([df_long.reset_index(drop=True),
                         parsed.reset_index(drop=True)], axis=1)
    df_long = df_long.dropna(subset=["stressor", "density"])
    df_long["density"] = pd.to_numeric(df_long["density"], errors="coerce")
    df_long = df_long.dropna(subset=["density"])
    df_long = df_long.rename(columns={"Region.Name": "region"})

    df_long["stressor"]  = pd.Categorical(df_long["stressor"],
                                           categories=STRESSOR_ORDER, ordered=True)
    df_long["timepoint"] = pd.Categorical(df_long["timepoint"],
                                           categories=TIMEPOINT_ORDER, ordered=True)

    if cfg["log_transform"]:
        df_long["log_density"] = np.log10(df_long["density"].clip(lower=1e-15))
        print(f"  Log10-transform applied.")

    obs_per_region = df_long.groupby("region").size()
    keep = obs_per_region[obs_per_region >= cfg["min_obs"]].index
    df_long = df_long[df_long["region"].isin(keep)]

    print(f"  Long-format shape : {df_long.shape}")
    print(f"  Regions kept      : {df_long['region'].nunique()}")
    print(f"  Stressors         : {sorted(df_long['stressor'].dropna().unique().tolist())}")
    print(f"  Timepoints        : {sorted(df_long['timepoint'].dropna().unique().tolist())}")
    return df_long


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2+3 – ANOVA per region
# ═════════════════════════════════════════════════════════════════════════════

def _ols_type3(sub: pd.DataFrame, dep: str) -> dict:
    """
    OLS Type-III two-way ANOVA: dep ~ stressor * timepoint
    Returns F-stats, p-values, partial η² for each effect.
    Partial η² = SS_effect / (SS_effect + SS_residual)
    """
    # Need ≥ 2 levels per factor and ≥ 6 observations
    n_s = sub["stressor"].nunique()
    n_t = sub["timepoint"].nunique()
    n   = len(sub)
    if n_s < 2 or n_t < 2 or n < 6:
        return None

    formula = f"{dep} ~ C(stressor) * C(timepoint)"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.ols(formula, data=sub).fit()
            aov = anova_lm(fit, typ=3)
    except Exception:
        return None

    # anova_lm Type-III rows: Intercept, C(stressor), C(timepoint),
    #                          C(stressor):C(timepoint), Residual
    def _extract(row_name):
        if row_name not in aov.index:
            return np.nan, np.nan, np.nan
        ss_e   = aov.loc[row_name, "sum_sq"]
        ss_res = aov.loc["Residual", "sum_sq"]
        F      = aov.loc[row_name, "F"]
        p      = aov.loc[row_name, "PR(>F)"]
        denom  = ss_e + ss_res
        eta2p  = float(ss_e / denom) if denom > 0 else np.nan
        return float(F), float(p), eta2p

    F_s,  p_s,  e_s  = _extract("C(stressor)")
    F_t,  p_t,  e_t  = _extract("C(timepoint)")
    F_st, p_st, e_st = _extract("C(stressor):C(timepoint)")

    return dict(
        n=n,
        F_stressor=F_s,   p_stressor=p_s,   eta2p_stressor=e_s,
        F_time=F_t,       p_time=p_t,       eta2p_time=e_t,
        F_inter=F_st,     p_inter=p_st,     eta2p_inter=e_st,
    )


def _kruskal_wallis(sub: pd.DataFrame, dep: str) -> dict:
    """
    Kruskal-Wallis H-test for stressor and timepoint main effects separately.
    Non-parametric: no normality assumption, but cannot test interaction.
    """
    out = {}
    for factor, key in [("stressor", "kw_stressor"), ("timepoint", "kw_time")]:
        groups = [g[dep].dropna().values
                  for _, g in sub.groupby(factor, observed=True)
                  if len(g[dep].dropna()) >= 2]
        if len(groups) >= 2:
            H, p = scipy_stats.kruskal(*groups)
            # Eta² approximation for KW: η² = (H - k + 1) / (n - k)
            k = len(groups)
            n = sum(len(g) for g in groups)
            eta2 = max((H - k + 1) / (n - k), 0.0) if n > k else np.nan
            out[f"H_{key}"]    = float(H)
            out[f"p_{key}"]    = float(p)
            out[f"eta2_{key}"] = float(eta2)
        else:
            out[f"H_{key}"]    = np.nan
            out[f"p_{key}"]    = np.nan
            out[f"eta2_{key}"] = np.nan
    return out


def fit_anova_all_regions(df_long: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Fit one OLS Type-III ANOVA (and optionally Kruskal-Wallis) per region.

    Returns anova_df with columns:
        region
        n
        F_stressor, p_stressor, eta2p_stressor, q_stressor
        F_time,     p_time,     eta2p_time,     q_time
        F_inter,    p_inter,    eta2p_inter,    q_inter
        [H_kw_stressor, p_kw_stressor, eta2_kw_stressor,  ← if kw or both]
         H_kw_time,     p_kw_time,     eta2_kw_time]
    """
    print("\n" + "═" * 60)
    print("STAGE 2+3 – Fitting ANOVA per brain region")
    print("═" * 60)

    cache = cfg.get("anova_cache")
    if cache and os.path.exists(cache):
        print(f"  Loading cached ANOVA results from: {cache}")
        return pd.read_pickle(cache)

    dep_var = "log_density" if cfg["log_transform"] else "density"
    method  = cfg.get("anova_method", "ols")
    regions = df_long["region"].unique()
    records = []

    for region in tqdm(regions, desc="  Fitting ANOVAs"):
        sub = df_long[df_long["region"] == region].copy()
        row = {"region": region}

        # ── OLS ──────────────────────────────────────────────────────────────
        if method in ("ols", "both"):
            ols_res = _ols_type3(sub, dep_var)
            if ols_res:
                row.update(ols_res)
            else:
                for k in ("n", "F_stressor", "p_stressor", "eta2p_stressor",
                          "F_time", "p_time", "eta2p_time",
                          "F_inter", "p_inter", "eta2p_inter"):
                    row.setdefault(k, np.nan)

        # ── Kruskal-Wallis ────────────────────────────────────────────────────
        if method in ("kw", "both"):
            kw_res = _kruskal_wallis(sub, dep_var)
            row.update(kw_res)

        records.append(row)

    anova_df = pd.DataFrame(records)

    # ── FDR correction (Benjamini-Hochberg) per effect ────────────────────────
    for effect, pcol, qcol in [
        ("stressor", "p_stressor", "q_stressor"),
        ("time",     "p_time",     "q_time"),
        ("inter",    "p_inter",    "q_inter"),
    ]:
        if pcol in anova_df.columns:
            valid = anova_df[pcol].notna()
            qs = np.full(len(anova_df), np.nan)
            if valid.sum() > 0:
                _, q_vals, _, _ = multipletests(
                    anova_df.loc[valid, pcol].values,
                    method="fdr_bh"
                )
                qs[valid.values] = q_vals
            anova_df[qcol] = qs

    # KW FDR correction
    for pcol, qcol in [("p_kw_stressor", "q_kw_stressor"),
                        ("p_kw_time",     "q_kw_time")]:
        if pcol in anova_df.columns:
            valid = anova_df[pcol].notna()
            qs = np.full(len(anova_df), np.nan)
            if valid.sum() > 0:
                _, q_vals, _, _ = multipletests(
                    anova_df.loc[valid, pcol].values, method="fdr_bh")
                qs[valid.values] = q_vals
            anova_df[qcol] = qs

    n_sig_s = (anova_df.get("q_stressor", pd.Series()) < cfg["fdr_alpha"]).sum()
    n_sig_t = (anova_df.get("q_time",     pd.Series()) < cfg["fdr_alpha"]).sum()
    n_sig_i = (anova_df.get("q_inter",    pd.Series()) < cfg["fdr_alpha"]).sum()
    print(f"  Regions fitted  : {len(anova_df)}")
    print(f"  FDR sig stressor: {n_sig_s}  |  time: {n_sig_t}  |  inter: {n_sig_i}")

    if cache:
        anova_df.to_pickle(cache)
        print(f"  Saved to cache  : {cache}")

    return anova_df


# ═════════════════════════════════════════════════════════════════════════════
# ANOVA summary visualisations
# ═════════════════════════════════════════════════════════════════════════════
def plot_anova_summary(anova_df: pd.DataFrame, cfg: dict, df_long=None):
    """Distributions of partial η² and volcano-style plots."""
    os.makedirs(cfg["out_dir"], exist_ok=True)
    alpha = cfg["fdr_alpha"]

    # ── η² distributions ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, (col, qcol, label, color) in zip(axes, [
        ("eta2p_stressor", "q_stressor", "Stressor",       "#7F77DD"),
        ("eta2p_time",     "q_time",     "Timepoint",      "#1D9E75"),
        ("eta2p_inter",    "q_inter",    "Interaction",    "#D85A30"),
    ]):
        if col not in anova_df.columns:
            continue
        vals = anova_df[col].dropna().clip(0, 1)
        ax.hist(vals, bins=30, color=color, edgecolor="white", lw=0.3)
        ax.axvline(vals.median(), ls="--", color="black", lw=0.8,
                   label=f"median={vals.median():.3f}")
        n_sig = (anova_df.get(qcol, pd.Series()) < alpha).sum()
        ax.set_xlabel(f"Partial η²  ({label})")
        ax.set_ylabel("N regions")
        ax.set_title(f"FDR sig: {n_sig}", fontsize=8)
        ax.legend(fontsize=7)
    plt.suptitle("Variance explained by each effect (ANOVA)", y=1.02)
    plt.tight_layout()
    fname = os.path.join(cfg["out_dir"], "anova_eta2_distributions.svg")
    plt.savefig(fname, format="svg", bbox_inches="tight")
    plt.close()
    print(f"  η² distributions → {fname}")

    # ── KW H-stat distributions (if present) ─────────────────────────────────
    if "H_kw_stressor" in anova_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        for ax, (col, label, color) in zip(axes, [
            ("H_kw_stressor", "Stressor (KW)", "#7F77DD"),
            ("H_kw_time",     "Time (KW)",     "#1D9E75"),
        ]):
            vals = anova_df[col].dropna()
            ax.hist(vals, bins=30, color=color, edgecolor="white", lw=0.3)
            ax.axvline(vals.median(), ls="--", color="black", lw=0.8,
                       label=f"median H={vals.median():.2f}")
            ax.set_xlabel(f"KW H-statistic  ({label})")
            ax.set_ylabel("N regions")
            ax.legend(fontsize=7)
        plt.suptitle("Kruskal-Wallis H-statistics across regions", y=1.02)
        plt.tight_layout()
        fname = os.path.join(cfg["out_dir"], "kw_H_distributions.svg")
        plt.savefig(fname, format="svg", bbox_inches="tight")
        plt.close()
        print(f"  KW H distributions → {fname}")

    # ── Volcano: stressor effect ─────────────────────────────────────────────
    if "eta2p_stressor" in anova_df.columns and "q_stressor" in anova_df.columns:
        valid = anova_df.dropna(subset=["eta2p_stressor", "q_stressor"])
        fig, ax = plt.subplots(figsize=(6, 5))
        x   = valid["eta2p_stressor"]
        y   = -np.log10(valid["q_stressor"].clip(lower=1e-30))
        sig = valid["q_stressor"] < alpha
        ax.scatter(x[~sig], y[~sig], s=5, alpha=0.4, color="#B4B2A9", label="n.s.")
        ax.scatter(x[sig],  y[sig],  s=8, alpha=0.7, color="#7F77DD", label=f"q<{alpha}")
        ax.axhline(-np.log10(alpha), ls="--", color="gray", lw=0.8)
        ax.set_xlabel("Partial η²  (Stressor)")
        ax.set_ylabel("-log₁₀(FDR q)")
        ax.set_title("Stressor effect – volcano plot")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fname = os.path.join(cfg["out_dir"], "anova_volcano_stressor.svg")
        plt.savefig(fname, format="svg", bbox_inches="tight")
        plt.close()
        print(f"  Volcano plot     → {fname}")

    # ── Comparison: OLS eta2p vs KW eta2  +  Shapiro-Wilk normality ─────────
    if "eta2p_stressor" in anova_df.columns and "eta2_kw_stressor" in anova_df.columns:
        dep_var = "log_density" if cfg.get("log_transform", True) else "density"
        valid   = anova_df.dropna(subset=["eta2p_stressor", "eta2_kw_stressor",
                                           "region"]).copy()

        # ── Shapiro-Wilk per region (test normality of residuals) ─────────────
        sw_failed = []
        for region in valid["region"].values:
            sub = df_long[df_long["region"] == region].copy() if df_long is not None else None
            if sub is None or len(sub) < 8:
                sw_failed.append(False)
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit      = smf.ols(f"{dep_var} ~ C(stressor) * C(timepoint)",
                                       data=sub).fit()
                    residuals = fit.resid.values
                    _, p_sw   = scipy_stats.shapiro(residuals)
                sw_failed.append(p_sw < 0.05)
            except Exception:
                sw_failed.append(False)

        valid["sw_failed"] = sw_failed
        normal  = valid[~valid["sw_failed"]]
        nonnorm = valid[valid["sw_failed"]]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(normal["eta2p_stressor"],  normal["eta2_kw_stressor"],
                   s=6, alpha=0.4, color="#378ADD", label="Normal residuals")
        ax.scatter(nonnorm["eta2p_stressor"], nonnorm["eta2_kw_stressor"],
                   s=8, alpha=0.7, color="#D85A30",
                   label=f"Non-normal (Shapiro p<0.05, n={len(nonnorm)})")

        lim = max(valid["eta2p_stressor"].max(),
                  valid["eta2_kw_stressor"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="identity")
        ax.set_xlabel("OLS partial η²  (Stressor)")
        ax.set_ylabel("KW η²  (Stressor)")
        ax.set_title("OLS vs Kruskal-Wallis – stressor effect size\n"
                     "(red = failed Shapiro-Wilk normality test)")
        ax.legend(fontsize=7)
        plt.tight_layout()
        fname = os.path.join(cfg["out_dir"], "anova_vs_kw_scatter.svg")
        plt.savefig(fname, format="svg", bbox_inches="tight")
        plt.close()
        pct = 100 * len(nonnorm) / max(len(valid), 1)
        print(f"  OLS vs KW scatter → {fname}  "
              f"({len(nonnorm)}/{len(valid)} regions failed normality, {pct:.1f}%)")


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4 – Build feature matrices
# ═════════════════════════════════════════════════════════════════════════════
def build_feature_matrices(df_long: pd.DataFrame,
                           anova_df: pd.DataFrame,
                           cfg: dict) -> tuple:
    """
    Build three sample × feature matrices:

        'raw'             – log-density for ALL regions
        'anova_selected'  – log-density for top-k regions ranked by ANOVA q-value
        'anova_enriched'  – same as selected + η² values appended as features

    Returns (feature_matrices dict, anova_df_sorted)
    """
    print("\n" + "═" * 60)
    print("STAGE 4 – Building feature matrices")
    print("═" * 60)

    dep_var   = "log_density" if cfg["log_transform"] else "density"
    rank_by   = cfg.get("anova_rank_by", "stressor")
    qcol      = f"q_{rank_by}" if rank_by != "inter" else "q_inter"
    eta_col   = f"eta2p_{rank_by}" if rank_by != "inter" else "eta2p_inter"
    top_k     = cfg["clf_top_k"]
    label_col = cfg["clf_label"]

    # ── Pivot: samples × regions ──────────────────────────────────────────────
    pivot = (df_long
             .pivot_table(index=["stressor", "timepoint", "animal"],
                          columns="region",
                          values=dep_var,
                          aggfunc="mean")
             .reset_index())

    region_cols = [c for c in pivot.columns
                   if c not in ("stressor", "timepoint", "animal")]

    # Drop rows with too many missing regions
    row_ok = pivot[region_cols].notna().mean(axis=1) >= 0.80
    pivot  = pivot[row_ok].reset_index(drop=True)

    # Impute remaining NaNs with column median
    for col in region_cols:
        pivot[col] = pivot[col].fillna(pivot[col].median())

    sample_meta = pivot[["stressor", "timepoint", "animal"]].copy()

    # ── Labels ────────────────────────────────────────────────────────────────
    le    = LabelEncoder()
    y_raw = le.fit_transform(pivot[label_col].astype(str))
    print(f"  Samples  : {len(pivot)}")
    print(f"  Label    : {label_col} → {dict(enumerate(le.classes_))}")

    # ── Rank regions by ANOVA q-value ─────────────────────────────────────────
    ranked = (anova_df
              .dropna(subset=[qcol, eta_col])
              .sort_values([qcol, eta_col], ascending=[True, False]))

    top_k_regions = [r for r in ranked["region"].tolist()
                     if r in region_cols][:top_k]

    if len(top_k_regions) == 0:
        print("  [WARN] No ANOVA-ranked regions found – falling back to all regions")
        top_k_regions = region_cols[:top_k]

    print(f"  Top-{top_k} regions selected (of {len(region_cols)} available)")

    # ── Raw matrix ────────────────────────────────────────────────────────────
    X_raw = pivot[region_cols].to_numpy(dtype=float)

    # ── ANOVA-selected matrix ─────────────────────────────────────────────────
    X_sel = pivot[top_k_regions].to_numpy(dtype=float)

    # ── ANOVA-enriched matrix: density + η² values appended ──────────────────
    eta_cols_all = [c for c in anova_df.columns if c.startswith("eta2")]
    if eta_cols_all:
        anova_top = (anova_df
                     .set_index("region")
                     .reindex(top_k_regions)[eta_cols_all]
                     .fillna(0)
                     .to_numpy(dtype=float))
        anova_broadcast = np.tile(anova_top.flatten(), (len(pivot), 1))
        X_enr = np.hstack([X_sel, anova_broadcast])
        print(f"  Enriched: {X_sel.shape[1]} density cols + "
              f"{anova_broadcast.shape[1]} η² cols = {X_enr.shape[1]} total")
    else:
        X_enr = X_sel.copy()
        print("  [WARN] No η² columns found – enriched = selected")

    print(f"  'raw'              shape: {X_raw.shape}")
    print(f"  'anova_selected'   shape: {X_sel.shape}")
    print(f"  'anova_enriched'   shape: {X_enr.shape}")

    feat_matrices = {
        "raw":             (X_raw, y_raw, sample_meta, region_cols,    le),
        "anova_selected":  (X_sel, y_raw, sample_meta, top_k_regions,  le),
        "anova_enriched":  (X_enr, y_raw, sample_meta, None,           le),
    }
    return feat_matrices, ranked


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 5 – Classifier
# ═════════════════════════════════════════════════════════════════════════════
def _build_pipeline(X: np.ndarray, cfg: dict) -> Pipeline:
    steps = []
    if cfg["clf_scale"]:
        steps.append(("power", PowerTransformer(method="yeo-johnson",
                                                standardize=False)))
        steps.append(("scale", RobustScaler()))
    if X.shape[1] > 200:
        steps.append(("sel", SelectKBest(f_classif,
                                         k=min(200, X.shape[1]))))
    solver = "lbfgs" if X.shape[0] < 5000 else "saga"
    if cfg["clf_model"] == "LogReg":
        clf = LogisticRegression(
            penalty="l2", solver=solver, multi_class="multinomial",
            max_iter=10000, tol=1e-4, random_state=cfg["clf_rand_seed"])
    else:
        clf = SVC(kernel="linear", probability=True,
                  random_state=cfg["clf_rand_seed"])
    steps.append(("clf", clf))
    return Pipeline(steps)


def _run_single_clf(X: np.ndarray, y: np.ndarray, cfg: dict) -> tuple:
    """
    Run stratified shuffle-split CV.
    Returns (accs, all_ytrue, all_ypred, last_pipe) where
    all_ytrue / all_ypred are CONCATENATED test-set labels across all splits.
    """
    pipe = _build_pipeline(X, cfg)
    sss  = StratifiedShuffleSplit(
        n_splits=cfg["clf_n_splits"],
        test_size=cfg["clf_test_size"],
        random_state=cfg["clf_rand_seed"])
    accs, all_ytrue, all_ypred = [], [], []
    for tr, te in sss.split(X, y):
        pipe.fit(X[tr], y[tr])
        preds = pipe.predict(X[te])
        accs.append(accuracy_score(y[te], preds))
        all_ytrue.extend(y[te].tolist())
        all_ypred.extend(preds.tolist())
    return accs, np.array(all_ytrue), np.array(all_ypred), pipe


def run_classifier(feature_matrices: dict, cfg: dict) -> dict:
    """Train / evaluate classifier for each feature set."""
    print("\n" + "═" * 60)
    print("STAGE 5 – Classifier")
    print("═" * 60)
    os.makedirs(cfg["out_dir"], exist_ok=True)

    results = {}
    for feat_name, (X, y, meta, feat_labels, le) in feature_matrices.items():
        print(f"\n  ── Feature set: {feat_name} ──")
        print(f"     X shape: {X.shape}   classes: {le.classes_}")

        if X.shape[1] == 0:
            print("     [SKIP] 0 features."); continue
        if len(np.unique(y)) < 2:
            print("     [SKIP] Only 1 class."); continue
        if np.bincount(y).min() < 2:
            print("     [SKIP] Smallest class < 2 samples."); continue

        accs, all_ytrue, all_ypred, pipe = _run_single_clf(X, y, cfg)
        mean_acc = float(np.mean(accs))
        std_acc  = float(np.std(accs))
        print(f"     Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")

        # ── Confusion matrix ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            all_ytrue, all_ypred, display_labels=le.classes_,
            ax=ax, colorbar=False, cmap="Blues", normalize="true")
        ax.set_title(f"{feat_name}  acc={mean_acc:.3f}±{std_acc:.3f}\n"
                     f"(pooled test sets, {cfg['clf_n_splits']} splits, normalised)")
        plt.tight_layout()
        fname = os.path.join(cfg["out_dir"],
                             f"confusion_{feat_name}_{cfg['clf_model']}.svg")
        plt.savefig(fname, format="svg", bbox_inches="tight")
        plt.close()
        print(f"     Confusion matrix (test) → {fname}")

        results[feat_name] = dict(
            accs=accs, mean_acc=mean_acc, std_acc=std_acc,
            all_ytrue=all_ytrue, all_ypred=all_ypred, pipeline=pipe)

    # ── Accuracy comparison bar chart ─────────────────────────────────────────
    if results:
        fig, ax = plt.subplots(figsize=(5, 3))
        names  = list(results.keys())
        means  = [results[n]["mean_acc"] for n in names]
        stds   = [results[n]["std_acc"]  for n in names]
        colors = ["#378ADD", "#1D9E75", "#D85A30"][:len(names)]
        ax.bar(names, means, yerr=stds, color=colors, capsize=5, alpha=0.85)
        ax.set_ylabel("Accuracy")
        ax.set_ylim([0, 1])
        ax.set_title(f"Classifier comparison  ({cfg['clf_label']})")
        n_classes = len(np.unique(list(results.values())[0]["all_ytrue"]))
        ax.axhline(1 / n_classes, ls="--", color="gray", lw=0.8, label="Chance")
        ax.legend()
        plt.tight_layout()
        fname = os.path.join(cfg["out_dir"], "classifier_comparison.svg")
        plt.savefig(fname, format="svg", bbox_inches="tight")
        plt.close()
        print(f"\n  Comparison bar chart → {fname}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 5b – SHAP
# ═════════════════════════════════════════════════════════════════════════════
def run_shap(clf_results: dict, feat_matrices: dict, cfg: dict):
    """SHAP importance for the best-performing feature set."""
    if not SHAP_AVAILABLE:
        print("\n[SKIP] shap not installed.")
        return

    print("\n" + "═" * 60)
    print("STAGE 5b – SHAP region importance")
    print("═" * 60)

    best = max(
        [(k, v) for k, v in clf_results.items() if "pipeline" in v],
        key=lambda kv: kv[1]["mean_acc"],
        default=(None, None))
    if best[0] is None:
        print("  No trained pipeline found."); return

    feat_name, res = best
    print(f"  Using feature set: {feat_name}  (acc={res['mean_acc']:.3f})")

    X, y, meta, feat_labels, le = feat_matrices[feat_name]
    pipe = res["pipeline"]
    pipe.fit(X, y)

    # Transform through all steps except final classifier
    pre_steps = pipe.steps[:-1]
    if pre_steps:
        from sklearn.pipeline import Pipeline as _Pipe
        X_t = _Pipe(pre_steps).fit_transform(X, y)
    else:
        X_t = X

    # Resolve feature names
    if feat_labels is not None:
        fnames = np.array(feat_labels, dtype=str)
    else:
        if "anova_selected" in feat_matrices:
            rnames = np.array(feat_matrices["anova_selected"][3] or [], dtype=str)
        else:
            rnames = np.array([], dtype=str)
        n_extra = X.shape[1] - len(rnames)
        fnames  = np.concatenate([rnames,
                                   [f"eta2_feat_{i}" for i in range(max(n_extra, 0))]])

    if "sel" in pipe.named_steps:
        mask = pipe.named_steps["sel"].get_support()
        if len(mask) == len(fnames):
            fnames = fnames[mask]

    n_out = X_t.shape[1]
    if len(fnames) > n_out:
        fnames = fnames[:n_out]
    elif len(fnames) < n_out:
        fnames = np.concatenate([fnames, [f"feat_{i}" for i in
                                           range(len(fnames), n_out)]])

    clf_step   = pipe.named_steps["clf"]
    n_bg       = min(50, X_t.shape[0])
    background = shap.kmeans(X_t, n_bg)

    print(f"  Computing SHAP (model={cfg['clf_model']}, "
          f"n_features={X_t.shape[1]}, n_samples={X_t.shape[0]})...")

    try:
        masker    = shap.maskers.Independent(X_t, max_samples=n_bg)
        explainer = shap.LinearExplainer(clf_step, masker)
        shap_values = explainer(X_t).values
        if shap_values.ndim == 3:
            shap_values = shap_values.transpose(2, 0, 1)
        else:
            shap_values = shap_values[np.newaxis, :, :]
    except Exception as e:
        print(f"  [WARN] LinearExplainer failed ({e}), falling back to KernelExplainer...")
        explainer   = shap.KernelExplainer(clf_step.predict_proba, background)
        shap_values = explainer.shap_values(X_t, nsamples=100)
        shap_values = np.array(shap_values)

    shap_arr      = np.array(shap_values)
    mean_abs_shap = np.abs(shap_arr).mean(axis=(0, 1))
    top_n         = min(20, len(fnames))
    top_idx       = np.argsort(mean_abs_shap)[::-1][:top_n]

    # Beeswarm
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_arr.mean(axis=0)[:, top_idx], X_t[:, top_idx],
                      feature_names=fnames[top_idx],
                      plot_type="dot", show=False, max_display=top_n,
                      plot_size=None, color_bar=True)
    plt.title(f"SHAP beeswarm – top {top_n} regions ({feat_name})", fontsize=9)
    plt.tight_layout()
    fname = os.path.join(cfg["out_dir"], f"shap_beeswarm_{feat_name}.svg")
    plt.savefig(fname, format="svg", bbox_inches="tight")
    plt.close()
    print(f"  Beeswarm → {fname}")

    # CSV ranking
    rows = [{"rank": r+1, "region": fnames[fi],
              "mean_abs_shap": round(float(mean_abs_shap[fi]), 6)}
            for r, fi in enumerate(np.argsort(mean_abs_shap)[::-1])]
    pd.DataFrame(rows).to_csv(
        os.path.join(cfg["out_dir"], f"shap_top_regions_{feat_name}.csv"),
        index=False)
    print(f"  CSV → shap_top_regions_{feat_name}.csv")


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 5c – Top-k sweep
# ═════════════════════════════════════════════════════════════════════════════
def run_k_sweep(df_long: pd.DataFrame,
                anova_df: pd.DataFrame,
                cfg: dict) -> pd.DataFrame:
    """Sweep over k to find how many ANOVA-ranked regions are needed."""
    if not cfg.get("k_sweep_enabled", True):
        return None

    print("\n" + "═" * 60)
    print("STAGE 5c – Top-k region sweep")
    print("═" * 60)

    records = []

    # Raw baseline once
    cfg_raw = {**cfg, "clf_top_k": cfg.get("clf_top_k", 50)}
    fm_raw, _ = build_feature_matrices(df_long, anova_df, cfg_raw)
    X_raw, y_raw, _, _, _ = fm_raw["raw"]
    raw_accs, _, _, _ = _run_single_clf(X_raw, y_raw, cfg)
    raw_mean  = float(np.mean(raw_accs))
    raw_std   = float(np.std(raw_accs))
    n_raw     = X_raw.shape[1]
    print(f"  raw baseline  k={n_raw}  acc={raw_mean:.3f} ± {raw_std:.3f}")
    records.append(dict(k=n_raw, label="raw baseline",
                        mean_acc=raw_mean, std_acc=raw_std))

    for k in cfg["k_sweep_values"]:
        cfg_k = {**cfg, "clf_top_k": k}
        fm_k, _ = build_feature_matrices(df_long, anova_df, cfg_k)
        X_k, y_k, _, _, _ = fm_k["anova_selected"]
        if X_k.shape[1] == 0:
            continue
        accs, _, _, _ = _run_single_clf(X_k, y_k, cfg)
        m, s = float(np.mean(accs)), float(np.std(accs))
        print(f"  anova_selected  k={k:4d}  acc={m:.3f} ± {s:.3f}")
        records.append(dict(k=k, label="anova_selected", mean_acc=m, std_acc=s))

    sweep_df = pd.DataFrame(records)

    sel = sweep_df[sweep_df["label"] == "anova_selected"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sel["k"], sel["mean_acc"],
            color="#7F77DD", marker="o", ms=5, lw=1.8, label="anova_selected")
    ax.fill_between(sel["k"],
                    sel["mean_acc"] - sel["std_acc"],
                    sel["mean_acc"] + sel["std_acc"],
                    alpha=0.15, color="#7F77DD")
    ax.axhline(raw_mean, color="#D85A30", lw=1.5, ls="--",
               label=f"raw ({n_raw} regions) = {raw_mean:.3f}")
    ax.axhspan(raw_mean - raw_std, raw_mean + raw_std,
               alpha=0.10, color="#D85A30")
    ax.set_xlabel("N regions (k)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Top-k sweep  ({cfg['clf_label']})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = os.path.join(cfg["out_dir"], "topk_sweep.svg")
    plt.savefig(fname, format="svg", bbox_inches="tight")
    plt.close()
    print(f"  Sweep plot → {fname}")

    return sweep_df


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 6 – GNN  (PyTorch Geometric)
# ═════════════════════════════════════════════════════════════════════════════
class GAT_Classifier(nn.Module):
    """
    Graph Attention Network for brain-state classification.
    Set return_attention=True to also return edge attention weights
    from both GAT layers — used for explainability.
    """
    def __init__(self, in_channels: int, hidden: int,
                 heads: int, n_classes: int):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden // heads, heads=heads,
                             dropout=0.3, concat=True)
        self.gat2 = GATConv(hidden, hidden, heads=1,
                             dropout=0.3, concat=False)
        self.lin  = nn.Linear(hidden, n_classes)

    def forward(self, x, edge_index, edge_attr, batch,
                return_attention: bool = False):
        x1, (ei1, a1) = self.gat1(x,  edge_index, return_attention_weights=True)
        x1 = F_torch.elu(x1)
        x2, (ei2, a2) = self.gat2(x1, edge_index, return_attention_weights=True)
        x2 = F_torch.elu(x2)
        out = self.lin(global_mean_pool(x2, batch))
        if return_attention:
            attn1 = a1.mean(dim=1)
            attn2 = a2.squeeze(-1)
            return out, (ei1, attn1), (ei2, attn2)
        return out


def build_graph_dataset(df_long: pd.DataFrame,
                        anova_df: pd.DataFrame,
                        cfg: dict):
    """
    Build one PyG Data object per sample (animal × stressor × timepoint).

    Node features per region (for each sample):
        [log_density | eta2p_stressor | eta2p_time | eta2p_inter |
         F_stressor  | F_time         | F_inter]

    Edges: Pearson |correlation| >= cfg['gnn_edge_corr'] across samples.
    """
    if not GNN_AVAILABLE:
        print("\n[SKIP] PyTorch Geometric not installed.")
        return None, None

    print("\n" + "═" * 60)
    print("STAGE 6 – Building GNN dataset")
    print("═" * 60)

    dep_var = "log_density" if cfg["log_transform"] else "density"

    all_regions = sorted(
        set(df_long["region"].unique()) & set(anova_df["region"].values)
    )
    region_idx = {r: i for i, r in enumerate(all_regions)}
    n_regions  = len(all_regions)

    anova_feat_cols = [c for c in anova_df.columns
                       if c.startswith("eta2") or c.startswith("F_")]
    if not anova_feat_cols:
        anova_feat_cols = ["eta2p_stressor", "eta2p_time", "eta2p_inter"]

    anova_feat = (anova_df
                  .set_index("region")
                  .reindex(all_regions)[anova_feat_cols]
                  .fillna(0)
                  .to_numpy(dtype=float))

    anova_feat = ((anova_feat - anova_feat.mean(axis=0)) /
                  (anova_feat.std(axis=0) + 1e-8))

    pivot_all = (df_long[df_long["region"].isin(all_regions)]
                 .pivot_table(index=["stressor", "timepoint", "animal"],
                              columns="region",
                              values=dep_var,
                              aggfunc="mean")[all_regions])
    pivot_all = pivot_all.fillna(pivot_all.median())

    #corr_mat    = np.corrcoef(pivot_all.T)       #change for proper pearson correlation with p-values
    corr_mat = pivot_all.corr(method='spearman').to_numpy()  # Spearman correlation for robustness  
    threshold   = cfg["gnn_edge_corr"]
    rows_e, cols_e = np.where(np.abs(corr_mat) >= threshold)
    mask           = rows_e != cols_e
    rows_e, cols_e = rows_e[mask], cols_e[mask]
    edge_weights   = corr_mat[rows_e, cols_e].astype(np.float32)
    edge_index     = torch.tensor(np.vstack([rows_e, cols_e]), dtype=torch.long)
    edge_attr      = torch.tensor(edge_weights, dtype=torch.float)

    print(f"  Regions (nodes) : {n_regions}")
    print(f"  Edges (|corr|≥{threshold}): {edge_index.shape[1]}")

    label_col = cfg["gnn_label"]
    samples   = (df_long
                 .groupby(["stressor", "timepoint", "animal"])
                 .size().reset_index(name="n"))
    le = LabelEncoder()
    le.fit(samples[label_col].astype(str))
    n_classes = len(le.classes_)
    print(f"  Label: {label_col} → {dict(enumerate(le.classes_))}")

    pivot_reset = pivot_all.reset_index()
    data_list   = []

    for i in range(len(pivot_reset)):
        srow = pivot_reset.iloc[i]
        s = str(srow["stressor"])
        t = str(srow["timepoint"])
        a = str(srow["animal"])

        density_vec = pivot_reset.iloc[i][all_regions].to_numpy(dtype=float).flatten()

        if len(density_vec) != n_regions:
            continue

        density_vec = ((density_vec - density_vec.mean()) /
                       (density_vec.std() + 1e-8))

        node_feats = np.hstack([density_vec[:, None], anova_feat])
        x = torch.tensor(node_feats, dtype=torch.float)

        y_label = le.transform([s if label_col == "stressor" else t])[0]
        y       = torch.tensor([y_label], dtype=torch.long)

        data_list.append(Data(x=x, edge_index=edge_index,
                              edge_attr=edge_attr, y=y))

    in_channels = 1 + anova_feat.shape[1]
    print(f"  Graphs built    : {len(data_list)}")
    print(f"  Node in_channels: {in_channels}  "
          f"(1 density + {anova_feat.shape[1]} ANOVA stats)")
    return data_list, (le, n_classes, in_channels)


def plot_brain_networks(df_long: pd.DataFrame,
                        anova_df: pd.DataFrame,
                        cfg: dict):
    """
    Save correlation-based brain region network plots.

    Produces:
        gnn_network_overall.svg          – full network, all samples
        gnn_network_<condition>.svg      – one plot per stressor (or timepoint)
    """
    try:
        import networkx as nx
    except ImportError:
        print("  [SKIP] networkx not installed – pip install networkx")
        return

    print("\n  Generating brain network plots...")
    dep_var   = "log_density" if cfg["log_transform"] else "density"
    threshold = cfg["gnn_edge_corr"]
    label_col = cfg["gnn_label"]
    os.makedirs(cfg["out_dir"], exist_ok=True)

    rank_col = f"eta2p_{cfg.get('anova_rank_by', 'stressor')}"
    if rank_col not in anova_df.columns:
        rank_col = "eta2p_stressor"

    def _make_network_plot(pivot_sub: pd.DataFrame,
                           all_regions: list,
                           title: str,
                           fname: str):
        import matplotlib.patches as mpatches
        import matplotlib.patheffects as pe

        #corr_mat  = np.corrcoef(pivot_sub.T)     #change for proper pearson correlation with p-values
        corr_mat = pivot_all.corr(method='spearman').to_numpy()  # Spearman correlation for robustness
        n_regions = len(all_regions)

        G = nx.Graph()
        G.add_nodes_from(range(n_regions))
        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                w = corr_mat[i, j]
                if abs(w) >= threshold:
                    G.add_edge(i, j, weight=float(w))

        if G.number_of_edges() == 0:
            print(f"    [WARN] No edges above threshold for {title} – skipping")
            return

        connected_nodes = sorted([n for n in G.nodes() if G.degree(n) > 0])
        G = G.subgraph(connected_nodes).copy()
        regions_connected = [all_regions[i] for i in connected_nodes]
        n_connected = len(connected_nodes)
        mapping = {old_n: new_n for new_n, old_n in enumerate(connected_nodes)}
        G = nx.relabel_nodes(G, mapping)

        if n_connected == 0:
            print(f"    [WARN] No connected nodes for {title} – skipping")
            return

        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = greedy_modularity_communities(G)
            community_map = {}
            for comm_id, comm in enumerate(communities):
                for node in comm:
                    community_map[node] = comm_id
            n_communities = len(communities)
        except Exception:
            community_map = {n: 0 for n in G.nodes()}
            n_communities = 1

        comm_palette = plt.cm.Pastel1(np.linspace(0, 1, max(n_communities, 2)))
        node_colors  = [comm_palette[community_map[n] % len(comm_palette)]
                        for n in G.nodes()]

        eta_vals = (anova_df
                    .set_index("region")
                    .reindex(regions_connected)[rank_col]
                    .fillna(0).values)
        node_sizes = 25 + 250 * (eta_vals - eta_vals.min()) / (
            eta_vals.max() - eta_vals.min() + 1e-8)

        try:
            pos = nx.kamada_kawai_layout(G, weight="weight")
        except Exception:
            pos = nx.spring_layout(G, weight="weight",
                                   seed=cfg["clf_rand_seed"], k=0.6)

        fig, ax = plt.subplots(figsize=(14, 12), facecolor="white")
        ax.set_facecolor("white")

        edge_norm = plt.Normalize(vmin=-1, vmax=1)
        edge_cmap = plt.cm.RdBu_r

        neg_edges = [(u, v) for u, v in G.edges() if G[u][v]["weight"] < 0]
        pos_edges = [(u, v) for u, v in G.edges() if G[u][v]["weight"] >= 0]

        def _draw_curved_edges(edgelist, style, base_alpha):
            for u, v in edgelist:
                w   = G[u][v]["weight"]
                col = edge_cmap(edge_norm(w))
                lw  = 0.15 + 0.8 * abs(w)
                alpha = base_alpha * (0.3 + 0.7 * abs(w))
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                rad = 0.12 * (1 if (u + v) % 2 == 0 else -1)
                arrow = mpatches.FancyArrowPatch(
                    posA=(x0, y0), posB=(x1, y1),
                    arrowstyle="-",
                    connectionstyle=f"arc3,rad={rad}",
                    color=col, linewidth=lw, alpha=alpha,
                    linestyle=style, zorder=1)
                ax.add_patch(arrow)

        _draw_curved_edges(neg_edges, "dashed", base_alpha=0.55)
        _draw_curved_edges(pos_edges, "solid",  base_alpha=0.45)

        nx.draw_networkx_nodes(G, pos, ax=ax,
                               node_size=node_sizes,
                               node_color=node_colors,
                               alpha=0.92, linewidths=0.6,
                               edgecolors="#cccccc")

        top15_idx = np.argsort(eta_vals)[::-1][:15]
        for i in top15_idx:
            x, y = pos[i]
            txt = ax.text(x, y, regions_connected[i][:22],
                          fontsize=4.5, ha="center", va="center",
                          color="#222222", zorder=5,
                          fontfamily="sans-serif")
            txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

        n_pos_e = len(pos_edges)
        n_neg_e = len(neg_edges)
        title_str = (f"{title}\n"
                    f"nodes={n_connected}  edges={G.number_of_edges()} "
                    f"(+{n_pos_e} / -{n_neg_e})  |r|>={threshold}\n"
                    f"node size ~ eta2p  color=community  "
                    f"solid=+corr  dashed=-corr")
        ax.set_title(title_str, fontsize=8, color="#333333", pad=12)
        ax.axis("off")

        sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.3, pad=0.02, aspect=20)
        cbar.set_label("Pearson r", fontsize=8, color="#333333")
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.ax.tick_params(labelsize=7)
        cbar.outline.set_visible(False)

        if n_communities > 1:
            legend_elements = [
                mpatches.Patch(
                    facecolor=comm_palette[i % len(comm_palette)],
                    edgecolor="#cccccc", linewidth=0.5,
                    label=f"Community {i+1}  "
                          f"(n={sum(1 for v in community_map.values() if v==i)})")
                for i in range(min(n_communities, 10))
            ]
            leg = ax.legend(handles=legend_elements, fontsize=6,
                            loc="lower left", framealpha=0.6,
                            edgecolor="#cccccc", facecolor="white")
            leg.get_frame().set_linewidth(0.5)

        plt.tight_layout()
        plt.savefig(fname, format="svg", bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"    Network plot → {fname}  "
              f"({n_connected} nodes, {G.number_of_edges()} edges, "
              f"{n_communities} communities)")

    candidate_regions = sorted(
        set(df_long["region"].unique()) & set(anova_df["region"].values))

    top100_regions = (anova_df
                      [anova_df["region"].isin(candidate_regions)]
                      .dropna(subset=[rank_col])
                      .sort_values(rank_col, ascending=False)
                      .head(cfg.get("network_top_k", 100))
                      ["region"].tolist())
    all_regions = top100_regions
    print(f"  Network plots: using top {len(all_regions)} regions by {rank_col}")

    pivot_all = (df_long[df_long["region"].isin(all_regions)]
                 .pivot_table(index=["stressor", "timepoint", "animal"],
                              columns="region",
                              values=dep_var,
                              aggfunc="mean")
                 .reindex(columns=all_regions))
    pivot_all = pivot_all.fillna(pivot_all.median())

    _make_network_plot(
        pivot_all, all_regions,
        title="Brain region network – all samples",
        fname=os.path.join(cfg["out_dir"], "gnn_network_overall.svg"))

    conditions = sorted(df_long[label_col].dropna().unique().tolist())
    for cond in conditions:
        sub_df = df_long[df_long[label_col].astype(str) == str(cond)]
        pivot_cond = (sub_df[sub_df["region"].isin(all_regions)]
                      .pivot_table(index=["stressor", "timepoint", "animal"],
                                   columns="region",
                                   values=dep_var,
                                   aggfunc="mean")
                      .reindex(columns=all_regions))
        pivot_cond = pivot_cond.fillna(pivot_cond.median())

        present_regions = [r for r in all_regions if pivot_cond[r].notna().any()]
        pivot_cond = pivot_cond[present_regions]

        if len(pivot_cond) < 3:
            print(f"    [SKIP] {cond}: only {len(pivot_cond)} samples")
            continue

        _make_network_plot(
            pivot_cond, present_regions,
            title=f"Brain region network – {label_col}={cond}",
            fname=os.path.join(cfg["out_dir"], f"gnn_network_{cond}.svg"))


def train_gnn(data_list: list, cfg: dict, meta: tuple):
    """Train GAT and evaluate on held-out test split."""
    if not GNN_AVAILABLE or data_list is None:
        return

    print("\n" + "═" * 60)
    print("STAGE 6 – Training GAT classifier")
    print("═" * 60)

    le, n_classes, in_channels = meta
    os.makedirs(cfg["out_dir"], exist_ok=True)

    np.random.seed(cfg["clf_rand_seed"])
    idx      = np.random.permutation(len(data_list))
    n_test   = max(1, int(len(idx) * cfg["gnn_test_size"]))
    test_idx  = idx[:n_test]
    train_idx = idx[n_test:]

    train_loader = DataLoader([data_list[i] for i in train_idx],
                              batch_size=16, shuffle=True)
    test_loader  = DataLoader([data_list[i] for i in test_idx],
                              batch_size=len(test_idx))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = GAT_Classifier(in_channels,
                             cfg["gnn_hidden"],
                             cfg["gnn_heads"],
                             n_classes).to(device)
    opt   = torch.optim.Adam(model.parameters(),
                              lr=cfg["gnn_lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["gnn_epochs"])

    train_losses, test_losses, test_accs = [], [], []

    for epoch in range(1, cfg["gnn_epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out  = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = F_torch.cross_entropy(out, batch.y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        sched.step()
        model.eval()
        with torch.no_grad():
            train_eval_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                train_eval_loss += F_torch.cross_entropy(out, batch.y).item()
        train_losses.append(train_eval_loss / len(train_loader))
        model.train()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    t_loss = F_torch.cross_entropy(out, batch.y).item()
                    pred = out.argmax(dim=1)
                    acc = (pred == batch.y).float().mean().item()
                    test_losses.append(t_loss)
                    test_accs.append(acc)
            print(f"  Epoch {epoch:3d}/{cfg['gnn_epochs']} | "
                  f"loss={train_losses[-1]:.4f} | test acc={acc:.3f}")

    # Training curve
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(train_losses, label="train loss", color="#378ADD")
    test_epochs = list(range(9, cfg["gnn_epochs"], 10))
    if len(test_epochs) == len(test_losses):
        ax.plot(test_epochs, test_losses, label="val loss",
                color="#E07B54", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("GAT training curve (ANOVA pipeline)")
    ax.legend()
    plt.tight_layout()
    fname = os.path.join(cfg["out_dir"], "gnn_training_curve.svg")
    plt.savefig(fname, format="svg", bbox_inches="tight")
    plt.close()
    print(f"  Training curve → {fname}")

    # Final evaluation
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out   = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_pred.extend(out.argmax(dim=1).cpu().numpy())
            all_true.extend(batch.y.cpu().numpy())

    from sklearn.metrics import classification_report
    print(f"\n  Final test accuracy: {accuracy_score(all_true, all_pred):.3f}")
    print(f"\n  {classification_report(all_true, all_pred, target_names=le.classes_)}")

    torch.save(model.state_dict(),
               os.path.join(cfg["out_dir"], "gat_model.pt"))
    print(f"  Model saved → {cfg['out_dir']}/gat_model.pt")
    return model


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 6b – GAT Explainability
# ═════════════════════════════════════════════════════════════════════════════
def run_gat_explainability(model,
                           data_list: list,
                           all_regions: list,
                           le,
                           cfg: dict):
    """
    Extract GAT attention weights from the trained model and produce:

        gat_node_importance_<class>.svg
        gat_attention_network_<class>.svg
        gat_attention_top_regions.csv
    """
    if not GNN_AVAILABLE:
        return

    try:
        import networkx as nx
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [SKIP] networkx not installed")
        return

    print("\n" + "═" * 60)
    print("STAGE 6b – GAT Explainability (attention weights)")
    print("═" * 60)
    os.makedirs(cfg["out_dir"], exist_ok=True)

    device    = next(model.parameters()).device
    n_regions = len(all_regions)
    classes   = le.classes_
    n_classes = len(classes)

    attn_sum       = {c: None for c in range(n_classes)}
    attn_count     = {c: 0    for c in range(n_classes)}
    aug_edge_index = None

    model.eval()
    with torch.no_grad():
        for data in data_list:
            data  = data.to(device)
            label = int(data.y.item())

            batch_vec = torch.zeros(data.x.shape[0], dtype=torch.long,
                                    device=device)
            _, (_, a1), (ei2, a2) = model(
                data.x, data.edge_index, data.edge_attr,
                batch_vec, return_attention=True)

            attn_np = a2.cpu().numpy()

            if attn_sum[label] is None:
                attn_sum[label]  = attn_np.copy()
                if aug_edge_index is None:
                    aug_edge_index = ei2.cpu().numpy()
            else:
                attn_sum[label] += attn_np
            attn_count[label] += 1

    mean_attn = {}
    for c in range(n_classes):
        mean_attn[c] = attn_sum[c] / attn_count[c] if attn_count[c] > 0 else None

    src_nodes = aug_edge_index[0]
    dst_nodes = aug_edge_index[1]

    non_self  = src_nodes != dst_nodes
    src_nodes = src_nodes[non_self]
    dst_nodes = dst_nodes[non_self]

    node_importance = {}
    for c in range(n_classes):
        if mean_attn[c] is None:
            continue
        attn_c = mean_attn[c][non_self]
        imp = np.zeros(n_regions)
        cnt = np.zeros(n_regions)
        for e_idx, (s, d) in enumerate(zip(src_nodes, dst_nodes)):
            if s < n_regions and d < n_regions:
                imp[s] += attn_c[e_idx]
                imp[d] += attn_c[e_idx]
                cnt[s] += 1
                cnt[d] += 1
        imp = np.where(cnt > 0, imp / cnt, 0.0)
        node_importance[c] = imp

    # ── 1. Bar chart: top-20 regions per class ───────────────────────────────
    top_n_bar = 20
    for c, cls_name in enumerate(classes):
        if c not in node_importance:
            continue
        imp  = node_importance[c]
        top  = np.argsort(imp)[::-1][:top_n_bar]
        vals = imp[top]
        regs = [all_regions[i][:28] for i in top]

        fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.95, top_n_bar))[::-1]
        ax.barh(range(top_n_bar), vals[::-1], color=colors[::-1],
                edgecolor="none", height=0.7)
        ax.set_yticks(range(top_n_bar))
        ax.set_yticklabels(regs[::-1], fontsize=7)
        ax.set_xlabel("Mean GAT attention (layer 2)", fontsize=9)
        ax.set_title(
            f"GAT node importance – {cls_name}\n"
            f"(top {top_n_bar} regions, n={attn_count[c]} samples)",
            fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        fname = os.path.join(cfg["out_dir"],
                             f"gat_node_importance_{cls_name}.svg")
        plt.savefig(fname, format="svg", bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Node importance bar → {fname}")

    # ── 2. Attention network plot per class ──────────────────────────────────
    top_k_edges = cfg.get("gat_explain_top_edges", 150)

    for c, cls_name in enumerate(classes):
        if mean_attn[c] is None:
            continue

        attn = mean_attn[c][non_self]

        top_edge_idx = np.argsort(attn)[::-1][:top_k_edges]
        sel_src   = src_nodes[top_edge_idx]
        sel_dst   = dst_nodes[top_edge_idx]
        sel_attn  = attn[top_edge_idx]

        active_nodes = sorted(set(sel_src.tolist()) | set(sel_dst.tolist()))
        active_nodes = [n for n in active_nodes if n < n_regions]
        node_remap   = {old_n: new_n for new_n, old_n in enumerate(active_nodes)}

        G = nx.DiGraph()
        G.add_nodes_from(range(len(active_nodes)))
        for s, d, w in zip(sel_src, sel_dst, sel_attn):
            if s in node_remap and d in node_remap:
                G.add_edge(node_remap[s], node_remap[d], weight=float(w))

        if G.number_of_edges() == 0:
            continue

        imp_active = node_importance[c][active_nodes]
        node_sizes = 20 + 300 * (imp_active - imp_active.min()) / (
            imp_active.max() - imp_active.min() + 1e-8)

        node_cmap   = plt.cm.YlOrRd
        node_norm   = plt.Normalize(vmin=imp_active.min(),
                                    vmax=imp_active.max())
        node_colors = [node_cmap(node_norm(v)) for v in imp_active]

        try:
            pos = nx.kamada_kawai_layout(G.to_undirected(), weight="weight")
        except Exception:
            pos = nx.spring_layout(G, seed=cfg["clf_rand_seed"], k=0.5)

        fig, ax = plt.subplots(figsize=(13, 11), facecolor="white")
        ax.set_facecolor("white")

        import matplotlib.patches as mpatches_inner
        attn_norm = plt.Normalize(vmin=sel_attn.min(), vmax=sel_attn.max())
        edge_cmap = plt.cm.Oranges

        for u, v, d_edge in G.edges(data=True):
            w   = d_edge["weight"]
            col = edge_cmap(attn_norm(w))
            lw  = 0.2 + 1.8 * attn_norm(w)
            alp = 0.25 + 0.65 * attn_norm(w)
            rad = 0.10 * (1 if (u + v) % 2 == 0 else -1)
            ax.add_patch(mpatches_inner.FancyArrowPatch(
                posA=pos[u], posB=pos[v],
                arrowstyle="-",
                connectionstyle=f"arc3,rad={rad}",
                color=col, linewidth=lw, alpha=alp, zorder=1))

        nx.draw_networkx_nodes(G, pos, ax=ax,
                               node_size=node_sizes,
                               node_color=node_colors,
                               alpha=0.92, linewidths=0.5,
                               edgecolors="#dddddd")

        top12 = np.argsort(imp_active)[::-1][:12]
        import matplotlib.patheffects as pe_inner
        for i in top12:
            if i not in pos:
                continue
            x, y = pos[i]
            txt = ax.text(x, y, all_regions[active_nodes[i]][:22],
                          fontsize=4.5, ha="center", va="center",
                          color="#222222", zorder=5)
            txt.set_path_effects(
                [pe_inner.withStroke(linewidth=2, foreground="white")])

        ax.set_title(
            f"GAT attention network – {cls_name}\n"
            f"top {top_k_edges} edges by attention  ·  "
            f"node size & color = importance  ·  "
            f"edge darkness = attention weight\n"
            f"(n={attn_count[c]} samples of class {cls_name})",
            fontsize=8, color="#333333", pad=10)
        ax.axis("off")

        sm_node = plt.cm.ScalarMappable(cmap=node_cmap, norm=node_norm)
        sm_node.set_array([])
        cb1 = plt.colorbar(sm_node, ax=ax, shrink=0.28, pad=0.01,
                           location="right", aspect=18)
        cb1.set_label("Node importance", fontsize=7)
        cb1.ax.tick_params(labelsize=6)
        cb1.outline.set_visible(False)

        plt.tight_layout()
        fname = os.path.join(cfg["out_dir"],
                             f"gat_attention_network_{cls_name}.svg")
        plt.savefig(fname, format="svg", bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Attention network  → {fname}")

    # ── 3. CSV: region × class attention table ───────────────────────────────
    rows = []
    for i, region in enumerate(all_regions):
        row = {"region": region}
        for c, cls_name in enumerate(classes):
            row[f"attn_{cls_name}"] = (
                round(float(node_importance[c][i]), 6)
                if c in node_importance else np.nan)
        rows.append(row)

    att_df = pd.DataFrame(rows)
    att_df["max_attn"] = att_df[
        [f"attn_{c}" for c in classes]].max(axis=1)
    att_df = att_df.sort_values("max_attn", ascending=False).drop(
        columns="max_attn")

    csv_path = os.path.join(cfg["out_dir"], "gat_attention_top_regions.csv")
    att_df.to_csv(csv_path, index=False)
    print(f"  Attention CSV      → {csv_path}")
    print(f"\n  Top 5 regions by max attention:")
    print(att_df.head(5).to_string(index=False))


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def run_pipeline(cfg: dict = CFG):
    # ── SVG font settings for Affinity Designer compatibility ─────────────────
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"]  = "Arial"

    print("\n" + "█" * 60)
    print("  ANOVA → Classifier + GNN  –  full pipeline")
    print("█" * 60)
    os.makedirs(cfg["out_dir"], exist_ok=True)

    # Stage 1
    df_long = load_and_melt(cfg)

    # Stage 2+3
    anova_df = fit_anova_all_regions(df_long, cfg)
    plot_anova_summary(anova_df, cfg, df_long=df_long)

    # Save results table
    anova_df.to_csv(
        os.path.join(cfg["out_dir"], "anova_results.csv"), index=False)
    print(f"\n  ANOVA results → {cfg['out_dir']}/anova_results.csv")

    # Stage 4
    feat_matrices, ranked = build_feature_matrices(df_long, anova_df, cfg)

    # Stage 5
    clf_results = run_classifier(feat_matrices, cfg)

    # Stage 5b
    run_shap(clf_results, feat_matrices, cfg)

    # Stage 5c
    run_k_sweep(df_long, anova_df, cfg)

    # Stage 6
    data_list, meta = build_graph_dataset(df_long, anova_df, cfg)
    if GNN_AVAILABLE and data_list:
        trained_model = train_gnn(data_list, cfg, meta)
        if trained_model is not None:
            all_gnn_regions = sorted(
                set(df_long["region"].unique()) & set(anova_df["region"].values))
            _, gnn_le, _ = meta[1], meta[0], meta[2]
            run_gat_explainability(
                trained_model, data_list,
                all_gnn_regions, meta[0], cfg)

    # Network plots
    plot_brain_networks(df_long, anova_df, cfg)

    print("\n" + "█" * 60)
    print(f"  Pipeline complete. Outputs in: ./{cfg['out_dir']}/")
    print("█" * 60)

    return anova_df, feat_matrices, clf_results


if __name__ == "__main__":
    run_pipeline()
