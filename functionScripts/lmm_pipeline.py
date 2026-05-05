"""
lmm_pipeline.py  –  Full end-to-end pipeline
==============================================
Stage 1 : Load & reshape data  (wide CSV → long format)
Stage 2 : Linear Mixed Model per brain region (statsmodels)
            density ~ stressor * timepoint + (1|animal)
Stage 3 : Extract LMM features (betas, t-values, p-values, ω²)
Stage 4 : Build enriched sample × feature matrices
Stage 5 : Classifier  (LogReg / SVM via sklearn, mirrors classifyFunctions.py)
Stage 6 : GNN         (PyTorch Geometric – optional; graceful skip if not installed)

Requirements:
    pip install statsmodels scikit-learn matplotlib seaborn tqdm
    pip install torch torch-geometric          # for Stage 6

Usage:
    python lmm_pipeline.py
    or import and call run_pipeline() from a notebook.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import re
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.preprocessing import LabelEncoder, RobustScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
from sklearn.feature_selection import SelectKBest, f_classif

# GNN – optional
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    print("[INFO] PyTorch Geometric not installed – Stage 6 (GNN) will be skipped.\n"
          "       Install with: pip install torch torch-geometric")

# SHAP – optional
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[INFO] shap not installed – SHAP plots will be skipped.\n"
          "       Install with: pip install shap")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  –  edit these to change behaviour
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # ── data ─────────────────────────────────────────────────────────────────
    csv_path        = "merged_wide_with_acro.csv",
    csv_sep         = ";",
    min_obs         = 20,       # skip regions with fewer total observations
    log_transform   = True,     # log10-transform density (recommended: spans many OOM)
    ref_stressor    = "Ctrl",   # reference level for LMM fixed effects
    ref_timepoint   = "7D",     # reference level for LMM fixed effects

    # ── LMM ──────────────────────────────────────────────────────────────────
    lmm_cache       = "lmm_results_cache.pkl",  # None to always refit
    lmm_method      = "lbfgs",  # 'lbfgs' (fast) | 'nm' | 'powell'
    lmm_reml        = True,     # REML estimation

    # ── classifier ───────────────────────────────────────────────────────────
    clf_label       = "stressor",      # "stressor" | "timepoint"
    clf_features    = "lmm_selected",  # "raw" | "lmm_selected" | "lmm_enriched"
    clf_top_k       = 200,              # regions kept by LMM p-value filter
    clf_model       = "LogReg",        # "LogReg" | "SVM"
    clf_n_splits    = 100,
    clf_test_size   = 0.25,
    clf_rand_seed   = 42,
    clf_scale       = True,

    # ── GNN ──────────────────────────────────────────────────────────────────
    gnn_label       = "stressor",  # "stressor" | "timepoint"
    gnn_edge_corr   = 0.7,         # correlation threshold for edge creation
    gnn_hidden      = 64,
    gnn_heads       = 4,           # GAT attention heads
    gnn_epochs      = 150,
    gnn_lr          = 1e-3,
    gnn_test_size   = 0.2,

    # ── k-sweep ──────────────────────────────────────────────────────────────
    k_sweep_enabled = False,   # run the top-k sweep after the main classifier
    k_sweep_values  = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 661],

    # ── original classifier bridge ───────────────────────────────────────────
    # Set to True to run classifyFunctions pipeline on LMM-filtered features.
    # Requires classifyFunctions.py + helperFunctions.py in sys.path.
    use_original_classifier = True,
    orig_clf_model          = "LogRegL2",   # "LogRegL2" | "LogRegL1" | "svm"
    orig_clf_featureSel     = "Boruta",     # "Univar" | "mutInfo" | "Boruta" | "Fdr" | "Fwe" | "None"
    orig_clf_featureSel_k   = [10, 25, 50], # used when featureSel is "Univar" or "mutInfo"
    orig_clf_balance        = True,         # RandomUnderSampler class balancing
    orig_clf_shuffle        = True,         # run a permutation (shuffle) baseline
    orig_clf_CV_count       = 100,
    orig_clf_CVstrat        = "ShuffleSplit",  # "ShuffleSplit" | "StratKFold"
    orig_clf_multiclass     = "multinomial",
    orig_clf_max_iter       = 10000,

    # ── output ────────────────────────────────────────────────────────────────
    out_dir         = "lmm_pipeline_output",
)

STRESSOR_ORDER  = ["Ctrl", "FS", "FSW", "RS", "TS"]
TIMEPOINT_ORDER = ["Acute", "7D", "14D", "21D"]

os.makedirs(CFG["out_dir"], exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 – Load & reshape
# ═════════════════════════════════════════════════════════════════════════════
def load_and_melt(cfg: dict) -> pd.DataFrame:
    """
    Read the wide CSV and return a tidy long-format DataFrame with columns:
        region | stressor | timepoint | animal | density [| log_density]
    """
    print("\n" + "═" * 60)
    print("STAGE 1 – Loading and reshaping data")
    print("═" * 60)

    df = pd.read_csv(cfg["csv_path"], sep=cfg["csv_sep"])
    print(f"  Wide CSV shape : {df.shape}  ({df.shape[1]-1} samples, {df.shape[0]} regions)")

    # Melt to long
    df_long = df.melt(id_vars=["Region.Name"],
                      var_name="sample_col",
                      value_name="density")

    # Parse sample column  →  stressor | timepoint | animal
    pat = re.compile(r"(Ctrl|FS(?:W)?|RS|TS)_(7D|14D|21D|Acute)_(F\d+)_normalized")
    parsed = df_long["sample_col"].str.extract(pat)
    parsed.columns = ["stressor", "timepoint", "animal"]
    df_long = pd.concat([df_long.reset_index(drop=True),
                         parsed.reset_index(drop=True)], axis=1)

    df_long = df_long.dropna(subset=["stressor", "density"])
    df_long["density"] = pd.to_numeric(df_long["density"], errors="coerce")
    df_long = df_long.dropna(subset=["density"])
    df_long = df_long.rename(columns={"Region.Name": "region"})

    # Ordered categoricals (helps with reference levels later)
    df_long["stressor"]  = pd.Categorical(df_long["stressor"],
                                           categories=STRESSOR_ORDER,  ordered=True)
    df_long["timepoint"] = pd.Categorical(df_long["timepoint"],
                                           categories=TIMEPOINT_ORDER, ordered=True)

    if cfg["log_transform"]:
        # Density spans 1e-13 … 1e-8 → log10 is much better behaved for LMM
        df_long["log_density"] = np.log10(df_long["density"].clip(lower=1e-15))
        print(f"  Log10-transform applied. Range: "
              f"[{df_long['log_density'].min():.1f}, {df_long['log_density'].max():.1f}]")

    # Drop regions with too few observations
    obs_per_region = df_long.groupby("region").size()
    keep = obs_per_region[obs_per_region >= cfg["min_obs"]].index
    df_long = df_long[df_long["region"].isin(keep)]
    print(f"  Long-format shape (after min_obs={cfg['min_obs']} filter): {df_long.shape}")
    print(f"  Regions kept     : {df_long['region'].nunique()}")
    print(f"  Samples          : {df_long['sample_col'].nunique()}")

    return df_long


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 + 3 – Fit LMM per region & extract features
# ═════════════════════════════════════════════════════════════════════════════
def fit_lmm_all_regions(df_long: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Fit one LMM per brain region:
        log_density ~ C(stressor) * C(timepoint) + (1 | animal)

    Returns
    -------
    lmm_df : pd.DataFrame
        One row per region. Columns:
            region
            beta_<effect>   – fixed-effect coefficient
            tval_<effect>   – t-statistic
            pval_<effect>   – p-value
            omega2_stressor – partial ω² for stressor (variance explained)
            omega2_time     – partial ω² for timepoint
            omega2_inter    – partial ω² for interaction
            converged       – bool
    """
    print("\n" + "═" * 60)
    print("STAGE 2+3 – Fitting LMM per brain region")
    print("═" * 60)

    cache = cfg.get("lmm_cache")
    if cache and os.path.exists(cache):
        print(f"  Loading cached LMM results from: {cache}")
        return pd.read_pickle(cache)

    dep_var = "log_density" if cfg["log_transform"] else "density"
    ref_s   = cfg["ref_stressor"]
    ref_t   = cfg["ref_timepoint"]

    formula_interaction = (
        f"{dep_var} ~ "
        f"C(stressor, Treatment('{ref_s}')) * "
        f"C(timepoint, Treatment('{ref_t}'))"
    )
    formula_additive = (
        f"{dep_var} ~ "
        f"C(stressor, Treatment('{ref_s}')) + "
        f"C(timepoint, Treatment('{ref_t}'))"
    )
    regions  = df_long["region"].unique()   # ← this line was missing
    records  = []
    
    for region in tqdm(regions, desc="  Fitting LMMs"):
        rdf = df_long[df_long["region"] == region].copy()

        # Need at least 2 animals for random intercept
        if rdf["animal"].nunique() < 2:
            continue

        # Choose formula based on cell coverage
        cell_counts  = rdf.groupby(["stressor", "timepoint"]).size()
        use_formula  = formula_interaction if cell_counts.min() >= 2 else formula_additive
        model_type   = "interaction" if use_formula == formula_interaction else "additive"

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                result = None
                for reml, method in [(True,  "lbfgs"),
                                     (True,  "nm"),
                                     (False, "lbfgs"),
                                     (False, "nm")]:
                    try:
                        model  = smf.mixedlm(use_formula, data=rdf, groups=rdf["animal"])
                        result = model.fit(reml=reml, method=method,
                                           warn_convergence=False)
                        break
                    except Exception:
                        continue

                if result is None:
                    raise RuntimeError("All fitting methods failed")

            row = {"region": region, "converged": result.converged,
                   "model_type": model_type}

            for param in result.fe_params.index:
                clean = _clean_param_name(param, ref_s, ref_t)
                row[f"beta_{clean}"] = result.fe_params[param]
                row[f"tval_{clean}"] = result.tvalues[param]
                row[f"pval_{clean}"] = result.pvalues[param]

            stressor_params = [p for p in result.fe_params.index
                               if "stressor" in p and "timepoint" not in p]
            time_params     = [p for p in result.fe_params.index
                               if "timepoint" in p and "stressor" not in p]
            inter_params    = [p for p in result.fe_params.index
                               if "stressor" in p and "timepoint" in p]
            N = len(rdf)

            row["omega2_stressor"] = _omega2_approx(result.tvalues, stressor_params, N)
            row["omega2_time"]     = _omega2_approx(result.tvalues, time_params,     N)
            row["omega2_inter"]    = _omega2_approx(result.tvalues, inter_params,    N)

            records.append(row)

        except Exception:
            records.append({"region": region, "converged": False,
                            "model_type": model_type})

    lmm_df = pd.DataFrame(records).fillna(0)

    n_ok = lmm_df["converged"].sum()
    print(f"  Converged : {n_ok}/{len(regions)} regions")
    print(f"  LMM feature matrix: {lmm_df.shape}")

    if cache:
        lmm_df.to_pickle(cache)
        print(f"  Saved to cache: {cache}")

    return lmm_df


def _clean_param_name(param: str, ref_s: str, ref_t: str) -> str:
    """Convert statsmodels parameter name to a short, filesystem-safe key."""
    name = param
    # Remove verbose Treatment(...) wrappers
    name = re.sub(r"C\(stressor,\s*Treatment\('.*?'\)\)\[T\.", "stressor_", name)
    name = re.sub(r"C\(timepoint,\s*Treatment\('.*?'\)\)\[T\.", "time_", name)
    name = name.replace("]", "").replace(" ", "_").replace(":", "__X__")
    name = name.replace("Intercept", "intercept")
    return name


def _omega2_approx(tvalues: pd.Series, params: list, N: int) -> float:
    """Approximate partial ω² from t-statistics for a group of parameters."""
    if not params:
        return 0.0
    df_e = len(params)
    # Sum of squared t-values approximates the Wald chi-square for the group
    chi2 = np.sum(tvalues[params] ** 2)
    # Convert to F-statistic (each coef has df=1, so F ≈ chi2 / df_e)
    F = chi2 / df_e
    omega2 = (df_e * (F - 1)) / (df_e * F + N - df_e)
    return float(np.clip(omega2, 0, 1))


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4 – Build sample × feature matrices
# ═════════════════════════════════════════════════════════════════════════════
def build_feature_matrices(df_long: pd.DataFrame,
                           lmm_df: pd.DataFrame,
                           cfg: dict) -> dict:
    """
    Returns a dict with three matrices (samples × features):
        'raw'          – density for each region
        'lmm_selected' – density for LMM-significant regions only
        'lmm_enriched' – density + LMM betas as additional features

    Each entry is (X, y, sample_meta) where:
        X           : np.ndarray  (n_samples, n_features)
        y           : np.ndarray  (n_samples,)  – integer labels
        sample_meta : pd.DataFrame – stressor / timepoint / animal per row
    """
    print("\n" + "═" * 60)
    print("STAGE 4 – Building feature matrices")
    print("═" * 60)

    dep_var = "log_density" if cfg["log_transform"] else "density"

    # ── Pivot: samples × regions ───────────────────────────────────────────
    # Average over animals within each (stressor, timepoint) for the classifier
    # – or keep individual animals as separate samples for more power
    # Here we keep individual animals (more samples, proper cross-validation)
    pivot = (df_long
             .pivot_table(index=["stressor", "timepoint", "animal"],
                          columns="region",
                          values=dep_var,
                          aggfunc="mean"))
    pivot = pivot.reset_index()

    # Samples that have all regions (drop rows with too many NaNs)
    region_cols = [c for c in pivot.columns
                   if c not in ("stressor", "timepoint", "animal")]

    # Keep samples with at least 80% of regions observed
    row_completeness = pivot[region_cols].notna().mean(axis=1)
    pivot = pivot[row_completeness >= 0.80].reset_index(drop=True)

    # Impute remaining NaNs with column median
    for col in region_cols:
        pivot[col] = pivot[col].fillna(pivot[col].median())

    sample_meta = pivot[["stressor", "timepoint", "animal"]].copy()

    # ── Labels ─────────────────────────────────────────────────────────────
    label_col = cfg["clf_label"]
    le = LabelEncoder()
    y_raw = le.fit_transform(pivot[label_col].astype(str))

    print(f"  Samples (rows) : {len(pivot)}")
    print(f"  Label           : {label_col} → {dict(enumerate(le.classes_))}")

    # ── LMM-based region ranking ────────────────────────────────────────────
    converged_lmm = lmm_df[lmm_df["converged"] == True].copy()
    print(f"  Converged LMM regions available : {len(converged_lmm)}")

    # ── Diagnostic: show actual column names so mismatches are visible ───────
    all_lmm_cols = [c for c in converged_lmm.columns if c != "region"]
    print(f"  LMM feature columns (sample)   : {all_lmm_cols[:8]}")

    # Identify stressor p-value columns – try several naming patterns
    pval_cols = [c for c in converged_lmm.columns
                 if "pval" in c and "stressor" in c and "__X__" not in c]
    if not pval_cols:
        # Fallback: any pval column that is not an interaction term
        pval_cols = [c for c in converged_lmm.columns
                     if "pval" in c and "__X__" not in c
                     and c not in ("converged", "region")]
    print(f"  Stressor p-value columns found : {pval_cols}")

    if pval_cols:
        # Convert to numeric defensively (sometimes stored as object after fillna)
        for pc in pval_cols:
            converged_lmm[pc] = pd.to_numeric(converged_lmm[pc], errors="coerce")
        converged_lmm["min_stressor_pval"] = converged_lmm[pval_cols].min(axis=1)
    else:
        print("  [WARN] No stressor p-value columns found – ranking by omega2_stressor")
        if "omega2_stressor" in converged_lmm.columns:
            converged_lmm["min_stressor_pval"] = 1.0 - converged_lmm["omega2_stressor"]
        else:
            converged_lmm["min_stressor_pval"] = 1.0

    ranked_regions = converged_lmm.sort_values("min_stressor_pval")["region"].tolist()

    # Only keep regions that actually appear as columns in the pivot
    top_k_regions = [r for r in ranked_regions if r in region_cols]
    top_k         = min(cfg["clf_top_k"], len(top_k_regions))

    if top_k == 0:
        # Last-resort fallback: use all available region columns
        print("  [WARN] top_k_regions is empty after filtering – "
              "falling back to all region columns")
        top_k_regions = region_cols
        top_k         = len(region_cols)
    else:
        top_k_regions = top_k_regions[:top_k]

    print(f"  Top-{top_k} regions selected (of {len(region_cols)} available)")

    # ── Build the three matrices ────────────────────────────────────────────
    X_raw = pivot[region_cols].to_numpy(dtype=float)
    X_sel = pivot[top_k_regions].to_numpy(dtype=float)

    # Enriched: raw density for top regions + LMM betas broadcast to all samples
    beta_cols = [c for c in converged_lmm.columns
                 if ("beta_stressor" in c or "beta_time" in c)]
    if not beta_cols:
        # Fallback: use any beta column
        beta_cols = [c for c in converged_lmm.columns
                     if c.startswith("beta_") and c != "beta_intercept"]

    valid_top = [r for r in top_k_regions if r in converged_lmm["region"].values]

    if beta_cols and valid_top:
        lmm_top = (converged_lmm
                   .set_index("region")
                   .loc[valid_top, beta_cols]
                   .to_numpy(dtype=float))
        lmm_broadcast = np.tile(lmm_top.flatten(), (len(pivot), 1))
        X_enr = np.hstack([X_sel, lmm_broadcast])
    else:
        print("  [WARN] No beta columns found for enriched matrix – "
              "using lmm_selected as enriched fallback")
        X_enr = X_sel.copy()

    # ── Final shape guard ────────────────────────────────────────────────────
    for name, arr in [("raw", X_raw), ("lmm_selected", X_sel), ("lmm_enriched", X_enr)]:
        if arr.shape[1] == 0:
            raise ValueError(
                f"Feature matrix '{name}' has 0 columns. "
                f"region_cols={len(region_cols)}, top_k_regions={len(top_k_regions)}, "
                f"converged_lmm rows={len(converged_lmm)}. "
                f"Check that lmm_cache ('{cfg.get('lmm_cache')}') is not stale – "
                f"delete it and rerun to regenerate fresh LMM results."
            )

    print(f"  'raw'          shape: {X_raw.shape}")
    print(f"  'lmm_selected' shape: {X_sel.shape}")
    print(f"  'lmm_enriched' shape: {X_enr.shape}")

    return {
        "raw":          (X_raw, y_raw, sample_meta, region_cols,          le),
        "lmm_selected": (X_sel, y_raw, sample_meta, top_k_regions,        le),
        "lmm_enriched": (X_enr, y_raw, sample_meta, None,                 le),
    }, converged_lmm


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 5 – Classifier
# ═════════════════════════════════════════════════════════════════════════════
def run_classifier(feature_matrices: dict, cfg: dict) -> dict:
    """
    Train / evaluate a classifier using repeated stratified shuffled splits.
    Compares 'raw', 'lmm_selected', and 'lmm_enriched' feature sets.
    """
    print("\n" + "═" * 60)
    print("STAGE 5 – Classifier")
    print("═" * 60)

    results = {}

    for feat_name, (X, y, meta, feat_labels, le) in feature_matrices.items():
        print(f"\n  ── Feature set: {feat_name} ──")
        print(f"     X shape: {X.shape}   classes: {le.classes_}")

        # Guard: skip if matrix has no features
        if X.shape[1] == 0:
            print(f"     [SKIP] {feat_name} has 0 features – skipping.")
            continue

        # Guard: skip if only one class
        if len(np.unique(y)) < 2:
            print("     [SKIP] Only one class present.")
            continue

        # Guard: skip if smallest class has fewer than 2 samples
        if np.bincount(y).min() < 2:
            print("     [SKIP] Smallest class has fewer than 2 samples.")
            continue

        # Build pipeline
        steps = []
        if cfg["clf_scale"]:
            # PowerTransformer (Yeo-Johnson) handles remaining skew in log-density.
            # RobustScaler then centres and scales robustly against outlier regions.
            steps.append(("power", PowerTransformer(method="yeo-johnson", standardize=False)))
            steps.append(("scale", RobustScaler()))
        if feat_name == "raw":
            # For raw (all regions), apply a fast F-test top-k selector
            steps.append(("sel", SelectKBest(f_classif, k=min(cfg["clf_top_k"], X.shape[1]))))

        if cfg["clf_model"] == "LogReg":
            # lbfgs converges faster than saga on small-medium datasets (<10k samples).
            # saga is only preferable when n_samples > 100k or with l1/elasticnet.
            # PowerTransformer before scaling helps saga too if lbfgs is swapped back.
            n_samples = X.shape[0]
            solver    = "lbfgs" if n_samples < 5000 else "saga"
            clf = LogisticRegression(
                penalty="l2", solver=solver, multi_class="multinomial",
                max_iter=10000, tol=1e-4, random_state=cfg["clf_rand_seed"]
            )
        else:
            clf = SVC(kernel="linear", probability=True,
                      random_state=cfg["clf_rand_seed"])
        steps.append(("clf", clf))

        pipe = Pipeline(steps)

        # Repeated stratified split
        sss = StratifiedShuffleSplit(
            n_splits=cfg["clf_n_splits"],
            test_size=cfg["clf_test_size"],
            random_state=cfg["clf_rand_seed"],
        )

        accs     = []
        all_ytrue, all_ypred = [], []

        for train_idx, test_idx in sss.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)
            accs.append(accuracy_score(y_te, y_pred))
            all_ytrue.extend(y_te)
            all_ypred.extend(y_pred)

        mean_acc = np.mean(accs)
        std_acc  = np.std(accs)
        print(f"     Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
        print(f"\n     {classification_report(all_ytrue, all_ypred, target_names=le.classes_)}")

        # ── Confusion matrix – row-normalised percentages ───────────────────
        cm_raw  = confusion_matrix(all_ytrue, all_ypred)
        # Normalise each row by its true-class total → recall per class
        cm_pct  = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True) * 100

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

        # Left panel: raw counts (kept for reference)
        disp_raw = ConfusionMatrixDisplay(cm_raw, display_labels=le.classes_)
        disp_raw.plot(ax=axes[0], colorbar=False, cmap="Blues")
        axes[0].set_title("Counts")

        # Right panel: row-normalised percentages
        im = axes[1].imshow(cm_pct, interpolation="nearest",
                            cmap="Blues", vmin=0, vmax=100)
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="%")
        tick_marks = np.arange(len(le.classes_))
        axes[1].set_xticks(tick_marks); axes[1].set_xticklabels(le.classes_, rotation=45, ha="right")
        axes[1].set_yticks(tick_marks); axes[1].set_yticklabels(le.classes_)
        axes[1].set_xlabel("Predicted label")
        axes[1].set_ylabel("True label")
        axes[1].set_title("Row-normalised (% recall)")
        # Annotate cells: white text on dark cells, dark text on light cells
        thresh = 50
        for i in range(len(le.classes_)):
            for j in range(len(le.classes_)):
                color = "white" if cm_pct[i, j] > thresh else "black"
                axes[1].text(j, i, f"{cm_pct[i, j]:.1f}%",
                             ha="center", va="center",
                             fontsize=8, color=color, fontweight="bold")

        fig.suptitle(
            f"{cfg['clf_model']} | {feat_name} | acc={mean_acc:.3f}±{std_acc:.3f}",
            fontsize=10, fontweight="bold"
        )
        plt.tight_layout()
        fname = os.path.join(cfg["out_dir"], f"cm_{feat_name}_{cfg['clf_model']}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"     Confusion matrix saved → {fname}")

        results[feat_name] = dict(
            accs=accs, mean_acc=mean_acc, std_acc=std_acc,
            all_ytrue=all_ytrue, all_ypred=all_ypred, pipeline=pipe
        )

    # ── Accuracy comparison bar chart ────────────────────────────────────────
    if results:
        fig, ax = plt.subplots(figsize=(5, 3), dpi=150)
        names  = list(results.keys())
        means  = [results[n]["mean_acc"] for n in names]
        stds   = [results[n]["std_acc"]  for n in names]
        colors = ["#378ADD", "#1D9E75", "#D85A30"][:len(names)]
        ax.bar(names, means, yerr=stds, color=colors, capsize=5, alpha=0.85)
        ax.set_ylabel("Accuracy")
        ax.set_ylim([0, 1])
        ax.set_title(f"Classifier comparison  ({cfg['clf_label']})")
        ax.axhline(1 / len(np.unique(list(results.values())[0]["all_ytrue"])),
                   ls="--", color="gray", lw=0.8, label="Chance")
        ax.legend()
        plt.tight_layout()
        fname = os.path.join(cfg["out_dir"], "classifier_comparison.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Comparison bar chart saved → {fname}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 6 – GNN  (PyTorch Geometric)
# ═════════════════════════════════════════════════════════════════════════════
class GAT_Classifier(nn.Module):
    """
    Graph Attention Network for brain-state classification.

    Architecture:
        GATConv (in → hidden)  → ELU
        GATConv (hidden → hidden) → ELU
        global_mean_pool
        Linear → n_classes
    """
    def __init__(self, in_channels: int, hidden: int,
                 heads: int, n_classes: int):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden // heads, heads=heads,
                             dropout=0.3, concat=True)
        self.gat2 = GATConv(hidden, hidden, heads=1,
                             dropout=0.3, concat=False)
        self.lin  = nn.Linear(hidden, n_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = global_mean_pool(x, batch)     # graph-level readout
        return self.lin(x)


def build_graph_dataset(df_long: pd.DataFrame,
                        lmm_df: pd.DataFrame,
                        cfg: dict):
    """
    Build one PyG Data object per sample (animal × stressor × timepoint).

    Node features (per region):
        - log_density at this sample
        - LMM betas for stressor, timepoint, interaction (shared across samples)
        - partial ω² for stressor, time, interaction

    Edges:
        Pearson correlation across samples > cfg['gnn_edge_corr']
        (computed once from the global sample matrix, then fixed for all graphs)
    """
    if not GNN_AVAILABLE:
        print("\n[SKIP] PyTorch Geometric not installed.")
        return None, None

    print("\n" + "═" * 60)
    print("STAGE 6 – Building GNN dataset")
    print("═" * 60)

    dep_var = "log_density" if cfg["log_transform"] else "density"

    # ── Ordered list of regions ──────────────────────────────────────────────
    converged_regions = lmm_df[lmm_df["converged"] == True]["region"].tolist()
    all_regions = sorted(set(df_long["region"].unique()) & set(converged_regions))
    region_idx  = {r: i for i, r in enumerate(all_regions)}
    n_regions   = len(all_regions)

    # ── LMM feature matrix: regions × (betas + omegas) ──────────────────────
    lmm_feat_cols = (
        [c for c in lmm_df.columns if c.startswith("beta_")]
        + ["omega2_stressor", "omega2_time", "omega2_inter"]
    )
    lmm_feat = (lmm_df[lmm_df["region"].isin(all_regions)]
                .set_index("region")
                .loc[all_regions, lmm_feat_cols]
                .to_numpy(dtype=float))

    # Normalise LMM features
    lmm_feat = (lmm_feat - lmm_feat.mean(axis=0)) / (lmm_feat.std(axis=0) + 1e-8)

    # ── Edge index from correlation matrix ───────────────────────────────────
    # Compute correlation from sample-level wide matrix
    pivot_all = (df_long[df_long["region"].isin(all_regions)]
                 .pivot_table(index=["stressor", "timepoint", "animal"],
                              columns="region",
                              values=dep_var,
                              aggfunc="mean")
                 [all_regions])
    pivot_all = pivot_all.fillna(pivot_all.median())

    corr_mat = np.corrcoef(pivot_all.T)  # shape: (n_regions, n_regions)

    threshold = cfg["gnn_edge_corr"]
    rows, cols = np.where(np.abs(corr_mat) >= threshold)
    # Remove self-loops
    mask      = rows != cols
    rows, cols = rows[mask], cols[mask]
    edge_weights = corr_mat[rows, cols].astype(np.float32)
    edge_index   = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    edge_attr    = torch.tensor(edge_weights, dtype=torch.float)

    print(f"  Regions (nodes) : {n_regions}")
    print(f"  Edges (|corr| ≥ {threshold}): {edge_index.shape[1]}")

    # ── Label encoder ────────────────────────────────────────────────────────
    label_col = cfg["gnn_label"]
    samples   = (df_long
                 .groupby(["stressor", "timepoint", "animal"])
                 .size()
                 .reset_index(name="n"))
    le = LabelEncoder()
    le.fit(samples[label_col].astype(str))
    n_classes = len(le.classes_)
    print(f"  Label            : {label_col} → {dict(enumerate(le.classes_))}")

    # ── Per-sample density matrix ─────────────────────────────────────────────
    pivot_idx = pivot_all.index.to_frame(index=False)

    # ── Build one Data object per sample ─────────────────────────────────────
    data_list = []
    for _, row in pivot_idx.iterrows():
        s, t, a = str(row["stressor"]), str(row["timepoint"]), str(row["animal"])
        mask_row = ((pivot_idx["stressor"].astype(str) == s) &
                    (pivot_idx["timepoint"].astype(str) == t) &
                    (pivot_idx["animal"].astype(str)   == a))
        density_vec = pivot_all.loc[mask_row].to_numpy(dtype=float).flatten()

        if len(density_vec) != n_regions:
            continue

        # Normalise density for this sample
        density_vec = (density_vec - density_vec.mean()) / (density_vec.std() + 1e-8)

        # Node feature = [density | lmm_betas | omegas]
        node_feats = np.hstack([density_vec[:, None], lmm_feat])  # (n_regions, 1+n_lmm)
        x = torch.tensor(node_feats, dtype=torch.float)

        y_label = le.transform([s if label_col == "stressor" else t])[0]
        y = torch.tensor([y_label], dtype=torch.long)

        data_list.append(Data(x=x, edge_index=edge_index,
                              edge_attr=edge_attr, y=y))

    print(f"  Graphs built    : {len(data_list)}")
    return data_list, (le, n_classes, lmm_feat.shape[1] + 1)


def train_gnn(data_list: list, cfg: dict, meta):
    """Train the GAT and evaluate on a held-out test split."""
    if not GNN_AVAILABLE or data_list is None:
        return

    print("\n" + "═" * 60)
    print("STAGE 6 – Training GAT classifier")
    print("═" * 60)

    le, n_classes, in_channels = meta

    # Train / test split
    np.random.seed(cfg["clf_rand_seed"])
    idx = np.random.permutation(len(data_list))
    n_test  = max(1, int(len(idx) * cfg["gnn_test_size"]))
    test_idx  = idx[:n_test]
    train_idx = idx[n_test:]

    train_loader = DataLoader([data_list[i] for i in train_idx],
                              batch_size=16, shuffle=True)
    test_loader  = DataLoader([data_list[i] for i in test_idx],
                              batch_size=len(test_idx))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = GAT_Classifier(in_channels, cfg["gnn_hidden"],
                             cfg["gnn_heads"],   n_classes).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=cfg["gnn_lr"],
                              weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["gnn_epochs"])

    train_losses, test_accs = [], []

    for epoch in range(1, cfg["gnn_epochs"] + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out  = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        sched.step()
        train_losses.append(total_loss / len(train_loader))

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    out   = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    pred  = out.argmax(dim=1)
                    acc   = (pred == batch.y).float().mean().item()
                    test_accs.append(acc)
            print(f"  Epoch {epoch:3d}/{cfg['gnn_epochs']} | "
                  f"loss={train_losses[-1]:.4f} | test acc={acc:.3f}")

    # ── Training curve ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3), dpi=150)
    ax.plot(train_losses, label="train loss", color="#378ADD")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title("GAT training curve")
    ax.legend()
    plt.tight_layout()
    fname = os.path.join(cfg["out_dir"], "gnn_training_curve.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()

    # ── Final test evaluation ────────────────────────────────────────────────
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out   = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred  = out.argmax(dim=1).cpu().numpy()
            all_pred.extend(pred)
            all_true.extend(batch.y.cpu().numpy())

    print(f"\n  Final test accuracy: {accuracy_score(all_true, all_pred):.3f}")
    print(f"\n  {classification_report(all_true, all_pred, target_names=le.classes_)}")

    # Save model weights
    torch.save(model.state_dict(),
               os.path.join(cfg["out_dir"], "gat_model.pt"))
    print(f"  Model saved → {cfg['out_dir']}/gat_model.pt")


# ═════════════════════════════════════════════════════════════════════════════
# BONUS – LMM summary visualisations
# ═════════════════════════════════════════════════════════════════════════════
def plot_lmm_summary(lmm_df: pd.DataFrame, cfg: dict):
    """
    Plot the distribution of partial ω² across regions for each effect,
    and a volcano-style plot (effect size vs. -log10 p-value) for stressor.
    """
    print("\n" + "═" * 60)
    print("BONUS – LMM summary visualisations")
    print("═" * 60)

    ok = lmm_df[lmm_df["converged"] == True].copy()

    # ── ω² distributions ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), dpi=150, sharey=False)
    for ax, (col, label, color) in zip(
        axes,
        [("omega2_stressor", "Stressor", "#7F77DD"),
         ("omega2_time",     "Timepoint", "#1D9E75"),
         ("omega2_inter",    "Interaction", "#D85A30")]
    ):
        if col in ok.columns:
            ax.hist(ok[col].clip(0, 1), bins=30, color=color, edgecolor="white", lw=0.3)
            ax.set_xlabel(f"Partial ω²  ({label})")
            ax.set_ylabel("N regions")
            ax.axvline(ok[col].median(), ls="--", color="black", lw=0.8,
                       label=f"median={ok[col].median():.3f}")
            ax.legend(fontsize=7)
    plt.suptitle("Variance explained by each fixed effect (across regions)", y=1.02)
    plt.tight_layout()
    fname = os.path.join(cfg["out_dir"], "lmm_omega2_distributions.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ω² distributions → {fname}")

    # ── Volcano: stressor effect (pick first non-reference stressor) ─────────
    beta_cols_s = [c for c in ok.columns if c.startswith("beta_stressor")
                   and "__X__" not in c]
    pval_cols_s = [c.replace("beta_", "pval_") for c in beta_cols_s
                   if c.replace("beta_", "pval_") in ok.columns]

    if beta_cols_s and pval_cols_s:
        # Use first non-reference stressor contrast
        bc, pc = beta_cols_s[0], pval_cols_s[0]
        stressor_name = bc.replace("beta_stressor_", "")

        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        x = ok[bc]
        y = -np.log10(ok[pc].clip(lower=1e-30))
        sig = ok[pc] < 0.05
        ax.scatter(x[~sig], y[~sig], s=5, alpha=0.4, color="#B4B2A9", label="n.s.")
        ax.scatter(x[sig],  y[sig],  s=8, alpha=0.7, color="#7F77DD", label="p < 0.05")
        ax.axhline(-np.log10(0.05), ls="--", lw=0.8, color="gray")
        ax.axvline(0, ls="--", lw=0.5, color="gray")
        ax.set_xlabel(f"β  (Ctrl vs {stressor_name})")
        ax.set_ylabel("-log10(p-value)")
        ax.set_title(f"Volcano plot – {stressor_name} vs Ctrl (per region LMM)")
        ax.legend(markerscale=2)

        # Annotate top 5 regions by |beta| × -log10(p)
        score = (x.abs() * y).fillna(0)
        top5  = score.nlargest(5).index
        for idx in top5:
            ax.annotate(ok.loc[idx, "region"][:20],
                        xy=(x[idx], y[idx]),
                        xytext=(5, 3), textcoords="offset points",
                        fontsize=5, alpha=0.8)

        plt.tight_layout()
        fname = os.path.join(cfg["out_dir"], f"lmm_volcano_{stressor_name}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Volcano plot     → {fname}")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════
# STAGE 5b – SHAP region importance
# ═════════════════════════════════════════════════════════════════════════════
def run_shap(clf_results: dict,
             feat_matrices: dict,
             lmm_df: pd.DataFrame,
             cfg: dict):
    """
    Compute SHAP values for the best-performing feature set and produce:
        1. shap_beeswarm_<feat>.png  – beeswarm (top-20 regions, all classes)
        2. shap_bar_<feat>.png       – mean |SHAP| per region (global importance)
        3. shap_classwise_<feat>.png – mean |SHAP| heatmap: regions × classes
        4. shap_top_regions.csv      – ranked region table saved to disk

    Uses a LinearExplainer (exact, fast) for LogReg and a
    KernelExplainer (model-agnostic, slower) for SVM.
    """
    if not SHAP_AVAILABLE:
        print("\n[SKIP] shap not installed – run: pip install shap")
        return

    print("\n" + "═" * 60)
    print("STAGE 5b – SHAP region importance")
    print("═" * 60)

    # Pick the best-scoring feature set that has a stored pipeline
    best_feat = max(
        [(k, v) for k, v in clf_results.items() if "pipeline" in v],
        key=lambda kv: kv[1]["mean_acc"],
        default=(None, None)
    )
    if best_feat[0] is None:
        print("  No trained pipeline found in clf_results.")
        return

    feat_name, res = best_feat
    print(f"  Using feature set : {feat_name}  (acc={res['mean_acc']:.3f})")

    X, y, meta, feat_labels, le = feat_matrices[feat_name]

    # Re-fit once on the full dataset for SHAP (all data, no train/test split)
    pipe = res["pipeline"]
    pipe.fit(X, y)

    # Transform X through all pipeline steps except the final classifier
    steps_except_clf = pipe.steps[:-1]
    if steps_except_clf:
        from sklearn.pipeline import Pipeline as _Pipe
        pre = _Pipe(steps_except_clf)
        X_transformed = pre.fit_transform(X, y)
    else:
        X_transformed = X

    clf_step = pipe.named_steps["clf"]

    # ── Feature names ────────────────────────────────────────────────────────────────────────
    # Start from the original feature labels BEFORE any pipeline reduction.
    # feat_labels for lmm_selected / raw  = list of region name strings.
    # feat_labels for lmm_enriched        = None (mixed density + beta cols).
    if feat_labels is not None:
        feature_names = np.array(feat_labels, dtype=str)
    else:
        # lmm_enriched: first columns are region names, rest are LMM betas.
        if "lmm_selected" in feat_matrices:
            region_names = np.array(feat_matrices["lmm_selected"][3] or [], dtype=str)
        else:
            region_names = np.array([], dtype=str)
        n_extra    = X.shape[1] - len(region_names)
        beta_names = np.array([f"beta_feat_{i}" for i in range(max(n_extra, 0))], dtype=str)
        feature_names = np.concatenate([region_names, beta_names])

    # Apply SelectKBest mask BEFORE the length check so names stay aligned.
    # The mask reflects the columns that survived feature selection on full X.
    if "sel" in pipe.named_steps:
        sel_mask = pipe.named_steps["sel"].get_support()
        if len(sel_mask) == len(feature_names):
            feature_names = feature_names[sel_mask]
        elif len(sel_mask) == X.shape[1]:
            feature_names = np.array(feature_names, dtype=str)[sel_mask]

    # Final safety: truncate or pad to match X_transformed width exactly
    n_out = X_transformed.shape[1]
    if len(feature_names) > n_out:
        feature_names = feature_names[:n_out]
    elif len(feature_names) < n_out:
        pad = np.array([f"feat_{i}" for i in range(len(feature_names), n_out)], dtype=str)
        feature_names = np.concatenate([feature_names, pad])

    print(f"  Feature names resolved : {len(feature_names)} names "
          f"| first 3: {list(feature_names[:3])}")

    # ── Choose explainer ─────────────────────────────────────────────────────
    model_name = cfg["clf_model"]
    n_background = min(50, X_transformed.shape[0])
    background   = shap.kmeans(X_transformed, n_background)

    print(f"  Computing SHAP values (model={model_name}, "
          f"n_features={X_transformed.shape[1]}, n_samples={X_transformed.shape[0]})...")

    if model_name == "LogReg":
        explainer   = shap.LinearExplainer(clf_step, X_transformed,
                                            feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_transformed)   # list[n_classes] of (n, p)
    else:
        # KernelExplainer: model-agnostic but slower
        explainer   = shap.KernelExplainer(clf_step.predict_proba, background)
        shap_values = explainer.shap_values(X_transformed, nsamples=100)

    # shap_values is list[n_classes] of shape (n_samples, n_features)
    # Stack to (n_classes, n_samples, n_features)
    shap_arr = np.array(shap_values)   # (n_classes, n_samples, n_features)

    # ── Global importance: mean |SHAP| across classes and samples ────────────
    mean_abs_shap = np.abs(shap_arr).mean(axis=(0, 1))   # (n_features,)
    sorted_idx    = np.argsort(mean_abs_shap)[::-1]
    top_n         = min(20, len(feature_names))
    top_idx       = sorted_idx[:top_n]

    # ── 1. Beeswarm – top-20 regions, all classes pooled ────────────────────
    # Flatten across classes for a single pooled beeswarm
    shap_pooled = shap_arr.mean(axis=0)   # (n_samples, n_features)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    shap.summary_plot(
        shap_pooled[:, top_idx],
        X_transformed[:, top_idx],
        feature_names=feature_names[top_idx],
        plot_type="dot",
        show=False,
        max_display=top_n,
        plot_size=None,
        color_bar=True,
    )
    plt.title(f"SHAP beeswarm – top {top_n} regions ({feat_name})", fontsize=9)
    plt.tight_layout()
    fname = os.path.join(cfg["out_dir"], f"shap_beeswarm_{feat_name}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Beeswarm saved      → {fname}")

    # ── 2. Bar chart – mean |SHAP| global ranking ────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, top_n))[::-1]
    ax.barh(range(top_n), mean_abs_shap[top_idx][::-1],
            color=colors, edgecolor="none")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(feature_names[top_idx][::-1], fontsize=7)
    ax.set_xlabel("Mean |SHAP value|  (global importance)")
    ax.set_title(f"Top {top_n} discriminative regions – {feat_name}", fontsize=9)
    plt.tight_layout()
    fname = os.path.join(cfg["out_dir"], f"shap_bar_{feat_name}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Bar chart saved     → {fname}")

    # ── 3. Class-wise heatmap – mean |SHAP| per class × top regions ──────────
    # shap_arr shape: (n_classes, n_samples, n_features)
    classwise = np.abs(shap_arr[:, :, top_idx]).mean(axis=1)   # (n_classes, top_n)

    fig, ax = plt.subplots(figsize=(max(6, top_n * 0.45), len(le.classes_) * 0.7 + 1), dpi=150)
    sns.heatmap(
        classwise,
        xticklabels=feature_names[top_idx],
        yticklabels=le.classes_,
        cmap="YlOrRd",
        linewidths=0.3,
        ax=ax,
        cbar_kws={"label": "Mean |SHAP|", "shrink": 0.6},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
    ax.set_title(f"Class-wise SHAP importance – {feat_name}", fontsize=9)
    plt.tight_layout()
    fname = os.path.join(cfg["out_dir"], f"shap_classwise_{feat_name}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Class-wise heatmap  → {fname}")

    # ── 4. CSV table ─────────────────────────────────────────────────────────
    rows = []
    for rank, fi in enumerate(sorted_idx):
        row = {
            "rank":            rank + 1,
            "region":          feature_names[fi],
            "mean_abs_shap":   round(float(mean_abs_shap[fi]), 6),
        }
        for ci, cls in enumerate(le.classes_):
            row[f"shap_{cls}"] = round(float(np.abs(shap_arr[ci, :, fi]).mean()), 6)
        rows.append(row)

    shap_table = pd.DataFrame(rows)
    fname_csv  = os.path.join(cfg["out_dir"], f"shap_top_regions_{feat_name}.csv")
    shap_table.to_csv(fname_csv, index=False)
    print(f"  Region table saved  → {fname_csv}")

    # Print top-10 to console
    print("\n  Top-10 discriminative regions:")
    print(shap_table.head(10).to_string(index=False))

    return shap_table


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 5c – Top-k sweep: find the optimal number of LMM-selected regions
# ═════════════════════════════════════════════════════════════════════════════
def run_k_sweep(df_long: pd.DataFrame,
                lmm_df: pd.DataFrame,
                cfg: dict) -> pd.DataFrame:
    """
    Sweep the number of LMM-ranked regions k and plot accuracy vs k.

    The key fix vs. the previous version: we build the full pivot matrix
    ONCE, compute the LMM ranking ONCE with fuzzy string matching, then
    simply slice columns to k inside the loop.  This avoids the repeated
    build_feature_matrices() calls that were silently hitting the fallback
    (returning all 639 regions every time) due to region-name mismatches.
    """
    if not cfg.get("k_sweep_enabled", True):
        return None

    print("\n" + "═" * 60)
    print("STAGE 5c – Top-k region sweep")
    print("═" * 60)

    dep_var  = "log_density" if cfg["log_transform"] else "density"

    # ── 1. Build full sample × region pivot once ───────────────────────────
    pivot = (df_long
             .pivot_table(index=["stressor", "timepoint", "animal"],
                          columns="region",
                          values=dep_var,
                          aggfunc="mean")
             .reset_index())

    region_cols = [c for c in pivot.columns
                   if c not in ("stressor", "timepoint", "animal")]

    # Drop samples with > 20% NaN regions
    row_ok = pivot[region_cols].notna().mean(axis=1) >= 0.80
    pivot  = pivot[row_ok].reset_index(drop=True)
    for col in region_cols:
        pivot[col] = pivot[col].fillna(pivot[col].median())

    label_col = cfg["clf_label"]
    le = LabelEncoder()
    y_all = le.fit_transform(pivot[label_col].astype(str))
    X_all = pivot[region_cols].to_numpy(dtype=float)

    print(f"  Pivot shape      : {X_all.shape}")
    print(f"  Label            : {label_col} → {dict(enumerate(le.classes_))}")

    # ── 2. Build the LMM ranking with fuzzy name matching ──────────────
    converged_lmm = lmm_df[lmm_df["converged"] == True].copy()

    # Normalise both sides: strip whitespace, lowercase for matching
    pivot_norm = {r.strip().lower(): r for r in region_cols}        # normalised → original
    lmm_norm   = converged_lmm["region"].str.strip().str.lower()    # normalised lmm names

    pval_cols = [c for c in converged_lmm.columns
                 if "pval" in c and "stressor" in c and "__X__" not in c]
    if not pval_cols:
        pval_cols = [c for c in converged_lmm.columns
                     if "pval" in c and "__X__" not in c
                     and c not in ("converged", "region")]

    print(f"  Stressor p-val cols found : {pval_cols}")

    if pval_cols:
        for pc in pval_cols:
            converged_lmm[pc] = pd.to_numeric(converged_lmm[pc], errors="coerce")
        converged_lmm["_min_pval"] = converged_lmm[pval_cols].min(axis=1)
    else:
        converged_lmm["_min_pval"] = converged_lmm.get("omega2_stressor",
                                                         pd.Series(1.0, index=converged_lmm.index))
        converged_lmm["_min_pval"] = 1.0 - converged_lmm["_min_pval"]

    # Sort LMM by p-value and build matched list in pivot column order
    lmm_sorted = converged_lmm.sort_values("_min_pval")
    ranked_orig = []   # original (pivot) column names in LMM rank order
    for lmm_name_norm in lmm_sorted["region"].str.strip().str.lower():
        if lmm_name_norm in pivot_norm:
            ranked_orig.append(pivot_norm[lmm_name_norm])

    n_matched = len(ranked_orig)
    print(f"  LMM-pivot matches: {n_matched} / {len(region_cols)} regions "
          f"({'%.1f' % (100*n_matched/max(len(region_cols),1))}%)")

    if n_matched == 0:
        print("  [WARN] No LMM-pivot name matches found. "
              "Falling back to F-statistic ranking for the sweep.")
        # Rank by univariate F-statistic as fallback
        from sklearn.feature_selection import f_classif as _fc
        fvals, _ = _fc(X_all, y_all)
        fvals    = np.nan_to_num(fvals, nan=0.0)
        ranked_orig = [region_cols[i] for i in np.argsort(fvals)[::-1]]

    # Ranked column indices into X_all
    ranked_idx = [region_cols.index(r) for r in ranked_orig]

    # ── 3. Raw baseline (all regions, no LMM filter) ──────────────────
    records  = []
    raw_accs = _run_single_clf(X_all, y_all, cfg)
    raw_mean = float(np.mean(raw_accs))
    raw_std  = float(np.std(raw_accs))
    n_raw    = X_all.shape[1]
    print(f"\n  raw baseline  k={n_raw:4d}  acc={raw_mean:.3f} ± {raw_std:.3f}")
    records.append(dict(k=n_raw, label="raw baseline",
                        mean_acc=raw_mean, std_acc=raw_std,
                        n_features_actual=n_raw))

    # ── 4. Sweep k ─────────────────────────────────────────────────────────
    for k in cfg["k_sweep_values"]:
        n_actual = min(k, len(ranked_idx))
        if n_actual == 0:
            print(f"  k={k:4d}  → 0 matched regions, skipping")
            continue

        idx_k = ranked_idx[:n_actual]
        X_k   = X_all[:, idx_k]

        accs     = _run_single_clf(X_k, y_all, cfg)
        mean_acc = float(np.mean(accs))
        std_acc  = float(np.std(accs))

        print(f"  lmm_selected  k={k:4d}  "
              f"(actual={n_actual:4d})  acc={mean_acc:.3f} ± {std_acc:.3f}")
        records.append(dict(k=k, label="lmm_selected",
                            mean_acc=mean_acc, std_acc=std_acc,
                            n_features_actual=n_actual))

    sweep_df = pd.DataFrame(records)

    # ── 5. Plot ──────────────────────────────────────────────────────────────
    sel_df  = sweep_df[sweep_df["label"] == "lmm_selected"].copy()
    raw_row = sweep_df[sweep_df["label"] == "raw baseline"].iloc[0]
    chance  = 1.0 / len(np.unique(y_all))

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)

    ax.plot(sel_df["n_features_actual"], sel_df["mean_acc"],
            color="#7F77DD", marker="o", ms=5, lw=1.8, label="lmm_selected")
    ax.fill_between(
        sel_df["n_features_actual"],
        sel_df["mean_acc"] - sel_df["std_acc"],
        sel_df["mean_acc"] + sel_df["std_acc"],
        alpha=0.15, color="#7F77DD"
    )
    ax.axhline(raw_row["mean_acc"], color="#D85A30", lw=1.5, ls="--",
               label=f"raw ({n_raw} regions) = {raw_row['mean_acc']:.3f}")
    ax.axhspan(raw_row["mean_acc"] - raw_row["std_acc"],
               raw_row["mean_acc"] + raw_row["std_acc"],
               alpha=0.10, color="#D85A30")
    ax.axhline(chance, color="#888780", lw=1, ls=":",
               label=f"chance ({chance:.2f})")

    cross = sel_df[sel_df["mean_acc"] >= raw_row["mean_acc"] - raw_row["std_acc"]]
    if not cross.empty:
        kx = cross.iloc[0]["n_features_actual"]
        ax.axvline(kx, color="#1D9E75", lw=1, ls="-.",
                   label=f"crossover at k={int(kx)}")
        ax.annotate(f"k={int(kx)}",
                    xy=(kx, cross.iloc[0]["mean_acc"]),
                    xytext=(kx + 8, cross.iloc[0]["mean_acc"] - 0.025),
                    fontsize=8, color="#0F6E56",
                    arrowprops=dict(arrowstyle="->", color="#0F6E56", lw=0.8))

    ax.set_xlabel("Number of LMM-selected regions (k)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim([max(0, chance - 0.05), min(1.0, raw_row["mean_acc"] + 0.15)])
    ax.set_title(
        f"Top-k sweep – {cfg['clf_label']} classification ({cfg['clf_model']})",
        fontsize=10
    )
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    fname = os.path.join(cfg["out_dir"], "k_sweep_accuracy.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Sweep plot saved → {fname}")

    # ── 6. Save CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(cfg["out_dir"], "k_sweep_results.csv")
    sweep_df.to_csv(csv_path, index=False)
    print(f"  Sweep table saved → {csv_path}")
    print("\n  Summary:")
    print(sweep_df[["label", "n_features_actual",
                    "mean_acc", "std_acc"]].to_string(index=False))

    return sweep_df


def _run_single_clf(X: np.ndarray, y: np.ndarray, cfg: dict) -> list:
    """
    Fit one classifier configuration and return per-split accuracy list.
    Shared helper used by run_classifier and run_k_sweep.
    """
    steps = []
    if cfg["clf_scale"]:
        steps.append(("power", PowerTransformer(method="yeo-johnson",
                                                standardize=False)))
        steps.append(("scale", RobustScaler()))

    # For high-dimensional inputs apply SelectKBest as additional safeguard
    if X.shape[1] > 200:
        steps.append(("sel", SelectKBest(f_classif,
                                         k=min(200, X.shape[1]))))

    n_samples = X.shape[0]
    solver    = "lbfgs" if n_samples < 5000 else "saga"
    if cfg["clf_model"] == "LogReg":
        clf = LogisticRegression(
            penalty="l2", solver=solver, multi_class="multinomial",
            max_iter=10000, tol=1e-4, random_state=cfg["clf_rand_seed"]
        )
    else:
        clf = SVC(kernel="linear", probability=True,
                  random_state=cfg["clf_rand_seed"])
    steps.append(("clf", clf))
    pipe = Pipeline(steps)

    sss  = StratifiedShuffleSplit(
        n_splits=cfg["clf_n_splits"],
        test_size=cfg["clf_test_size"],
        random_state=cfg["clf_rand_seed"],
    )
    accs = []
    for tr, te in sss.split(X, y):
        pipe.fit(X[tr], y[tr])
        accs.append(accuracy_score(y[te], pipe.predict(X[te])))
    return accs


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 5d – Bridge: LMM features → original classifyFunctions pipeline
# ═════════════════════════════════════════════════════════════════════════════
def run_original_classifier_with_lmm(feat_matrices: dict, cfg: dict) -> dict:
    """
    Feed LMM-filtered feature matrices into the full classifyFunctions pipeline.

    This gives you everything classifyFunctions adds on top of a plain LogReg:
        • Boruta / Fdr / Fwe / mutual-info feature selection inside CV folds
        • RandomUnderSampler class balancing
        • Shuffle permutation baseline (are results above chance?)
        • Probability outputs per fold
        • StratKFold option
        • Model save/load cache

    The input X matrices come straight from build_feature_matrices() so the
    LMM pre-filtering (ranking by stressor p-value) has already been applied.

    Parameters
    ----------
    feat_matrices : dict   output of build_feature_matrices()
    cfg           : dict   pipeline config (must include orig_clf_* keys)

    Returns
    -------
    dict  keyed by feat_name, each value is the classifyFunctions result dict
    """
    if not cfg.get("use_original_classifier", False):
        print("\n[SKIP] use_original_classifier=False – set to True in CFG to enable.")
        return {}

    print("\n" + "═" * 60)
    print("STAGE 5d – Original classifier bridge (classifyFunctions)")
    print("═" * 60)

    # ── import classifyFunctions (must be on sys.path) ────────────────────
    try:
        import classifyFunctions as cf
    except ImportError as e:
        print(f"  [ERROR] Cannot import classifyFunctions: {e}")
        print("  Make sure classifyFunctions.py and helperFunctions.py are in "
              "your working directory or on sys.path.")
        return {}

    # ── build classifyDict from cfg ────────────────────────────────────────
    classifyDict = dict(
        model                = cfg["orig_clf_model"],
        multiclass           = cfg["orig_clf_multiclass"],
        max_iter             = cfg["orig_clf_max_iter"],
        model_featureSel     = cfg["orig_clf_featureSel"],
        model_featureSel_k   = cfg["orig_clf_featureSel_k"],
        model_featureSel_mode= "single",    # one pipeline, not modelPer
        model_featureSel_alpha = 0.05,
        model_featureScale   = True,
        model_featureTransform = True,
        balance              = cfg["orig_clf_balance"],
        shuffle              = cfg["orig_clf_shuffle"],
        CV_count             = cfg["orig_clf_CV_count"],
        CVstrat              = cfg["orig_clf_CVstrat"],
        test_size            = cfg["clf_test_size"],
        randSeed             = cfg["clf_rand_seed"],
        randState            = cfg["clf_rand_seed"],
        saveLoadswitch       = False,   # no cache for bridge runs
        label                = cfg["clf_label"],
        data                 = "density_norm",
        featurefilt          = False,
        featureAgg           = False,
        pGrid                = {},
        adaptive_test_size   = True,
        small_class_threshold= 7,
        test_size_small      = 0.40,
        test_size_large      = 0.25,
    )

    results = {}

    for feat_name, (X, y, meta, feat_labels, le) in feat_matrices.items():

        print(f"\n  ── Feature set: {feat_name} ──")
        print(f"     X : {X.shape}   classes : {le.classes_}")

        if X.shape[1] == 0 or len(np.unique(y)) < 2:
            print("     [SKIP]")
            continue

        # ── Build pipeline from classifyFunctions ─────────────────────────
        try:
            modelList, cvFxn, rsFxn, paramGrid = cf.build_pipeline(classifyDict)
        except Exception as e:
            print(f"     [ERROR] build_pipeline failed: {e}")
            continue

        # ── Run Real + optional Shuffle fits ───────────────────────────
        fits = ["Real"] + (["Shuffle"] if classifyDict["shuffle"] else [])
        feat_results = {}

        for fit_name in fits:
            print(f"     └─ fit={fit_name}")
            for pipe in modelList:
                all_ytrue, all_ypred, all_yprob = [], [], []
                accs = []

                for train_idx, test_idx in cvFxn.split(X, y):
                    X_tr, X_te = X[train_idx].copy(), X[test_idx].copy()
                    y_tr, y_te = y[train_idx].copy(), y[test_idx].copy()

                    # Shuffle labels for permutation baseline
                    if fit_name == "Shuffle":
                        np.random.shuffle(y_tr)

                    # Class balancing
                    if rsFxn is not None:
                        try:
                            X_tr, y_tr = rsFxn.fit_resample(X_tr, y_tr)
                        except Exception:
                            pass  # skip if too few samples

                    try:
                        pipe.fit(X_tr, y_tr)
                        y_pred = pipe.predict(X_te)
                        accs.append(accuracy_score(y_te, y_pred))
                        all_ytrue.extend(y_te)
                        all_ypred.extend(y_pred)
                        if hasattr(pipe, "predict_proba"):
                            all_yprob.extend(pipe.predict_proba(X_te).tolist())
                    except Exception as ex:
                        print(f"       [WARN] fold failed: {ex}")
                        continue

                if not accs:
                    continue

                mean_acc = float(np.mean(accs))
                std_acc  = float(np.std(accs))
                print(f"       acc = {mean_acc:.3f} ± {std_acc:.3f}")
                if fit_name == "Real":
                    print(f"\n       {classification_report(all_ytrue, all_ypred, target_names=le.classes_)}")

                # ── Percentage confusion matrix ──────────────────────────────
                if fit_name == "Real":
                    cm_raw = confusion_matrix(all_ytrue, all_ypred)
                    cm_pct = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True) * 100

                    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

                    ConfusionMatrixDisplay(cm_raw,
                                          display_labels=le.classes_).plot(
                        ax=axes[0], colorbar=False, cmap="Blues")
                    axes[0].set_title("Counts")

                    im = axes[1].imshow(cm_pct, cmap="Blues",
                                        vmin=0, vmax=100, interpolation="nearest")
                    plt.colorbar(im, ax=axes[1], fraction=0.046,
                                 pad=0.04, label="%")
                    ticks = np.arange(len(le.classes_))
                    axes[1].set_xticks(ticks)
                    axes[1].set_xticklabels(le.classes_, rotation=45, ha="right")
                    axes[1].set_yticks(ticks)
                    axes[1].set_yticklabels(le.classes_)
                    axes[1].set_xlabel("Predicted label")
                    axes[1].set_ylabel("True label")
                    axes[1].set_title("Row-normalised (% recall)")
                    thresh = 50
                    for i in range(len(le.classes_)):
                        for j in range(len(le.classes_)):
                            col = "white" if cm_pct[i, j] > thresh else "black"
                            axes[1].text(j, i, f"{cm_pct[i, j]:.1f}%",
                                         ha="center", va="center",
                                         fontsize=8, color=col, fontweight="bold")
                    fig.suptitle(
                        f"Original clf | {cfg['orig_clf_model']} | "
                        f"{feat_name} | acc={mean_acc:.3f}±{std_acc:.3f}",
                        fontsize=10, fontweight="bold"
                    )
                    plt.tight_layout()
                    fname = os.path.join(
                        cfg["out_dir"],
                        f"cm_orig_{feat_name}_{cfg['orig_clf_model']}.png"
                    )
                    plt.savefig(fname, dpi=150, bbox_inches="tight")
                    plt.close()
                    print(f"       Confusion matrix → {fname}")

                feat_results[fit_name] = dict(
                    accs=accs, mean_acc=mean_acc, std_acc=std_acc,
                    all_ytrue=all_ytrue, all_ypred=all_ypred,
                    all_yprob=all_yprob, pipeline=pipe,
                )

        # ── Shuffle vs real bar chart ──────────────────────────────────
        if "Real" in feat_results and "Shuffle" in feat_results:
            r_real    = feat_results["Real"]
            r_shuffle = feat_results["Shuffle"]
            fig, ax   = plt.subplots(figsize=(4, 3), dpi=150)
            ax.bar(["Real", "Shuffle"],
                   [r_real["mean_acc"], r_shuffle["mean_acc"]],
                   yerr=[r_real["std_acc"], r_shuffle["std_acc"]],
                   color=["#7F77DD", "#B4B2A9"], capsize=5, alpha=0.85)
            chance = 1.0 / len(np.unique(y))
            ax.axhline(chance, ls="--", color="#888780", lw=0.8,
                       label=f"chance ({chance:.2f})")
            ax.set_ylim([0, 1])
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Real vs shuffle – {feat_name}", fontsize=9)
            ax.legend(fontsize=7, frameon=False)
            plt.tight_layout()
            fname = os.path.join(cfg["out_dir"],
                                 f"shuffle_test_{feat_name}.png")
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"       Shuffle test plot → {fname}")

        results[feat_name] = feat_results

    return results


def run_pipeline(cfg: dict = CFG):
    print("\n" + "█" * 60)
    print("  LMM → Classifier + GNN  –  full pipeline")
    print("█" * 60)

    # ── Stale cache guard ───────────────────────────────────────────────
    cache = cfg.get("lmm_cache")
    if cache and os.path.exists(cache):
        cached = pd.read_pickle(cache)
        pval_check = [c for c in cached.columns if "pval" in c and "stressor" in c]
        if not pval_check:
            print(f"[WARN] Cache '{cache}' has no stressor p-value columns.")
            print(f"       Columns found: {list(cached.columns[:12])}")
            print("[WARN] Deleting stale cache and refitting LMMs from scratch...")
            os.remove(cache)

    # Stage 1 – Load
    df_long = load_and_melt(cfg)

    # Stage 2+3 – Fit LMMs
    lmm_df = fit_lmm_all_regions(df_long, cfg)

    # Bonus – visualise LMM summary
    plot_lmm_summary(lmm_df, cfg)

    # Stage 4 – Feature matrices
    feat_matrices, lmm_df_conv = build_feature_matrices(df_long, lmm_df, cfg)

    # Stage 5 – Classifier
    clf_results = run_classifier(feat_matrices, cfg)

    # Stage 5b – SHAP
    run_shap(clf_results, feat_matrices, lmm_df, cfg)

    # Stage 5c – Top-k sweep
    run_k_sweep(df_long, lmm_df, cfg)

    # Stage 5d – Original classifier bridge
    run_original_classifier_with_lmm(feat_matrices, cfg)

    # Stage 6 – GNN
    data_list, meta = build_graph_dataset(df_long, lmm_df, cfg)
    if GNN_AVAILABLE and data_list:
        train_gnn(data_list, cfg, meta)

    print("\n" + "█" * 60)
    print(f"  Pipeline complete. Outputs in: ./{cfg['out_dir']}/")
    print("█" * 60)

    return lmm_df, feat_matrices, clf_results


if __name__ == "__main__":
    run_pipeline()
