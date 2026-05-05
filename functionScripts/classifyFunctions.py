"""
classifyFunctions.py  –  adapted for stressor data
====================================================
Key changes vs. original drug version:
  • reformat_pandasdf():
    - drug-specific dataset ordering removed
    - label-mapping now queries 'stressor' column (with 'drug' as fallback)
    - drug-name fixups (6FDET→6-F-DET, DMT→5MEO) removed
    - customOrder updated to stressor order: Ctrl, FS, FSW, RS, TS
  • classifySamples(), build_pipeline(), return_train_test_data():
    - unchanged (they use column names from classifyDict, so they work as-is)
  • LO_ analysis labels: kept for legacy compatibility
All other helper classes (Boruta, MRMR, SelectFwe_Holms) unchanged.
"""

import os
import pandas as pd
import numpy as np
import pickle as pkl

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import helperFunctions as hf
import plotFunctions as pf

from copy import deepcopy

from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.feature_selection import (SelectKBest, f_classif, mutual_info_classif,
                                       SequentialFeatureSelector, SelectFdr, SelectFwe)
from sklearn.feature_selection._univariate_selection import _BaseFilter

from sklearn.model_selection import StratifiedKFold, GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from numbers import Real
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from boruta import BorutaPy
from imblearn.under_sampling import RandomUnderSampler

# ─────────────────────────────────────────────────────────────────────────────
# Stressor order (mirrors configFunctions.STRESSOR_ORDER)
# ─────────────────────────────────────────────────────────────────────────────
STRESSOR_ORDER = ['Ctrl', 'FS', 'FSW', 'RS', 'TS']
TIMEPOINT_ORDER = ['Acute', '7D', '14D', '21D'] # for timepoint-specific classifiers (e.g. FS vs RS at 7D)


# ─────────────────────────────────────────────────────────────────────────────
def correlationMatrixPlot(X_data, col_names):
    corrMat = np.corrcoef(X_data, rowvar=False)
    plt.figure(figsize=(40, 40))
    sns.set(font_scale=1)
    sns.heatmap(corrMat, cmap="coolwarm", xticklabels=col_names, yticklabels=col_names)
    plt.title("Correlation Matrix Heatmap")
    plt.show()
    return corrMat


def extract_sorted_correlations(correlation_matrix, variable_names):
    num_variables = len(variable_names)
    correlation_pairs = []
    for i in range(num_variables):
        for j in range(i + 1, num_variables):
            correlation_pairs.append([variable_names[i], variable_names[j], correlation_matrix[i, j]])
    correlation_pairs.sort(key=lambda x: -abs(x[2]))
    return correlation_pairs


def bootstrap_fstat(pandasdf, classifyDict, dirDict):
    dirDict = hf.dataStrPathGen(classifyDict, dirDict)
    X, y, featureNames, labelDict = reformat_pandasdf(pandasdf, classifyDict, dirDict)
    outDirPath = os.path.join(dirDict['outDir'], 'Fstat')
    if not os.path.exists(outDirPath):
        os.mkdir(outDirPath)
    pandasdf_labeled = pd.DataFrame(np.hstack([y.reshape(-1, 1), X]), columns=np.append('label', featureNames))
    feature_scores = []
    for idx in np.arange(1e3):
        pdf_sample = pandasdf_labeled.sample(frac=.75, replace=True)
        f_stat, p_val = f_classif(pdf_sample.drop(['label'], axis=1), pdf_sample['label'])
        feature_scores.append(f_stat)
    feature_scores = np.array(feature_scores)
    feature_scores_df = pd.DataFrame(feature_scores, columns=featureNames)
    column_means = np.round(feature_scores_df.median(), 2)
    sorted_columns = column_means.sort_values(ascending=False)
    sorted_df = feature_scores_df[sorted_columns.index]
    new_col_names = [f"{x} ({num})" for x, num in zip(sorted_df.columns, sorted_columns.values)]
    new_col_dict = dict(zip(sorted_df.columns, new_col_names))
    regionSets = ['Top', 'Control']
    for regionL in regionSets:
        if regionL == 'Top':
            topVar = 10
            num_pieces = 1
            sorted_df_top = sorted_df.iloc[:, 0:topVar - 1]
            fontSizePlot = 14
        else:
            regionList = ['ACAd', 'ACAv', 'ILA', 'PL', 'MOs']
            sorted_df_top = sorted_df.loc[:, sorted_df.columns.isin(regionList)]
            num_pieces = 1
            fontSizePlot = 8
        sorted_df_top = sorted_df_top.rename(columns=new_col_dict)
        piece_size = int(np.ceil(len(sorted_df_top.columns) / num_pieces))
        fig = plt.figure(figsize=(num_pieces * 4, 5))
        for idx in np.arange(num_pieces):
            start_col = idx * piece_size
            end_col = start_col + piece_size
            data_slice = sorted_df_top.iloc[:, start_col:end_col]
            plt.subplot(1, num_pieces, idx + 1)
            sns.violinplot(data=data_slice, orient='h')
            plt.xlim([np.nanpercentile(data_slice.values, q=5), np.nanpercentile(data_slice.values, q=95)])
        fig.suptitle(f" {regionL} Regions {classifyDict['data']} ranked by F stat on {classifyDict['label']}",
                     fontsize=fontSizePlot, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(outDirPath, f"{classifyDict['data']}_{classifyDict['label']}_{regionL}.png"),
                    format='png', bbox_inches='tight')
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
def classifySamples(pandasdf, classifyDict, plotDict, dirDict):

    # ── Data and Pipeline Building ──────────────────────────────────────────
    classifyDict, dirDict = hf.dataStrPathGen(classifyDict, dirDict)
    X_orig, y_orig, y_dataset, featureNames, labelDict = reformat_pandasdf(pandasdf, classifyDict, dirDict)

    # ── Adaptive test_size: use 60/40 split for small per-class N ───────────
    # For timepoint-specific 5v5 comparisons (or any binary with ≤7 per class
    # after balancing) shrink the test set so that training still has
    # enough samples to learn from.
    oob_preds = {}
    
    if classifyDict.get('adaptive_test_size', False):
        n_per_class_after_balance = min(
            np.unique(y_orig, return_counts=True)[1]
        )
        threshold = classifyDict.get('small_class_threshold', 7)
        if n_per_class_after_balance <= threshold:
            effective_test_size = classifyDict.get('test_size_small', 0.40)
            print(f"  [adaptive split] n_per_class={n_per_class_after_balance} ≤ {threshold}"
                  f" → test_size set to {effective_test_size} (60% training)")
        else:
            effective_test_size = classifyDict.get('test_size_large', 0.25)
        # Override for this run without mutating the caller's dict
        classifyDict = {**classifyDict, 'test_size': effective_test_size}

    modelList, cvFxn, rsFxn, paramGrid = build_pipeline(classifyDict)

    fits = ['Real']
    if classifyDict['shuffle']:
        fits = fits + ['Shuffle']

    YtickLabs = list(labelDict.keys())
    n_classes = len(YtickLabs)

    for fit in fits:
        for clf in modelList:
            X = X_orig
            y = y_orig.copy()

            modelStr, saveStr, dirDict = hf.modelStrPathGen(clf, dirDict, cvFxn.n_splits, fit, classifyDict['randSeed'])
            saveFilePath = dirDict['tempDir_outdata']
            print("\nMODEL SAVE PATH:")
            print(saveFilePath)

            if classifyDict['saveLoadswitch'] and os.path.exists(saveFilePath):
                print(f"loading model: {modelStr}")
                with open(saveFilePath, 'rb') as f:
                    [classifyDict, modelList, modelStr, saveStr, featureSelSwitch,
                     y_real_lab, y_prob, conf_matrix_list_of_arrays, X_test_trans_list,
                     scores, selected_features_list, selected_features_params,
                     baseline_val, shap_values_list, oob_preds] = pkl.load(f)
            else:
                print(f"evaluating model: {modelStr}")

                # sklearn ≥1.8 deprecated the 'penalty' attribute; use coef_ sparsity instead
                penaltyStr = None
                if hasattr(clf['classif'], 'penalty'):
                    try:
                        penaltyStr = clf['classif'].penalty
                    except Exception:
                        pass
                featureSelSwitch = False
                if fit != 'Shuffle' and ('featureSel' in clf.named_steps.keys() or penaltyStr not in ('l2', None)):
                    featureSelSwitch = True

                cv_count = cvFxn.n_splits
                empty_list  = [None] * cv_count, [None] * cv_count, [None] * cv_count, [None] * cv_count
                empty_list2 = [None] * cv_count, [None] * cv_count, [None] * cv_count, [None] * cv_count
                y_real_lab, y_prob, conf_matrix_list_of_arrays, X_test_trans_list  = empty_list
                selected_features_list, selected_features_params, explainers, scores = empty_list2

                best_params = dict()

                

                if 'LO_' in classifyDict['label']:
                    if classifyDict['label'] == 'LO_6FDET':
                        LO_drug = ['6-F-DET']
                    elif classifyDict['label'] == 'LO_6FDET_SSRI':
                        LO_drug = ['6-F-DET', 'A-SSRI', 'C-SSRI']
                    elif classifyDict['label'] == 'LO_SSRI':
                        LO_drug = ['A-SSRI', 'C-SSRI']
                    else:
                        raise ValueError(f"{classifyDict['label']} has no LO_drug defined.")

                if 'LO_' in classifyDict['label']:
                    vecSize = range(n_classes - len(LO_drug))
                else:
                    vecSize = range(n_classes)

                baseline_val, shap_values_list = [[] for _ in vecSize], [[] for _ in vecSize]

                print(f"Performing CV split ", end='')
                for idx, (train_index, test_index) in enumerate(cvFxn.split(X_orig, y_orig)):
                    print(f"{idx + 1} ", end='')

                    X_train, X_test = X_orig[train_index], X_orig[test_index]
                    y_train, y_test = y_orig[train_index], y_orig[test_index]

                        # Undersample majority class in training fold only — keeps test set clean
                    if rsFxn is not None:
                        X_train, y_train = rsFxn.fit_resample(X_train, y_train)

                    if 'LO_' in classifyDict['label']:
                        bool_vec     = np.array([e not in LO_drug for e in y_train])
                        bool_lab_vec = np.array([e not in LO_drug for e in YtickLabs])
                        YtickLabs_train = np.array(YtickLabs.copy())
                        X_train = X_train[bool_vec]
                        y_train = y_train[bool_vec]
                        YtickLabs_train = YtickLabs_train[bool_lab_vec]

                    if fit == 'Shuffle':
                        y_test  = shuffle(y_test,  random_state=classifyDict['randState'])
                        y_train = shuffle(y_train, random_state=classifyDict['randState'])

                    X_train = pd.DataFrame(X_train, columns=featureNames)
                    X_test  = pd.DataFrame(X_test,  columns=featureNames)

                    if classifyDict['gridCV'] and paramGrid:
                        grid_search = GridSearchCV(clf, paramGrid, cv=classifyDict['innerFold'],
                                                   scoring='neg_log_loss', n_jobs=-1)
                        grid_search.fit(X_train, y_train)
                        best_params = grid_search.best_params_
                        clf.set_params(**best_params)

                    try:
                        clf.fit(X_train, y_train)
                    except Exception as e:
                        print(f"\n Failed to fit CV {idx}: {e}")
                        continue

                    if 'featureSel' in clf.named_steps.keys():
                        feature_selected = featureNames[clf['featureSel'].get_support()]
                    else:
                        feature_selected = featureNames

                    if fit != 'Shuffle':
                        X_test_trans = pd.DataFrame(clf[:-1].transform(X_test),
                                                    columns=feature_selected, index=test_index)
                        X_train_trans = pd.DataFrame(clf[:-1].transform(X_train),   # ← add this
                                 columns=feature_selected)
                        # Only collect SHAP values if the transformed data has no NaN
                        # (unexpected NaN can reach here if a feature had zero variance
                        # after imputation, causing the PowerTransformer to produce NaN)
                        if not X_test_trans.isna().any().any():
                            explainers, shap_values_list, baseline_val = hf.collect_shap_values(
                                idx, explainers, shap_values_list, baseline_val, n_classes, clf,
                                X_test_trans, feature_selected, test_index, classifyDict['featurePert'],
                                X_train_trans=X_train_trans)  # ← pass transformed training data as background for SHAP
                        else:
                            print(f"\n  [SHAP] Skipped CV split {idx}: NaN in transformed test data")
                        X_test_trans_list[idx] = X_test_trans.reset_index()

                    x_test_predict = clf.predict(X_test)

                    # ── OOB: store per-sample predictions ──────────────────────
                    if fit == 'Real':
                        for sample_pos, sample_idx in enumerate(test_index):
                            pred  = x_test_predict[sample_pos]
                            truth = y_test[sample_pos] if hasattr(y_test, '__getitem__') else list(y_test)[sample_pos]
                            oob_preds.setdefault(int(sample_idx), []).append((pred, truth))
                    conf_matrix = confusion_matrix(y_test, x_test_predict, labels=YtickLabs)
                    if np.max(np.sum(conf_matrix, axis=1)) != 1:
                        sums = np.sum(conf_matrix, axis=1)
                        conf_matrix = conf_matrix / sums[:, np.newaxis]
                    conf_matrix_list_of_arrays[idx] = conf_matrix
                    scores[idx] = np.mean(np.diag(conf_matrix))

                    y_scores = clf.predict_proba(X_test)
                    if len(clf.classes_) != len(YtickLabs):
                        y_prob[idx] = y_scores
                    elif not np.all(clf.classes_ == YtickLabs):
                        mapping = {e: i for i, e in enumerate(clf.classes_)}
                        index = [mapping[e] for e in YtickLabs]
                        y_prob[idx] = y_scores[:, index]
                    else:
                        y_prob[idx] = y_scores
                    y_real_lab[idx] = y_test

                    if featureSelSwitch:
                        selected_features_list = hf.feature_selection_info_gather(
                            idx, clf, featureNames, penaltyStr, selected_features_list)
                        if classifyDict['gridCV']:
                            selected_features_params[idx] = best_params

                if classifyDict['saveLoadswitch']:
                    saveList = [classifyDict, modelList, modelStr, saveStr, featureSelSwitch,
                                y_real_lab, y_prob, conf_matrix_list_of_arrays, X_test_trans_list,
                                scores, selected_features_list, selected_features_params,
                                baseline_val, shap_values_list, oob_preds]
                    with open(saveFilePath, 'wb') as f:
                        pkl.dump(saveList, f)

            # ── Post-CV cleanup ──────────────────────────────────────────────
            remove_Idx = [x is not None for x in y_real_lab]
            y_real_lab               = [e for e, f in zip(y_real_lab, remove_Idx) if f]
            y_prob                   = [e for e, f in zip(y_prob, remove_Idx) if f]
            scores                   = [e for e, f in zip(scores, remove_Idx) if f]
            conf_matrix_list_of_arrays = [e for e, f in zip(conf_matrix_list_of_arrays, remove_Idx) if f]

            if 'LO_' not in classifyDict['label']:
                auc_dict = pf.plotPRcurve(n_classes, y_real_lab, y_prob, labelDict,
                                          YtickLabs, modelStr, plotDict['plot_PRcurve'],
                                          fit, dirDict)

                # ── OOB error rate ──────────────────────────────────────────
                # For each sample, take the majority vote across all splits
                # in which it appeared in the test set, then compare to truth.
                oob_error = np.nan
                if fit == 'Real' and oob_preds:
                    correct, total = 0, 0
                    for preds_list in oob_preds.values():
                        truth      = preds_list[0][1]          # true label (constant)
                        all_preds  = [p[0] for p in preds_list]
                        # Majority vote
                        from collections import Counter as _Ctr
                        majority   = _Ctr(all_preds).most_common(1)[0][0]
                        correct   += int(majority == truth)
                        total     += 1
                    oob_error = 1.0 - correct / total
                    print(f"  OOB error rate ({total} samples): {oob_error:.3f}")

                score_dict = dict(
                    auc                  = auc_dict,
                    auc_per_split        = auc_dict.get('PerSplit', []),
                    auc_per_split_class  = auc_dict.get('PerSplit_perClass', {}),   # ← add this
                    scores               = scores,
                    featuresPerModel     = selected_features_list,
                    compLabel            = ' vs '.join(labelDict.keys()),
                    oob_error            = oob_error,
                    test_size            = classifyDict['test_size'],
                    n_classes            = n_classes,
                )
                dictPath = os.path.join(dirDict['outDir_model'], f'scoreDict_{fit}.pkl')
                with open(dictPath, 'wb') as f:
                    pkl.dump(score_dict, f)

            if plotDict['featureCorralogram']:
                featureCountLists = hf.feature_model_count(selected_features_list)
                feature_df = pd.DataFrame(X, columns=featureNames, index=y_dataset)
                pf.correlation_subset(feature_df, pandasdf, featureCountLists,
                                      plotDict['shapSummaryThres'], classifyDict, dirDict)

            if fit != 'Shuffle':
                # Only attempt SHAP plots when values were actually collected.
                # If all CV splits produced NaN in the transformed data (common with
                # small binary classes + PowerTransformer), every split is skipped and
                # shap_values_list[0] stays as [], causing IndexError downstream.
                shap_collected = any(len(s) > 0 for s in shap_values_list)
                if not shap_collected:
                    print("  [SHAP] No SHAP values were collected across any CV split — skipping SHAP plots.")

                if plotDict['plot_SHAPforce'] and n_classes == 2 and shap_collected:
                    pf.plot_shap_force(X_test_trans_list, shap_values_list[0],
                                       baseline_val[0], y_real_lab, labelDict, plotDict, dirDict)
                if plotDict['plot_SHAPsummary'] and shap_collected:
                    pf.plot_shap_summary(X_test_trans_list, shap_values_list, y,
                                         n_classes, plotDict, classifyDict, dirDict, labelDict=labelDict)

            if plotDict['plot_ConfusionMatrix']:
                pf.plotConfusionMatrix(scores, YtickLabs, conf_matrix_list_of_arrays, fit, saveStr, dirDict)

            if featureSelSwitch:
                hf.stringReportOut(selected_features_list, selected_features_params, YtickLabs, dirDict)


# ─────────────────────────────────────────────────────────────────────────────
def _pivot_with_na_handling(pandasdf, classifyDict):
    """
    Pivot long → wide and handle NaN values in two steps:

    Step 1 – Region (column) dropout
        Regions missing in more than `na_region_thresh` fraction of datasets
        are dropped entirely.  These are brain structures that were simply not
        detectable in many animals; imputing them would introduce noise.
        Controlled by:  classifyDict['na_region_thresh']  (default 0.20)

    Step 2 – Cell imputation
        Remaining NaN cells (single animals missing a region) are filled with
        the per-region mean across all datasets in the pivot.  This is a
        conservative imputation: it replaces a missing value with the average
        of that region, minimally distorting the feature distribution.
        Controlled by:  classifyDict['na_impute']  ('mean' | 'zero')

    Why not drop rows (datasets)?
        Every dataset is a precious animal.  Dropping datasets with any NaN
        would remove all 95 samples, since every dataset has at least some
        missing regions (~5–23% per dataset).

    Why impute rather than just drop columns?
        Dropping all 311 regions that have ANY NaN (strict complete-case
        analysis) would cut the feature space from 661 → 350, discarding 47%
        of brain regions.  With a 20% threshold and mean-imputation we keep
        578 regions while filling only the sparse remaining NaN cells.
    """
    # pivot_table uses aggfunc='mean' to handle any (unlikely) duplicates
    ls_data_agg = pandasdf.pivot_table(
        index   = 'dataset',
        columns = classifyDict['feature'],
        values  = classifyDict['data'],
        aggfunc = 'mean'
    )

    # ── Step 1: drop high-missingness regions ────────────────────────────────
    na_thresh = classifyDict.get('na_region_thresh', 0.20)
    nan_frac  = ls_data_agg.isna().mean(axis=0)
    keep_mask = nan_frac <= na_thresh
    n_dropped = (~keep_mask).sum()
    if n_dropped > 0:
        print(f"  [NA] dropped {n_dropped} regions with >{na_thresh*100:.0f}% missing "
              f"→ {keep_mask.sum()} / {len(keep_mask)} regions kept")
    ls_data_agg = ls_data_agg.loc[:, keep_mask]

    # ── Step 2: impute remaining sparse NaN cells ────────────────────────────
    n_remaining = int(ls_data_agg.isna().sum().sum())
    if n_remaining > 0:
        impute_method = classifyDict.get('na_impute', 'mean')
        if impute_method == 'mean':
            col_means   = ls_data_agg.mean(axis=0)      # mean per region, ignores NaN
            ls_data_agg = ls_data_agg.fillna(col_means)
            print(f"  [NA] imputed {n_remaining} sparse NaN cells with per-region mean")
        elif impute_method == 'zero':
            ls_data_agg = ls_data_agg.fillna(0)
            print(f"  [NA] imputed {n_remaining} sparse NaN cells with 0")

    return ls_data_agg


# ─────────────────────────────────────────────────────────────────────────────
def reformat_pandasdf(pandasdf, classifyDict, dirDict):
    """
    Pivot the long-format DataFrame into (samples × features) for the classifier.

    ── Changes from drug version ──────────────────────────────────────────────
    1. Label column: queries 'stressor' first, falls back to 'drug'.
    2. No drug-specific dataset ordering (PSI/KET/SAL reindex removed).
    3. No drug-name fixups (6FDET→6-F-DET, DMT→5MEO removed).
    4. customOrder updated to STRESSOR_ORDER: ['Ctrl','FS','FSW','RS','TS'].
    """
    import warnings
    conv_dict = hf.create_drugClass_dict(classifyDict)
    data_param_string = dirDict['data_param_string']
    filtAggFileName = os.path.join(dirDict['tempDir_data'], "data_preprocessed.pkl")

    # ── Determine which column carries the class label ───────────────────────
    label_source_col = 'stressor' if 'stressor' in pandasdf.columns else 'drug'

    # ── Apply label conversion if a conv_dict exists ─────────────────────────
    if classifyDict['label'] not in ('stressor', 'drug') and bool(conv_dict):
        pandasdf = pandasdf.copy()
        pandasdf[classifyDict['label']] = pandasdf[label_source_col].map(conv_dict)
        pandasdf = pandasdf.dropna(subset=[classifyDict['label']])

    # ── Feature filtering / aggregation ──────────────────────────────────────
    if classifyDict['featurefilt'] or classifyDict['featureAgg']:
        if not os.path.exists(filtAggFileName):
            print(f"Generating filtered/aggregated data file... {data_param_string}")
            if classifyDict['featurefilt']:
                pandasdf = hf.filter_features(pandasdf, classifyDict)
            if classifyDict['featureAgg']:
                ls_data_agg = hf.agg_cluster(pandasdf, classifyDict, dirDict)
            else:
                ls_data_agg = _pivot_with_na_handling(pandasdf, classifyDict)
            ls_data_agg.to_pickle(filtAggFileName)
        else:
            print(f"Loading filtered/aggregated data from file... {data_param_string}")
            ls_data_agg = pd.read_pickle(filtAggFileName)
    else:
        ls_data_agg = _pivot_with_na_handling(pandasdf, classifyDict)

    # ── Shape data ────────────────────────────────────────────────────────────
    X          = np.array(ls_data_agg.values)
    y_dataset  = ls_data_agg.index

    # Extract label from dataset index: "Ctrl_7D_F1" → "Ctrl"
    y = np.array([str(x).split('_')[0] for x in ls_data_agg.index])

    # ── Convert labels using conv_dict (pairwise comparisons) ────────────────
    if bool(conv_dict):
        y = np.array([conv_dict.get(item, np.nan) for item in y])
        labels = [s for i, s in enumerate(y) if s not in y[:i]]
        # Canonicalise order so compLabel is deterministic regardless of data sort
        labels = sorted(labels, key=lambda x:
            STRESSOR_ORDER.index(x)  if x in STRESSOR_ORDER  else
            TIMEPOINT_ORDER.index(x) if x in TIMEPOINT_ORDER else
            len(STRESSOR_ORDER) + len(TIMEPOINT_ORDER)
        )
    else:
    # All stressors: use canonical order, keep only those present
        if set(y) <= set(TIMEPOINT_ORDER):          # ← timepoint classifier
            y = pd.Categorical(y, categories=TIMEPOINT_ORDER, ordered=True)
            labels = [lab for lab in TIMEPOINT_ORDER if lab in y]
        else: # All stressors: use canonical order, keep only those present
            y = pd.Categorical(y, categories=STRESSOR_ORDER, ordered=True)
            labels = [lab for lab in STRESSOR_ORDER if lab in y.categories and lab in y]
        # Fallback if no stressor labels found (e.g. pure drug data)
        if not labels:
            labels = list(y.unique())

    # Filter out any NaN rows (samples whose stressor isn't in conv_dict)
    valid_mask = np.array([v not in (np.nan, None, 'nan') and not (isinstance(v, float) and np.isnan(v))
                           for v in y])
    X         = X[valid_mask]
    y         = np.array(y)[valid_mask]
    y_dataset = y_dataset[valid_mask]

    y_Int_dict   = dict(zip(labels, range(len(labels))))
    featureNames = np.array(ls_data_agg.columns.tolist())

    return X, y, y_dataset, featureNames, y_Int_dict


# ─────────────────────────────────────────────────────────────────────────────
def return_train_test_data(randUndSamp, train_index, test_index, X, y, y_bin, featureNames, classifyDict):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_bin_test = y_bin[test_index, :]
    if classifyDict['balance']:
        X_train, y_train = randUndSamp.fit_resample(X_train, y_train)
    X_train = pd.DataFrame(X_train, columns=featureNames)
    X_test  = pd.DataFrame(X_test,  columns=featureNames)
    return X_train, X_test, y_train, y_test, y_bin_test


# ─────────────────────────────────────────────────────────────────────────────
def build_pipeline(classifyDict):
    """Build sklearn Pipeline with feature selection, scaling and classifier."""

    pipelineList = []

    # ── Feature transform ─────────────────────────────────────────────────────
    if classifyDict['model_featureTransform']:
        pipelineList.append(
            ('featureTrans', PowerTransformer(method='yeo-johnson', standardize=False))
        )

    # ── Feature scaling ───────────────────────────────────────────────────────
    if classifyDict['model_featureScale']:
        pipelineList.append(('featureScale', RobustScaler()))

    # ── Feature selection ─────────────────────────────────────────────────────
    fsMethod = classifyDict['model_featureSel']
    fsVar = None

    if fsMethod == 'Univar':
        selector = SelectKBest(f_classif, k=classifyDict['model_featureSel_k'][0])
        fsVar = 'k'
        pipelineList.append(('featureSel', selector))
    elif fsMethod == 'mutInfo':
        selector = SelectKBest(mutual_info_classif, k=classifyDict['model_featureSel_k'][0])
        fsVar = 'k'
        pipelineList.append(('featureSel', selector))
    elif fsMethod == 'Fdr':
        pipelineList.append(('featureSel', SelectFdr(f_classif, alpha=classifyDict['model_featureSel_alpha'])))
    elif fsMethod == 'Fwe':
        pipelineList.append(('featureSel', SelectFwe(f_classif, alpha=classifyDict['model_featureSel_alpha'])))
    elif fsMethod == 'Fwe_BH':
        pipelineList.append(('featureSel', SelectFwe_Holms(f_classif, alpha=classifyDict['model_featureSel_alpha'])))
    elif fsMethod == 'Boruta':
        pipelineList.append(('featureSel', BorutaFeatureSelector(
            feature_sel='confirmed', random_seed=classifyDict['randSeed'],
            random_state=classifyDict['randState']
        )))
    elif fsMethod == 'None':
        pass

    # ── Classifier ────────────────────────────────────────────────────────────
    model = classifyDict['model']
    if model == 'LogRegL2':
        clf_step = linear_model.LogisticRegression(
            penalty='l2', multi_class=classifyDict['multiclass'],
            solver='saga', max_iter=classifyDict['max_iter'],
            random_state=classifyDict['randState'], C=1
        )
    elif model == 'LogRegL1':
        clf_step = linear_model.LogisticRegression(
            penalty='l1', multi_class=classifyDict['multiclass'],
            solver='saga', max_iter=classifyDict['max_iter'],
            random_state=classifyDict['randState'], C=1
        )
    elif model == 'svm':
        clf_step = svm.SVC(kernel='linear', probability=True, random_state=classifyDict['randState'])
    else:
        raise ValueError(f"Unknown model: {model}")

    pipelineList.append(('classif', clf_step))
    clf = Pipeline(pipelineList)

    # ── Cross-validation ──────────────────────────────────────────────────────
    if classifyDict['CVstrat'] == 'ShuffleSplit':
        cvFxn = StratifiedShuffleSplit(n_splits=classifyDict['CV_count'],
                                        test_size=classifyDict['test_size'],
                                        random_state=classifyDict['randSeed'])
    elif classifyDict['CVstrat'] == 'StratKFold':
        cvFxn = StratifiedKFold(n_splits=classifyDict['CV_count'],
                                 shuffle=True, random_state=classifyDict['randSeed'])

    # ── Resampler ─────────────────────────────────────────────────────────────
    if classifyDict['balance']:
        rsFxn = RandomUnderSampler(sampling_strategy='not minority',
                                    random_state=classifyDict['randState'])
    else:
        rsFxn = None

    # ── Build model list ──────────────────────────────────────────────────────
    modelList = []
    paramGrid = classifyDict.get('pGrid', {})

    if classifyDict['model_featureSel_mode'] == 'modelPer' and fsVar is not None:
        for k_feat in classifyDict['model_featureSel_k']:
            clf_copy = deepcopy(clf)
            clf_copy.set_params(**{f"featureSel__{fsVar}": k_feat})
            modelList.append(clf_copy)
    else:
        modelList.append(clf)

    return modelList, cvFxn, rsFxn, paramGrid


# ─────────────────────────────────────────────────────────────────────────────
# Feature selection classes (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
class BorutaFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_sel='all', random_seed=0, random_state=None, n_workers=None):
        self.feature_sel  = feature_sel
        self.random_state = random_state
        self.n_workers    = n_workers
        self.random_seed  = random_seed

    def __str__(self):
        return f"BorFS(random_state_seed={self.random_seed})"

    def fit(self, X, y=None):
        boruta = BorutaPy(estimator=RandomForestClassifier(n_jobs=self.n_workers),
                          n_estimators='auto', max_iter=100, perc=95,
                          verbose=0, random_state=self.random_state)
        boruta.fit(np.array(X), np.array(y))
        self.selected_features_ = boruta.support_ | boruta.support_weak_ if self.feature_sel == 'all' else boruta.support_
        return self

    def get_support(self, indices=True):
        return self.selected_features_

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_features_]
        return X[:, self.selected_features_]


class SelectFwe_Holms(_BaseFilter):
    _parameter_constraints: dict = {
        **_BaseFilter._parameter_constraints,
        "alpha": [Interval(Real, 0, 1, closed="both")],
    }

    def __init__(self, score_func=f_classif, *, alpha=5e-2):
        super().__init__(score_func=score_func)
        self.alpha = alpha

    def _get_support_mask(self):
        check_is_fitted(self)
        m = len(self.pvalues_)
        j = np.arange(1, m + 1)
        threshold = self.alpha / np.array(m + 1 - j)
        sorted_indices  = np.argsort(self.pvalues_)
        reverse_indices = np.argsort(sorted_indices)
        sorted_pvalues  = self.pvalues_[sorted_indices]
        below_threshold = sorted_pvalues < threshold
        return below_threshold[reverse_indices]
