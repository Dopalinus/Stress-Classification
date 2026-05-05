"""
helperFunctions.py  –  adapted for stressor data
=================================================
Key changes vs. original drug version:
  • create_color_dict()       – stressor colours added / drug colours kept
  • create_translation_dict() – stressor translations added
  • create_drugClass_dict()   – stressor-pair classification labels added
  • dataStrPathGen()          – unchanged (calls create_drugClass_dict internally)
  • All other utility functions kept verbatim.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import os, sys
import pickle as pkl
from collections import defaultdict, Counter

# ─────────────────────────────────────────────────────────────────────────────
# Stressor metadata (shared across modules)
# ─────────────────────────────────────────────────────────────────────────────
STRESSOR_ORDER  = ['Ctrl', 'FS', 'FSW', 'RS', 'TS']
TIMEPOINT_ORDER = ['Acute', '7D', '14D', '21D']


# ─────────────────────────────────────────────────────────────────────────────
def create_drugClass_dict(classifyDict):
    """
    Return a label-conversion dict for the requested classification type.
    Covers both original drug-based labels (legacy) and new stressor-based labels.
    """
    conv_dict = dict()
    label = classifyDict['label']

    # ── Stressor vs stressor (pairwise) ─────────────────────────────────────
    if label == 'class_CtrlFS':
        conv_dict = {'Ctrl': 'Ctrl', 'FS': 'FS'}
    elif label == 'class_CtrlFSW':
        conv_dict = {'Ctrl': 'Ctrl', 'FSW': 'FSW'}
    elif label == 'class_CtrlRS':
        conv_dict = {'Ctrl': 'Ctrl', 'RS': 'RS'}
    elif label == 'class_CtrlTS':
        conv_dict = {'Ctrl': 'Ctrl', 'TS': 'TS'}
    elif label == 'class_FSRS':
        conv_dict = {'FS': 'FS', 'RS': 'RS'}
    elif label == 'class_FSTS':
        conv_dict = {'FS': 'FS', 'TS': 'TS'}
    elif label == 'class_FSWRS':
        conv_dict = {'FSW': 'FSW', 'RS': 'RS'}
    elif label == 'class_FSWTS':
        conv_dict = {'FSW': 'FSW', 'TS': 'TS'}
    elif label == 'class_RSTS':
        conv_dict = {'RS': 'RS', 'TS': 'TS'}
    elif label == 'class_FSFSW':
        conv_dict = {'FS': 'FS', 'FSW': 'FSW'}

    # ── All stressors (multiclass) ───────────────────────────────────────────
    elif label == 'stressor':
        # No conversion needed – labels already are the stressor names.
        # Return empty dict so the pipeline uses raw stressor labels.
        conv_dict = {}
    # --── Timepoint groups ─────────────────────────────────────────────────────
    elif label == 'class_tp_all':
        conv_dict = {'Acute': 'Acute', '7D': '7D', '14D': '14D', '21D': '21D'}
        
    # ── Stressor groups ──────────────────────────────────────────────────────
    elif label == 'class_acute_vs_chronic':
        # Acute stressors vs repeated stressors
        conv_dict = {'FS': 'Acute', 'FSW': 'Repeated', 'RS': 'Repeated', 'TS': 'Repeated'}

    elif label == 'class_physical_vs_psychological':
        # Physical (FS, FSW, TS) vs Psychological/Restraint (RS)
        conv_dict = {'FS': 'Physical', 'FSW': 'Physical', 'TS': 'Physical', 'RS': 'Restraint'}

    elif label == 'class_CtrlAll':
        # Control (all timepoints) vs all stressors pooled (FS, FSW, RS, TS – all timepoints)
        # Gives the maximum statistical power by pooling everything.
        conv_dict = {'Ctrl': 'Ctrl', 'FS': 'Stress', 'FSW': 'Stress', 'RS': 'Stress', 'TS': 'Stress'}

    elif label == 'class_CtrlAll_balanced':
        # Same as class_CtrlAll but the pipeline will undersample the majority class,
        # so the label map is identical – the balance flag handles the rest.
        conv_dict = {'Ctrl': 'Ctrl', 'FS': 'Stress', 'FSW': 'Stress', 'RS': 'Stress', 'TS': 'Stress'}

    # ── Legacy drug labels (kept for backward compatibility) ─────────────────
    elif label == 'class_5HT2A':
        conv_dict = {'PSI': 'PSI/5MEO', '5MEO': 'PSI/5MEO', 'MDMA': 'MDMA'}
    elif label == 'class_PsiKet':
        conv_dict = {'PSI': 'PSI', 'KET': 'KET'}
    elif label == 'class_Psi5MEO':
        conv_dict = {'PSI': 'PSI', '5MEO': '5MEO'}
    elif label == 'class_PsiMDMA':
        conv_dict = {'PSI': 'PSI', 'MDMA': 'MDMA'}
    elif label == 'class_PsiSSRI':
        conv_dict = {'PSI': 'PSI', 'A-SSRI': 'A-SSRI'}
    elif label == 'class_SSRI':
        conv_dict = {'A-SSRI': 'A-SSRI', 'C-SSRI': 'C-SSRI'}
    elif label in ('drug', 'LO_6FDET', 'LO_SSRI'):
        conv_dict = {}   # raw drug labels used

    # ── Unrecognised label ───────────────────────────────────────────────────
    elif label != 'stressor':
        raise KeyError(
            f"No classification dict found for label='{label}'. "
            "Add it to helperFunctions.create_drugClass_dict()."
        )

    return conv_dict


# ─────────────────────────────────────────────────────────────────────────────
def create_color_dict(dictType='stressor', rgbSwitch=0, alpha_value=1, scaleVal=False):
    """
    Return a colour dictionary for stressors, brain areas, or drugs (legacy).
    """
    color_dict = dict()

    if dictType in ('stressor', 'drug'):
        # ── Stressor colours ─────────────────────────────────────────────────
        color_dict['Ctrl']    = '#BBBBBB'   # grey  (control)
        color_dict['FS']      = '#228833'   # green (foot shock)
        color_dict['FSW']     = '#4477AA'   # blue  (forced swimm)
        color_dict['RS']      = '#AA3377'   # purple (restraint stress)
        color_dict['TS']      = '#CC3311'   # red   (tail suspension)
        # ── Timepoint colours (sequential light→dark blue-grey) ──────────────────
        color_dict['Acute'] = '#88CCEE'   # light blue
        color_dict['7D']    = '#44AA99'   # teal
        color_dict['14D']   = '#117733'   # green
        color_dict['21D']   = '#332288'   # dark purple

        # ── Legacy drug colours (kept so existing plot calls don't break) ────
        color_dict['PSI']     = '#228833'
        color_dict['KET']     = '#AA3377'
        color_dict['5MEO']    = '#4477AA'
        color_dict['6-F-DET'] = '#66CCEE'
        color_dict['MDMA']    = '#CCBB44'
        color_dict['A-SSRI']  = '#CC3311'
        color_dict['C-SSRI']  = '#EE6677'
        color_dict['SAL']     = '#BBBBBB'

        # ── Group colours ─────────────────────────────────────────────────────
        color_dict['Stress']   = '#CC3311'
        color_dict['Repeated'] = '#AA3377'
        color_dict['Physical'] = '#228833'
        color_dict['Restraint']= '#4477AA'

    elif dictType == 'brainArea':
        color_dict['Olfactory']       = '#377eb8'
        color_dict['Cortex']          = '#ff7f00'
        color_dict['Hippo']           = '#4daf4a'
        color_dict['StriatumPallidum']= '#f781bf'
        color_dict['Thalamus']        = '#a65628'
        color_dict['Hypothalamus']    = '#984ea3'
        color_dict['MidHindMedulla']  = '#999999'
        color_dict['Cerebellum']      = '#e41a1c'
        color_dict['Unknown']         = '#cccccc'

    if rgbSwitch:
        for key, val in list(color_dict.items()):
            r = int(val[1:3], 16)
            g = int(val[3:5], 16)
            b = int(val[5:7], 16)
            color_dict[key] = (r, g, b)
        if scaleVal:
            color_dict = {k: tuple(c/255 for c in v) for k, v in color_dict.items()}
        if alpha_value != 0:
            color_dict = {k: v + (alpha_value,) for k, v in color_dict.items()}

    return color_dict


# ─────────────────────────────────────────────────────────────────────────────
def create_translation_dict(dictType='stressor'):
    translation_dict = dict()

    if dictType == 'stressor':
        translation_dict['Ctrl'] = 'Control'
        translation_dict['FS']   = 'Foot Shock'
        translation_dict['FSW']  = 'Forced Swim'
        translation_dict['RS']   = 'Restraint Stress'
        translation_dict['TS']   = 'Tail Suspension'

    elif dictType == 'drug':
        translation_dict['PSI']     = 'Psilocybin'
        translation_dict['KET']     = 'Ketamine'
        translation_dict['5MEO']    = '5-MeO-DMT'
        translation_dict['6-F-DET'] = '6-Fluoro-DET'
        translation_dict['MDMA']    = '3,4-MDMA'
        translation_dict['A-SSRI']  = 'Acute SSRI'
        translation_dict['C-SSRI']  = 'Chronic SSRI'
        translation_dict['SAL']     = 'Saline'

    elif dictType == 'brainArea':
        translation_dict['Olfactory']        = 'Olfactory'
        translation_dict['Cortex']           = 'Cortex'
        translation_dict['Hippo']            = 'Hippo'
        translation_dict['StriatumPallidum'] = 'Stri+Pall'
        translation_dict['Thalamus']         = 'Thalamus'
        translation_dict['Hypothalamus']     = 'Hypothalamus'
        translation_dict['MidHindMedulla']   = 'Mid Hind Medulla'
        translation_dict['Cerebellum']       = 'Cerebellum'
        translation_dict['Unknown']          = 'Unknown'

    return translation_dict


# ─────────────────────────────────────────────────────────────────────────────
# All functions below are unchanged from the original – copied verbatim.
# ─────────────────────────────────────────────────────────────────────────────

def find_middle_occurrences(lst):
    positions = defaultdict(list)
    first_occurrences, middle_occurrences, last_occurrences, items = [], [], [], []
    for idx, elem in enumerate(lst):
        positions[elem].append(idx)
    for elem, pos_list in positions.items():
        middle_idx = len(pos_list) // 2
        middle_occurrences.append(pos_list[middle_idx])
        items.append(elem)
    for item in items:
        first_occurrences.append(positions[item][0])
        last_occurrences.append(positions[item][-1])
    position_dict = {item: [first_occurrences[i], middle_occurrences[i], last_occurrences[i]]
                     for i, item in enumerate(items)}
    return position_dict


def agg_cluster(lightsheet_data, classifyDict, dirDict):
    from sklearn import cluster, preprocessing
    from scipy.spatial import distance
    from scipy.cluster import hierarchy
    import seaborn as sns

    cluster_count = classifyDict['cluster_count']
    brainAreaList = list(lightsheet_data['Brain_Area'].unique()) if 'Brain_Area' in lightsheet_data.columns else ['Unknown']
    AreaIdx = dict(zip(brainAreaList, np.arange(len(brainAreaList))))
    brainAreaColorDict = create_color_dict(dictType='brainArea', rgbSwitch=0, alpha_value=1, scaleVal=False)

    colList = [classifyDict['feature'], 'Brain_Area']
    # If Brain_Area is missing, add a placeholder
    if 'Brain_Area' not in lightsheet_data.columns:
        lightsheet_data = lightsheet_data.copy()
        lightsheet_data['Brain_Area'] = 'Unknown'

    regionArea = lightsheet_data.loc[:, colList].drop_duplicates()
    regionArea['Brain_Area_Idx'] = [AreaIdx.get(x, 0) for x in regionArea.loc[:, 'Brain_Area']]
    regionArea['Region_Color'] = [brainAreaColorDict.get(x, '#cccccc') for x in regionArea.loc[:, 'Brain_Area']]
    regionArea.sort_values(by='abbreviation', inplace=True)

    df_Tilted = lightsheet_data.pivot(index='dataset', columns=classifyDict['feature'], values=classifyDict['data'])
    X = df_Tilted.values

    X_scaled = X

    col_row_link = hierarchy.linkage(X_scaled.T, method=classifyDict['featureSel_linkage'],
                                     metric=classifyDict['featureSel_distance'], optimal_ordering=True)
    titleStr = (f"{classifyDict['data']}, Metric: {classifyDict['featureSel_distance']}, "
                f"Linkage: {classifyDict['featureSel_linkage']}")

    plt.figure(figsize=(40, 10))
    dendObj = hierarchy.dendrogram(col_row_link, labels=df_Tilted.columns, leaf_rotation=90,
                                   leaf_font_size=10, color_threshold=classifyDict['cluster_thres'])
    plt.title(titleStr, fontsize=30)
    ax = plt.gca()
    for xtick in ax.get_xticklabels():
        colorCode = regionArea[regionArea[classifyDict['feature']] == xtick._text]['Region_Color']
        if len(colorCode):
            xtick.set_color(colorCode.values[0])

    plt.axhline(y=classifyDict['cluster_thres'], color='r', linestyle='--', linewidth=3)
    plt.show()

    new_labels = hierarchy.fcluster(col_row_link, classifyDict['cluster_thres'], criterion='distance')
    unique_values, counts = np.unique(new_labels, return_counts=True)
    df_agged = pd.DataFrame(index=df_Tilted.index)

    single_Idx = counts == 1
    singleFeatureIdx = np.in1d(new_labels, unique_values[single_Idx])
    df_agged = df_agged.join(df_Tilted.loc[:, singleFeatureIdx])

    unique_values = unique_values[~single_Idx]
    counts = counts[~single_Idx]
    print(f'Clustering done: {len(counts)} clusters generated from {np.sum(counts)} features.')

    file = open(os.path.join(dirDict['outDir_data'], 'Cluster_members.txt'), 'w')
    ClusterCount = 1
    for idx, val in enumerate(unique_values):
        featureList = list(df_Tilted.columns[new_labels == val])
        if len(featureList) < 5:
            new_feature_name = '-'.join(featureList)
        else:
            new_feature_name = f"Clus{ClusterCount}_n{len(featureList)}"
            ClusterCount += 1
            file.write(f'{new_feature_name}: {featureList} \n\n')
        df_agged[f'{new_feature_name}'] = df_Tilted[featureList].apply(lambda x: x.mean(), axis=1)
    file.close()
    return df_agged


def create_region_to_area_dict(lightsheet_data, dataFeature):
    brainAreas = lightsheet_data['Brain_Area'].unique() if 'Brain_Area' in lightsheet_data.columns else ['Unknown']
    AreaIdx = dict(zip(brainAreas, np.arange(len(brainAreas))))
    if isinstance(dataFeature, str):
        dataFeature = [dataFeature]
    dataFeature = list(dataFeature) + ['Brain_Area']
    if 'Brain_Area' not in lightsheet_data.columns:
        lightsheet_data = lightsheet_data.copy()
        lightsheet_data['Brain_Area'] = 'Unknown'
    regionArea = lightsheet_data.loc[:, dataFeature].drop_duplicates()
    regionArea['Brain_Area_Idx'] = [AreaIdx.get(x, 0) for x in regionArea.loc[:, 'Brain_Area']]
    regionArea.sort_values(by='Brain_Area_Idx', inplace=True)
    return regionArea


def create_brainArea_dict(dictType):
    brainAreaList = ['Olfactory', 'Cortex', 'Hippo', 'StriatumPallidum',
                     'Thalamus', 'Hypothalamus', 'MidHindMedulla', 'Cerebellum', 'Unknown']
    if dictType == 'short':
        brainAreaListPlot = ['Olfactory', 'Cortex', 'Hippo', 'Stri+Pall',
                             'Thalamus', 'Hypothalamus', 'Mid Hind Medulla', 'Cerebellum', 'Unknown']
    else:
        brainAreaListPlot = ['Olfactory', 'Cortex', 'Hippocampus', 'Striatum and Pallidum',
                             'Thalamus', 'Hypothalamus', 'Midbrain, Hind Brain, and Medulla',
                             'Cerebellum', 'Unknown']
    return dict(zip(brainAreaList, brainAreaListPlot))


def replace_strings_with_dict(input_strings, translate_dict):
    replaced_strings = []
    for string in input_strings:
        for key, value in translate_dict.items():
            string = string.replace(key, value)
        replaced_strings.append(string)
    return replaced_strings


def filter_features(pandasdf, classifyDict):
    if classifyDict['filtType'] == 'max':
        thres = np.percentile(pandasdf[classifyDict['data']], 99.5)
        pandasdf_over = pandasdf[pandasdf[classifyDict['data']] >= thres]
        features_remove = pandasdf_over[classifyDict['feature']].unique()
    if classifyDict['filtType'] == 'min':
        thres = np.percentile(pandasdf[classifyDict['data']], .5)
        pandasdf_under = pandasdf[pandasdf[classifyDict['data']] <= thres]
        features_remove = pandasdf_under[classifyDict['feature']].unique()
    pandasdf_filt = pandasdf[~pandasdf[classifyDict['feature']].isin(features_remove)]
    feature_n_old = len(pandasdf[classifyDict['feature']].unique())
    feature_n_new = len(pandasdf_filt[classifyDict['feature']].unique())
    print(f"feature count: {feature_n_old} → {feature_n_new}")
    return pandasdf_filt


def flatten_list(nested_list):
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def replace_ones_with_integers(binary_vector, integer_array):
    result = []
    integer_index = 0
    for element in binary_vector:
        if element == 1:
            if integer_index < len(integer_array):
                result.append(integer_array[integer_index])
                integer_index += 1
            else:
                raise ValueError("Not enough integers in integer_array")
        else:
            result.append(0)
    return result


def check_phrase_in_keys(phrase, dictionary):
    import re
    phrase = re.escape(phrase).replace('\\*', '.*')
    pattern = '^' + phrase + '$'
    regex = re.compile(pattern)
    return [key for key in dictionary.keys() if regex.match(key)]


def convert_to_highest_entry_array(input_arr):
    output_arr = np.zeros_like(input_arr)
    for i, row in enumerate(input_arr):
        output_arr[i, np.argmax(row)] = 1
    return output_arr


def conciseStringReport(strings, counts):
    data_dict = dict(zip(strings, counts))
    reverse_dict = {}
    for key, value in data_dict.items():
        reverse_dict.setdefault(value, []).append(key)
    result_string = ""
    for count in sorted(reverse_dict.keys()):
        string_group = ', '.join(reverse_dict[count])
        result_string += f"Present {count}x: {len(reverse_dict[count])} - {string_group}\n"
    return result_string + '\n '


def modelStrPathGen(clf, dirDict, n_splits, fit, randSeed):
    import re
    from sklearn.base import BaseEstimator
    elements = []
    for name, step in clf.steps:
        elements.append(str(step))
    modelStr = ' -> '.join(elements)
    elements = []
    for name, step in clf.steps:
        elements.append(type(step).__name__ if isinstance(step, BaseEstimator) else str(step))
    figSaveStr = '_'.join(elements)
    elements = []
    for name, step in clf.steps:
        elements.append(f"{name}_{step}")
    modelParamStr = '_'.join(elements)
    ssDict = save_string_dict()
    for key, value in ssDict.items():
        modelParamStr = modelParamStr.replace(key, value)
    modelParamStr += f"_CV{n_splits}"
    modelParamStr = re.sub(r'\n\s+', '', modelParamStr)
    modelStr = re.sub(r'\n\s+', '', modelStr)
    modelParamStr = re.sub(r',random_state=[^,]*,', f',randStateSeed={randSeed},', modelParamStr)
    modelStr = re.sub(r',random_state=[^,]*,', f',randStateSeed={randSeed},', modelStr)
    tmpDict = {
        'tempDir_model': os.path.join(dirDict['tempDir_data'], modelParamStr),
        'outDir_model': os.path.join(dirDict['outDir_data'], modelParamStr),
    }
    for key, path in tmpDict.items():
        if not os.path.isdir(path):
            os.makedirs(path)
    dirDict.update(tmpDict)
    dirDict['tempDir_outdata'] = os.path.join(tmpDict['tempDir_model'], f"{fit}_outdata.pkl")
    return modelStr, figSaveStr, dirDict


def dataStrPathGen(classifyDict, dirDict):
    ssDict = save_string_dict()
    conv_dict = create_drugClass_dict(classifyDict)
    keys_to_keep = ['data', 'label']
    if classifyDict['featureAgg']:
        keys_to_keep += ['featureAgg', 'featureSel_linkage', 'featureSel_distance', 'cluster_thres']
    if classifyDict['featurefilt']:
        keys_to_keep += ['featurefilt', 'filtType']
    if classifyDict['gridCV']:
        keys_to_keep.append('gridCV')
    smallDict = {key: value for key, value in classifyDict.items() if key in keys_to_keep}
    data_param_string = "-".join([f"{key}={value}" for key, value in smallDict.items()])
    for key, value in ssDict.items():
        data_param_string = data_param_string.replace(key, value)
    tempDataStr = os.sep.join([dirDict['tempDir'], data_param_string])
    outDataStr = os.sep.join([dirDict['classifyDir'], data_param_string])
    dirDict['data_param_string'] = data_param_string
    tmpDir = {'tempDir_data': tempDataStr, 'outDir_data': outDataStr}
    classifyDict['tempDir_cacheDir'] = tmpDir['tempDir_data']
    for key, path in tmpDir.items():
        if not os.path.isdir(path):
            os.makedirs(path)
    dirDict.update(tmpDir)
    return classifyDict, dirDict


def save_string_dict():
    saveStringDict = {
        'label=class_': '',
        'max_iter=1000, ': '',
        'label=drug': 'drug',
        'label=stressor': 'stressor',
        'featurefilt=True-filtType=min': 'filtMin',
        'featureAgg=True': 'featAgg',
        'featureSel_linkage=average-featureSel_distance=correlation': 'avgCorrClus',
        'cluster_thres=': 'clusThres',
        'n_jobs=-1': '',
        'n_workers=-1,  ': '',
        'gridCV=True': 'gridCV',
        'featureTrans_PowerTransformer(standardize=False)': 'PowTrans',
        'featureSel': 'fSel',
        'featureScale_RobustScaler()': 'RobScal',
        'classif': 'clf',
        'BorutaFeatureSelector(': 'BorFS(',
        'MRMRFeatureSelector': 'MrmrFS',
        'n_features_to_select=': '',
        'RobustScaler()': 'SelFroMod',
        'LogisticRegression(': 'LogReg(',
        "multi_class='multinomial'": 'multinom',
        ", solver='saga'": '',
    }
    return saveStringDict


def stringReportOut(selected_features_list, selected_features_params, YtickLabs, dirDict):
    if len(YtickLabs) == 2:
        YtickLabs = [f"{YtickLabs[0]} vs {YtickLabs[1]}"]
    else:
        YtickLabs = [' vs '.join(YtickLabs)]
    featurePerModel = [len(x) for x in selected_features_list]
    featurePerModelStr = str(featurePerModel)
    paramStr = ''
    if selected_features_params[0]:
        for key in selected_features_params[0].keys():
            keyVals = [x[key] for x in selected_features_params]
            paramStr += f"{key}: {str(keyVals)} \n"
    if np.sum(featurePerModel) == 0:
        return
    labels, counts = feature_model_count(selected_features_list)
    finalStr = conciseStringReport(labels, counts)
    print(f'==== {YtickLabs} ==== \n Features per Model: {featurePerModelStr}')
    print(f'Parameters: \n {paramStr}')
    print(f'Total Regions = {str(len(labels))} \n {finalStr}')
    file = open(os.path.join(dirDict['outDir_model'], 'featureSelReadout.txt'), 'w')
    file.write(f'==== {YtickLabs} ==== \n Features per Model: {featurePerModelStr} \n')
    file.write(f'Parameters: \n {paramStr}')
    file.write(f'Total Regions = {str(len(labels))} \n {finalStr}')
    file.close()


def feature_model_count(selected_features_list):
    regionList = np.concatenate(selected_features_list)
    regionDict = dict(Counter(regionList))
    return list(regionDict.keys()), list(regionDict.values())


def retrieve_dict_data(dirDict, sortedNames, classifyDict):
    targDir = dirDict['classifyDir']
    tagList = classifyDict['crossComp_tagList']
    print(f"Looking for 'scoreDict_Real.pkl' files in directories containing {tagList}")
    score_dict_paths = []
    for root, dirs, files in os.walk(targDir):
        if 'scoreDict_Real.pkl' in files:
            if all(tag in root for tag in tagList):
                score_dict_paths.append(os.path.join(root, 'scoreDict_Real.pkl'))
    assert len(score_dict_paths) > 0, f"No files found in {targDir} with tag {tagList}"

    aucScores, meanScores, aucScrambleScores, meanScrambleScores = [], [], [], []
    accStd, oobErrors, aucPerSplit = [], [], []
    featureLists, countNames = [], []

    for path in score_dict_paths:
        with open(path, 'rb') as f:
            featureDict = pkl.load(f)
            featureLists.append(featureDict['featuresPerModel'])
            scores_arr = np.array(featureDict['scores'])
            meanScores.append(float(np.mean(scores_arr)))
            accStd.append(float(np.std(scores_arr)))
            aucScores.append(featureDict['auc']['Mean'])
            oobErrors.append(featureDict.get('oob_error', np.nan))
            aucPerSplit.append(featureDict.get('auc_per_split', []))
        countNames.append(featureDict['compLabel'])

        scramblePath = path.replace('_Real', '_Shuffle')
        try:
            with open(scramblePath, 'rb') as f:
                featureDict = pkl.load(f)
                meanScrambleScores.append(float(np.mean(np.array(featureDict['scores']))))
                aucScrambleScores.append(featureDict['auc']['Mean'])
        except FileNotFoundError:
            print(f"  [retrieve_dict_data] WARNING: Shuffle pkl missing — filling with NaN:\n    {scramblePath}")
            meanScrambleScores.append(np.nan)
            aucScrambleScores.append(np.nan)

    sortIdx = sort_comparison_idx(sortedNames, countNames)
    featureLists       = [featureLists[i]       for i in sortIdx]
    countNames         = [countNames[i]         for i in sortIdx]
    aucScores          = [aucScores[i]          for i in sortIdx]
    meanScores         = [meanScores[i]         for i in sortIdx]
    accStd             = [accStd[i]             for i in sortIdx]
    oobErrors          = [oobErrors[i]          for i in sortIdx]
    aucPerSplit        = [aucPerSplit[i]         for i in sortIdx]
    aucScrambleScores  = [aucScrambleScores[i]  for i in sortIdx]
    meanScrambleScores = [meanScrambleScores[i] for i in sortIdx]
    countNames = [score.replace('/', ' & ') for score in countNames]

    return (featureLists, countNames, aucScores, meanScores, accStd,
            oobErrors, aucPerSplit, aucScrambleScores, meanScrambleScores)


def listToCounterFilt(listArray, filterByFreq=0):
    counter_u = Counter(listArray)
    if filterByFreq > 0:
        return Counter({k: v for k, v in counter_u.items() if v >= filterByFreq})
    return counter_u


def overlapCounter(list1, list2, filterByFreq=0):
    counter_u = listToCounterFilt(list1, filterByFreq)
    counter_v = listToCounterFilt(list2, filterByFreq)
    list1 = list(counter_u.keys())
    list2 = list(counter_v.keys())
    return list(set(list1) - set(list2)), list(set(list2) - set(list1)), list(set(list1) & set(list2))


def sort_comparison_idx(orderedList, dataList):
    orderedListNew = [name for name in orderedList if name in dataList]
    return [dataList.index(name) for name in orderedListNew]


def weighted_jaccard_similarity(u, v, filt):
    counter_u, counter_v = Counter(u), Counter(v)
    if filt:
        counter_u = Counter({k: v for k, v in counter_u.items() if v > filt})
        counter_v = Counter({k: v for k, v in counter_v.items() if v > filt})
    intersection = sum((counter_u & counter_v).values())
    union = sum((counter_u | counter_v).values())
    return intersection / union if union != 0 else 0


def feature_selection_info_gather(idx_o, clf, featureNames, penaltyStr, selected_features_list):
    if 'featureSel' in clf.named_steps.keys():
        featureNamesSub = featureNames[clf['featureSel'].get_support(indices=True)]
    else:
        featureNamesSub = featureNames
    if penaltyStr not in ('l2', None):
        bool_array = clf['classif'].coef_ != 0
        featureNamesSub = featureNamesSub[bool_array.flatten()]
    selected_features_list[idx_o] = featureNamesSub
    return selected_features_list


def collect_shap_values(idx_o, explainers, shap_values_list, baseline_val, n_classes, clf,
                        X_test_trans, feature_selected, test_index, featurePert,
                        X_train_trans=None):
    """
    Collect SHAP values for one CV split.

    Fixes vs. original
    ──────────────────
    1. Modern SHAP (≥0.41) returns a LIST of arrays from .shap_values() even for
       binary classifiers — one array per class.  The original code blindly wrapped
       the whole return value in a DataFrame, so for binary the DataFrame had 2 rows
       (one per class) instead of n_test rows.  Downstream code then tried
       shap_values_list[0][0].shape[0] on a list that was never populated past slot 0,
       raising "list index out of range".

       Fix: detect whether shap_values() returned a list and, for binary, keep only
       the first element (positive-class SHAP values).

    2. Whole function wrapped in try/except so a SHAP failure never aborts the
       classification loop — SHAP values are optional diagnostic outputs.

    3. Pass numpy arrays to SHAP to avoid pandas index confusion — SHAP internally
       treats pandas index values as positional row numbers, causing
       "index N is out of bounds for axis 0 with size N" when test_index contains
       original sample positions (e.g. [3, 7, 12, ...]).

    4. Use X_train_trans as the SHAP background when provided — training data is
       the statistically correct background for LinearExplainer; falling back to
       X_test_trans when not supplied.
    """
    try:
        import shap

        is_binary = len(clf._final_estimator.classes_) == 2

        # Use training data as background if available (correct choice for
        # LinearExplainer); fall back to test data for backwards compatibility.
        background = (X_train_trans if X_train_trans is not None
                      else X_test_trans)
        background_arr   = background.values
        X_test_trans_arr = X_test_trans.values

        if is_binary:
            explainer = shap.LinearExplainer(
                clf._final_estimator, background_arr,
                feature_perturbation=featurePert
            )
        else:
            # Multiclass must use interventional perturbation
            explainer = shap.LinearExplainer(
                clf._final_estimator, background_arr,
                feature_perturbation='interventional'
            )

        raw_shap = explainer.shap_values(X_test_trans_arr)

        # ── Normalise the return format ───────────────────────────────────────
        # Old SHAP (binary): raw_shap is a 2-D array  (n_test, n_features)
        # New SHAP (binary): raw_shap is a list of two 2-D arrays
        # Multiclass:        raw_shap is a list of n_class 2-D arrays
        if is_binary:
            if isinstance(raw_shap, list):
                # New API: take the positive-class array (index 1)
                shap_array = raw_shap[1]
            else:
                # Old API: single 2-D array
                shap_array = raw_shap
            shap_values_test = [
                pd.DataFrame(shap_array, columns=feature_selected, index=test_index)
            ]
            # expected_value may also be a list for new API — take the scalar for class 1
            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                base_val = base_val[1]
        else:
            shap_values_test = [
                pd.DataFrame(x, columns=feature_selected, index=test_index)
                for x in raw_shap
            ]
            base_val = explainer.expected_value

        # ── Store results ─────────────────────────────────────────────────────
        explainers[idx_o] = explainer
        for slot_idx, shap_val in enumerate(shap_values_test):
            # Guard: only write into slots that were pre-allocated
            if slot_idx < len(shap_values_list):
                shap_values_list[slot_idx].append(shap_val.reset_index())
                ev = base_val[slot_idx] if isinstance(base_val, (list, np.ndarray)) else base_val
                baseline_val[slot_idx].append(ev)

    except Exception as shap_err:
        print(f"\n  [SHAP] Skipped CV split {idx_o}: {shap_err}")

    return explainers, shap_values_list, baseline_val


def flatten(lst):
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(flatten(item))
        else:
            flattened.append(item)
    return flattened


def generate_region_csv(lightSheetData, dirDict):
    regionList = lightSheetData['Region_ID'].unique()
    regionList = pd.DataFrame(regionList, columns=['Region_ID'])
    regionList.to_csv(os.path.join(dirDict['atlasDir'], 'dataRegions.csv'), index=False)


def extract_stats_per_box(dataframe, group_col='stressor', value_col='density_norm'):
    def calculate_stats(group):
        return pd.Series({
            'Median': group.median(), 'Q1': group.quantile(0.25),
            'Q3': group.quantile(0.75), 'Min': group.min(), 'Max': group.max()
        })
    grouped_stats = dataframe.groupby(group_col)[value_col].apply(calculate_stats)
    for label in dataframe[group_col].unique():
        print(f"{group_col}: {label}")
        print(grouped_stats.loc[label])
        print()


def feature_barplot(selected_features_list, selected_features_params, YtickLabs):
    if len(YtickLabs) == 2:
        YtickLabs = [f"{YtickLabs[0]} vs {YtickLabs[1]}"]
    else:
        YtickLabs = [' vs '.join(YtickLabs)]
    import seaborn as sns
    regionList = np.concatenate(selected_features_list[0])
    regionDict = dict(Counter(regionList))
    labels, counts = list(regionDict.keys()), list(regionDict.values())
    counts = np.array(counts) / len(selected_features_list[0])
    pd_df = pd.DataFrame({'Region': labels, 'Count': counts}).sort_values(by='Count')
    pd_df.plot.barh(x='Region', y='Count', rot=0, figsize=(10, 20))
    plt.title(f'CV Split Feature Presence, Fraction: {YtickLabs}')
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# permutation_stats  –  statistical comparison of real vs shuffle per-split AUC
# ─────────────────────────────────────────────────────────────────────────────
import re

def _label_from_path(real_path, stored_label):
    """
    Enrich a stored compLabel with folder-derived context that isn't
    captured in the pkl itself (timepoint, stressor, pooled tag, etc.).
    """
    path_norm = real_path.replace('\\', '/')

    # Per-timepoint multiclass:  .../classif_Acute/...  → "Acute | Ctrl vs FS vs ..."
    for tp in ('Acute', '7D', '14D', '21D'):
        if f'/classif_{tp}/' in path_norm:
            return f'{tp} | {stored_label}'

    # Per-stressor timepoint:    .../classif_tp_RS/...  → "tp_RS | Ctrl vs RS"
    m = re.search(r'/classif_tp_([^/]+)/', path_norm)
    if m:
        return f'tp_{m.group(1)} | {stored_label}'

    # Pooled timepoint multiclass
    if '/classif_tp_pooled/' in path_norm:
        return f'tp_pooled | {stored_label}'

    return stored_label   # binary / main multiclass — label is already unique




def compute_permutation_stats(dirDict, tagList=None, alpha=0.05):
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    # ── Collect all unique scoreDict_Real.pkl paths across BOTH roots ────────
    search_roots = []
    for key in ('classifyDir', 'outDir'):
        p = dirDict.get(key)
        if p and os.path.isdir(p):
            search_roots.append(p)

    seen_paths = set()
    pkl_paths  = []
    for root_dir in search_roots:
        for root, dirs, files in os.walk(root_dir):
            if 'scoreDict_Real.pkl' not in files:
                continue
            if tagList and not all(tag in root for tag in tagList):
                continue
            real_path = os.path.join(root, 'scoreDict_Real.pkl')
            if real_path in seen_paths:
                continue
            seen_paths.add(real_path)
            pkl_paths.append(real_path)

    rows = []
    for real_path in pkl_paths:
        with open(real_path, 'rb') as f:
            real = pkl.load(f)

        auc_real = np.array(real.get('auc_per_split', []))
        acc_real = np.array(real.get('scores', []))
        label    = _label_from_path(real_path, real.get('compLabel', os.path.basename(os.path.dirname(real_path))))

        shuffle_path = real_path.replace('scoreDict_Real', 'scoreDict_Shuffle')
        if not os.path.exists(shuffle_path):
            print(f"  [permutation_stats] Missing shuffle pkl for: {label}")
            continue

        with open(shuffle_path, 'rb') as f:
            shuf = pkl.load(f)

        auc_shuf = np.array(shuf.get('auc_per_split', []))
        acc_shuf = np.array(shuf.get('scores', []))

        if len(auc_real) == 0 or len(auc_shuf) == 0:
            print(f"  [permutation_stats] ⚠ Empty auc_per_split for: {label} — rerun to regenerate pkl")
            continue

        per_class_real = real.get('auc_per_split_class', {})
        per_class_shuf = shuf.get('auc_per_split_class', {})

        if per_class_real and per_class_shuf:
            # Multiclass with per-class splits — one row per class
            for cls in per_class_real:
                if cls not in per_class_shuf:
                    continue
                r_arr = np.array(per_class_real[cls])
                s_arr = np.array(per_class_shuf[cls])
                stat, p = mannwhitneyu(r_arr, s_arr, alternative='greater')
                rows.append(dict(
                    classifier       = f'{label} [{cls}]',
                    n_splits_real    = len(r_arr),
                    n_splits_shuffle = len(s_arr),
                    mean_auc_real    = float(np.mean(r_arr)),
                    mean_auc_shuffle = float(np.mean(s_arr)),
                    auc_delta        = float(np.mean(r_arr) - np.mean(s_arr)),
                    U_stat           = float(stat),
                    p_value          = float(p),
                    mean_acc_real    = float(np.mean(acc_real)) if len(acc_real) else np.nan,
                    mean_acc_shuffle = float(np.mean(acc_shuf)) if len(acc_shuf) else np.nan,
                ))
        else:
            # Binary or old multiclass pkl without per-class splits — single mean row
            stat, p = mannwhitneyu(auc_real, auc_shuf, alternative='greater')
            rows.append(dict(
                classifier       = label,
                n_splits_real    = len(auc_real),
                n_splits_shuffle = len(auc_shuf),
                mean_auc_real    = float(np.mean(auc_real)),
                mean_auc_shuffle = float(np.mean(auc_shuf)),
                auc_delta        = float(np.mean(auc_real) - np.mean(auc_shuf)),
                U_stat           = float(stat),
                p_value          = float(p),
                mean_acc_real    = float(np.mean(acc_real)) if len(acc_real) else np.nan,
                mean_acc_shuffle = float(np.mean(acc_shuf)) if len(acc_shuf) else np.nan,
            ))

    if not rows:
        print("  [permutation_stats] No valid pkl pairs found in classifyDir or outDir.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values('classifier').reset_index(drop=True)
    reject, p_adj, _, _ = multipletests(df['p_value'], alpha=alpha, method='fdr_bh')
    df['p_adj']       = p_adj
    df['significant'] = reject

    return df