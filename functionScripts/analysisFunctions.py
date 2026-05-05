"""
analysisFunctions.py  –  adapted for stressor data
====================================================
Key changes vs. original drug version:
  • drug_stats_and_changes()   → stressor_stats_and_changes()
    - 'SAL' reference replaced by 'Ctrl'
    - 'drug' column queries use the data column that equals 'stressor'
    - 'drug_diff' / 'drugComp' labels → 'stressor_diff' / 'stressorComp'
  • compareAnimals()   – updated column references (drug→stressor)
  • brainRegionTrend() – unchanged structurally; drugA/B still work
  • collect_CI()       – 'SAL' → 'Ctrl' reference updated
  • gen_gene_corr()    – kept for completeness; not applicable without gene data
"""

import os
import pandas as pd
import numpy as np
from os.path import exists
from math import isnan
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
def stressor_stats_and_changes(databaseFrame, stressorList, ctrlSwitch=True):
    """
    Compute per-region statistics and pairwise percent-change for each stressor.

    Parameters
    ----------
    databaseFrame : pd.DataFrame
        Long-format dataframe with at least columns:
        ['stressor', 'dataset', 'Region_Name', 'density_norm']
    stressorList : list of str
        Ordered list of stressor names to compare (e.g. ['Ctrl','FS','RS','TS']).
    ctrlSwitch : bool
        If True, only return Ctrl-vs-X comparisons.
        If False, return all pairwise comparisons.

    Returns
    -------
    stressorPairDB    : pd.DataFrame   – per-region pairwise percent change (means)
    stressorStatsAll  : pd.DataFrame   – per-region descriptive statistics
    stressorPairDataList : list        – raw per-region pairwise data (for CIs)
    stressorCompNames : list of str    – comparison labels e.g. ['Ctrl-FS', ...]
    """
    dataCol = 'density_norm'

    stressorCount = len(stressorList)
    stressorDataFrames = np.empty(stressorCount, dtype=object)
    stressorStatsAll   = None

    # ── Per-stressor descriptive statistics ───────────────────────────────────
    # dropna(subset=[dataCol]): exclude this animal-region pair only.
    # The count column reflects actual non-NA observations per region.
    for s_i, stressor in enumerate(stressorList):
        stressorTemp = databaseFrame[databaseFrame['stressor'] == stressor].dropna(subset=[dataCol])
        stressorStats = stressorTemp.groupby('Region_Name').agg(
            {dataCol: ['mean', 'std', 'count']}
        )
        stressorStats.columns = stressorStats.columns.droplevel(0)
        stressorStats.columns = [
            f'{stressor}_average',
            f'{stressor}_standard_deviation',
            f'{stressor}_observations'
        ]
        if stressorStatsAll is None:
            stressorStatsAll = stressorStats
        else:
            stressorStatsAll = pd.merge(stressorStatsAll, stressorStats,
                                        on='Region_Name', how='inner')
        stressorDataFrames[s_i] = stressorTemp

    stressorStatsAll = stressorStatsAll.reset_index()

    # ── Build pairwise comparison list ────────────────────────────────────────
    stressorPairList = [(a, b) for idx, a in enumerate(stressorList)
                        for b in stressorList[idx + 1:]]
    stressorPairIndList = [(a, b) for idx, a in enumerate(range(stressorCount))
                           for b in range(stressorCount)[idx + 1:]]

    if ctrlSwitch:
        ctrl_idx = stressorList.index('Ctrl')
        stressorPairList    = [p for p in stressorPairList    if 'Ctrl' in p]
        stressorPairIndList = [p for p in stressorPairIndList if ctrl_idx in p]

    stressorCompNames  = [f'{A}-{B}' for A, B in stressorPairList]
    stressorPairDataList = []

    # initialise result table
    id_cols = ['Region_ID', 'abbreviation', 'Region_Name']
    # only keep id columns that exist in the dataframe
    id_cols_avail = [c for c in id_cols if c in databaseFrame.columns]
    stressorPairDB = databaseFrame.drop_duplicates(subset='Region_Name')[id_cols_avail].copy()

    for compName, (sA_idx, sB_idx) in zip(stressorCompNames, stressorPairIndList):
        sA_data = stressorDataFrames[sA_idx]
        sB_data = stressorDataFrames[sB_idx]

        # inner merge: only include regions present with valid data in BOTH groups
        sA_clean = sA_data.dropna(subset=[dataCol])
        sB_clean = sB_data.dropna(subset=[dataCol])
        merged = pd.merge(sA_clean, sB_clean, on='Region_Name', how='inner',
                          suffixes=('_A', '_B'))

        # Percent change: (A - B) / B * 100
        merged[compName] = (
            (merged[f'{dataCol}_A'] - merged[f'{dataCol}_B'])
            / merged[f'{dataCol}_B']
        ) * 100

        # Keep only needed columns
        keep_cols = ['Region_Name', compName]
        if 'abbreviation_A' in merged.columns:
            merged = merged.rename(columns={'abbreviation_A': 'abbreviation'})
            keep_cols = ['Region_Name', 'abbreviation', compName]
        if 'Region_ID_A' in merged.columns:
            merged = merged.rename(columns={'Region_ID_A': 'Region_ID'})
            keep_cols = ['Region_Name', 'Region_ID', 'abbreviation', compName]

        merged = merged[[c for c in keep_cols if c in merged.columns]]
        stressorPairDataList.append(merged)

        # Mean per region (numeric columns only to avoid dtype errors)
        groupCols = [c for c in ['Region_Name', 'Region_ID', 'abbreviation'] if c in merged.columns]
        merged_mean = merged.groupby(groupCols)[compName].mean().reset_index()

        merge_on = [c for c in ['Region_Name', 'Region_ID'] if c in stressorPairDB.columns and c in merged_mean.columns]
        stressorPairDB = pd.merge(stressorPairDB, merged_mean[[*merge_on, compName]],
                                  on=merge_on, how='inner')

    # ── Clean up ──────────────────────────────────────────────────────────────
    stressorPairDB = stressorPairDB.loc[:, ~stressorPairDB.columns.duplicated()]
    bad_regions = stressorPairDB.Region_Name[
        stressorPairDB.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    ]
    stressorPairDB   = stressorPairDB[~stressorPairDB['Region_Name'].isin(bad_regions)]
    stressorStatsAll = stressorStatsAll[~stressorStatsAll['Region_Name'].isin(bad_regions)]
    for i, comp in enumerate(stressorPairDataList):
        stressorPairDataList[i] = comp[~comp['Region_Name'].isin(bad_regions)]

    return stressorPairDB, stressorStatsAll, stressorPairDataList, stressorCompNames


# ─────────────────────────────────────────────────────────────────────────────
# Alias so existing notebook cells that call drug_stats_and_changes still work.
# ─────────────────────────────────────────────────────────────────────────────
def drug_stats_and_changes(databaseFrame, stressorList, ctrlSwitch=True):
    """
    Alias for stressor_stats_and_changes().
    Accepts the same arguments – replaces the drug-based original.
    """
    return stressor_stats_and_changes(databaseFrame, stressorList, ctrlSwitch)


# ─────────────────────────────────────────────────────────────────────────────
def compareAnimals(lightsheetDB, stressorA, stressorB, compColumn, dirDict):
    """
    Scatter-plot grid comparing individual animals across two stressor groups.
    Adapted from original compareAnimals() – 'drug' → 'stressor'.
    """
    plotTitle = f'{stressorA} vs {stressorB} – {compColumn}'

    stressorAdata = lightsheetDB[lightsheetDB['stressor'].str.match(stressorA)]
    stressorBdata = lightsheetDB[lightsheetDB['stressor'].str.match(stressorB)]

    setsA = stressorAdata['dataset'].unique()
    setsB = stressorBdata['dataset'].unique()

    fig, axs = plt.subplots(len(setsA), len(setsB), figsize=(8, 8), dpi=200)
    fig.suptitle(plotTitle, y=0.92)
    fontdict2 = {'fontsize': 5}

    for i, setA in enumerate(setsA):
        for j, setB in enumerate(setsB):
            dataA = lightsheetDB[lightsheetDB['dataset'] == setA]
            dataB = lightsheetDB[lightsheetDB['dataset'] == setB]

            colA = setA + '_'
            colB = setB

            dataA = dataA.rename(columns={compColumn: colA})
            dataA[colB] = dataB[compColumn].to_numpy()

            bad = dataA['Region_Name'][dataA.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
            dataA = dataA[~dataA['Region_Name'].isin(bad)]

            pA = np.percentile(dataA[colA].to_numpy(), 95)
            pB = np.percentile(dataA[colB].to_numpy(), 95)

            ax = axs[i][j] if len(setsA) > 1 and len(setsB) > 1 else axs
            sns.scatterplot(x=colA, y=colB, data=dataA, s=5, ax=ax)
            tmpSet = np.vstack((np.asarray(dataA[colA]), np.asarray(dataA[colB])))
            r = np.corrcoef(tmpSet)[0][1]
            ax.set_title(f'r = {round(r, 2)}', fontdict=fontdict2, y=0.98)
            ax.tick_params(axis='x', labelsize=4)
            ax.tick_params(axis='y', labelsize=4)
            ax.set_xlim([0, pA])
            ax.set_ylim([0, pB])
            maxNum = np.amax([pA, pB])
            ax.plot([0, maxNum], [0, maxNum], linewidth=1, alpha=0.5)

    saveTitle = plotTitle.replace('/', ' per ')
    plt.savefig(os.path.join(dirDict['debugDir'], saveTitle + '.png'),
                format='png', bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
def brainRegionTrend(lightsheetDBAll, dataColumn, stressorA, stressorB, ylimMax=0):
    """
    Plot percent-difference between two stressors across brain regions.
    Arguments identical to original – drugA/drugB renamed stressorA/stressorB.
    """
    lightsheetDB = lightsheetDBAll.groupby(
        ['abbreviation', 'Region_Name', 'Brain_Area', 'stressor']
    )[dataColumn].mean().reset_index()

    lsA = lightsheetDB.query("stressor == @stressorA").reset_index()
    lsB = lightsheetDB.query("stressor == @stressorB").reset_index()

    diff = lsA.copy().rename(columns={dataColumn: stressorA})
    diff[stressorB] = lsB[dataColumn]
    diff['stressor_diff'] = (diff[stressorA] - diff[stressorB]) / diff[stressorB] * 100
    diff = diff.sort_values('stressor_diff', ascending=False)

    brainAreaList = list(diff.Brain_Area.unique())
    for area in brainAreaList:
        dataSet = np.asarray(diff.query("Brain_Area == @area")['stressor_diff'])
        plt.plot(dataSet)
    plt.legend(brainAreaList)
    plt.xlabel('Region Index')
    plt.ylabel('Percent Difference (A−B)/B × 100')
    if ylimMax != 0:
        plt.ylim([-100, ylimMax])
    plt.title(f'{stressorA} vs {stressorB} {dataColumn}')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
def collect_CI(stressor_diff, stressor_diff_labels, stressorList, dirDict):
    """
    Bootstrap 95% CIs for each Ctrl-vs-X comparison.
    Adapted from original collect_CI – SAL → Ctrl.
    """
    tmpFileName = 'stressor_ci_range_db.pkl'
    tempDataFilename = os.path.join(dirDict['tempDir'], tmpFileName)

    if os.path.exists(tempDataFilename):
        print(f'Loading {tmpFileName}...')
        return pd.read_pickle(tempDataFilename)

    print(f'Generating {tmpFileName}...')

    stressorPairList = [f'{a}-{b}' for idx, a in enumerate(stressorList)
                        for b in stressorList[idx + 1:]]
    ctrlList = [p for p in stressorPairList if 'Ctrl' in p]

    ciRangeDB = pd.DataFrame()

    for idx, stressorPair in enumerate(tqdm(ctrlList)):
        dataInd  = list(stressor_diff_labels).index(stressorPair)
        dataCol  = stressor_diff_labels[dataInd]
        dataTable = stressor_diff[dataInd]
        regionList = dataTable.Region_Name.unique()

        plt.figure(figsize=(20, 160))
        ax = sns.pointplot(y='Region_Name', x=dataCol, data=dataTable,
                           errorbar=('ci', 95), join=False, errwidth=0.5,
                           color='red', dodge=True)
        if idx == 0:
            ciRangeDB['Region_Name'] = [t.get_text() for t in ax.axes.get_yticklabels()]
        ciRangeDB[stressorPair + '_upper'] = [line.get_xdata().max() for line in ax.lines]
        ciRangeDB[stressorPair + '_lower'] = [line.get_xdata().min() for line in ax.lines]
        plt.clf()

    ciRangeDB.to_pickle(tempDataFilename)
    return ciRangeDB
