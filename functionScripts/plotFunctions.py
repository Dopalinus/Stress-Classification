import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import exists, join
from math import isnan
from tqdm import tqdm

import matplotlib.patches as patches
import matplotlib.ticker as tkr
import helperFunctions as hf
import scipy.stats as stats
from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import textwrap
import matplotlib.ticker as tkr
from statannotations.Annotator import Annotator
import configFunctions as config 
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['svg.fonttype'] = 'none'   

sys.path.append('dependencies')

def plot_headTwitchTotal(dirDict):

    htrDataPath = os.sep.join([dirDict['dataDir'],'behavioral','HTR_summary_data.csv'])
    df = pd.read_csv(htrDataPath)

    # Some parameters
    figHeight = 1.8
    figWidthHT = 4.15

    colorDict = hf.create_color_dict()
    linWidth = plt.rcParams['axes.linewidth']

    # Plot the box and whisker plot
    plt.figure(figsize=(figWidthHT, figHeight))

    plotOrder = ['Saline', '6-Fluoro-DET', 'Psilocybin', '5-MeO-DMT']
    tickLabels = ['SAL', '6-F-DET', 'PSI', '5MEO']

    colorPal = [colorDict[tickLabel] for tickLabel in tickLabels]

    boxprops = dict() # alpha=0.7
    ax = sns.boxplot(data=df, x= 'drug', y= 'total_HTR', order=plotOrder, palette=colorPal, boxprops=boxprops, linewidth=linWidth) 
    ax.legend().remove()
    sns.swarmplot(data=df, x= 'drug', y= 'total_HTR', color='black', size = 3, order=plotOrder, palette=colorPal)

    hf.extract_stats_per_box(df)

    # Stats - Place here to keep lines within the figure.
    pairs = [('Saline', 'Psilocybin'), ('6-Fluoro-DET', 'Psilocybin'), ('Saline', '5-MeO-DMT'), ('6-Fluoro-DET', '5-MeO-DMT')]
    annotator = Annotator(ax, pairs, data=df, x='drug', y='total_HTR', order=plotOrder)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()

    # Label the axes
    plt.ylim(0, 150)
    plt.xticks(ticks=[0, 1, 2, 3], labels=tickLabels)
    plt.yticks(np.arange(0, 126, 25))
    ax.yaxis.set_tick_params(which = 'both', length=1, width=linWidth)
    ax.xaxis.set_tick_params(which = 'both', length=1, width=linWidth)

    plt.ylabel('Head-twitch count', fontsize=7)
    plt.xlabel('') 

    savePath = os.path.join(dirDict['outDir'], 'HTR_total.svg')
    plt.savefig(savePath, format='svg', bbox_inches='tight')
    plt.show()
    plt.close()

def plotTotalPerDrug(pandasdf, column2Plot, dirDict):
    # Select a random region to collect 'total_cells' from
    totalCellCountData = pandasdf[pandasdf.Region_ID == 88]

    totalCellCountData['sex'] = totalCellCountData['sex'].replace({'M': 'Male', 'F': 'Female'})

    # Shift the color codes to RGB and add alpha
    colorDict = hf.create_color_dict('drug', 0)

    scaleFactor = 1
    figSize = (3.2*scaleFactor, 1.278*scaleFactor)

    plt.figure(figsize=figSize)
    ax = sns.boxplot(x="drug", y=column2Plot, data=totalCellCountData, whis=0, dodge=False, showfliers=False, linewidth=.5, hue='drug', palette=colorDict)

    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .7))

    # Cylce through x-axis labels and change their color to match the boxplot
    for idx, label in enumerate(ax.get_xticklabels()):
        label.set_color(colorDict[label.get_text()])
    
    ax2 = sns.scatterplot(x="drug", y=column2Plot, data=totalCellCountData, hue='drug', linewidth=0, style='sex', markers=True, s=5, palette=colorDict, ax=ax, edgecolor='black')

    # remove legend
    # plt.legend([], [], frameon=False)
    # Delete all but the final 2 legend entries
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[-2:], labels=labels[-2:], loc='upper right')

    # cleanup
    ax2.spines['left'].set_linewidth(0.5)
    ax2.spines['bottom'].set_linewidth(0.5)
    ax2.yaxis.set_tick_params(which = 'both', length=1.5, width=1)
    ax2.xaxis.set_tick_params(which = 'both', length=1, width=1)
    # ax2.yaxis.set_tick_params(which = 'both', length=1, width=0.5)
    # ax2.yaxis.set_tick_params(length=1, width=0.2)

    ax.set_yscale('log')
    ax.set(ylim=(7.9e5, 1e7))
    ax.set(xlabel='')
    ax.set_ylabel('c-Fos+ cell count', fontsize=7)
    # ax.set_ylabel()
    sns.despine()

    plt.savefig(dirDict['outDir'] + os.sep + 'totalCells', bbox_inches='tight')

    config.setup_figure_settings()

def plotLowDimEmbed(pandasdf, column2Plot, dirDict, dimRedMeth, classifyDict, ldaDict):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler, PowerTransformer
    from itertools import combinations
    from sklearn.pipeline import Pipeline

    # Build colour dict that covers both stressors and legacy drug names
    colorHex = hf.create_color_dict(dictType='stressor')
    # Plot
    config.setup_figure_settings()
    
    # If some filtering is desired, do so here
    if classifyDict['featurefilt']:
        pandasdf = hf.filter_features(pandasdf, classifyDict)
        filtTag = 'filt'
    else:
        filtTag = ''

    # Pivot the lightsheet data table — use pivot_table to handle NaN gracefully
    df_Tilted = pandasdf.pivot_table(index='dataset', columns='abbreviation',
                                     values=column2Plot, aggfunc='mean')

    # ── NA handling (mirrors _pivot_with_na_handling) ─────────────────────────
    na_thresh = classifyDict.get('na_region_thresh', 0.20)
    nan_frac  = df_Tilted.isna().mean(axis=0)
    df_Tilted = df_Tilted.loc[:, nan_frac <= na_thresh]
    df_Tilted = df_Tilted.fillna(df_Tilted.mean(axis=0))

    # Extract stressor label from dataset index: "Ctrl_7D_F1" → "Ctrl"
    y = np.array([str(x).split('_')[0] for x in df_Tilted.index])
    df_Tilted['y_vec'] = y

    n_comp = 2

    # Apply some preprocessing to mimic pipeline
    transMod = PowerTransformer(method='yeo-johnson', standardize=False)
    scaleMod = RobustScaler()

    if dimRedMeth == 'PCA':
        dimRedMod = PCA(n_components=n_comp)
        compName = 'Principal Component '
    elif dimRedMeth == 'LDA':
        dimRedMod = LDA(n_components=n_comp)
        compName = 'Linear discriminant '
    else:
        KeyError('dimRedMethod not recognized, pick LDA or PCA')

    pipelineList = [('transMod', transMod), ('scaleMod', scaleMod), ('dimRedMod', dimRedMod)]
    pipelineObj = Pipeline(pipelineList)

    ### Identify sets of training/testing data to loop through

    if not ldaDict:
        # Full set is used for training and testing
        analysisNames = 'All'
        trainSets = [set(df_Tilted.y_vec.unique())]
        testSets = [set(df_Tilted.y_vec.unique())]
    else:
        # Create a paired list of training sets and testing sets. 
        analysisNames = list(ldaDict.keys())
        trainSets = [ldaDict[aName][0] for aName in analysisNames]
        testSets = [ldaDict[aName][1] for aName in analysisNames]

    # ── Dynamic group order: stressors first, then any legacy drug names ──────
    stressor_order = ['Ctrl', 'FS', 'FSW', 'RS', 'TS']
    drug_order     = ['PSI', 'KET', '5MEO', '6-F-DET', 'MDMA', 'A-SSRI', 'C-SSRI', 'SAL']
    present_groups = list(df_Tilted.y_vec.unique())
    customOrder = ([g for g in stressor_order if g in present_groups] +
                   [g for g in drug_order     if g in present_groups] +
                   [g for g in present_groups if g not in stressor_order + drug_order])

    config.setup_LDA_settings()

    for aName, trainSet, testSet in zip(analysisNames, trainSets, testSets):

        df_train = df_Tilted[df_Tilted.y_vec.isin(trainSet)]
        df_plot = df_Tilted[df_Tilted.y_vec.isin(testSet)]
        testOnly = [item for item in testSet if item not in trainSet]

        if len(trainSet) == 2 and dimRedMeth == 'LDA':
            pipelineList[2][1].n_components = 1
            colNames = [compName]
        else:
            pipelineList[2][1].n_components = n_comp
            colNames = [f"{compName}{x}" for x in range(1, n_comp+1)]

        # Fit on train data, then transform plot data
        pipelineObj.fit(df_train.iloc[:, :-1], df_train.iloc[:, -1])
        df_plot_data_transformed = pipelineObj.transform(df_plot.iloc[:, :-1])

        # Extract explained variance ratio for PCA (not available for LDA)
        if dimRedMeth == 'PCA':
            evr = pipelineObj.named_steps['dimRedMod'].explained_variance_ratio_
        else:
            evr = None

        # Create average cases for each drug.
        dimRedData = pd.DataFrame(data=df_plot_data_transformed, index=df_plot.index, columns=colNames)
        if df_plot_data_transformed.shape[1] == 1:
            # dimRedData['null'] = np.zeros(dimRedData.shape[0])
            dimRedData['null'] = np.random.rand(dimRedData.shape[0])

        dimRedData.loc[:, 'drug'] = pd.Categorical(y, categories=customOrder, ordered=True)
        dimRedDrugMean = dimRedData.groupby(by='drug').mean()

        # Sort group means to match customOrder (works for any number of groups).
        # Original code used a hardcoded resortIdx=[1,2,3,0,4,5,6,7] which assumed
        # exactly 8 drug groups and crashed with fewer groups (IndexError).
        present_in_mean = [g for g in customOrder if g in dimRedDrugMean.index]
        dimRedDrugMean = dimRedDrugMean.reindex(present_in_mean)

        if trainSet == testSet:
            pairs = list(combinations(range(n_comp), 2))
            for comp_pair in pairs:
                col1 = colNames[comp_pair[0]]
                col2 = colNames[comp_pair[1]]

                plt.figure(figsize=(2.25, 2.25))  # Adjust the figure size as needed
                sns.scatterplot(x=col1, y=col2, hue='drug', data=dimRedData, s=10, alpha=0.75, palette=colorHex)
                sns.scatterplot(x=col1, y=col2, hue='drug', data=dimRedDrugMean, s=20, legend=False, edgecolor='black', palette=colorHex)
                plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6)

                # Customize the plot
                # plt.title(f"{dimRedMeth} of {column2Plot}", fontsize=20)
                # plt.title(f"Linear Discrimants of {aName}", fontsize=20)
                # Axis labels with explained variance (PCA only)
                if evr is not None:
                    i1, i2 = comp_pair
                    plt.xlabel(f"{col1} ({evr[i1]*100:.1f}%)")
                    plt.ylabel(f"{col2} ({evr[i2]*100:.1f}%)")

                # Save
                plt.savefig(dirDict['outDir'] + os.sep + f"dimRed_{aName}_{col1} x {col2}", bbox_inches='tight')

                plt.show()
                plt.close()
        else:
            pairs = list(combinations(range(n_comp), 2))
            for comp_pair in pairs:
                col1 = colNames[comp_pair[0]]
                col2 = colNames[comp_pair[1]]

                # Plot the training set, slightly lighter
                plt.figure(figsize=(2.25, 2.25))  # Adjust the figure size as needed
                sns.scatterplot(x=col1, y=col2, hue='drug', data=dimRedData[dimRedData.drug.isin(trainSet)], s=10, alpha=0.5, legend=False, palette=colorHex)
                sns.scatterplot(x=col1, y=col2, hue='drug', data=dimRedDrugMean[dimRedDrugMean.index.isin(trainSet)], s=20, alpha=0.5, legend=False, edgecolor='black', palette=colorHex)

                sns.scatterplot(x=col1, y=col2, hue='drug', data=dimRedData[dimRedData.drug.isin(testOnly)], s=12, edgecolor='black', palette=colorHex, marker="D")
                sns.scatterplot(x=col1, y=col2, hue='drug', data=dimRedDrugMean[dimRedDrugMean.index.isin(testOnly)], s=25, legend=False, edgecolor='black', palette=colorHex, marker="D")
                plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6)

                # Customize the plot
                # plt.title(f"{dimRedMeth} of {column2Plot}", fontsize=20)
                # plt.title(f"Linear Discrimants of {aName}", fontsize=20)
                #Axis labels with explained variance (PCA only)
                if evr is not None:
                    i1, i2 = comp_pair
                    plt.xlabel(f"{col1} ({evr[i1]*100:.1f}%)")
                    plt.ylabel(f"{col2} ({evr[i2]*100:.1f}%)")
                # Save
                plt.savefig(dirDict['outDir'] + os.sep + f"dimRed_{aName}_{col1} x {col2}", bbox_inches='tight')

                plt.show()
                plt.close()            
        # else:
        #     plt.figure(figsize=(2.25, 2.25))  # Adjust the figure size as needed
        #     sns.scatterplot(x=compName, y='null', hue='drug', data=dimRedData, s=10, alpha=0.75, palette=colorHex)
        #     sns.scatterplot(x=compName, y='null', hue='drug', data=dimRedDrugMean, s=20, legend=False, edgecolor='black', palette=colorHex)
        #     plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6)

        #     # # Customize the plot
        #     # # plt.title(f"{dimRedMeth} of {column2Plot}", fontsize=20)
        #     # # plt.title(f"Linear Discrimants of {aName}", fontsize=7)

        #     # Save
        #     plt.savefig(dirDict['outDir'] + os.sep + f"dimRed_{aName}_{filtTag}_{compName} x null", bbox_inches='tight')

        #     plt.show()
    # Reset changes made
    config.setup_figure_settings()

def compute_pca_centroid_distances(pandasdf, column2Plot, classifyDict):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler, PowerTransformer
    from sklearn.pipeline import Pipeline

    na_thresh = classifyDict.get('na_region_thresh', 0.20)
    df = pandasdf.pivot_table(index='dataset', columns='abbreviation',
                              values=column2Plot, aggfunc='mean')
    nan_frac = df.isna().mean(axis=0)
    df = df.loc[:, nan_frac <= na_thresh]
    df = df.fillna(df.mean(axis=0))

    y = np.array([str(x).split('_')[0] for x in df.index])

    pipeline = Pipeline([
        ('transform', PowerTransformer(method='yeo-johnson', standardize=False)),
        ('scale',     RobustScaler()),
        ('pca',       PCA(n_components=2))
    ])
    X_pca = pipeline.fit_transform(df.values)
    evr   = pipeline.named_steps['pca'].explained_variance_ratio_

    stressor_order = ['Ctrl', 'FS', 'FSW', 'RS', 'TS']
    present    = [s for s in stressor_order if s in y]
    centroids  = {s: X_pca[y == s].mean(axis=0) for s in present}

    dist_df = pd.DataFrame(
        [[np.linalg.norm(centroids[s1] - centroids[s2]) for s2 in present]
         for s1 in present],
        index=present, columns=present
    )

    print(f"Explained variance: PC1={evr[0]*100:.1f}%  PC2={evr[1]*100:.1f}%")
    print("\nCentroid coordinates (PC1, PC2):")
    for s, c in centroids.items():
        print(f"  {s:5s}  n={(y==s).sum()}  PC1={c[0]:+.4f}  PC2={c[1]:+.4f}")
    print("\nPairwise Euclidean distances between centroids:")
    print(dist_df.round(4).to_string())

    return dist_df, evr

def meanCountPerRegion(pandasdf):
    # Pull the mean count per region to help highlight.
    meanCountPerRegion = pandasdf.groupby(['Region_Name', 'Brain_Area'])['count', 'volume_(mm^3)'].mean().reset_index()
    plt.figure(figsize=(20, 20))
    sns.scatterplot(x='count', y='volume_(mm^3)', hue='Brain_Area', data=meanCountPerRegion)

    # Change the plot to have x and y axes limits at 80th percentile
    plt.xlim(0, np.percentile(meanCountPerRegion['count'], 95))
    plt.ylim(0, np.percentile(meanCountPerRegion['volume_(mm^3)'], 95))

    plt.show()
    plt.close()


    print('d')

def histPrePostScale(pandasdf, dataPerPlot, dirDict):
    # Create a grid of histograms of columns in a pandas dataframe.
    from sklearn.preprocessing import RobustScaler, PowerTransformer

    outDirPath = os.path.join(dirDict['outDir'], 'featureScale')
    if not os.path.exists(outDirPath):
        os.mkdir(outDirPath)

    pivotTables = []
    for dataV in dataPerPlot:
        pivotTables.append(pandasdf.pivot(index='dataset', columns='abbreviation', values=dataV))

    scaledData = PowerTransformer(method='yeo-johnson', standardize=False).fit_transform(pivotTables[1])
    scaledData_df = pd.DataFrame(scaledData, index=pivotTables[1].index, columns=pivotTables[1].columns)
    pivotTables.append(scaledData_df)
    
    scaledData = RobustScaler().fit_transform(pivotTables[2])
    scaledData_df = pd.DataFrame(scaledData, index=pivotTables[2].index, columns=pivotTables[2].columns)
    pivotTables.append(scaledData_df)

    featList = list(pivotTables[0].columns)
    featList = featList[0:1]

    for feat in featList:
        imgPath = os.path.join(outDirPath, f"scaleChain_{feat}.png")

        if os.path.exists(imgPath):
            continue

        fig, axes = plt.subplots(1, 4, figsize=(4*4, 2))

        axes[0].hist(pivotTables[0].loc[:, feat], bins=20, alpha=0.5)  # Adjust the number of bins as needed
        axes[0].title.set_text(f"{dataPerPlot[0]}: {feat}")
        axes[0].grid(False)

        axes[1].hist(pivotTables[1].loc[:, feat], bins=20, alpha=0.5)  # Adjust the number of bins as needed
        axes[1].title.set_text(f"{dataPerPlot[1]}: {feat}")
        axes[1].grid(False)

        axes[2].hist(pivotTables[2].loc[:, feat], bins=20, alpha=0.5)  # Adjust the number of bins as needed
        axes[2].title.set_text(f"yj norm: {feat}")
        axes[2].grid(False)

        axes[3].hist(pivotTables[3].loc[:, feat], bins=20, alpha=0.5)  # Adjust the number of bins as needed
        axes[3].title.set_text(f"Robust Scaled yj norm: {feat}")
        axes[3].grid(False)

        plt.savefig(os.sep.join([outDirPath, f"scaleChain_{feat}"]), bbox_inches='tight')

def distance_matrix(lightsheet_data, classifyDict, dirDict):
    import matplotlib.patheffects as PathEffects
    import numpy as np

    from scipy.spatial.distance import pdist, squareform
    from sklearn.metrics import pairwise_distances
    
    colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628','#984ea3','#999999', '#e41a1c'] #, '#dede00'

    dist_met_list = ['euclidean', 'minkowski', 'cityblock', 'seuclidean', 'mahalanobis', 'cosine']
    df_Tilted = lightsheet_data.pivot(index=classifyDict['feature'], columns='dataset', values=classifyDict['data'])

    for dist_met in dist_met_list:

        pairwise = pd.DataFrame(squareform(pdist(df_Tilted, metric=dist_met)),columns = df_Tilted.index,index = df_Tilted.index)

        plt.figure(figsize=(10,10))
        sns.heatmap(
            pairwise,
            # cmap='OrRd',
            # linewidth=1
        )
        plt.title(dist_met, fontsize=45)

        plt.savefig(dirDict['outDir'] + os.sep + f'Dist_{dist_met}_raw.png', format='png', bbox_inches='tight')

        sns.clustermap(pairwise, 
                       cmap='rocket', 
                       fmt='.2f', 
                        dendrogram_ratio = 0.1)

        plt.title(dist_met, fontsize=45)
        plt.savefig(dirDict['outDir'] + os.sep + f'Dist_{dist_met}_clustered.png', format='png', bbox_inches='tight')

        plt.show()
        plt.close()

def correlation_plot(lightsheet_data, classifyDict, dirDict):
    # Create an index for sorting the brain Areas. List below is for custom ordering.
    # brainAreaList = lightsheet_data['Brain_Area'].unique().tolist()
    brainAreaColorDict = hf.create_color_dict(dictType='brainArea', rgbSwitch=0)
    brainAreaPlotDict = hf.create_brainArea_dict('short')
    regionArea = hf.create_region_to_area_dict(lightsheet_data, classifyDict['feature'])
    regionArea['Region_Color'] = regionArea['Brain_Area'].map(brainAreaColorDict)

    df_Tilted = lightsheet_data.pivot(index='dataset', columns=classifyDict['feature'], values=classifyDict['data'])
    df_Tilted = df_Tilted.reindex(regionArea[classifyDict['feature']].tolist(), axis=1)

    # Calculate spearman and pearson correlation.
    corr_matrix = np.corrcoef(df_Tilted.values.T)
    s_corr_matrix = stats.spearmanr(df_Tilted.values)

    corr_mats = [corr_matrix, s_corr_matrix[0]]
    corr_names = ['Pearson', 'Spearman']

    # Plotting variables
    scalefactor = 12
    cmap = 'rocket'
    yticklabels = df_Tilted.columns.tolist()
    plotDim = len(yticklabels) * scalefactor * 0.015

    for corr_data, corr_name in zip(corr_mats, corr_names):
        # Plotting
        plt.figure(figsize=(plotDim*1.1, plotDim))
        # cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
        ax = sns.heatmap(corr_data, cmap=cmap, fmt='.2f', yticklabels=yticklabels, xticklabels=yticklabels, square=True)

        # Color code the labs
        for xtick, ytick in zip(ax.get_xticklabels(), ax.get_yticklabels()):
            colorCode = regionArea[regionArea[classifyDict['feature']] == ytick._text]['Region_Color'].values[0]
            xtick.set_color(colorCode)
            ytick.set_color(colorCode)

        # Adjust the color bar
        cbar = ax.collections[0].colorbar  # Get the colorbar object
        cbar.ax.tick_params(labelsize=45, width=1)  # Increase font size and adjust width

        # Add in vertical lines breaking up sample types
        # _, line_break_ind = np.unique(regionArea.Brain_Area, return_index=True)
        # for idx in line_break_ind:
        #     plt.axvline(x=idx, color='black', linewidth=6)
        #     plt.axhline(y=idx, color='black', linewidth=6)

        # Add additional text to denote which area each region belongs to
        positionDict = hf.find_middle_occurrences(regionArea.Brain_Area)
        mid_idx = [x[1] for x in positionDict.values()]
        items = positionDict.keys()
        for mid_sample_ind, label in zip(mid_idx, items):
            ax.text(mid_sample_ind, corr_matrix.shape[0] + 5, f"{brainAreaPlotDict[label]}", size = 30, ha='center', va='top', color=brainAreaColorDict[label]) #, transform=ax.transAxes
        
        # Add colored boxes
        for area in items:
            squarePos = positionDict[area][0]
            squareLen = positionDict[area][-1] - positionDict[area][0] + 1
            tmpVar = patches.Rectangle((squarePos, squarePos), squareLen, squareLen, linewidth=7, edgecolor='black', fill=False)
            # tmpVar = patches.Rectangle((squarePos, squarePos), squareLen, squareLen, linewidth=7, edgecolor=brainAreaColorDict[area], fill=False)
            ax.add_patch(tmpVar)

        # Labels    
        titleStr = f"{corr_name}, {classifyDict['data']} Correlations Across Regions"

        ax.set_ylabel("Region", fontsize=40)
        ax.set_xlabel('Region', fontsize=45, labelpad=40)
        plt.tick_params(axis='x', which='both', length=0)
        plt.title(titleStr, fontsize=65)
        
        plt.savefig(dirDict['classifyDir'] + os.sep + titleStr + '.png', format='png', bbox_inches='tight')
        plt.show()
        plt.close()

def correlation_plot_hier(lightsheet_data, classifyDict, dirDict):
    # Plot Hierarchical Correlation Clustering Heatmap

    brainAreaColorDict = hf.create_color_dict('brainArea', 0)
    brainAreas = list(brainAreaColorDict.keys())
    AreaIdx = dict(zip(brainAreas, np.arange(len(brainAreas))))

    colList = [classifyDict['feature'], 'Brain_Area']
    regionArea = lightsheet_data[colList]
    regionArea.drop_duplicates(inplace=True)
    regionArea['Brain_Area_Idx'] = [AreaIdx[x] for x in regionArea['Brain_Area']]
    regionArea['Region_Color'] = [brainAreaColorDict[x] for x in regionArea['Brain_Area']]
    regionArea.sort_values(by='Brain_Area_Idx', inplace=True)

    df_Tilted = lightsheet_data.pivot(index='dataset', columns=classifyDict['feature'], values=classifyDict['data'])
    df_Tilted = df_Tilted.reindex(regionArea[classifyDict['feature']].tolist(), axis=1)

    corrHier = True
    if not corrHier:
        ylabelStr = "Dataset"

        # Cluster the data directly
        ax = sns.clustermap(df_Tilted.values, cmap='rocket', fmt='.2f', yticklabels=df_Tilted.index, xticklabels=df_Tilted.columns, dendrogram_ratio = 0.1, figsize=(40, 10))
        ax.ax_heatmap.set_xticklabels(ax.ax_heatmap.get_xmajorticklabels(), fontsize = 3)
        ax.ax_heatmap.set_yticklabels(ax.ax_heatmap.get_ymajorticklabels(), fontsize = 8)

        for xtick, ytick in zip(ax.get_xticklabels(), ax.get_yticklabels()):
            colorCode = regionArea[regionArea[classifyDict['feature']] == ytick._text]['Region_Color'].values[0]
            xtick.set_color(colorCode)
            ytick.set_color(colorCode)

        # Adjust the color bar
        
        ax.ax_cbar.tick_params(width=1)  # Increase font size and adjust width

    else:
        ylabelStr = "Region"
        corr_matrix = np.corrcoef(df_Tilted.values.T)
        ax = sns.clustermap(corr_matrix, cmap='rocket', fmt='.2f', yticklabels=df_Tilted.columns, xticklabels=df_Tilted.columns, dendrogram_ratio = 0.1, figsize=(40, 40))

    # Labels    
    titleStr = f"{classifyDict['data']} - Clustering Across Regions"
    plt.ylabel(ylabelStr)
    plt.xlabel('Region')
    plt.tick_params(axis='x', which='both', length=0)
    plt.title(titleStr)
    
    plt.savefig(dirDict['classifyDir'] + os.sep + titleStr + '.png', format='png', bbox_inches='tight')
    plt.show()
    plt.close()

def correlation_subset(processed_data, lightsheet_data, modelCountDict, threshold, classifyDict, dirDict):
    # Generates a correlation matrix heatmap based on the correlation between specific features within a comparison.

    brainAreaColorDict = hf.create_color_dict('brainArea', 0)
    brainAreas = list(brainAreaColorDict.keys())
    AreaIdx = dict(zip(brainAreas, np.arange(len(brainAreas))))

    # Use the unprocessesd lightsheet_data to generate a structure for sorting the processed data
    colList = [classifyDict['feature'], 'Brain_Area']
    regionArea = lightsheet_data[colList]
    regionArea.drop_duplicates(inplace=True)
    regionArea['Brain_Area_Idx'] = [AreaIdx[x] for x in regionArea['Brain_Area']]
    regionArea['Region_Color'] = [brainAreaColorDict[x] for x in regionArea['Brain_Area']]
    regionArea.sort_values(by='Brain_Area_Idx', inplace=True)

    # Filter the lightsheet_data to only include data from the drug column which matches condition
    # df = lightsheet_data[lightsheet_data['drug'] == drug]
    df = processed_data

    # identify which regions in modelCount have higher or equal to the threshold
    regionKeepIdx = np.array(np.array(modelCountDict[1]).astype(int) > threshold)
    regionList = np.array(modelCountDict[0])[regionKeepIdx]
    df_Tilted = df[regionList]

    corr_matrix = np.corrcoef(df_Tilted.values.T)
    # ax = sns.heatmap(corr_matrix, cmap='rocket', fmt='.2f', yticklabels=regionList, xticklabels=regionList, square=True)
    ax = sns.clustermap(corr_matrix, cmap="vlag", center=0, fmt='.2f', yticklabels=df_Tilted.columns, xticklabels=df_Tilted.columns, dendrogram_ratio = 0.1, figsize=(10, 10))

    regionArea.set_index('abbreviation', inplace=True)
    
    for tick_label in ax.ax_heatmap.axes.get_yticklabels():
        tick_text = tick_label.get_text()
        regionColor = regionArea.loc[tick_text, 'Region_Color']
        tick_label.set_color(regionColor)

    for tick_label in ax.ax_heatmap.axes.get_xticklabels():
        tick_text = tick_label.get_text()
        regionColor = regionArea.loc[tick_text, 'Region_Color']
        tick_label.set_color(regionColor)
        
    # Label and save the image
    titleStr = f"Region {classifyDict['data']} correlation between {classifyDict['label']}"
    plt.tick_params(axis='x', which='both', length=0)
    plt.title(titleStr, fontsize=10)
    
    plt.savefig(os.sep.join([dirDict['outDir_model'], titleStr + '.svg']), format='svg', bbox_inches='tight')
    plt.show()
    plt.close()

def plot_data_heatmap(lightsheet_data, heatmapDict, dirDict):
    # Current Mode: Create plot with colorbar, then without, and grab the svg item and place it in the second plot to ensure even spacing
    # TODO: shift code to use 'GridSpec' and create a single image with 3 equally sized columns and a colorbar at once.
    # Creates the heatmap for the data
    # heatmapDict['feature'] = 'abbreviation'
    # heatmapDict['data'] = 'cell_density', 'count', 'count_norm', 'density_norm', 'count_norm_scaled'

    # Set variables
    colorMapCap = True
    dataFeature = heatmapDict['feature']
    dataValues = heatmapDict['data']
    blockCount = heatmapDict['blockCount']

    # take the mean across all the datasets for each region, then log transform
    if heatmapDict['logChangeSal'] == True:
        # Create a version of the dataset which is averaged and adjusted against SAL. 
        df_avg  = lightsheet_data.groupby([dataFeature, 'drug'])[dataValues].mean().reset_index()
        df_piv = df_avg.pivot(index=dataFeature, columns='drug', values=dataValues)
        df_piv = df_piv.div(df_piv['SAL'], axis=0)
        # df_piv.drop(columns=['SAL'], inplace=True)
        df_plot = np.log(df_piv)
        
    else:
        # Pivot data to represent samples, features, and data correctly for a heatmap.
        df_plot = lightsheet_data.pivot(index=dataFeature, columns='dataset', values=dataValues)

    # Resort for coherence across figures
    reIdx = heatmapDict['SortList']
    if heatmapDict['logChangeSal'] == False:
        # Identify the highest dataset number and use that to sort things. 
        # Will throw an error if dataset numbers are not the same across conditions.
        maxSampleNum = np.max(np.array([x[-1:] for x in list(lightsheet_data.dataset)]).astype(int)) + 1
        reIdx = [f'{item}{i}' for item in reIdx for i in range(1, maxSampleNum)]
    df_plot = df_plot[reIdx]

    # Create a dictionary of region to area
    regionArea = hf.create_region_to_area_dict(lightsheet_data, dataFeature)

    # Create indicies for dividing the data into the correct number of  sections regardless of the size
    row_idx_set = np.zeros((blockCount, 2), dtype=int)
    if heatmapDict['areaBlocks'] == True:
        # Make the blocks 2 roughly equal size blocks
        line_break_num, line_break_ind = np.unique(regionArea.Brain_Area_Idx, return_index=True)
        row_idx_set[0,:] = [line_break_ind[0], line_break_ind[5]]
        row_idx_set[1,:] = [line_break_ind[5], len(df_plot)]
    
    else:
        indices = np.linspace(0, len(df_plot), num=blockCount+1, dtype=int)
        for block_idx in range(blockCount):
            row_idx_set[block_idx][0] = indices[block_idx]
            row_idx_set[block_idx][1] = indices[block_idx+1]
    

    # Sort the data to be combined per larger area
    df_plot = df_plot.loc[regionArea[dataFeature]]

    # Find the ends of the colormap
    if colorMapCap:
        vmin, vmax = np.percentile(df_plot.values.flatten(), [1, 99])
    else:
        vmin = df_plot.min().min()
        vmax = df_plot.max().max()

    # Plotting variables
    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))

    # Create a version with and without the colorbar for purposes of keeping segments equally sized.
    colorBarSwitch = [True, False]

    for cbs in colorBarSwitch:

        if heatmapDict['logChangeSal'] == True:
            cmap = sns.diverging_palette(240, 10, as_cmap=True, center='light')
            scalefactor = 5
            figH = (scalefactor*8)/blockCount
            figW = blockCount * 3
        else:
            cmap = 'rocket'
            scalefactor = 12
            figH = (scalefactor*5)/blockCount
            figW = blockCount * 10

        fig, axes = plt.subplots(1, blockCount, figsize=(figW, figH))  # Adjust figsize as needed
        # figsize=(scalefactor*2.4, len(df_plot)/len(row_idx_set) * scalefactor * 0.0125)

        if blockCount == 1:
            axes = [axes]

        for idx, row_set in enumerate(row_idx_set):

            # Slice and modify previous structures to create segment
            df_plot_seg = df_plot.iloc[row_set[0]: row_set[1], :]
            regionArea_local = regionArea[regionArea[dataFeature].isin(df_plot_seg.index)]
            region_idx = regionArea_local.Brain_Area_Idx  # Extract for horizontal lines in plot later.

            matrix = df_plot_seg.values

            xticklabels = df_plot_seg.columns.values.tolist()
            yticklabels = df_plot_seg.index.values.tolist()
            if heatmapDict['logChangeSal'] == True:

                heatmap = sns.heatmap(matrix, cmap=cmap, ax=axes[idx] , fmt='.2f', cbar = cbs, yticklabels=yticklabels, xticklabels=xticklabels, vmin=vmin, vmax=vmax, cbar_kws={"format": formatter}, center=0)
                horzLineColor = 'black'

            else:
                xticklabels = [x[0:-1] for x in xticklabels]
                # Convert to x axis labels
                x_labels = ['' for _ in range(matrix.shape[1])]
                result = hf.find_middle_occurrences(xticklabels)
                for mid_sample_ind in result:
                    x_labels[result[mid_sample_ind][1]] = xticklabels[result[mid_sample_ind][1]]
                
                heatmap = sns.heatmap(matrix, cmap=cmap, ax=axes[idx], fmt='.2f', cbar = cbs, square=True, yticklabels=yticklabels, xticklabels=x_labels, vmin=vmin, vmax=vmax, cbar_kws={"format": formatter})
    
                # Clear the x-ticks
                heatmap.tick_params(axis='x', which='both', length=0, labelsize=14)

                # Add in vertical lines breaking up sample types
                _, line_break_ind = np.unique(xticklabels, return_index=True)
                for l_idx in line_break_ind:
                    axes[idx].axvline(x=l_idx, color='white', linewidth=1)
                horzLineColor = 'white'

            # Change
            # Shift the color codes to RGB and add alpha
            colorDict = hf.create_color_dict('drug', 0)
            # Cylce through x-axis labels and change their color to match the boxplot
            for _, label in enumerate(heatmap.get_xticklabels()):
                if label.get_text():
                    label.set_color(colorDict[label.get_text()])

            # Add in horizontal lines breaking up brain regions types.
            line_break_num, line_break_ind = np.unique(region_idx, return_index=True)
            for l_idx in line_break_ind[1:]:
                axes[idx].axhline(y=l_idx, color=horzLineColor, linewidth=1)
                
            # Set the ylabel on the first subplot.
            if idx == 0:
                axes[idx].set_ylabel("Region Names", fontsize=20)

            # if idx == 2:
            #     cbar = heatmap.collections[0].colorbar
            #     cbar.set_label('Colorbar Label', rotation=270, labelpad=5)

        titleStr = f"Data_{dataValues}_block_colorbar_{cbs}"  
        fig.suptitle(titleStr, fontsize=30, y=1.02)
        # fig.text(0.5, -.02, "Samples Per Group", ha='center', fontsize=20)
        plt.tight_layout(h_pad = 0, w_pad = .5)

        # Change the axis of the colorbar to represent multiples of 
        plt.savefig(dirDict['outDir'] + os.sep + f"{titleStr}", bbox_inches='tight')
        plt.show()
        plt.close()

def plot_data_heatmap_perArea(lightsheet_data, heatmapDict, dirDict):
    # Current Mode: Create plot with colorbar, then without, and grab the svg item and place it in the second plot to ensure even spacing
    # TODO: shift code to use 'GridSpec' and create a single image with 3 equally sized columns and a colorbar at once.
    # Creates the heatmap for the data
    # heatmapDict['feature'] = 'abbreviation'
    # heatmapDict['data'] = 'cell_density', 'count', 'count_norm', 'density_norm', 'count_norm_scaled'

    # Set variables
    colorMapCap = True
    dataFeature = heatmapDict['feature']
    dataValues = heatmapDict['data']
    blockCount =heatmapDict['blockCount']

    # take the mean across all the datasets for each region, then log transform
    if heatmapDict['logChangeSal'] == True:
        # Create a version of the dataset which is averaged and adjusted against SAL. 
        df_avg  = lightsheet_data.groupby([dataFeature, 'drug'])[dataValues].mean().reset_index()
        df_piv = df_avg.pivot(index=dataFeature, columns='drug', values=dataValues)
        df_piv = df_piv.div(df_piv['SAL'], axis=0)
        # df_piv.drop(columns=['SAL'], inplace=True)
        df_plot = np.log(df_piv)
        
    else:
        # Pivot data to represent samples, features, and data correctly for a heatmap.
        df_plot = lightsheet_data.pivot(index=dataFeature, columns='dataset', values=dataValues)

    # Resort for coherence across figures
    reIdx = heatmapDict['SortList']
    if heatmapDict['logChangeSal'] == False:
        # Identify the highest dataset number and use that to sort things. 
        # Will throw an error if dataset numbers are not the same across conditions.
        maxSampleNum = np.max(np.array([x[-1:] for x in list(lightsheet_data.dataset)]).astype(int)) + 1
        reIdx = [f'{item}{i}' for item in reIdx for i in range(1, maxSampleNum)]
    df_plot = df_plot[reIdx]

    # Create indicies for dividing the data into the correct number of  sections regardless of the size
    row_idx_set = np.zeros((blockCount, 2), dtype=int)
    indices = np.linspace(0, len(df_plot), num=blockCount+1, dtype=int)
    for block_idx in range(blockCount):
        row_idx_set[block_idx][0] = indices[block_idx]
        row_idx_set[block_idx][1] = indices[block_idx+1]
        
    # Create a dictionary of region to area
    regionArea = hf.create_region_to_area_dict(lightsheet_data, dataFeature)

    # Sort the data to be combined per larger area
    df_plot = df_plot.loc[regionArea[dataFeature]]

    # Find the ends of the colormap
    if colorMapCap:
        vmin, vmax = np.percentile(df_plot.values.flatten(), [1, 99])
    else:
        vmin = df_plot.min().min()
        vmax = df_plot.max().max()

    # Plotting variables
    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))

    # Create a version with and without the colorbar for purposes of keeping segments equally sized.
    colorBarSwitch = [True, False]

    for cbs in colorBarSwitch:

        if heatmapDict['logChangeSal'] == True:
            cmap = sns.diverging_palette(240, 10, as_cmap=True, center='light')
            scalefactor = 12
            figH = (scalefactor*5)/blockCount
            figW = blockCount * 3
        else:
            cmap = 'rocket'
            scalefactor = 12
            figH = (scalefactor*5)/blockCount
            figW = blockCount * 10

        fig, axes = plt.subplots(1, blockCount, figsize=(figW, figH))  # Adjust figsize as needed
        # figsize=(scalefactor*2.4, len(df_plot)/len(row_idx_set) * scalefactor * 0.0125)

        if blockCount == 1:
            axes = [axes]

        for idx, row_set in enumerate(row_idx_set):

            # Slice and modify previous structures to create segment
            df_plot_seg = df_plot.iloc[row_set[0]: row_set[1], :]
            regionArea_local = regionArea[regionArea[dataFeature].isin(df_plot_seg.index)]
            region_idx = regionArea_local.Brain_Area_Idx  # Extract for horizontal lines in plot later.

            matrix = df_plot_seg.values

            xticklabels = df_plot_seg.columns.values.tolist()
            yticklabels = df_plot_seg.index.values.tolist()
            if heatmapDict['logChangeSal'] == True:

                heatmap = sns.heatmap(matrix, cmap=cmap, ax=axes[idx] , fmt='.2f', cbar = cbs, yticklabels=yticklabels, xticklabels=xticklabels, vmin=vmin, vmax=vmax, cbar_kws={"format": formatter}, center=0)
                horzLineColor = 'black'

            else:
                xticklabels = [x[0:-1] for x in xticklabels]
                # Convert to x axis labels
                x_labels = ['' for _ in range(matrix.shape[1])]
                result = hf.find_middle_occurrences(xticklabels)
                for mid_sample_ind in result:
                    x_labels[result[mid_sample_ind][1]] = xticklabels[result[mid_sample_ind][1]]
                
                heatmap = sns.heatmap(matrix, cmap=cmap, ax=axes[idx], fmt='.2f', cbar = cbs, yticklabels=yticklabels, xticklabels=x_labels, vmin=vmin, vmax=vmax, cbar_kws={"format": formatter})
    
                # Clear the x-ticks
                heatmap.tick_params(axis='x', which='both', length=0, labelsize=14)

                # Add in vertical lines breaking up sample types
                _, line_break_ind = np.unique(xticklabels, return_index=True)
                for l_idx in line_break_ind:
                    axes[idx].axvline(x=l_idx, color='white', linewidth=1)
                horzLineColor = 'white'

            # Add in horizontal lines breaking up brain regions types.
            line_break_num, line_break_ind = np.unique(region_idx, return_index=True)
            for l_idx in line_break_ind[1:]:
                axes[idx].axhline(y=l_idx, color=horzLineColor, linewidth=1)
                
            # Set the ylabel on the first subplot.
            if idx == 0:
                axes[idx].set_ylabel("Region Names", fontsize=20)

            # if idx == 2:
            #     cbar = heatmap.collections[0].colorbar
            #     cbar.set_label('Colorbar Label', rotation=270, labelpad=5)

        titleStr = f"Data_{dataValues}_block_colorbar_{cbs}"  
        fig.suptitle(titleStr, fontsize=30, y=1.02)
        fig.text(0.5, -.02, "Samples Per Group", ha='center', fontsize=20)
        plt.tight_layout(h_pad = 0, w_pad = .5)

        # Change the axis of the colorbar to represent multiples of 

        plt.savefig(dirDict['classifyDir'] + os.sep + f"{titleStr}", bbox_inches='tight')
        plt.show()
        plt.close()

def create_heatmaps_allC(matrix, dim_to_loop=0, titleStatic='Heatmap', titleLoop=[], dirDict=[]):
    import seaborn as sns
    from mpl_toolkits.axes_grid1 import ImageGrid
    from matplotlib.colors import ListedColormap

    cmap = plt.cm.Blues

    loopRange = matrix.shape[dim_to_loop]

    if titleLoop == []:
        titleLoop = range(loopRange)

    fig, axes = plt.subplots(nrows=1, ncols=loopRange,
                             figsize=(loopRange * 1.5, 15))

    fullTitleStr = f"{titleStatic} Classification via L1 Regularization, Feature weights"

    fig.suptitle(fullTitleStr, fontsize=16, y=0.92)

    for i in range(loopRange):
        heatmap_data = matrix[i, :, :]
        plt.figure(figsize=(2, 10))

        # Only colorbar for the value all the way to the right
        sns.heatmap(heatmap_data, cmap=cmap,
                    cbar=False, fmt='.2f', ax=axes[i])

        axes[i].set_title(f"C = {str(titleLoop[i])}")

        axes[i].set_xlabel("Split")

        if i == 0:
            axes[i].set_ylabel("Feature")
        else:
            axes[i].tick_params(left=False, labelleft=False)

    plt.savefig(dirDict['classifyDir'] + os.sep + fullTitleStr, bbox_inches='tight')

    # Add a colorbar on the far right plot
    vmin = np.min(matrix)
    vmax = np.max(matrix)

    norm = plt.Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm)
    sm.set_array([])

    # [left, bottom, width, height] of the colorbar axis
    cbar_ax = fig.add_axes([0.93, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)

    plt.tight_layout()
    plt.show()
    plt.close()

def create_heatmaps_perDrug(matrix, titleStatic='Heatmap', titleLoop=[], xLab = [], perPlotXTicks=[], dirDict=[]):
    import seaborn as sns
    from mpl_toolkits.axes_grid1 import ImageGrid
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    # Create a TwoSlopeNorm normalization centered at 0
    vmin, vmax = matrix.min(), matrix.max()
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Pull the color map, and set 0 to white
    cmap = sns.color_palette("rocket", as_cmap=True)  # Use the viridis color map as a base
    linSpaceInts = np.linspace(vmin, vmax, cmap.N*2)
    zero_idx = np.argmin(np.abs(linSpaceInts))
    colors = cmap(linSpaceInts)  # Extract the colors from the color map

    # Attempting to set the 0 value and the numbers around to to grey does not make the final result grey. 
    # zero_color = [.75, 0.75, 0.75, 1]
    # colors[zero_idx - 1] = zero_color
    # colors[zero_idx] = zero_color
    # colors[zero_idx + 1] = zero_color

    custom_cmap = LinearSegmentedColormap.from_list('custom', colors)

    loopRange = matrix.shape[0]
    
    if titleLoop == []:
        titleLoop = range(loopRange)

    fig, axes = plt.subplots(nrows=1, ncols=loopRange,figsize=(loopRange * 1.5, 18))

    fullTitleStr = f"{titleStatic} Classification via L1 Regularization, Median Feature weights"

    fig.suptitle(fullTitleStr, fontsize=16, y=0.92)

    for i in range(loopRange):
        heatmap_data = matrix[i, :, :]
        plt.figure(figsize=(2, 10))

        # Only colorbar for the value all the way to the right
        sns.heatmap(heatmap_data, cmap=custom_cmap, vmin=vmin, vmax=vmax, cbar=False, fmt='.2f', ax=axes[i], norm=norm)

        axes[i].set_title(f"{str(titleLoop[i])}")

        axes[i].set_xlabel(xLab)
        perPlotXTicks = [str(x) for x in perPlotXTicks]
        # perPlotXTicks = [int(num) if isinstance(num, int) else str(num) if isinstance(num, float) else None for num in perPlotXTicks]
        axes[i].set_xticks(np.arange(0, len(perPlotXTicks))+0.5)
        axes[i].set_xticklabels(perPlotXTicks, fontdict={'fontsize':7})

        if i == 0:
            axes[i].set_ylabel("Feature")
            axes[i].set_yticklabels(labels=axes[i].get_yticklabels(), fontdict={'fontsize':8})
        else:
            axes[i].tick_params(left=False, labelleft=False)

        for idx in range(1, len(perPlotXTicks)): 
            axes[i].axvline(idx, color='black', linewidth=0.5)


    # Add a colorbar on the far right plot
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm) #
    sm.set_array([])

    # Not working
    # plt.savefig(dirDict['classifyDir'] + os.sep + fullTitleStr + '.png',
    #             format='png', bbox_inches='tight')

    # [left, bottom, width, height] of the colorbar axis
    cbar_ax = fig.add_axes([0.93, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)

    plt.tight_layout()
    plt.savefig(dirDict['classifyDir'] + os.sep + fullTitleStr + '.png', format='png', bbox_inches='tight')
    plt.show()
    plt.close() 

def plot_cFos_delta(cfos_diff, drugList, fileOutName):
    plt.figure(figsize=(20,50))
    drugPairList = [(a + '-'+ b) for idx, a in enumerate(drugList) for b in drugList[idx + 1:]]
    palette = sns.color_palette(n_colors=len(drugList))

    for drug_comp_i in range(0, len(cfos_diff)):
        #melt data for plotting and specify color
        data = pd.melt(cfos_diff[drug_comp_i], id_vars=['Region_Name'], value_vars=drugPairList[drug_comp_i],
                        var_name='Drug', value_name='Change')
        if drug_comp_i == 0:
            data_melted = data
        else:
            data_melted = pd.concat([data_melted, data], ignore_index=True, sort=False)

    # Focus on the comparisons involving SAL condition
    drugDataMelt = data_melted[data_melted.Drug.str.contains('SAL')]

    #point plot
    ax = sns.pointplot(y='Region_Name', x='Change', data = drugDataMelt, errorbar=('ci', 95), join=False, units=16, errwidth = 0.5,
                    hue='Drug', palette = palette, dodge=0.4, scale=0.5)

    #cleanup
    sns.despine()
    plt.xlabel('cFos density change (%)')
    plt.ylabel('')

    plt.axvline(x=0, color='grey', linestyle='--', lw=0.5)

    fig = plt.savefig(fileOutName, bbox_inches='tight')

def plot_cFos_delta_new(lightsheet_data, cfos_diff, cfos_diff_labels, drugList, fileOutName):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    data_melted = pd.DataFrame()
    for data, label in zip(cfos_diff, cfos_diff_labels):
        # Melt data for plotting and specify color
        dataMelt = pd.melt(data, id_vars=['Region_Name'], value_vars=label, var_name='Drug', value_name='Change')
        data_melted = pd.concat([data_melted, dataMelt], ignore_index=True, sort=False)

    # Focus on the comparisons involving SAL condition
    drugDataMelt = data_melted[data_melted.Drug.str.contains('SAL')]
    drugDataMelt['Drug'] = drugDataMelt['Drug'].str.replace('-SAL', '')

    # Testing - leave only KET and PSI
    drugDataMelt = drugDataMelt[drugDataMelt['Drug'].isin(['KET', 'PSI', '5MEO'])]

    # Generate the dictionary for region_name to brain_area
    brainAreaColorDict = hf.create_region_to_area_dict(lightsheet_data, ['Region_Name', 'Region_ID'])

    # merge the brainAreaColorDict with the drugDataMelt
    drugDataMelt = drugDataMelt.merge(brainAreaColorDict, left_on='Region_Name', right_on='Region_Name')

    # Switch drugs to categorical variables
    drugDataMelt['Drug'] = pd.Categorical(drugDataMelt['Drug'], categories=['PSI', 'KET', '5MEO', '6-F-DET', 'MDMA', 'A-SSRI', 'C-SSRI', 'SAL'], ordered=True)
    palette = hf.create_color_dict(dictType='drug')
                      
    # Cycle through brainAreas, filter drugDataMelt, and plot
    # for brainArea in brainAreas:

    # brainAreadata = drugDataMelt[drugDataMelt['Brain_Area'] == brainArea].sort_index()
    brainAreadata = drugDataMelt.sort_index()
    regionCount = brainAreadata.Region_Name.unique().shape[0]
    
    # Identify the index for the supraoptic nucleus
    supraoptic_idx = brainAreadata[brainAreadata['Region_Name'] == 'Supraoptic nucleus'].index[0]
    plot_idx_set = [[0, supraoptic_idx], [supraoptic_idx, len(brainAreadata)]]
    
    # Create a figure with two columns
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, regionCount*0.05), sharey=True)

    for plot_idx, data_idx in enumerate(plot_idx_set):

        plt.figure(figsize=(1.5, regionCount*0.03))

        # Plot in the first column
        ax = sns.pointplot(y='Region_Name', x='Change', data=brainAreadata.iloc[data_idx[0]:data_idx[1]], errorbar=('ci', 95), legend=False,
                      join=False, units=16, errwidth=0.5, hue='Drug', palette=palette, dodge=0.4, scale=0.25)

        # Limit the x axis to 500
        lowLim = -100
        upperLim = min(ax.get_xlim()[1], 1000)
        ax.set_xlim(lowLim, upperLim)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        plt.title(f'cFos Density Change', fontsize=10)

        # Cleanup
        sns.despine()
        ax.set_xlabel('cFos density change (%)')
        # axes[ax_idx].set_ylabel('')

        ax.axvline(x=0, color='grey', linestyle='--', lw=0.5)
        # plt.title(f'{brainArea} - cFos Density Change', fontsize=5)
        

        # plt.savefig(f'{fileOutName}_{brainArea}_Limited.png', bbox_inches='tight')
        plt.savefig(f'{fileOutName}_{plot_idx}_Limited.png', bbox_inches='tight')
        # plt.savefig(f'{fileOutName}_{plot_idx}.png', bbox_inches='tight')
        plt.show()
        plt.close()

### Classification based plots
def plotConfusionMatrix(scores, YtickLabs, conf_matrix_list_of_arrays, fit, titleStr, dirDict):

    conf_matrix_list_of_arrays = np.array(conf_matrix_list_of_arrays)

    print(f"{fit}: {np.mean(scores):.2f} accuracy with a standard deviation of {np.std(scores):.2f}")
    # Prepare the confusion matrix plot
    mean_of_conf_matrix_arrays = np.mean(conf_matrix_list_of_arrays, axis=0)

    if fit != '':
        fullTitleStr = fit + ': ' + titleStr
    else:
        fullTitleStr = titleStr

    if '\n' in titleStr:
        titleStr = titleStr.replace('\n                  ', '')

    config.setup_Confmatrix_settings()

    figSizeMat = np.array(mean_of_conf_matrix_arrays.shape)/3.33
    figSizeMat[0] = figSizeMat[0] + 1
    plt.figure(figsize=figSizeMat)
    ax = sns.heatmap(mean_of_conf_matrix_arrays,cmap='Reds', annot=True, fmt=".2f", square=True, cbar_kws={"shrink": 0.8})
    ax.set(xticklabels=YtickLabs, yticklabels=YtickLabs, xlabel='Predicted Label', ylabel='True Label')
    # plt.title(fullTitleStr, fontsize=figSizeMat[0]*1.5)

    # Save the plot
    plt.savefig(join(dirDict['outDir_model'], f"ConfusionMatrix_{fit}"), bbox_inches='tight')     
    plt.show()
    plt.close() 

def plotPRcurve(n_classes, y_real_lab, y_prob, labelDict, Yticklabs, daObjstr, plotSwitch, fit, dirDict):
    # n_classes = int, number of classes
    # y_real, y_prob = test set labels and probabilities assigned to test set samples.
    # y_real, y_prob are in a [n_splits, n_samples, n_classes] format

    from sklearn.metrics import precision_recall_curve, auc
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.font_manager import FontProperties
    from sklearn.preprocessing import label_binarize
    
    # Convert the labels to a binary format
    y_real_lab = [label_binarize(x, classes=Yticklabs) for x in y_real_lab]
    y_real_lab = np.array(y_real_lab)
    if n_classes == 2:
        y_real_lab = np.concatenate([1 - y_real_lab, y_real_lab], axis=2)
    y_real = y_real_lab

    y_prob = np.array(y_prob)

    y_real_all, y_prob_all = [], []

    if plotSwitch:
        figSizeMat = np.array((n_classes, n_classes))/2.2
        f = plt.figure(figsize=figSizeMat)
        axes = plt.axes()

    # Depending on feature list y passed, determine if the classes are numbers or strings
    if all(isinstance(key, str) for key in labelDict.keys()):
        labelDict = {value: key for key, value in labelDict.items()}

    auc_dict = dict()

    colorDict = hf.create_color_dict(dictType='stressor', rgbSwitch=0, alpha_value=0, scaleVal=False)

    n_splits = y_real.shape[0]

    for i in np.arange(n_classes):
        label_per_split = y_real[:, :, i]
        prob_per_split  = y_prob[:, :, i]

        label_per_split_reshape = label_per_split.reshape(prob_per_split.size, 1)
        prob_per_split_reshape  = prob_per_split.reshape(prob_per_split.size, 1)

        y_real_all.append(label_per_split_reshape)
        y_prob_all.append(prob_per_split_reshape)

        # Calculate the PR Curve (all splits concatenated)
        precision, recall, _ = precision_recall_curve(label_per_split_reshape, prob_per_split_reshape)
        auc_val = auc(recall, precision)
        lab = f'{labelDict[i]}=%.2f' % (auc_val)
        auc_dict[labelDict[i]] = np.round(auc_val, 2)

        if plotSwitch:
            col = colorDict.get(labelDict[i], '#888888')
            axes.step(recall, precision, label=lab, color=col, lw=2)

    # Mean PR Curve (all classes + all splits concatenated)
    precision, recall, _ = precision_recall_curve(
        np.concatenate(y_real_all), np.concatenate(y_prob_all)
    )
    auc_val_mean = auc(recall, precision)
    auc_dict['Mean'] = np.round(auc_val_mean, 2)

    # ── Per-split mean AUC ──────────────────────────────────────────────────
    # For each CV split compute the mean AUC across all classes so we have a
    # distribution of n_splits scalars for jitter plotting downstream.
    # around line 1211 — replace the existing auc_per_split block with:

    auc_per_split      = []          # mean across classes per split (existing)
    auc_per_split_cls  = {}          # {class_label: [auc_s1, auc_s2, ...]}  ← new

    for s in range(n_splits):
        split_aucs = {}
        for i in np.arange(n_classes):
            lbl  = y_real[s, :, i]
            prob = y_prob[s, :, i]
            try:
                p, r, _ = precision_recall_curve(lbl, prob)
                split_aucs[labelDict[i]] = auc(r, p)
            except Exception:
                pass
        if split_aucs:
            auc_per_split.append(float(np.mean(list(split_aucs.values()))))
            for cls, val in split_aucs.items():
                auc_per_split_cls.setdefault(cls, []).append(float(val))

    auc_dict['PerSplit']         = auc_per_split       # unchanged — backward compatible
    auc_dict['PerSplit_perClass']= auc_per_split_cls   # new: {class: [n_splits floats]}

    if plotSwitch:
        lab = f'Mean Curve=%.2f' % (auc_val_mean)
        axes.step(recall, precision, label=lab, lw=3, color='black')
        axes.set_xlabel('Recall')
        axes.set_ylabel('Precision')

        if n_classes == 2:
            legend = axes.legend(loc='lower left')
        else:
            legend = axes.legend(loc='lower left')
            for label in legend.get_lines():
                label.set_linewidth(.5)

        plt.savefig(join(dirDict['outDir_model'], f"PRcurve_{fit}"), bbox_inches='tight')
        plt.show()
        plt.close() 

    return auc_dict

def _shap_condition_label(classifyDict, labelDict=None):
    """
    Build a human-readable comparison string, e.g. 'Ctrl vs FS' or
    'Ctrl vs FS vs FSW vs RS vs TS'.
    Uses labelDict class names when available; falls back to parsing
    classifyDict['label'].
    """
    if labelDict is not None:
        return ' vs '.join(str(k) for k in labelDict.keys())
    # Fallback: strip 'class_' prefix and insert ' vs '
    raw = classifyDict.get('label', '')
    if raw.startswith('class_'):
        raw = raw[len('class_'):]
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# BEESWARM SUMMARY PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_summary(X_train_trans_list, shap_values_list, y_vec, n_classes,
                      plotDict, classifyDict, dirDict, labelDict=None):
    import shap

    n_splits   = len(X_train_trans_list)
    test_count = shap_values_list[0][0].shape[0]
    shap_threshold = np.ceil(n_splits * plotDict['shapSummaryThres'] / 100)

    X_train_trans_nonmean = pd.concat(X_train_trans_list, axis=0)
    # reset_index() in classifyFunctions promotes the original sample positions
    # into a column named 'index'.  Drop it now so it never enters feature
    # counting / threshold filtering (it has no NaN → count = n_splits, which
    # makes it look like the best feature and corrupts sortingIdx).
    X_train_trans_nonmean = X_train_trans_nonmean.drop(columns='index', errors='ignore')

    # ── Aggregate SHAP values: group BY CLASS across all splits ──────────────
    shap_values_nonmean = []
    if n_classes == 2:
        shap_values_nonmean.append(
            pd.concat(shap_values_list[0], axis=0)
        )
    else:
        n_cls = len(shap_values_list)          # outer dim = n_classes
        for cls_idx in range(n_cls):
            class_dfs = [shap_values_list[cls_idx][split]
                        for split in range(len(shap_values_list[cls_idx]))]
            if class_dfs:                      # skip if all splits failed for this class
                shap_values_nonmean.append(pd.concat(class_dfs, axis=0))

    # Save original sample positions BEFORE stripping — needed by binary branch
    # to map test samples back to y_vec labels.
    shap_sample_indices = [
        df['index'].values if 'index' in df.columns else np.arange(len(df))
        for df in shap_values_nonmean
    ]

    # Same 'index' artefact in SHAP DataFrames — strip before feature counting
    shap_values_nonmean = [
        df.drop(columns='index', errors='ignore') for df in shap_values_nonmean
    ]

    # Overall comparison string for the annotation (e.g. "Ctrl vs FS vs FSW")
    comparison_str = _shap_condition_label(classifyDict, labelDict)

    # Class name lookup for multiclass plots
    class_names = list(labelDict.keys()) if labelDict is not None else None

    cap_shap_values  = True
    max_abs_shap_val = 1

    # ── One beeswarm plot per class ───────────────────────────────────────────
    for class_idx, shap_vals in enumerate(shap_values_nonmean):

        # Count in how many CV splits each feature appeared
        shapValueCount    = shap_vals.agg(np.isnan).sum()
        feature_model_count = n_splits - shapValueCount / test_count
        svf_sorted        = feature_model_count.sort_values(ascending=False)

        if plotDict['shapSummaryThres'] is not None:
            svf_sorted = svf_sorted[svf_sorted >= shap_threshold]
            maxDisp    = len(svf_sorted) - 1
        else:
            maxDisp = plotDict['shapMaxDisplay']

        shap_vals_for_plot = shap_vals.copy()   # keep NaN intact for plotting
        shap_vals = shap_vals.fillna(0)         # filled version only for sorting/clipping below

        sortingIdx            = svf_sorted.index
        X_train_trans_sorted  = X_train_trans_nonmean[sortingIdx]
        shap_vals             = shap_vals[sortingIdx]
        shap_vals_for_plot    = shap_vals_for_plot[sortingIdx]

        # Guard: if threshold filtered out every feature, skip rather than crash
        if shap_vals.shape[1] == 0:
            print(f"  [SHAP] class {class_idx}: 0 features passed threshold — skipping beeswarm.")
            continue

        # ── Sorting / feature naming ──────────────────────────────────────────
        if n_classes == 2:
            shap_vals['drug'] = y_vec[shap_sample_indices[class_idx]]
            drugList          = list(shap_vals['drug'].unique())
            drugMedians       = pd.DataFrame(index=list(shap_vals.columns[1:-1]),
                                             columns=drugList)
            shap_vals.reset_index(inplace=True, drop=True)
            for drug in drugList:
                drugMedians.loc[:, drug] = shap_vals.loc[
                    shap_vals['drug'] == drug].median(numeric_only=True)
            drugMedians['medianDiff']  = abs(drugMedians[drugList[0]] -
                                             drugMedians[drugList[1]])
            drugMedians_sort           = drugMedians.sort_values(
                                             by='medianDiff', ascending=False)
            shap_vals            = shap_vals[drugMedians_sort.index]
            X_train_trans_sorted = X_train_trans_sorted[drugMedians_sort.index]
            # shap_vals_for_plot must get the same treatment: drop 'index' and
            # apply the same drugMedians re-sort, otherwise it has one extra
            # column and hits the shape AssertionError in shap.summary_plot
            if 'index' in shap_vals_for_plot.columns:
                shap_vals_for_plot = shap_vals_for_plot.drop('index', axis=1)
            shap_vals_for_plot = shap_vals_for_plot[drugMedians_sort.index]
            featureNames = [f"{feat} ({drugMedians_sort.medianDiff.round(2)[feat]})"
                            for feat in list(shap_vals.columns)]
        else:
            if 'index' in shap_vals.columns:
                shap_vals = shap_vals.drop('index', axis=1)
            if 'index' in shap_vals_for_plot.columns:          
                shap_vals_for_plot = shap_vals_for_plot.drop('index', axis=1)    
            if 'index' in X_train_trans_sorted.columns:
                X_train_trans_sorted = X_train_trans_sorted.drop('index', axis=1)
            featureNames = [f"{feat} ({int(svf_sorted[feat])})"
                            for feat in list(shap_vals.columns)]

        sortSHAP = False
        
        shared_features = [c for c in shap_vals.columns
                   if c in X_train_trans_sorted.columns]
        if len(shared_features) != len(shap_vals.columns) or \
        len(shared_features) != len(X_train_trans_sorted.columns):
            print(f"  [SHAP] Column mismatch — shap:{len(shap_vals.columns)} "
                f"vs X:{len(X_train_trans_sorted.columns)} — "
                f"aligning to {len(shared_features)} shared features")
            shap_vals            = shap_vals[shared_features]
            shap_vals_for_plot   = shap_vals_for_plot[shared_features]
            X_train_trans_sorted = X_train_trans_sorted[shared_features]
            featureNames = [f"{feat} ({int(svf_sorted[feat])})"
                            for feat in shared_features
                            if feat in svf_sorted.index]
    
        if cap_shap_values:
            shap_vals          = shap_vals.clip(lower=-max_abs_shap_val, upper=max_abs_shap_val)
            shap_vals_for_plot = shap_vals_for_plot.clip(lower=-max_abs_shap_val, upper=max_abs_shap_val)

        shap_values_min = shap_vals.min().min()
        shap_values_max = shap_vals.max().max()
        maxVal = max(abs(shap_values_min), shap_values_max) + 0.01

        # ── Build corner annotation string ────────────────────────────────────
        if n_classes == 2:
            # Binary: show full comparison ("Ctrl vs FS")
            corner_text = comparison_str
        else:
            # Multiclass: show current class + full comparison on second line
            current_class = (class_names[class_idx]
                             if class_names is not None
                             else f"Class {class_idx}")
            corner_text = f"{current_class}\n({comparison_str})"

        # ── File label ───────────────────────────────────────────────────────
        class_label = f"class{class_idx}" if n_classes > 2 else "binary"

        # ── Plot ─────────────────────────────────────────────────────────────
        def _annotate_corner(text):
            """Add condition label to top-right corner of current SHAP axes."""
            ax = plt.gcf().axes[0]
            ax.text(
                0.98, 0.98, text,
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=7, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec='none', alpha=0.7)
            )

        if 'MDMA' in classifyDict['label']:
            firstHalfIdx = int(len(featureNames) / 2)

            shap.summary_plot(shap_vals_for_plot.values[:, :firstHalfIdx],
                              X_train_trans_sorted.values[:, :firstHalfIdx],
                              feature_names=featureNames[:firstHalfIdx],
                              sort=sortSHAP, show=False, max_display=maxDisp,
                              cmap='PuOr_r', color_bar_label='cFos Score',
                              plot_size=.3)
            plt.xlim([-maxVal, maxVal])
            _annotate_corner(corner_text)
            plt.savefig(join(dirDict['outDir_model'],
                             f"SHAP_summary_{class_label}_1.svg"),
                        format='svg', bbox_inches='tight')
            plt.show()
            plt.close()

            shap.summary_plot(shap_vals_for_plot.values[:, firstHalfIdx:],
                              X_train_trans_sorted.values[:, firstHalfIdx:],
                              feature_names=featureNames[firstHalfIdx:],
                              sort=sortSHAP, show=False, max_display=maxDisp,
                              cmap='PuOr_r', color_bar_label='cFos Score',
                              plot_size=.3)
            plt.xlim([-maxVal, maxVal])
            _annotate_corner(corner_text)
            plt.savefig(join(dirDict['outDir_model'],
                             f"SHAP_summary_{class_label}_2.svg"),
                        format='svg', bbox_inches='tight')
            plt.show()
            plt.close()

        else:
            shap.summary_plot(shap_vals_for_plot.values, X_train_trans_sorted.values,
                              feature_names=featureNames,
                              sort=sortSHAP, show=False, max_display=maxDisp,
                              cmap='PuOr_r', color_bar_label='cFos Score',
                              plot_size=.45)
            plt.xlim([-maxVal, maxVal])
            _annotate_corner(corner_text)
            plt.savefig(join(dirDict['outDir_model'],
                             f"SHAP_summary_{class_label}.svg"),
                        format='svg', bbox_inches='tight')
            plt.show()
            plt.close()

        # ── Correlation matrix of top SHAP regions ───────────────────────────
        if plotDict.get('plot_SHAPcorr', True):
            n_top = min(plotDict.get('shapCorrTop', 20),
                        len(shap_vals_for_plot.columns))
            plot_shap_correlation_matrix(shap_vals_for_plot,
                                         X_train_trans_nonmean,
                                         featureNames, n_top,
                                         class_label, dirDict)

# ─────────────────────────────────────────────────────────────────────────────
def plot_shap_correlation_matrix(shap_vals_for_plot, X_train_trans_nonmean,
                                  featureNames, n_top, class_label, dirDict):
    """
    Pearson correlation matrix of the top-n SHAP regions (by mean |SHAP|),
    computed on the pooled transformed feature values across all CV splits.
    Saved as SHAP_corrmat_{class_label}.svg next to the beeswarm plots.
    """
    # Top n regions by mean absolute SHAP value
    mean_abs   = shap_vals_for_plot.abs().mean(axis=0)
    top_regions = mean_abs.nlargest(n_top).index.tolist()

    # Subset pooled transformed data and compute Pearson correlation
    X_sub = X_train_trans_nonmean[
        [r for r in top_regions if r in X_train_trans_nonmean.columns]
    ].dropna()
    corr = X_sub.corr(method='pearson')

    # Strip the "(N splits)" suffix added to featureNames for display
    col_to_label = {}
    for feat, label in zip(shap_vals_for_plot.columns, featureNames):
        col_to_label[feat] = label.split(' (')[0]
    short_labels = [col_to_label.get(r, r) for r in corr.columns]

    fig, ax = plt.subplots(figsize=(n_top * 0.45 + 1, n_top * 0.45 + 1))
    sns.heatmap(corr, ax=ax,
                xticklabels=short_labels, yticklabels=short_labels,
                cmap='RdBu_r', vmin=-1, vmax=1, center=0,
                square=True, linewidths=0.3, linecolor='white',
                cbar_kws={'shrink': 0.6, 'label': 'Pearson r'})
    ax.tick_params(labelsize=6, length=2)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title(f'Feature correlation — top {n_top} SHAP regions\n({class_label})',
              fontsize=7)
    plt.tight_layout()
    plt.savefig(join(dirDict['outDir_model'],
                     f'SHAP_corrmat_{class_label}.svg'),
                format='svg', bbox_inches='tight')
    plt.show()
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
def plot_shap_brainglobe(X_test_trans_list, shap_values_list, n_classes,
                         plotDict, classifyDict, dirDict,
                         orientation='frontal', slice_position=0.5,
                         colour_metric='mean_abs',
                         atlas_name='allen_mouse_25um'):
    try:
        import brainglobe_heatmap as bgh
    except ImportError:
        print("[brainglobe] Not installed — run: pip install brainglobe-heatmap brainglobe-atlasapi")
        return

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    # ── Replicate the >threshold logic from plot_shap_summary() ──────────────
    n_splits       = len(X_test_trans_list)
    shap_threshold = np.ceil(n_splits * plotDict['shapSummaryThres'] / 100)

    def _concat_class(shap_list_for_class):
        valid = [df for df in shap_list_for_class if df is not None and len(df) > 0]
        return pd.concat(valid, axis=0) if valid else None

    def _above_threshold(shap_df):
        if 'index' in shap_df.columns:
            shap_df = shap_df.drop(columns=['index'])
        test_count          = shap_df.shape[0] / n_splits
        nan_count           = shap_df.agg(np.isnan).sum()
        feature_model_count = n_splits - nan_count / test_count
        svf                 = feature_model_count.sort_values(ascending=False)
        return svf[svf >= shap_threshold]

    # ── Mean SHAP per region across all classes ───────────────────────────────
    all_abs, all_signed = [], []
    for ci in range(len(shap_values_list)):
        shap_df = _concat_class(shap_values_list[ci])
        if shap_df is None:
            continue
        selected = _above_threshold(shap_df).index.tolist()
        if not selected:
            continue
        filt = shap_df[selected].fillna(0)
        all_abs.append(filt.abs().mean(axis=0))
        all_signed.append(filt.mean(axis=0))

    if not all_abs:
        print("[brainglobe] No regions above threshold — nothing to render.")
        return

    mean_abs    = pd.concat(all_abs,    axis=1).mean(axis=1)
    mean_signed = pd.concat(all_signed, axis=1).mean(axis=1)

    colour_series = mean_abs    if colour_metric == 'mean_abs' else mean_signed
    cmap_name     = 'Reds'      if colour_metric == 'mean_abs' else 'RdBu_r'
    cmap_label    = 'Mean |SHAP|' if colour_metric == 'mean_abs' else 'Mean SHAP'
    region_values = colour_series.to_dict()

    print(f"[brainglobe] {len(region_values)} regions ≥{int(shap_threshold)} splits — rendering {orientation} …")

    # ── Render ────────────────────────────────────────────────────────────────
    try:
        hmap = bgh.Heatmap(
            values        = region_values,
            position      = slice_position,
            orientation   = orientation,
            atlas_name    = atlas_name,
            cmap          = cmap_name,
            vmin          = 0 if colour_metric == 'mean_abs' else None,
            label_regions = True,
            )
        fig, ax = hmap.plot()

        vals = np.array(list(region_values.values()))
        norm = mcolors.Normalize(vmin=vals.min(), vmax=vals.max()) \
               if colour_metric == 'mean_abs' \
               else mcolors.TwoSlopeNorm(vmin=vals.min(), vcenter=0,
                                         vmax=max(abs(vals.min()), vals.max()))
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap_name)),
                     ax=ax, shrink=0.6, pad=0.02, label=cmap_label)

        ax.set_title(f"SHAP regions (>{plotDict['shapSummaryThres']}% splits) — "
                     f"{classifyDict.get('label', '')} | {orientation}", fontsize=10)
        plt.tight_layout()

        save_path = join(dirDict['outDir_model'],
                         f"SHAP_brainglobe_{orientation}_{colour_metric}.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"[brainglobe] Saved → {save_path}")

    except Exception as e:
        print(f"[brainglobe] Render failed: {e}")

    # ── Save region table ─────────────────────────────────────────────────────
    pd.DataFrame({
        'acronym'          : list(region_values.keys()),
        'mean_abs_shap'    : [mean_abs.get(k, np.nan)    for k in region_values],
        'mean_signed_shap' : [mean_signed.get(k, np.nan) for k in region_values],
    }).sort_values('mean_abs_shap', ascending=False).to_csv(
        join(dirDict['outDir_model'],
             f"SHAP_brainglobe_regions_{colour_metric}.csv"),
        index=False
    )

def plot_shap_bar(explainers, X_train_trans_list, shap_values_list, y_vec, n_classes, plotDict, classifyDict, dirDict):
    import shap

    n_splits = len(X_train_trans_list)
    shap_threshold = np.ceil(n_splits * plotDict['shapSummaryThres']/100)

    X_train_trans_nonmean = pd.concat(X_train_trans_list, axis=0)
    shap_values_nonmean = []
    if n_classes == 2:
        shap_values_nonmean.append(pd.concat(shap_values_list, axis=0))
        test_count = shap_values_list[0].shape[0]
    else:
        test_count = shap_values_list[0][0].shape[0]
        for shap_x_df in shap_values_list:
            shap_values_nonmean.append(pd.concat(shap_x_df, axis=0))

    # Plot the SHAP values for each class
    cap_shap_values = True # Cap the SHAP values at 1 and -1
    max_abs_shap_val = 1

    # expOut = []
    # for explainer, x_data in zip(explainers, X_train_trans_list):
    #     expOut1 = explainer(x_data.drop('index', axis=1))



    for shap_vals in shap_values_nonmean:
        # determine how many models across all the splits each feature was included in
        shapValueCount = shap_vals.agg(np.isnan).sum()
        feature_model_count = n_splits - shapValueCount/test_count
        svf_sorted = feature_model_count.sort_values(ascending=False)

        if plotDict['shapSummaryThres'] is not None:
            svf_sorted = svf_sorted[svf_sorted >= shap_threshold]
            maxDisp = len(svf_sorted)-1
        else:
            maxDisp = plotDict['shapMaxDisplay']

        # For effective plotting and sorting purposes, NaNs -> 0s
        shap_vals = shap_vals.fillna(0)

        # Filter the data for the top features
        sortingIdx = svf_sorted.index
        X_train_trans_sorted = X_train_trans_nonmean[sortingIdx]
        shap_vals = shap_vals[sortingIdx]

        # if there are 2 classes, sort by the median difference
        if n_classes == 2:
            # Map the index column to a new 'drug' column using y_vec
            shap_vals['drug'] = y_vec[shap_vals['index']]

            # Identify unique drug names
            drugList = list(shap_vals['drug'].unique())

            drugMedians = pd.DataFrame(index=list(shap_vals.columns[1:-1]), columns=drugList)
            shap_vals.reset_index(inplace=True, drop=True)
            for drug in drugList:
                drugMedians.loc[:, drug] = shap_vals.loc[shap_vals['drug'] == drug].median()
            
            # Find the difference between the medians
            drugMedians['medianDiff'] = abs(drugMedians[drugList[0]] - drugMedians[drugList[1]])

            # Sort by the median difference
            drugMedians_sort = drugMedians.sort_values(by='medianDiff', ascending=False)

            # Resort data by the drug median
            shap_vals = shap_vals[drugMedians_sort.index]
            X_train_trans_sorted = X_train_trans_sorted[drugMedians_sort.index]

        sortSHAP = True
        parenVal = 'medianDiff'

        # explainer(x_data.drop('index', axis=1))
        thingOut = explainers[0](X_train_trans_list[0].drop('index', axis=1))
        shap.plots.bar(thingOut, max_display=12)
        plt.show()
        plt.close()


        if parenVal == 'count':
            # Adjust the feature names to include their counts.
            parenVal = [int(x) for x in svf_sorted.values[1:]]
            featureNames = [f"{feat} ({svf_sorted[feat]})" for feat in list(shap_vals.columns)]
        elif parenVal == 'medianDiff':
            parenVal = drugMedians_sort.medianDiff.round(2)
            featureNames = [f"{feat} ({parenVal[feat]})" for feat in list(shap_vals.columns)]

        if cap_shap_values:
            shap_vals = shap_vals.clip(lower=-max_abs_shap_val, upper=max_abs_shap_val)

        # Find the min and the max of the SHAP values
        shap_values_min = shap_vals.min().min()
        shap_values_max = shap_vals.max().max()

        # Find whether the min or max is larger in magnitude and assgin it to 'maxVal'
        if abs(shap_values_min) > shap_values_max:
            maxVal = abs(shap_values_min)
        else:
            maxVal = shap_values_max
        maxVal = maxVal + 0.01

        # Create a method that accounts for the number of features in the plot to set the figure size. Scale for purposes of other elements.
        scaleFactor = 5
        figW = 1.763
        figH = 0.141 + (0.0913125 * len(shap_vals.columns)) * 1.5
        figSize = (figW * scaleFactor, figH * scaleFactor)
        #, plot_size=figSize
               
        # Plot the SHAP values
        if 'MDMA' in classifyDict['label']:
            firstHalfIdx = int(len(featureNames)/2)
            pltHdl = shap.summary_plot(shap_vals.values[:, 0:firstHalfIdx], X_train_trans_sorted.values[:, 0:firstHalfIdx], feature_names=featureNames[0:firstHalfIdx], sort=sortSHAP, show=False, max_display=maxDisp, cmap='PuOr_r', color_bar_label='cFos Score')
            plt.xlim([-maxVal, maxVal])
            plt.savefig(join(dirDict['outDir_model'], f"SHAP_summary_1_Sym.svg"), format='svg', bbox_inches='tight')
            plt.show()
            plt.close()


            shap.summary_plot(shap_vals.values[:, firstHalfIdx:], X_train_trans_sorted.values[:, firstHalfIdx:], feature_names=featureNames[firstHalfIdx:], sort=sortSHAP, show=False, max_display=maxDisp, cmap='PuOr_r', color_bar_label='cFos Score')
            plt.xlim([-maxVal, maxVal])
            plt.savefig(join(dirDict['outDir_model'], f"SHAP_summary_2_Sym.svg"), format='svg', bbox_inches='tight')
            plt.show()
            plt.close()
        else:
            pltHdl = shap.summary_plot(shap_vals.values, X_train_trans_sorted.values, feature_names=featureNames, sort=sortSHAP, show=False, max_display=maxDisp, cmap='PuOr_r', color_bar_label='cFos Score')
            plt.xticks(fontsize=14)
            plt.xlim([-maxVal, maxVal])
            # Save the plot +/- Titling it.
            # Update the font on the x axis tick labels

            # shap.plots.bar(shap_vals.values, max_display=12)
            # plt.show()

            plt.savefig(join(dirDict['outDir_model'], f"SHAP_summary_Sym.svg"), format='svg', bbox_inches='tight') #, bbox_inches='tight'
            plt.show()
            plt.close()

def plot_shap_force(X_train_trans_list, shap_values_list, baseline_val,
                    y_real_lab, numYDict, plotDict, dirDict):
    import itertools
    import shap
    _saved_rc = {k: plt.rcParams[k] for k in
             ['font.family', 'svg.fonttype', 'font.size',
              'xtick.labelsize', 'ytick.labelsize']}
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.size']         = 5
    plt.rcParams['xtick.labelsize']   = 5
    plt.rcParams['ytick.labelsize']   = 5

    if plotDict['shapForcePlotCount'] == 0:
        return
    forcePlotCount = plotDict['shapForcePlotCount']

    class_count = len(numYDict.keys())
    n_splits    = len(X_train_trans_list)
    test_count  = shap_values_list[0].shape[0]

    labelDict  = {value: key for key, value in numYDict.items()}

    # Overall comparison string for corner annotation
    comparison_str = ' vs '.join(str(k) for k in numYDict.keys())

    
    # Rank CV splits by max absolute SHAP value across all features (no hardcoded regions)
    norm_vals = [
        shap_val_tab.iloc[:, 1:].abs().max().max()
        for shap_val_tab in shap_values_list]
    idx_db = list(np.argsort(norm_vals)[::-1])

    idx_db2     = np.tile(idx_db, [4, 1]).T.reshape(-1)
    testPointSet = [0, 1, 2, 3] * 100
    cvSplitTest  = list(zip(idx_db2, testPointSet))[:forcePlotCount]

    if class_count == 2:
        for cvSplit, testPoint in cvSplitTest:

            y_labels = y_real_lab[cvSplit]
            idStr    = f"CV{cvSplit}_Sample{testPoint}"
            titleStr = f'Test Sample of {y_labels[testPoint]}, {idStr}'

            featCount = shap_values_list[cvSplit].shape[1]
            shapVals  = np.round(
                shap_values_list[cvSplit].iloc[testPoint, 1:featCount].values, 2)
            testVals  = np.round(
                X_train_trans_list[cvSplit].iloc[testPoint, 1:featCount].values, 2)
            featNames = list(shap_values_list[cvSplit].columns[1:featCount])

            shap.plots.force(baseline_val[cvSplit],
                             shap_values=shapVals,
                             features=testVals,
                             feature_names=featNames,
                             out_names=None,
                             link='identity',
                             plot_cmap='RdBu',
                             matplotlib=True,
                             figsize=(14, 2),
                             show=False)

            # Title with sample info
            plt.title(titleStr, y=1.5)

            # ── Condition label: top-right corner ────────────────────────────
            fig = plt.gcf()
            fig.text(
                0.98, 0.98, comparison_str,
                ha='right', va='top',
                fontsize=6, fontweight='bold', color='black',
                transform=fig.transFigure,
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec='none', alpha=0.7)
            )

            plt.savefig(join(dirDict['outDir_model'],
                             f"SHAP_example_{idStr}.svg"),
                        format='svg', bbox_inches='tight')
            plt.show()
            plt.close()
            plt.rcParams.update(_saved_rc)

def plot_feature_scores(clf, featureNames):
    # A function which plots feature scores from a pipeline object if that feature selection method has scores/pvalues.
    # Examples - Fdr, Fwe, and Fwe_BH

    support_ = clf['featureSel'].get_support()
    scores_ = clf['featureSel'].scores_[support_]
    pvalues_ = clf['featureSel'].pvalues_[support_]
    
    # Sort arrays based on pvalues_
    sort_indices = np.argsort(pvalues_)
    sorted_score = scores_[sort_indices]
    sorted_pvalues = pvalues_[sort_indices]
    sorted_featureNames = [featureNames[i] for i in sort_indices]

    # Create the horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot horizontal bars
    bars = ax.barh(range(len(sorted_score)), sorted_score)

    # Add pvalues as text on top of each bar
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{sorted_pvalues[i]:.2e}", ha='center', va='center')

    # Set y-axis ticks and labels
    ax.set_yticks(range(len(sorted_score)))
    ax.set_yticklabels(sorted_featureNames)

    plt.tight_layout()
    plt.show()
    plt.close()

def plot_histogram(data, dirDict):
    plt.title('Feature Count per CV', fontdict={'fontsize': 20})
    plt.hist(data, bins=10, edgecolor='black')
    plt.savefig(os.sep.join([dirDict['outDir_model'], "featureCountHist.svg"]), format='svg', bbox_inches='tight')          
    plt.show()
    plt.close()

def plot_cross_model_AUC(scoreNames, aucScores, aucScrambleScores, dirDict,
                         meanScores=None, accStd=None, oobErrors=None,
                         meanScrambleScores=None, aucPerSplit=None):
    """
    Combined summary figure with three panels:

    Panel A — Mean Precision-Recall AUC (bar) + per-CV-split jitter dots
    Panel B — Mean CV accuracy ± std (bar with error) vs shuffled baseline
    Panel C — OOB error rate (bar; lower = better)

    aucPerSplit : list of lists — [[auc_s1, auc_s2, ...], ...] one inner list
                 per comparison.  When supplied, individual CV-split AUC values
                 are overlaid as semi-transparent jitter dots on Panel A so the
                 reader can see the full distribution, not just the mean.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    has_acc    = (meanScores   is not None)
    has_oob    = (oobErrors    is not None and not all(np.isnan(o) for o in oobErrors))
    has_jitter = (aucPerSplit  is not None and any(len(v) > 0 for v in aucPerSplit))
    n_panels   = 1 + int(has_acc) + int(has_oob)

    n_comp  = len(scoreNames)
    fig_h   = max(1.6, n_comp * 0.38 + 0.6)
    fig_w   = 1.9 * n_panels + 0.3

    dark_col  = np.array([80,  80,  80])  / 255
    light_col = np.array([180, 180, 180]) / 255
    acc_col   = np.array([60,  130, 200]) / 255
    oob_col   = np.array([200, 80,  60])  / 255
    jitter_col= np.array([255, 255, 255]) / 255   # white dots on dark bars

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(1, n_panels, figure=fig, wspace=0.55)

    rng    = np.random.default_rng(0)   # fixed seed → reproducible jitter
    panel  = 0
    yticks = np.arange(n_comp)

    # ── Panel A: Mean AUPRC + per-split jitter ────────────────────────────────
    ax = fig.add_subplot(gs[0, panel])
    ax.barh(yticks, aucScores,         color=dark_col,  label='Data',     height=0.55)
    ax.barh(yticks, aucScrambleScores, color=light_col, label='Shuffled', height=0.55)

    # Mean AUC label inside bar
    fs = plt.rcParams['font.size']
    for i, v in enumerate(aucScores):
        ax.text(max(v - 0.01, 0.01), i, f'{v:.0%}',
                ha='right', va='center', color='white', fontsize=fs)
    for i, v in enumerate(aucScrambleScores):
        ax.text(max(v - 0.01, 0.01), i, f'{v:.0%}',
                ha='right', va='center', fontsize=fs)

    # Per-split jitter overlay
    if has_jitter:
        for i, split_vals in enumerate(aucPerSplit):
            if not split_vals:
                continue
            arr    = np.array(split_vals)
            jitter = rng.uniform(-0.18, 0.18, size=len(arr))   # vertical spread
            ax.scatter(arr, i + jitter,
                       s=6, color='white', alpha=0.65,
                       linewidths=0, zorder=3,
                       label='_nolegend_')
            # Thin vertical line at the mean for clarity
            ax.axvline(np.mean(arr), ymin=(i + 0.1) / n_comp,
                       ymax=(i + 0.9) / n_comp,
                       color='white', lw=0.6, alpha=0.5, zorder=4)

    ax.set_yticks(yticks)
    ax.set_yticklabels(scoreNames)
    ax.set_xlabel('Mean PR-AUC')
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=fs - 1, frameon=False, loc='lower right')
    panel += 1

    # ── Panel B: Mean CV Accuracy ± std ──────────────────────────────────────
    if has_acc:
        ax = fig.add_subplot(gs[0, panel])
        xerr = accStd if accStd is not None else None
        ax.barh(yticks, meanScores, color=acc_col, height=0.55,
                xerr=xerr, error_kw=dict(ecolor='black', lw=0.8, capsize=2),
                label='Data')
        if meanScrambleScores is not None:
            ax.barh(yticks, meanScrambleScores, color=light_col, height=0.55,
                    label='Shuffled', alpha=0.75)
        for i, v in enumerate(meanScores):
            ax.text(max(v - 0.01, 0.01), i, f'{v:.0%}',
                    ha='right', va='center', color='white', fontsize=fs)
        ax.axvline(0.5, color='grey', linestyle='--', linewidth=0.6, label='Chance')
        ax.set_yticks(yticks)
        ax.set_yticklabels([''] * n_comp)
        ax.set_xlabel('Mean CV Accuracy')
        ax.set_xlim(0, 1.05)
        ax.legend(fontsize=fs - 1, frameon=False, loc='lower right')
        panel += 1

    # ── Panel C: OOB Error Rate ───────────────────────────────────────────────
    if has_oob:
        ax = fig.add_subplot(gs[0, panel])
        oob_vals = [o if not np.isnan(o) else 0 for o in oobErrors]
        ax.barh(yticks, oob_vals, color=oob_col, height=0.55)
        for i, v in enumerate(oob_vals):
            if not np.isnan(oobErrors[i]):
                ax.text(v + 0.01, i, f'{v:.0%}',
                        ha='left', va='center', fontsize=fs)
        ax.set_yticks(yticks)
        ax.set_yticklabels([''] * n_comp)
        ax.set_xlabel('OOB Error Rate')
        ax.set_xlim(0, 1.0)
        ax.text(0.98, -0.12, '← lower = better', transform=ax.transAxes,
                ha='right', va='top', fontsize=fs - 1, color='grey', style='italic')

    plt.savefig(os.sep.join([dirDict['crossComp_figDir'], 'Summary_AUC_Accuracy_OOB.svg']),
                format='svg', bbox_inches='tight')
    plt.show()
    plt.close()


def plot_cross_model_Accuracy(scoreNames, meanScores, meanScrambleScores, colorsList, saveDir):
    """Legacy standalone accuracy bar plot (kept for backward compatibility)."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 5))
    plt.barh(scoreNames, meanScores,        label='Data',     color=colorsList[0])
    plt.barh(scoreNames, meanScrambleScores, label='Shuffled', color=colorsList[1])
    plt.legend()
    plt.xlabel('Score')
    plt.ylabel('Classification')
    plt.title('Mean Accuracy across cross-validation')
    for i, v in enumerate(meanScores):
        plt.text(v - 0.01, i, f'{v:.0%}', ha='right', va='center', weight='bold', fontsize=10)
    for i, v in enumerate(meanScrambleScores):
        plt.text(v - 0.01, i, f'{v:.0%}', ha='right', va='center', weight='bold', fontsize=10)
    plt.savefig(os.sep.join([saveDir, 'MeanAcc_barplot.svg']), format='svg', bbox_inches='tight')
    plt.show()

def find_max_index(shap_values_list, regionSet):
    max_value = 0  # Initialize max_value to store the maximum value
    max_index = None  # Initialize max_index to store the corresponding index in shap_values_list
    max_row_index = None  # Initialize max_row_index to store the index of the row with the maximum values

    for idx, shap_val_tab in enumerate(shap_values_list):
        # Check if all elements in regionSet are present in the DataFrame
        if not all(region in shap_val_tab.columns for region in regionSet):
            # Skip to the next iteration if not all elements are present
            continue

        # Calculate the normalized values for 'VISpm' and 'LH'
        normalized_values = shap_val_tab.loc[:, regionSet].abs() / np.abs()

        # Calculate the sum of 'VISpm' and 'LH' for each row
        row_sums = normalized_values.sum(axis=1)

        # Find the index with the maximum sum
        current_max_value = row_sums.max()
        if current_max_value > max_value:
            max_value = current_max_value
            max_index = idx
            max_row_index = row_sums.idxmax()

    return max_index, max_row_index

def find_max_min_index(shap_values_list, regionSet):
    max_value = 0  # Initialize max_value to store the maximum value
    min_value = float('inf')  # Initialize min_value to store the minimum value
    max_index = None  # Initialize max_index to store the corresponding index in shap_values_list
    min_index = None  # Initialize min_index to store the corresponding index in shap_values_list
    max_row_index = None  # Initialize max_row_index to store the index of the row with the maximum values
    min_row_index = None  # Initialize min_row_index to store the index of the row with the minimum values

    idxList = []

    for idx, shap_val_tab in enumerate(shap_values_list):
        # Check if all elements in regionSet are present in the DataFrame
        if not all(region in shap_val_tab.columns for region in regionSet):
            # Skip to the next iteration if not all elements are present
            continue

        # Calculate the normalized values for the specified regions in regionSet
        normalized_values = shap_val_tab.loc[:, regionSet] / shap_val_tab.iloc[:, 1:].max().max()

        # Calculate the sum of the specified regions for each row
        row_sums = normalized_values.sum(axis=1)

        # Find the index with the maximum sum
        current_max_value = row_sums.max()
        if current_max_value > max_value:
            max_value = current_max_value
            max_index = idx
            max_row_index = row_sums.idxmax()
            idxList.append(idx)

        # Find the index with the minimum sum
        current_min_value = row_sums.min()
        if current_min_value < min_value:
            min_value = current_min_value
            min_index = idx
            min_row_index = row_sums.idxmin()

    max_index = idxList[-1]

    return max_index, max_row_index, min_index, min_row_index

def plot_featureCount_violin(scoreNames, featureLists, dirDict):
    import seaborn as sns
    import pandas as pd

    colorsList = [[82, 211, 216], [56, 135, 190]]
    colorsList = np.array(colorsList)/256

    # Your list of lists (sublists with numbers)
    data = [[len(sublist) for sublist in inner_list] for inner_list in featureLists]

    # Reverse the order of the data and scoreNames
    data = data[::-1]
    scoreNames = scoreNames[::-1]

    df = pd.melt(pd.DataFrame(data, index=scoreNames).T, var_name='Category', value_name='Values')

    # Create horizontally oriented violin plot
    plt.figure(figsize=(2, 1.95))  # Adjust the width and height as needed

    ax = sns.violinplot(x='Values', y='Category', data=df, orient='h', color=colorsList[0], linewidth=0.5)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    ax.spines['left'].set_linewidth(0.75)
    ax.spines['bottom'].set_linewidth(0.75)
    ax.xaxis.set_tick_params(length=2, width=1)
    ax.yaxis.set_tick_params(length=3, width=1)

    # Set plot labels and title
    plt.xlabel('Brain regions used by classifier')
    plt.ylabel('')

    plt.savefig(os.sep.join([dirDict['crossComp_figDir'], "RegionCountPerSplit_violin.svg"]), format='svg', bbox_inches='tight')     

    # Show the plot
    plt.show()
    plt.close()

def plot_similarity_matrix(scoreNames, featureLists, filterByFreq, dirDict):
    from matplotlib import cm

    modelCount = len(featureLists)
    # regionDict = dict(Counter(featureLists[0]))
    # labels, counts = list(regionDict.keys()), list(regionDict.values())

    # Initialize a grid
    grid = [[0 for _ in range(modelCount)] for _ in range(modelCount)]

    # Flatten every list
    featureListFlat = [[element for item in subList for element in item] for subList in featureLists]

    jacSim = False 

    # compare the mean distances across items of the list
    for idx_a, listA in enumerate(featureListFlat):
        for idx_b, listB in enumerate(featureListFlat):
            
            if jacSim:
                # Jaccard Sim
                grid[idx_a][idx_b] = hf.weighted_jaccard_similarity(listA, listB, 75)
            else:
                # Overlap count
                _, _, intersection = hf.overlapCounter(listA, listB, filterByFreq)
                grid[idx_a][idx_b] = len(intersection)

    # Plot the grid
    fig, ax = plt.subplots(figsize=(7,7))
    im = sns.heatmap(grid, cmap='Blues', annot=True, fmt='.0f', ax=ax, yticklabels=scoreNames, xticklabels=scoreNames, annot_kws={'size': 15})

    # Remove the colorbar
    cbar = ax.collections[0].colorbar
    cbar.remove()

    # Set font size for x-axis ticks and labels
    ax.tick_params(axis='x', labelsize=12)

    # Set font size for y-axis ticks and labels
    ax.tick_params(axis='y', labelsize=12, rotation=0)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Cycle through sns heatmap annotations and remove those that are equal to 0
    for text in ax.texts:
        if int(text.get_text()) == 0:
            text.set_text("")

    # plt.title(titleStr, fontdict={'fontsize': 18})
    plt.savefig(os.sep.join([dirDict['crossComp_figDir'], "MeanSimilarity_heatmap.svg"]), format='svg', bbox_inches='tight')     
    plt.show()
    plt.close()

def plot_featureOverlap_VennDiagram(scoreNames, featureLists, filterByFreq, dirDict):
    wrapper = textwrap.TextWrapper(width=12, break_on_hyphens=False)  # Adjust width as needed

    # Flatten every list
    featureListFlat = [[element for item in subList for element in item] for subList in featureLists]

    # Sample data
    for idx1, list1 in enumerate(featureListFlat):
        for idx2, list2 in enumerate(featureListFlat):

            # Skip the same list
            if idx1 == idx2:
                continue

            # Filter out features in each counter whose count is not above it.
            only_list1, only_list2, intersection = hf.overlapCounter(list1, list2, filterByFreq)

            # Skip if there is no overlap
            if intersection == []:
                continue

            # Create a Venn diagram
            venn_diagram = venn2(subsets=(len(only_list1), len(only_list2), len(intersection)/2),
                                set_labels=(scoreNames[idx1], scoreNames[idx2]))


            venn_labels = {'100': only_list1, '010': only_list2, '110': intersection}
            for idx, (labId, labels) in enumerate(venn_labels.items()):
                wrapped_labels = wrapper.fill(text='  '.join(labels))
                venn_diagram.get_label_by_id(labId).set_text(wrapped_labels)
                venn_diagram.get_label_by_id(labId).set_fontsize(8)  # Adjust font size if needed

            # # Customize the size of the Venn diagram
            # plt.gcf().set_size_inches(8, 8)
            figName = f'VD_{scoreNames[idx1]} and {scoreNames[idx2]}'
            figName = figName.replace('/', '+')
            figName = figName.replace(' ', '_')
            plt.savefig(os.sep.join([dirDict['crossComp_figDir'], f"{figName}.svg"]), format='svg', bbox_inches='tight')     

            # Display the plot
            plt.show()
            plt.close()

def plot_featureHeatMap(df_raw, scoreNames, featureLists, filterByFreq, dirDict):
    # Current Mode: Create plot with colorbar, then without, and grab the svg item and place it in the second plot to ensure even spacing
    # Creates the heatmap for the data

    scoreNames = scoreNames[::-1]
    featureLists = featureLists[::-1]

    # Set variables
    dataFeature = 'abbreviation'
    plt.rcParams['font.size'] = 6
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6

    sys.path.append('../dependencies/')

    # Create a sorted structure from the data for scaffolding the desired heatmap
    brainAreaColorDict = hf.create_color_dict(dictType='brainArea', rgbSwitch=0)
    regionArea = hf.create_region_to_area_dict(df_raw, 'abbreviation')
    regionArea['Region_Color'] = regionArea['Brain_Area'].map(brainAreaColorDict)

    value_col = 'count' if 'count' in df_raw.columns else 'density_norm'
    df_Tilted = df_raw.pivot(index='abbreviation', columns='dataset', values=value_col)
    df_Tilted = df_Tilted.reindex(regionArea['abbreviation'].tolist(), axis=0)

    featureListFlat = [[element for item in subList for element in item] for subList in featureLists]

    # Process the data from above
    featureListDicts = [hf.listToCounterFilt(x, filterByFreq=0) for x in featureListFlat]
    featureListArray = [list(x.keys()) for x in featureListDicts]

    # Add in columns for each of the actual comparisons.
    df_frame = df_Tilted.reindex(columns=scoreNames)
    df_frame.fillna(0, inplace=True)
        
    for idx, (comp, featureList) in enumerate(zip(df_frame.columns, featureListDicts)):
        for regionName in featureList.keys():
            df_frame.loc[regionName, comp] = featureList[regionName]
            
    # Remove any rows which are not above threshold
    df_plot = df_frame[df_frame.sum(axis=1) >= filterByFreq]
        
    # Remove the abbreviations from regionArea not represented in df_plot, Filter the regionArea for 'Cortex' and 'Thalamus'
    regionArea = hf.create_region_to_area_dict(df_raw, dataFeature)
    regionArea = regionArea[regionArea['abbreviation'].isin(df_plot.index)]
    regionArea = regionArea[regionArea['Brain_Area'].isin(['Cortex', 'Thalamus'])]

    # Sort the data to be combined per larger area
    df_plot = df_plot.loc[regionArea[dataFeature]]
    modelCount = len(df_plot.columns)

    # merge df_plot and regionArea, moving the Brain_Area_Idx and Brain_Area columns to df_plot
    df_plot_combo = df_plot.merge(regionArea, left_index=True, right_on=dataFeature)

    # Cycle through the df_plot_combo's distinct Brain_Area_Idx, and resort data by row sums
    newIdx = []
    for idx in regionArea.Brain_Area_Idx.unique():
        # Identify which regions have the same Brain_Area_Idx
        df_seg = df_plot_combo[df_plot_combo.Brain_Area_Idx == idx]
        sorted_seg_idx = df_seg.iloc[:, 0:modelCount].sum(axis=1).sort_values(ascending=False).index

        # Append to list
        newIdx = newIdx + list(sorted_seg_idx)

    # Resort the data
    df_plot = df_plot_combo.reindex(newIdx, axis=0)
    df_plot = df_plot.set_index('abbreviation')
    df_plot = df_plot.drop(columns=['Brain_Area_Idx'])

    # Drop the columns which do not include the string 'PSI'
    df_plot = df_plot.loc[:, df_plot.columns.str.contains('PSI') | (df_plot.columns == 'Brain_Area')]

    # Plotting variables
    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(False)
    formatter.set_powerlimits((-2, 2))

    colorbar = [False, True]
    axes = []
    regionSet = ['Cortex', 'Thalamus']

    for idx, regionName in enumerate(regionSet):

        # Slice and modify previous structures to create segment
        df_plot_seg = df_plot[df_plot.Brain_Area == regionName].drop(columns=['Brain_Area'])

        # Sort by highest sum of row
        df_plot_seg = df_plot_seg.loc[df_plot_seg.sum(axis=1).sort_values(ascending=False).index]

        matrix = df_plot_seg.values

        xticklabels = df_plot_seg.columns.values.tolist()
        yticklabels = df_plot_seg.index.values.tolist()

        figwidth = len(xticklabels)*0.1433
        figheight = len(yticklabels)*0.1433

        f = plt.figure(figsize=(figwidth, figheight))  # Adjust the width and height as needed
        ax = f.add_subplot(111)
        axes.append(ax)

        heatmap = sns.heatmap(matrix, cmap='crest', ax=axes[idx] , fmt='.2f', cbar = False, square=True, yticklabels=yticklabels, xticklabels=xticklabels, cbar_kws={"format": formatter}, center=0, linewidths=0.5, linecolor='black')
        axes[idx].tick_params(left=True, bottom=True, width=0.5, length=2)

        # Rotate the xticklabels 45 degrees
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        # if cbs:
        #     cbar = heatmap.figure.colorbar(heatmap.collections[0], ax=axes[idx], location='right', use_gridspec=True, pad=0.05)
        #     cbar.set_label('Feature Count', rotation=270, labelpad=5)
        #     cbar.ax.yaxis.set_major_formatter(formatter)

        plt.savefig(os.sep.join([dirDict['crossComp_figDir'], f"FeatureCountHeatmap_{regionName}.svg"]), format='svg', bbox_inches='tight')
        plt.show()
        plt.close()

def sortShap(shap_values_list, regionSet):
    max_value = 0  # Initialize max_value to store the maximum value
    min_value = float('inf')  # Initialize min_value to store the minimum value
    idxList = []
    databaseIdx = pd.DataFrame(index=np.arange(0, len(shap_values_list)), columns=['normVal'])

    for idx, shap_val_tab in enumerate(shap_values_list):
        # Check if all elements in regionSet are present in the DataFrame
        if not all(region in shap_val_tab.columns for region in regionSet):
            # Skip to the next iteration if not all elements are present
            continue

        # Calculate the normalized values for the specified regions in regionSet
        normalized_values = shap_val_tab.loc[:, regionSet] / shap_val_tab.iloc[:, 1:].max().max()

        # Calculate the sum of the specified regions for each row
        row_sums = normalized_values.sum(axis=1)

        databaseIdx.loc[idx, 'normVal'] = row_sums.max()
        # Find the index with the maximum sum
        current_max_value = row_sums.max()
        if current_max_value > max_value:
            max_value = current_max_value
            max_index = idx
            max_row_index = row_sums.idxmax()
            idxList.append(idx)

        # Find the index with the minimum sum
        current_min_value = row_sums.min()
        if current_min_value < min_value:
            min_value = current_min_value

    return databaseIdx

def genereate_cFos_gene_corr_plots(geneDict, geneColorDict, setNames, regionSet, plotNameDict, dirDict):

    unique_genes = list(set(hf.flatten(geneDict.values())))
    genePlotColorPalette = sns.color_palette("Spectral", as_cmap=True, n_colors=len(unique_genes))

    rows = len(geneDict)
    cols = np.max(list({key: len(value) for key, value in geneDict.items()}.values()))

    for set_i, set_name in enumerate(regionSet):

        fig, axs = plt.subplots(rows, cols, figsize=(10, len(geneDict.keys())*1.67))

        for drug_i, drug in enumerate(geneDict.keys()):

            genePlotList = geneDict[drug]
            
            for geneSet_i, genePlotList_data in enumerate(genePlotList):
            
                #plot the distribution for all the gene correlations
                plt.figure(figsize=(2,1))
                sns.set(style="ticks")
                drug_data = pd.read_pickle(os.path.join(dirDict['geneCorrDir'], f'{drug}_{set_name}_corr_db.h5'))
                axHand = axs[drug_i, geneSet_i]

                ax = sns.histplot(data=drug_data , x = drug + " correlation", element = 'step', fill = False, color='grey', ax=axHand) #, lw=7
                sns.despine()

                trans = ax.get_xaxis_transform()

                # Plot the individual
                genes_of_interest = drug_data[drug_data['gene'].isin(genePlotList_data)]
                corrData = list(genes_of_interest[drug + ' correlation'])

                genes_of_interest['colorInd'] = [geneColorDict[drug] for drug in genes_of_interest.gene]
                genes_of_interest = genes_of_interest.sort_values(drug + ' correlation')

                # Generate spots for the text above the plots to go. Move text over if it is overlapping with text to the left
            
                textXVals = np.array(genes_of_interest[drug + ' correlation'])
                minDist = .03

                xLimits = ax.get_xlim()
                xLimLeftLabel = textXVals[0] 
                xLimLeftLabel = xLimits[0]
                if abs(textXVals[0]-textXVals[-1]) < 0.2:
                    xLimRightLabel = textXVals[-1] + .2
                else:
                    xLimRightLabel = textXVals[-1] #xLimits[0]+(xLimits[1]*.7)

                geneCount = len(genePlotList_data)
                textXAxes = np.linspace(xLimLeftLabel, xLimLeftLabel+(geneCount*0.1), num:=geneCount)
                # textXAxes = np.linspace(xLimLeftLabel, xLimLeftLabel+(geneCount*0.05), num:=geneCount)

                plt.sca(axHand)
                plt.xlim(-0.85, 0.85)
                plt.ylim(0, 1050)
                y_pos = 0.90

                # For plotting, flip the order of genes.
                genes_of_interest = genes_of_interest.sort_values('percentile', ascending=False)

                config.setup_mRNA_corr_settings()

                # Iterate across genes of interest to plot lines and text
                for gene_i, gene in enumerate(genes_of_interest.gene):
                    
                    geneData = genes_of_interest[genes_of_interest.gene == gene]
                    corrVal = float(geneData[drug + ' correlation'])
                    prcVal = round(float(geneData['percentile'])*100)
                    lineCol = genePlotColorPalette(geneData.colorInd)[0]

                    # Draw lines and text
                    plt.axvline(x=corrVal, color=lineCol, linestyle='--') #, lw=10
                    plt.text(-0.83, y_pos, f'{gene} ({str(prcVal)}%)', transform=trans, color=lineCol, fontsize=8) #, rotation=45
                    
                    y_pos -= 0.15

                # plotLineWidth = 5

                if drug_i == len(geneDict.keys())-1:
                    plt.xlabel('Correlation', labelpad=1)
                else:
                    plt.xlabel('')

                if geneSet_i == 0:
                    plt.ylabel('Number of genes', labelpad=.5)
                else:
                    plt.ylabel('')

                plotTitle = f'{plotNameDict[drug]} vs {setNames[geneSet_i]} ({set_name})'
                plt.title(plotTitle, loc='center', fontsize=8, pad=10) #pad=300
                plt.subplots_adjust(hspace=0.6, wspace=0.2) 

        plt.savefig(os.path.join(dirDict['outDir'], f'{set_name}_Drug_vs_Receptors.svg'), bbox_inches='tight')

        plt.show()
        plt.clf()
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# plot_timepoint_PRAUC_summary
# ─────────────────────────────────────────────────────────────────────────────
def plot_timepoint_PRAUC_summary(dirDict,
                                  stressor_list=None,
                                  timepoint_order=None,
                                  show_mean_marker=True,
                                  show_per_split_jitter=True,
                                  figsize=(3.5, 3.0),
                                  save_name='timepoint_PRAUC_summary'):
    """
    Summary line plot of per-timepoint PR-AUC for the stressor-wise timepoint
    classifiers (classif_tp_{stressor}).

    Loads from cached scoreDict_Real.pkl — no rerun needed.

    Parameters
    ----------
    dirDict : dict
        Must contain 'outDir' (where classif_tp_* folders live) and
        'crossComp_figDir' for saving.
    stressor_list : list, optional
        Stressors to include.  Defaults to ['FS', 'FSW', 'RS', 'TS', 'Ctrl'].
    timepoint_order : list, optional
        Canonical timepoint order.  Defaults to ['Acute','7D','14D','21D'].
    show_mean_marker : bool
        Overlay a star marker at the per-stressor mean AUC (horizontal position
        = rightmost available timepoint + small offset).
    show_per_split_jitter : bool
        Draw individual CV-split AUCs as small semi-transparent dots at each
        timepoint (jittered x).  Requires 'PerSplit_perClass' stored in the
        scoreDict; falls back gracefully when absent.
    figsize : tuple
        Figure size in inches.
    save_name : str
        Output file stem (SVG + PNG saved).
    """
    import pickle as pkl
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import helperFunctions as hf

    if stressor_list is None:
        stressor_list = ['FS', 'FSW', 'RS', 'TS', 'Ctrl']
    if timepoint_order is None:
        timepoint_order = ['Acute', '7D', '14D', '21D']

    # Numeric x-positions that reflect actual time gaps
    tp_x = {'Acute': 0, '7D': 7, '14D': 14, '21D': 21}

    colorDict = hf.create_color_dict(dictType='stressor', rgbSwitch=0,
                                      alpha_value=1, scaleVal=False)

    fig, ax = plt.subplots(figsize=figsize)

    found_any = False

    for stressor in stressor_list:
        tp_dir = os.path.join(dirDict['outDir'], f'classif_tp_{stressor}')
        if not os.path.isdir(tp_dir):
            print(f"  [PRAUC summary] No folder found: {tp_dir} — skipping {stressor}")
            continue

        # Walk to find scoreDict_Real.pkl
        score_pkl = None
        for root, _, files in os.walk(tp_dir):
            if 'scoreDict_Real.pkl' in files:
                score_pkl = os.path.join(root, 'scoreDict_Real.pkl')
                break  # take the first match

        if score_pkl is None:
            print(f"  [PRAUC summary] No scoreDict_Real.pkl in {tp_dir} — skipping {stressor}")
            continue

        with open(score_pkl, 'rb') as f:
            sd = pkl.load(f)

        auc_d = sd.get('auc', {})
        color = colorDict.get(stressor, '#888888')

        # ── Collect per-timepoint AUC ───────────────────────────────────────
        x_vals, y_vals = [], []
        for tp in timepoint_order:
            if tp in auc_d:
                x_vals.append(tp_x[tp])
                y_vals.append(float(auc_d[tp]))

        if len(x_vals) == 0:
            print(f"  [PRAUC summary] No timepoint AUC keys in auc_dict for {stressor} — skipping")
            continue

        found_any = True

        # ── Main line ───────────────────────────────────────────────────────
        ax.plot(x_vals, y_vals,
                color=color, lw=1.8, marker='o', markersize=4,
                label=stressor, zorder=3)

        # ── Mean AUC star marker ─────────────────────────────────────────────
        if show_mean_marker and 'Mean' in auc_d:
            mean_val = float(auc_d['Mean'])
            # Place at a small x-offset past the last timepoint
            x_star = max(x_vals) + 1.5
            ax.plot(x_star, mean_val,
                    marker='*', markersize=8, color=color,
                    linestyle='none', zorder=4, alpha=0.9)
            # Dashed connector from last data point to star
            ax.plot([max(x_vals), x_star], [y_vals[-1], mean_val],
                    color=color, lw=0.8, linestyle=':', zorder=2, alpha=0.6)

        # ── Per-split jitter ─────────────────────────────────────────────────
        # 'auc_per_split' is the mean-across-classes per split (list of floats).
        # We draw them lightly at the rightmost timepoint as a proxy for spread.
        if show_per_split_jitter:
            per_split = sd.get('auc_per_split', [])
            if len(per_split) > 0:
                rng = np.random.default_rng(42)
                jit = rng.uniform(-0.6, 0.6, size=len(per_split))
                x_jit = max(x_vals) + jit
                ax.scatter(x_jit, per_split,
                           color=color, s=6, alpha=0.35, zorder=2,
                           linewidths=0)

    if not found_any:
        print("  [PRAUC summary] No data loaded — check that classif_tp_* folders exist "
              "and saveLoadswitch was True during the run.")
        plt.close()
        return

    # ── Axes formatting ──────────────────────────────────────────────────────
    tick_x   = [tp_x[tp] for tp in timepoint_order]
    tick_lab = timepoint_order

    # If mean markers are shown, add an extra x-tick label
    if show_mean_marker:
        extra_x = max(tick_x) + 1.5
        tick_x_all  = tick_x  + [extra_x]
        tick_lab_all = tick_lab + ['Mean\nAUC']
    else:
        tick_x_all  = tick_x
        tick_lab_all = tick_lab

    ax.set_xticks(tick_x_all)
    ax.set_xticklabels(tick_lab_all, fontsize=7)
    ax.set_xlabel('Time', fontsize=8)
    ax.set_ylabel('PR-AUC', fontsize=8)
    ax.tick_params(axis='both', labelsize=7)
    ax.set_ylim([0, 1.05])
    ax.axhline(1/len(timepoint_order), color='grey', lw=0.8, linestyle='--',
               alpha=0.5, zorder=1, label='Chance')

    # Legend — stressors only (no star marker in legend, it's self-explanatory)
    handles, labels = ax.get_legend_handles_labels()
    # keep only stressor entries (not 'Chance')
    h_s = [(h, l) for h, l in zip(handles, labels) if l != 'Chance']
    chance_entry = [(h, l) for h, l in zip(handles, labels) if l == 'Chance']
    h_all = h_s + chance_entry
    ax.legend([h for h, l in h_all], [l for h, l in h_all],
              fontsize=6, frameon=False, loc='upper right',
              handlelength=1.5, labelspacing=0.3)

    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────────
    fig_dir = dirDict.get('crossComp_figDir', dirDict['outDir'])
    for ext in ('svg', 'png'):
        plt.savefig(os.path.join(fig_dir, f'{save_name}.{ext}'),
                    format=ext, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    print(f"  Saved: {save_name}.svg / .png  →  {fig_dir}")

# ─────────────────────────────────────────────────────────────────────────────
# plot_timepoint_multiclass_PRAUC
# ─────────────────────────────────────────────────────────────────────────────
def plot_timepoint_multiclass_PRAUC(
        dirDict,
        stressor_list=None,
        timepoint_order=None,
        show_mean_line=True,
        show_per_split_jitter=True,
        figsize=(3.5, 3.0),
        save_name='timepoint_multiclass_PRAUC'):
    """
    Line plot of per-stressor-class PR-AUC from the per-timepoint multiclass
    classifiers (classif_Acute, classif_7D, classif_14D, classif_21D).

    At each timepoint a multiclass 'stressor' classifier is run on all animals
    sacrificed at that timepoint.  The resulting scoreDict_Real.pkl contains
    auc_dict with one PR-AUC value per stressor class.  This function
    connects those values across timepoints — one coloured line per stressor.

    Loads purely from cached scoreDict_Real.pkl — no rerun needed.

    Parameters
    ----------
    dirDict : dict
        Must contain 'outDir' (where classif_Acute / classif_7D / … live).
    stressor_list : list, optional
        Stressors to trace.  Defaults to ['Ctrl','FS','FSW','RS','TS'].
    timepoint_order : list, optional
        Canonical order.  Defaults to ['Acute','7D','14D','21D'].
    show_mean_line : bool
        Overlay the per-timepoint macro-mean AUC as a dashed black line.
    show_per_split_jitter : bool
        Draw individual CV-split mean AUCs as small semi-transparent dots.
    figsize : tuple
    save_name : str
    """
    import os
    import pickle as pkl
    import numpy as np
    import matplotlib.pyplot as plt
    import helperFunctions as hf

    if stressor_list is None:
        stressor_list = ['Ctrl', 'FS', 'FSW', 'RS', 'TS']
    if timepoint_order is None:
        timepoint_order = ['Acute', '7D', '14D', '21D']

    # Numeric x-positions reflecting real time gaps
    tp_x = {'Acute': 0, '7D': 7, '14D': 14, '21D': 21}

    colorDict = hf.create_color_dict(dictType='stressor', rgbSwitch=0,
                                      alpha_value=1, scaleVal=False)

    # ── Load one scoreDict per timepoint ────────────────────────────────────
    # Structure: {timepoint: {'auc': {...}, 'auc_per_split': [...], ...}}
    tp_data = {}
    for tp in timepoint_order:
        tp_dir = os.path.join(dirDict['outDir'], f'classif_{tp}')
        if not os.path.isdir(tp_dir):
            print(f"  [multiclass PRAUC] folder not found: {tp_dir} — skipping {tp}")
            continue

        score_pkl = None
        for root, _, files in os.walk(tp_dir):
            if 'scoreDict_Real.pkl' in files:
                score_pkl = os.path.join(root, 'scoreDict_Real.pkl')
                break

        if score_pkl is None:
            print(f"  [multiclass PRAUC] no scoreDict_Real.pkl in {tp_dir} — skipping {tp}")
            continue

        with open(score_pkl, 'rb') as f:
            sd = pkl.load(f)
        tp_data[tp] = sd
        print(f"  Loaded {tp}: classes = {list(sd.get('auc', {}).keys())}")

    if not tp_data:
        print("  [multiclass PRAUC] No data found — check classif_{tp} folders exist.")
        return

    # ── Build per-stressor traces ────────────────────────────────────────────
    # stressor_traces[stressor] = list of (x, auc_val) tuples
    stressor_traces = {s: [] for s in stressor_list}
    mean_trace   = []   # (x, mean_auc)
    split_points = []   # (x, auc_val) for per-split jitter

    for tp in timepoint_order:
        if tp not in tp_data:
            continue
        sd    = tp_data[tp]
        auc_d = sd.get('auc', {})
        x     = tp_x[tp]

        for stressor in stressor_list:
            if stressor in auc_d:
                stressor_traces[stressor].append((x, float(auc_d[stressor])))

        if 'Mean' in auc_d:
            mean_trace.append((x, float(auc_d['Mean'])))

        if show_per_split_jitter:
            per_split = sd.get('auc_per_split', [])
            for val in per_split:
                split_points.append((x, float(val)))

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    # Per-split jitter (draw first so lines sit on top)
    if show_per_split_jitter and split_points:
        rng = np.random.default_rng(42)
        xs  = np.array([p[0] for p in split_points], dtype=float)
        ys  = np.array([p[1] for p in split_points])
        xs += rng.uniform(-0.4, 0.4, size=len(xs))
        ax.scatter(xs, ys, color='grey', s=5, alpha=0.25, zorder=1,
                   linewidths=0, label='_nolegend_')

    # Per-stressor lines
    plotted_any = False
    for stressor in stressor_list:
        pts = stressor_traces[stressor]
        if len(pts) == 0:
            continue
        pts  = sorted(pts, key=lambda p: p[0])
        xs   = [p[0] for p in pts]
        ys   = [p[1] for p in pts]
        col  = colorDict.get(stressor, '#888888')

        ax.plot(xs, ys, color=col, lw=1.8, marker='o', markersize=4,
                label=stressor, zorder=3)
        plotted_any = True

    if not plotted_any:
        print("  [multiclass PRAUC] No stressor AUC values found in loaded scoreDicts.")
        plt.close()
        return

    # Mean AUC dashed line
    if show_mean_line and mean_trace:
        mean_trace = sorted(mean_trace, key=lambda p: p[0])
        mx = [p[0] for p in mean_trace]
        my = [p[1] for p in mean_trace]
        ax.plot(mx, my, color='black', lw=1.4, linestyle='--',
                marker='s', markersize=3.5, label='Mean', zorder=4)

    # Chance line — 1 / n_classes at each timepoint
    # n_classes can vary per timepoint (FS missing at 21D → 4 instead of 5)
    for tp in timepoint_order:
        if tp not in tp_data:
            continue
        auc_d    = tp_data[tp].get('auc', {})
        n_cls    = tp_data[tp].get('n_classes',
                   sum(1 for k in auc_d if k not in ('Mean', 'PerSplit')))
        if n_cls > 0:
            chance = 1.0 / n_cls
            x      = tp_x[tp]
            ax.plot([x - 0.8, x + 0.8], [chance, chance],
                    color='grey', lw=0.8, linestyle=':', alpha=0.6, zorder=1)

    # ── Axes ─────────────────────────────────────────────────────────────────
    tick_positions = [tp_x[tp] for tp in timepoint_order]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(timepoint_order, fontsize=7)
    ax.set_xlabel('Timepoint', fontsize=8)
    ax.set_ylabel('PR-AUC', fontsize=8)
    ax.set_ylim([0, 1.05])
    ax.tick_params(axis='both', labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)

    # Add a small note about the dotted chance line
    ax.text(0.02, 0.03, 'dotted = chance (1/n classes)',
            transform=ax.transAxes, fontsize=5, color='grey', va='bottom')

    ax.legend(fontsize=6, frameon=False, loc='upper right',
              handlelength=1.5, labelspacing=0.3)

    plt.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────────
    fig_dir = dirDict.get('crossComp_figDir', dirDict.get('outDir', '.'))
    for ext in ('svg', 'png'):
        plt.savefig(os.path.join(fig_dir, f'{save_name}.{ext}'),
                    format=ext, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    print(f"  Saved {save_name}.svg / .png  →  {fig_dir}")

# ─────────────────────────────────────────────────────────────────────────────
# plot_boruta_regions_timepoint
# ─────────────────────────────────────────────────────────────────────────────
def plot_boruta_regions_timepoint(
        csv_path,
        pkl_path=None,
        txt_path=None,
        dirDict=None,
        cv_count=None,
        min_count=1,
        group_by_area=True,
        show_fraction=True,
        figsize=None,
        save_name='boruta_regions_timepoint_pooled'):
    """
    Horizontal bar chart of Boruta-selected brain regions for the pooled
    timepoint classifier (classif_tp_pooled).  Loads purely from cache —
    no classifier rerun needed.

    Exactly ONE of `pkl_path` or `txt_path` must be supplied.

    Parameters
    ----------
    csv_path : str
        Path to merged_wide_with_acro.csv  (used to build acronym → brain-area
        mapping via the same rule-based assign_structure logic).
    pkl_path : str or None
        Path to the Real_outdata.pkl cache file (full CV output).
        selected_features_list is at index 10 in the unpickled list.
    txt_path : str or None
        Path to featureSelReadout.txt  (alternative if pkl is unavailable).
    dirDict : dict or None
        If provided, figure is saved to dirDict['crossComp_figDir'] or
        dirDict['outDir'].  If None, saved to current directory.
    cv_count : int or None
        Total number of CV splits.  Used to compute the selection fraction
        (count / cv_count).  Inferred from data when None.
    min_count : int
        Drop regions selected in fewer than this many CV folds (default 1 =
        keep everything selected at least once).
    group_by_area : bool
        If True, regions are clustered by brain area and colour-coded.
        If False, sorted globally by count descending.
    show_fraction : bool
        If True, x-axis shows fraction of CV folds (0–1).
        If False, shows raw integer count.
    figsize : tuple or None
        Override figure size.  Auto-scales by number of regions if None.
    save_name : str
        Output file stem (SVG + PNG).
    """
    import os, re, pickle as pkl
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import helperFunctions as hf

    # ── Import assign_structure from stressor_heatmap ───────────────────────
    try:
        from stressor_heatmap import assign_structure
    except ImportError:
        def assign_structure(name, acro):
            return 'Unknown'

    # ── 1. Load region counts ────────────────────────────────────────────────
    if pkl_path is not None:
        with open(pkl_path, 'rb') as f:
            cache = pkl.load(f)
        # cache = [classifyDict, modelList, modelStr, saveStr, featureSelSwitch,
        #          y_real_lab, y_prob, conf_matrix_list_of_arrays, X_test_trans_list,
        #          scores, selected_features_list, selected_features_params,
        #          baseline_val, shap_values_list, oob_preds]
        selected_features_list = cache[10]
        # Remove None entries (failed CV splits)
        selected_features_list = [s for s in selected_features_list if s is not None]
        if cv_count is None:
            cv_count = len(selected_features_list)

        from collections import Counter
        all_regions = np.concatenate(selected_features_list)
        region_counts = Counter(all_regions)

    elif txt_path is not None:
        # ── Parse text file ──────────────────────────────────────────────────
        # Format:   "Present Nx: M - acro1, acro2, ..."
        region_counts = {}
        n_folds_found = None
        with open(txt_path, 'r') as f:
            text = f.read()
        for line in text.splitlines():
            m = re.match(r'Present\s+(\d+)x:\s*\d+\s*-\s*(.+)', line.strip())
            if m:
                count = int(m.group(1))
                regions = [r.strip() for r in m.group(2).split(',') if r.strip()]
                for r in regions:
                    region_counts[r] = count
        if cv_count is None:
            # Infer from max count
            cv_count = max(region_counts.values()) if region_counts else 1
    else:
        raise ValueError("Supply either pkl_path or txt_path.")

    if len(region_counts) == 0:
        print("[plot_boruta_regions_timepoint] No regions found — check file path.")
        return

    # ── 2. Filter by min_count ───────────────────────────────────────────────
    region_counts = {k: v for k, v in region_counts.items() if v >= min_count}
    if len(region_counts) == 0:
        print(f"[plot_boruta_regions_timepoint] No regions survive min_count={min_count}.")
        return

    # ── 3. Build acronym → (Region.Name, Brain_Area) lookup from CSV ─────────
    df_wide = pd.read_csv(csv_path, sep=';', na_values=['NA', 'na', 'N/A', ''])
    acro_to_name = dict(zip(df_wide['acronym'].astype(str),
                            df_wide['Region.Name'].astype(str)))
    acro_to_area = {}
    for acro, name in acro_to_name.items():
        acro_to_area[acro] = assign_structure(name, acro)

    # For acronyms not in the CSV (shouldn't happen), fall back
    for acro in region_counts:
        if acro not in acro_to_area:
            acro_to_area[acro] = assign_structure('', acro)

    # ── 4. Build plotting DataFrame ──────────────────────────────────────────
    rows = []
    for acro, cnt in region_counts.items():
        area = acro_to_area.get(acro, 'Unknown')
        frac = cnt / cv_count
        rows.append({'acro': acro, 'count': cnt, 'fraction': frac, 'area': area})

    df = pd.DataFrame(rows)

    # ── 5. Sort ───────────────────────────────────────────────────────────────
    # Use assign_structure return values as the canonical order
    AREA_ORDER = ['Isocortex', 'Olfactory', 'Hippocampus',
                  'Striatum & Pallidum', 'Thalamus', 'Hypothalamus',
                  'Midbrain & Hindbrain', 'Cerebellum', 'Other', 'Unknown']

    area_color = hf.create_color_dict(dictType='brainArea', rgbSwitch=0,
                                       alpha_value=1, scaleVal=False)

    # assign_structure() returns different key names than create_color_dict().
    # Normalise so every assign_structure label maps to its colour.
    AREA_KEY_MAP = {
        'Isocortex':          'Cortex',
        'Hippocampus':        'Hippo',
        'Striatum & Pallidum':'StriatumPallidum',
        'Midbrain & Hindbrain':'MidHindMedulla',
    }
    area_color_ext = {**area_color}
    for struct_key, color_key in AREA_KEY_MAP.items():
        if color_key in area_color:
            area_color_ext[struct_key] = area_color[color_key]

    if group_by_area:
        df['area_rank'] = df['area'].apply(
            lambda a: AREA_ORDER.index(a) if a in AREA_ORDER else len(AREA_ORDER)
        )
        df = df.sort_values(['area_rank', 'count'], ascending=[True, False])
    else:
        df = df.sort_values('count', ascending=False)

    # ── 6. Plot ───────────────────────────────────────────────────────────────
    n_regions = len(df)
    if figsize is None:
        fig_h = max(3.5, n_regions * 0.22)
        fig_w = 5.5 if show_fraction else 4.5
        figsize = (fig_w, fig_h)

    fig, axes = plt.subplots(1, 2,
                              figsize=figsize,
                              gridspec_kw={'width_ratios': [4, 1],
                                           'wspace': 0.05})
    ax  = axes[0]   # main bar chart
    ax2 = axes[1]   # area-level summary

    xval    = 'fraction' if show_fraction else 'count'
    xlim_max = 1.0 if show_fraction else cv_count

    bar_colors = [area_color_ext.get(a, '#cccccc') for a in df['area']]

    # Horizontal bars — bottom-most = lowest count (matplotlib y=0 is bottom)
    y_pos = np.arange(n_regions)
    ax.barh(y_pos, df[xval].values, color=bar_colors,
            height=0.75, edgecolor='none', zorder=3)

    # Thin separator lines between brain area groups
    if group_by_area:
        prev_area = None
        for i, area in enumerate(df['area'].values):
            if prev_area is not None and area != prev_area:
                ax.axhline(i - 0.5, color='white', lw=1.5, zorder=4)
            prev_area = area

    # Annotate bars with count
    for i, (cnt, frac) in enumerate(zip(df['count'].values, df['fraction'].values)):
        label = f'{cnt}/{cv_count}' if show_fraction else str(cnt)
        ax.text(df[xval].values[i] + xlim_max * 0.01, i,
                label, va='center', ha='left', fontsize=5.5, color='#333333')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['acro'].values, fontsize=6)
    ax.set_xlim([0, xlim_max * 1.18])
    ax.set_xlabel('Fraction of CV folds' if show_fraction else 'CV fold count',
                  fontsize=8)
    ax.set_title(f'Boruta-selected regions\n(pooled timepoint, n={n_regions})',
                 fontsize=8, pad=4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.invert_yaxis()   # highest count at top

    # ── Area-level summary bar (right panel) ──────────────────────────────────
    area_totals = df.groupby('area')['count'].sum().reset_index()
    area_totals['area_rank'] = area_totals['area'].apply(
        lambda a: AREA_ORDER.index(a) if a in AREA_ORDER else len(AREA_ORDER)
    )
    area_totals = area_totals.sort_values('area_rank')
    a_colors = [area_color_ext.get(a, '#cccccc') for a in area_totals['area']]
    a_pos    = np.arange(len(area_totals))

    ax2.barh(a_pos, area_totals['count'].values,
             color=a_colors, height=0.7, edgecolor='none', zorder=3)
    ax2.set_yticks(a_pos)
    ax2.set_yticklabels(area_totals['area'].values, fontsize=6)
    ax2.set_xlabel('Total\ncount', fontsize=6)
    ax2.set_title('By area', fontsize=7, pad=4)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.tick_params(axis='y', length=0)
    ax2.invert_yaxis()

    # ── Legend ────────────────────────────────────────────────────────────────
    present_areas = df['area'].unique().tolist()
    legend_patches = [
        mpatches.Patch(color=area_color_ext.get(a, '#cccccc'), label=a)
        for a in AREA_ORDER if a in present_areas
    ]
    ax.legend(handles=legend_patches, fontsize=5.5, frameon=False,
              loc='lower right', handlelength=1, labelspacing=0.25,
              ncol=2 if len(legend_patches) > 6 else 1)

    # ── Save ─────────────────────────────────────────────────────────────────
    if dirDict is not None:
        out_dir = dirDict.get('crossComp_figDir', dirDict.get('outDir', '.'))
    else:
        out_dir = '.'

    for ext in ('svg', 'png'):
        plt.savefig(os.path.join(out_dir, f'{save_name}.{ext}'),
                    format=ext, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    print(f"  Saved {save_name}.svg / .png  →  {out_dir}")
    print(f"  Total regions plotted: {n_regions}  |  CV folds: {cv_count}")

    return df  # return df for further inspection / export