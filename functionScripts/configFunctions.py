"""
configFunctions.py  –  adapted for stressor data
=================================================
All drug-specific settings (PSI, KET, SAL, …) replaced with
stressor-specific settings (Ctrl, FS, FSW, RS, TS).
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Stressor metadata
# ─────────────────────────────────────────────────────────────────────────────
STRESSOR_ORDER  = ['Ctrl', 'FS', 'FSW', 'RS', 'TS']
TIMEPOINT_ORDER = ['Acute', '7D', '14D', '21D']


# ─────────────────────────────────────────────────────────────────────────────
def return_heatmapDict():
    heatmapDict = dict()
    heatmapDict['data']         = 'density_norm'   # column with numerical values
    heatmapDict['feature']      = 'abbreviation'   # feature column (brain region acronym)
    heatmapDict['blockCount']   = 2
    heatmapDict['logChangeSal'] = False
    heatmapDict['areaBlocks']   = True
    heatmapDict['areaPerBlock'] = 4
    heatmapDict['SortList']     = STRESSOR_ORDER   # order for heatmap columns
    return heatmapDict


# ─────────────────────────────────────────────────────────────────────────────
# Figure settings (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
def _apply_rcparams(font_size, extra=None):
    import matplotlib as mpl
    import seaborn as sns
    mpl.rcParams['font.size'] = font_size
    sns.set_style('ticks')
    sns.despine()
    base = {
        'font.family'        : 'Arial',
        'svg.fonttype'       : 'none',
        'savefig.dpi'        : 300,
        'figure.dpi'         : 72,
        'xtick.major.pad'    : 2,
        'ytick.major.pad'    : 0.5,
        'axes.labelpad'      : 0,
        'legend.frameon'     : False,
        'legend.loc'         : 'upper right',
        'figure.frameon'     : False,
        'axes.linewidth'     : 0.75,
        'legend.markerscale' : 1,
        'savefig.format'     : 'svg',
        'axes.spines.right'  : False,
        'axes.spines.top'    : False,
    }
    if extra:
        base.update(extra)
    mpl.rcParams.update(base)
    for key in ['xtick.labelsize', 'ytick.labelsize',
                'axes.titlesize', 'axes.labelsize', 'legend.fontsize']:
        mpl.rcParams[key] = font_size


def setup_figure_settings():          _apply_rcparams(4)
def setup_figure_settings_HTC():      _apply_rcparams(6, {'axes.linewidth': 0.5})
def setup_mRNA_corr_settings():       _apply_rcparams(8)
def setup_saldiff_settings():         _apply_rcparams(4)
def setup_figure_changeFonts(fs):     _apply_rcparams(fs)


def setup_LDA_settings():
    import matplotlib as mpl
    import seaborn as sns
    mpl.rcParams['font.size'] = 6
    sns.set_style('ticks')
    sns.despine()
    mpl.rcParams.update({
        'font.family': 'Arial', 'svg.fonttype': 'none',
        'savefig.dpi': 300, 'figure.dpi': 300,
        'xtick.direction': 'out',
        'xtick.minor.size': 2.5, 'xtick.major.size': 2.5,
        'xtick.minor.width': 0.75, 'xtick.major.width': 0.75,
        'ytick.minor.size': 2.5, 'ytick.major.size': 2.5,
        'ytick.minor.width': 0.75, 'ytick.major.width': 0.75,
        'axes.spines.right': False, 'axes.spines.top': False,
        'axes.linewidth': 0.75, 'axes.labelpad': 0,
        'legend.frameon': False, 'legend.loc': 'upper right',
        'figure.frameon': False, 'legend.markerscale': 1,
        'savefig.format': 'svg',
    })
    mpl.rcParams['xtick.labelsize'] = mpl.rcParams['ytick.labelsize'] = 6
    mpl.rcParams['axes.labelsize'] = mpl.rcParams['legend.fontsize'] = mpl.rcParams['axes.titlesize'] = 7


def setup_Confmatrix_settings():
    import matplotlib as mpl
    import seaborn as sns
    mpl.rcParams['font.size'] = 6
    sns.set_style('ticks')
    sns.despine()
    mpl.rcParams.update({
        'font.family': 'Arial', 'svg.fonttype': 'none',
        'savefig.dpi': 300, 'figure.dpi': 300,
        'xtick.direction': 'out', 'axes.labelpad': 0,
        'xtick.minor.size': 2.5, 'xtick.major.size': 2.5,
        'xtick.minor.width': 0.75, 'xtick.major.width': 0.75,
        'ytick.minor.size': 2.5, 'ytick.major.size': 2.5,
        'ytick.minor.width': 0.75, 'ytick.major.width': 0.75,
        'axes.spines.right': False, 'axes.spines.top': False,
        'axes.linewidth': 0.75, 'legend.frameon': False,
        'legend.loc': 'upper right', 'figure.frameon': False,
        'legend.markerscale': 1, 'savefig.format': 'svg',
    })
    mpl.rcParams['xtick.labelsize'] = mpl.rcParams['ytick.labelsize'] = 6
    mpl.rcParams['axes.labelsize'] = mpl.rcParams['legend.fontsize'] = mpl.rcParams['axes.titlesize'] = 7


# ─────────────────────────────────────────────────────────────────────────────
# Classification dictionaries
# ─────────────────────────────────────────────────────────────────────────────
def return_classifyDict_default():
    classifyDict = dict()

    # Reproducibility
    classifyDict['randSeed']  = 82590
    classifyDict['randState'] = np.random.RandomState(classifyDict['randSeed'])

    # Data columns
    classifyDict['data']    = 'density_norm'   # numerical feature column
    classifyDict['feature'] = 'abbreviation'   # brain-region column
    classifyDict['label']   = 'stressor'       # classification target

    # ── NA handling for the classifier pivot ────────────────────────────────
    # Regions (features) with a fraction of NaN values ABOVE na_region_thresh
    # are dropped before imputation.  Set to 1.0 to keep all regions.
    # After region dropout, remaining NaN cells are filled with the per-region
    # mean across all datasets ('mean'), or with 0 ('zero').
    classifyDict['na_region_thresh'] = 0.20   # drop regions missing in >20% of datasets
    classifyDict['na_impute']        = 'mean' # 'mean' | 'zero'

    # Feature filtering / aggregation
    classifyDict['featurefilt']          = False
    classifyDict['filtType']             = 'min'
    classifyDict['featureAgg']           = False
    classifyDict['featureSel_linkage']   = 'average'
    classifyDict['featureSel_distance']  = 'correlation'
    classifyDict['cluster_count']        = 100
    classifyDict['cluster_thres']        = 0.2

    # Preprocessing & feature selection
    classifyDict['model_featureTransform'] = True
    classifyDict['model_featureScale']     = True
    classifyDict['model_featureSel']       = 'Boruta'   # 'Univar','Boruta','None',…
    classifyDict['model_featureSel_alpha'] = 0.05
    classifyDict['model_featureSel_mode']  = 'modelPer'
    classifyDict['model_featureSel_k']     = [30]

    # Classifier
    classifyDict['model']      = 'LogRegL2'
    classifyDict['multiclass'] = 'multinomial'
    classifyDict['max_iter']   = 1000
    classifyDict['CVstrat']    = 'ShuffleSplit'

    # Parameter grid
    classifyDict['pGrid'] = {'classif__C': [1]}

    classifyDict['shuffle']        = True
    classifyDict['gridCV']         = False
    classifyDict['saveLoadswitch'] = True
    classifyDict['test_size']      = 1 / 4
    classifyDict['innerFold']      = 4
    classifyDict['CV_count']       = 100

    classifyDict['balance']     = True
    classifyDict['featurePert'] = 'correlation_dependent'

    # ── Adaptive train/test split ────────────────────────────────────────────
    # When the smallest class has ≤ this many samples (after undersampling),
    # the test fraction is automatically raised to `test_size_small` so that
    # at least 60% of samples are used for training.
    # For well-powered comparisons (e.g. all timepoints pooled, ≥15 per class)
    # the default 25% test fraction applies.
    classifyDict['adaptive_test_size']      = True
    classifyDict['small_class_threshold']   = 7    # ≤7 per class → small
    classifyDict['test_size_small']         = 0.40  # 60 % training for small N
    classifyDict['test_size_large']         = 0.25  # 75 % training default

    classifyDict['crossComp_tagList'] = [
        f"data={classifyDict['data']}-",
        "clf_LogReg(",
        f"_CV{classifyDict['CV_count']}"
    ]

    return classifyDict


def return_classifyDict_testing():
    """Faster settings for quick validation runs."""
    classifyDict = return_classifyDict_default()
    classifyDict['model_featureSel'] = 'Univar'
    classifyDict['CV_count']         = 20
    classifyDict['featurePert']      = 'interventional'
    classifyDict['saveLoadswitch']   = True
    classifyDict['crossComp_tagList'] = [
        f"data={classifyDict['data']}-",
        "clf_LogReg(",
        f"_CV{classifyDict['CV_count']}"
    ]
    return classifyDict


# ─────────────────────────────────────────────────────────────────────────────
# Plot dictionary
# ─────────────────────────────────────────────────────────────────────────────
def return_plotDict():
    plotDict = dict()
    plotDict['shapForcePlotCount']  = 20
    plotDict['shapSummaryThres']    = 75
    plotDict['shapMaxDisplay']      = 10
    plotDict['plot_ConfusionMatrix'] = True
    plotDict['plot_PRcurve']         = True
    plotDict['plot_SHAPsummary']     = False
    plotDict['plot_SHAPforce']       = False
    plotDict['featureCorralogram']   = False
    plotDict['plot_SHAPcorr']        = True   # correlation matrix of top SHAP regions
    plotDict['shapCorrTop']          = 20     # how many top regions to include
    return plotDict
