"""
initFunctions.py  –  adapted for stressor data
===============================================
`setPath_createDirs` is unchanged – it still creates the same Output/Temp/Debug
directory structure.

`loadLightSheetData` is replaced by `loaderFunctions.load_stressor_data`.
The old multi-batch loader stubs are kept for import compatibility.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
def setPath_createDirs():
    """
    Create all output directories and return a directory dictionary.
    Works identically to the original – no changes needed.
    """
    dirDict = dict()
    rootDir = os.getcwd()

    # Input / atlas directories (may not exist for stressor data – that's fine)
    dirDict['atlasDir']  = os.sep.join([rootDir, 'Atlas'])
    dirDict['dataDir']   = os.sep.join([rootDir, 'Data'])

    # Output directories
    outDirDict = dict()
    outDirDict['debugDir']         = os.sep.join([rootDir, 'Debug'])
    outDirDict['tempDir']          = os.sep.join([rootDir, 'Temp'])
    outDirDict['geneCorrDir']      = os.sep.join([outDirDict['tempDir'], 'geneCorr'])
    outDirDict['outDir']           = os.sep.join([rootDir, 'Output'])
    outDirDict['classifyDir']      = os.sep.join([outDirDict['outDir'], 'classif'])
    outDirDict['crossComp_figDir'] = os.sep.join([outDirDict['outDir'], 'crossComp'])

    for key, path in outDirDict.items():
        if not os.path.isdir(path):
            os.makedirs(path)

    dirDict.update(outDirDict)
    return dirDict


# ─────────────────────────────────────────────────────────────────────────────
def debugReport(pdDataFrame=None, sheetName=None, debugPath=None,
                roiColName=None, debug_ROI=None):
    if debug_ROI is None:
        roiTag = ''
        expObj = pdDataFrame
    else:
        roiTag = '_ROI'
        for Roi_i, Roi in enumerate(debug_ROI):
            tmp = pdDataFrame.loc[pdDataFrame[roiColName].str.contains(debug_ROI[Roi_i])]
            expObj = tmp if Roi_i == 0 else pd.concat([expObj, tmp])
    if os.path.exists(debugPath):
        writeObj = pd.ExcelWriter(debugPath, mode='a', if_sheet_exists='replace')
    else:
        writeObj = pd.ExcelWriter(debugPath)
    with writeObj as writer:
        expObj.to_excel(writer, sheetName + roiTag)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy stubs – not used for stressor data
# ─────────────────────────────────────────────────────────────────────────────
def loadLightSheetData(*args, **kwargs):
    raise NotImplementedError(
        "loadLightSheetData() is not used for stressor data. "
        "Use loaderFunctions.load_stressor_data(csv_path) instead."
    )
