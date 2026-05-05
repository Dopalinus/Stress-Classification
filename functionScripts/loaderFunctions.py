"""
loaderFunctions.py  –  adapted for stressor wide-format CSV
============================================================
Replaces the original multi-batch loader with a single function that reads
merged_wide_with_acro.csv (semicolon-separated, wide format) and converts it
to the long-format DataFrame that the rest of the pipeline expects.

Column format in CSV:  {Stressor}_{Timepoint}_{Animal}_normalized
  Stressors : Ctrl, FS, FSW, RS, TS
  Timepoints: 7D, 14D, 21D, Acute
  Animals   : F1 – F5  (all female)

Output long-format columns (mirrors the original pipeline):
  dataset      – e.g.  "Ctrl_7D_F1"    (stressor_timepoint_animal)
  stressor     – e.g.  "Ctrl"
  timepoint    – e.g.  "7D"
  animal       – e.g.  "F1"
  density_norm – normalised cFos density (the numerical value)
  count_norm   – alias for density_norm (backward-compatibility)
  Region_Name  – full brain region name
  Region_ID    – Allen Brain Atlas numeric ID
  abbreviation – region acronym (used as classifier feature)
  drug         – alias for stressor  (backward-compatibility with classify / plot fns)
  sex          – "F" for all animals
  Brain_Area   – "Unknown" placeholder (set properly if atlas files are available)
"""

import os
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Public ordered lists (used throughout the pipeline)
# ─────────────────────────────────────────────────────────────────────────────
STRESSOR_ORDER  = ['Ctrl', 'FS', 'FSW', 'RS', 'TS']
TIMEPOINT_ORDER = ['Acute', '7D', '14D', '21D']


# ─────────────────────────────────────────────────────────────────────────────
def load_stressor_data(csv_path: str) -> pd.DataFrame:
    """
    Load merged_wide_with_acro.csv and return a long-format DataFrame.

    Parameters
    ----------
    csv_path : str
        Path to the semicolon-separated wide-format CSV file.

    Returns
    -------
    pd.DataFrame
        Long-format data ready for classification and plotting.
    """

    print(f"Loading stressor data from: {csv_path}")

    # ── 1. Read wide CSV ────────────────────────────────────────────────────
    df_wide = pd.read_csv(
        csv_path,
        sep=';',
        na_values=['NA', 'na', 'N/A', ''],
        dtype={'Region ID': str}     # keep as string to avoid float conversion
    )

    # ── 2. Identify column groups ───────────────────────────────────────────
    meta_cols  = ['Region.Name', 'Region ID', 'acronym']
    value_cols = [c for c in df_wide.columns if c not in meta_cols]

    # ── 3. Melt to long format ──────────────────────────────────────────────
    df_long = df_wide.melt(
        id_vars    = meta_cols,
        value_vars = value_cols,
        var_name   = '_dataset_full',
        value_name = 'density_norm'
    )

    # NAs are KEPT here so every downstream step handles them appropriately:
    #   • groupby / per-region stats  → pandas ignores NaN by default
    #   • classifier pivot            → reformat_pandasdf() imputes or drops
    #   • plotting                    → seaborn/matplotlib skip NaN natively
    # To drop globally: df = df.dropna(subset=['density_norm'])
    n_na = df_long['density_norm'].isna().sum()
    print(f"  NaN values: {n_na:,} / {len(df_long):,} ({n_na/len(df_long)*100:.1f}%) — kept for step-wise handling")

    # ── 4. Parse stressor / timepoint / animal from the column name ─────────
    # Column format: "{Stressor}_{Timepoint}_{Animal}_normalized"
    clean = df_long['_dataset_full'].str.replace('_normalized', '', regex=False)
    parts = clean.str.split('_', expand=True)   # 0=stressor, 1=timepoint, 2=animal

    df_long['stressor']   = parts[0].values
    df_long['timepoint']  = parts[1].values
    df_long['animal']     = parts[2].values
    df_long['dataset']    = clean.values          # e.g. "Ctrl_7D_F1"

    # ── 5. Rename metadata columns ──────────────────────────────────────────
    df_long = df_long.rename(columns={
        'Region.Name': 'Region_Name',
        'Region ID':   'Region_ID',
        'acronym':     'abbreviation',
    })

    # ── 6. Backward-compatibility aliases ───────────────────────────────────
    df_long['drug']       = df_long['stressor']          # many functions query 'drug'
    df_long['count_norm'] = df_long['density_norm']      # some functions use count_norm
    df_long['sex']        = 'F'                          # all animals are female
    df_long['Brain_Area'] = 'Unknown'                    # atlas mapping not available

    # ── 7. Ordered categoricals ─────────────────────────────────────────────
    df_long['stressor']  = pd.Categorical(df_long['stressor'],
                                          categories=STRESSOR_ORDER, ordered=True)
    df_long['timepoint'] = pd.Categorical(df_long['timepoint'],
                                          categories=TIMEPOINT_ORDER, ordered=True)
    df_long['drug']      = df_long['stressor']

    # ── 8. Convert Region_ID to int where possible ──────────────────────────
    df_long['Region_ID'] = pd.to_numeric(df_long['Region_ID'], errors='coerce')

    # ── 9. Drop helper column ───────────────────────────────────────────────
    df_long = df_long.drop(columns=['_dataset_full'])

    # ── 10. Reset index ─────────────────────────────────────────────────────
    df_long = df_long.reset_index(drop=True)

    print(f"  Loaded {len(df_long):,} rows | "
          f"{df_long['Region_Name'].nunique()} regions | "
          f"{df_long['dataset'].nunique()} datasets")
    print(f"  Stressors : {sorted(df_long['stressor'].dropna().unique().tolist())}")
    print(f"  Timepoints: {sorted(df_long['timepoint'].dropna().unique().tolist())}")

    return df_long


# ─────────────────────────────────────────────────────────────────────────────
def filter_by_timepoint(df: pd.DataFrame, timepoint: str) -> pd.DataFrame:
    """Return rows for a single timepoint.  timepoint e.g. '7D'. """
    return df[df['timepoint'] == timepoint].copy()


def filter_by_stressor(df: pd.DataFrame, stressors: list) -> pd.DataFrame:
    """Return rows for a subset of stressors.  stressors e.g. ['Ctrl','FS']. """
    return df[df['stressor'].isin(stressors)].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Legacy stubs kept for import compatibility (not used for stressor data)
# ─────────────────────────────────────────────────────────────────────────────
def loadLightSheetData(**kwargs):
    raise NotImplementedError(
        "loadLightSheetData() is not used for stressor data. "
        "Use load_stressor_data(csv_path) instead."
    )

def createDirs(*args, **kwargs):
    raise NotImplementedError(
        "createDirs() is not used for stressor data. "
        "Use initFunctions.setPath_createDirs() instead."
    )
