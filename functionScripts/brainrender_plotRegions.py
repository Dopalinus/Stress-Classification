"""
brainrender_plotRegions.py
==========================
Two-step pipeline for brainrender figures:

  Step 1 – generate_brainrender_csvs()
      Reads classifier cache pkl files (Real_outdata.pkl), extracts SHAP values
      for binary comparisons, filters regions by SHAP consistency threshold, and
      colours regions by the direction of mean SHAP — i.e. which condition a high
      cFos score pushes the classifier toward.  Saves one br_<comparison>.csv per
      binary classifier result.

      Multiclass comparisons are skipped (n_classes != 2).

  Step 2 – plot_brain_regions()
      Reads every br_*.csv in a folder and renders a brainrender PNG +
      SVG text-label panel for each comparison.

Typical call (in notebook or script)
--------------------------------------
    from brainrender_plotRegions import generate_brainrender_csvs, plot_brain_regions

    br_dir = os.path.join(dirDict['outDir'], 'brainrender')

    generate_brainrender_csvs(
        lightsheet_data = lightsheet_data,
        dirDict         = dirDict,
        classifyDict    = classifyDict,
        output_dir      = br_dir,
        shap_thresh     = 50,   # region must appear in ≥50 % of CV splits
    )

    plot_brain_regions(
        csv_dir       = br_dir,
        output_dir    = br_dir,
        camera_preset = 'laptop',
    )

Cache pkl layout (saveList indices)
-------------------------------------
  0  classifyDict
  1  modelList
  2  modelStr
  3  saveStr
  4  featureSelSwitch
  5  y_real_lab
  6  y_prob
  7  conf_matrix_list_of_arrays
  8  X_test_trans_list
  9  scores
  10 selected_features_list
  11 selected_features_params
  12 baseline_val
  13 shap_values_list      ← used here
  14 oob_preds

SHAP direction convention (binary)
-------------------------------------
  SHAP values are computed w.r.t. the positive class (class index 1).
  Canonical class order follows STRESSOR_ORDER: ['Ctrl','FS','FSW','RS','TS'].
  For "Ctrl vs FS": class 0 = Ctrl, class 1 = FS.
    mean SHAP > 0  →  high cFos pushes toward class 1  →  coloured as class 1
    mean SHAP < 0  →  high cFos pushes toward class 0  →  coloured as class 0
"""

import glob
import re
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vedo
import pickle as pkl

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size']   = 6
plt.rcParams['svg.fonttype'] = 'none'

# ── Camera presets ────────────────────────────────────────────────────────────

CAMERA_PRESETS = {
    'desktop': {
        'pos':           (4284, -2674, 26722),
        'viewup':        (0, -1, 0),
        'clipping_range': (26768, 57715),
        'focalPoint':    (3678, 4091, -6418),
        'distance':      33828,
    },
    'laptop': {
        'pos':           (5084, -2583, 26726),
        'viewup':        (0, -1, 0),
        'clipping_range': (19255, 50998),
        'focalPoint':    (4478, 4182, -6413),
        'distance':      33828,
    },
    'video': {
        'pos':           (34302, -11155, 20007),
        'viewup':        (0, -1, 0),
        'clipping_range': (20102, 68096),
        'focalPoint':    (5059, 3895, -6370),
        'distance':      42158,
    },
}


# ── Title-cleaning helper ─────────────────────────────────────────────────────

def _clean_title(raw_title: str) -> str:
    """
    Translate short stressor / drug codes in a filename-derived title to
    human-readable names using helperFunctions.create_translation_dict().

    Falls back gracefully if helperFunctions is not importable.
    """
    try:
        import helperFunctions as hf
        for dict_type in ('stressor', 'drug'):
            trans = hf.create_translation_dict(dict_type)
            for key, value in trans.items():
                raw_title = raw_title.replace(key, value)
    except Exception:
        pass  # No translation available – use the raw title
    return raw_title


# ── SHAP aggregation helper ───────────────────────────────────────────────────

def _aggregate_shap_binary(shap_values_list, shap_thresh_pct):
    """
    Aggregate binary SHAP values across CV splits.

    Parameters
    ----------
    shap_values_list : list
        shap_values_list[0] is a list of DataFrames, one per CV split.
        Each DataFrame: rows = test samples, columns = region names,
        values = SHAP w.r.t. positive class (class index 1).
        Regions not selected in a given split are NaN.
    shap_thresh_pct : float
        Percentage of CV splits a region must appear in (non-NaN) to be kept.
        E.g. 50 means ≥50 % of splits.

    Returns
    -------
    mean_shap : pd.Series
        Mean SHAP value per region for regions passing the threshold.
        Sign encodes direction: positive → pushes toward class 1,
        negative → pushes toward class 0.
    """
    split_dfs = shap_values_list[0]          # list of DataFrames, one per split
    n_splits  = len(split_dfs)

    if n_splits == 0:
        return pd.Series(dtype=float)

    # Test-set size (samples per split) — used to convert NaN count → split count
    test_count = split_dfs[0].shape[0]

    # Drop the 'index' column that collect_shap_values adds via reset_index()
    clean_dfs = [df.drop(columns=['index'], errors='ignore') for df in split_dfs]

    # Concatenate across all splits: shape = (n_splits * test_count, n_features)
    shap_concat = pd.concat(clean_dfs, axis=0)

    # Count how many CV splits each region appeared in (non-NaN rows)
    nan_count         = shap_concat.isna().sum()          # total NaN per column
    feature_split_count = n_splits - nan_count / test_count   # splits with values

    # Apply consistency threshold
    thresh_abs    = np.ceil(n_splits * shap_thresh_pct / 100)
    passing       = feature_split_count[feature_split_count >= thresh_abs].index
    shap_filtered = shap_concat[passing].fillna(0)

    mean_shap = shap_filtered.mean(axis=0)   # signed mean per region

    print(f"  [SHAP] {len(mean_shap)} regions passed threshold "
          f"(≥{thresh_abs:.0f}/{n_splits} splits, {shap_thresh_pct}%)")

    return mean_shap


# ── CSV generation ───────────────────────────────────────────────────────────

def generate_brainrender_csvs(
    lightsheet_data: pd.DataFrame,
    dirDict: dict,
    classifyDict: dict,
    output_dir: str    = None,
    shap_thresh: float = 50,
    data_col: str      = 'density_norm',
):
    """
    Convert binary classifier cache results (Real_outdata.pkl) into br_*.csv
    files ready for plot_brain_regions().

    Regions are selected and coloured based on SHAP values:
      - Only regions present in ≥ shap_thresh % of CV splits are included.
      - Colour direction: mean SHAP > 0 → region coloured as the positive class
        (class 1 in labelDict order); mean SHAP < 0 → negative class (class 0).

    Multiclass comparisons (n_classes != 2) are skipped with a warning.

    Parameters
    ----------
    lightsheet_data : pd.DataFrame
        Long-format data as returned by loaderFunctions.load_stressor_data().
    dirDict : dict
        Directory dict from initFunctions.setPath_createDirs().
        Must contain 'classifyDir' and 'tempDir' keys.
    classifyDict : dict
        Classifier settings dict (needs 'crossComp_tagList' key).
    output_dir : str, optional
        Where to save the br_*.csv files.
        Defaults to dirDict['outDir']/brainrender.
    shap_thresh : float
        Percentage of CV splits a region must appear in to be included.
        Default 50 (= ≥50 % of splits).  Mirrors plotDict['shapSummaryThres'].
    data_col : str
        Column in lightsheet_data to average per region per group.
        Default is 'density_norm'.
    """
    if output_dir is None:
        output_dir = os.path.join(dirDict['outDir'], 'brainrender')
    os.makedirs(output_dir, exist_ok=True)

    tag_list   = classifyDict['crossComp_tagList']
    target_dir = dirDict['classifyDir']   # root where scoreDict_Real.pkl files live
    temp_root  = dirDict['tempDir']       # root where Real_outdata.pkl files live

    # ── Find all scoreDict_Real.pkl files matching the tag list ──────────────
    score_pkl_paths = []
    for root, dirs, files in os.walk(target_dir):
        if 'scoreDict_Real.pkl' in files:
            if all(tag in root for tag in tag_list):
                score_pkl_paths.append(os.path.join(root, 'scoreDict_Real.pkl'))

    if not score_pkl_paths:
        print(f"[brainrender] No scoreDict_Real.pkl files found in: {target_dir}")
        print(f"              (tag filter: {tag_list})")
        print("              Run the classifier cells first.")
        return

    print(f"[brainrender] Found {len(score_pkl_paths)} classifier result(s). "
          f"Generating br_*.csv from SHAP values ...")

    for score_pkl_path in score_pkl_paths:

        # ── Load scoreDict for metadata (compLabel, n_classes) ───────────────
        with open(score_pkl_path, 'rb') as f:
            score_dict = pkl.load(f)

        comp_label = score_dict['compLabel']          # e.g. "Ctrl vs FS"
        n_classes  = score_dict.get('n_classes', 2)

        if n_classes != 2:
            print(f"  Skipping '{comp_label}' — brainrender SHAP mode is binary only "
                  f"(n_classes={n_classes}).")
            continue

        # ── Derive cache pkl path from scoreDict path ────────────────────────
        # scoreDict: {classifyDir}/{rel_path}/scoreDict_Real.pkl
        # cache pkl: {tempDir}/{rel_path}/Real_outdata.pkl
        model_dir     = os.path.dirname(score_pkl_path)
        rel_path      = os.path.relpath(model_dir, target_dir)
        cache_pkl_path = os.path.join(temp_root, rel_path, 'Real_outdata.pkl')

        if not os.path.exists(cache_pkl_path):
            print(f"  [WARN] Cache pkl not found for '{comp_label}': {cache_pkl_path}")
            print(f"         Run the classifier with saveLoadswitch=True first.")
            continue

        # ── Load SHAP values from cache ──────────────────────────────────────
        with open(cache_pkl_path, 'rb') as f:
            save_list = pkl.load(f)

        shap_values_list = save_list[13]

        shap_collected = any(len(s) > 0 for s in shap_values_list)
        if not shap_collected:
            print(f"  [WARN] No SHAP values in cache for '{comp_label}' — skipping.")
            continue

        # ── Aggregate mean signed SHAP per region ────────────────────────────
        mean_shap = _aggregate_shap_binary(shap_values_list, shap_thresh)

        if mean_shap.empty:
            print(f"  [WARN] No regions passed SHAP threshold for '{comp_label}' — skipping.")
            continue

        selected_regions = list(mean_shap.index)

        # ── Parse group names from compLabel ─────────────────────────────────
        # labelDict canonical order: class 0 = group_names[0], class 1 = group_names[1]
        # SHAP is w.r.t. class 1:
        #   mean_shap > 0  →  pushes toward group_names[1]
        #   mean_shap < 0  →  pushes toward group_names[0]
        omp_label  = score_dict['compLabel']
        plt_title   = comp_label.replace('/', '+')          # safe for filenames
        group_names = comp_label.replace('+', '/').split(' vs ')

        # ── NEW: skip multiclass (non-binary) comparisons ────────────────────────
        if len(group_names) != 2:
            print(f"  [SKIP] '{comp_label}' is not a binary comparison "
                f"({len(group_names)} groups) – brainrender supports binary only.")
            continue

        # ── Subset lightsheet data ────────────────────────────────────────────
        df = lightsheet_data[
            lightsheet_data['stressor'].isin(group_names) &
            lightsheet_data['abbreviation'].isin(selected_regions)
        ].copy()

        if df.empty:
            print(f"  [WARN] No lightsheet data matched for '{comp_label}' — skipping.")
            continue

        # ── Mean cFos per region per group ───────────────────────────────────
        means = (
            df.groupby(['abbreviation', 'stressor'])[data_col]
            .mean()
            .unstack('stressor')
        )

        available = [g for g in group_names if g in means.columns]

        # ── Guard: need exactly 2 matched groups for a meaningful binary render
        if len(available) != 2:
            unmatched = [g for g in group_names if g not in means.columns]
            print(f"  [SKIP] '{comp_label}': group(s) {unmatched} not found in "
                  f"lightsheet data – cannot generate binary CSV.")
            continue

        means     = means[available].reset_index()
        means['Diff'] = means[available[0]] - means[available[1]]
        means = means.sort_values('Diff').reset_index(drop=True)

        out_path = os.path.join(output_dir, f'br_{plt_title}.csv')
        means.to_csv(out_path, index=False)
        print(f"  Saved → {out_path}  ({len(means)} regions)")
    print("[brainrender] CSV generation complete.")


# ── Core rendering function ───────────────────────────────────────────────────

def plot_brain_regions(
    csv_dir: str,
    output_dir: str,
    out_mode: str      = 'still',
    labels_flag: bool  = False,
    just_labels: bool  = False,
    camera_preset: str = 'laptop',
    zoom: float        = 1.0,
    color_dict: dict   = None,
):
    """
    Iterate over every ``br_*.csv`` in *csv_dir* and produce:
      • A brainrender PNG (unless *just_labels* is True)
      • A matching SVG text-label panel with region names coloured by SHAP direction

    The 'Diff' column in each CSV encodes the mean signed SHAP value:
      positive → region coloured as the positive class (class 1)
      negative → region coloured as the negative class (class 0)

    Parameters
    ----------
    csv_dir : str
        Directory that contains the ``br_*.csv`` input files.
    output_dir : str
        Directory where PNG / SVG outputs are written.
    out_mode : {'still', 'vid'}
        'still'  → single screenshot PNG
        'vid'    → rotating MP4 video (requires VideoMaker)
    labels_flag : bool
        If True, adds native brainrender text labels to the 3-D scene.
        If False (default), labels are written to a separate SVG text panel.
    just_labels : bool
        If True, skips the 3-D render entirely and only produces the SVG panel.
    camera_preset : {'laptop', 'desktop', 'video'}
        Which camera position preset to use.
    zoom : float
        Zoom level passed to brainrender's ``render()`` call.
    color_dict : dict, optional
        Custom ``{name: hex_colour}`` mapping.  If None, colours are loaded
        from ``helperFunctions.create_color_dict()``.
    """

    # ── Lazy imports (brainrender is heavy) ───────────────────────────────────
    from brainrender import Scene
    from brainrender.camera import set_camera

    # ── Resolve paths ─────────────────────────────────────────────────────────
    csv_dir    = os.path.abspath(csv_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ── Colour dictionary ─────────────────────────────────────────────────────
    if color_dict is None:
        try:
            import helperFunctions as hf
            color_dict = hf.create_color_dict()
        except Exception:
            # Minimal fallback palette
            color_dict = {
                'Ctrl': '#BBBBBB', 'FS': '#228833', 'FSW': '#4477AA',
                'RS': '#AA3377',   'TS': '#CC3311',
            }

    # ── Camera ────────────────────────────────────────────────────────────────
    if camera_preset not in CAMERA_PRESETS:
        raise ValueError(
            f"Unknown camera_preset '{camera_preset}'. "
            f"Choose from: {list(CAMERA_PRESETS.keys())}"
        )
    camera = CAMERA_PRESETS[camera_preset]

    # ── vedo backend ──────────────────────────────────────────────────────────
    vedo.settings.default_backend = 'vtk'

    # ── Find CSV files ────────────────────────────────────────────────────────
    file_pattern   = os.path.join(csv_dir, 'br_*.csv')
    matching_files = glob.glob(file_pattern, recursive=True)

    if not matching_files:
        print(f"[brainrender] No 'br_*.csv' files found in: {csv_dir}")
        return

    print(f"[brainrender] Found {len(matching_files)} file(s) to process.")

    # ── Process each CSV ──────────────────────────────────────────────────────
    for csv_path in matching_files:
        print(f"\n  Processing: {os.path.basename(csv_path)}")

        pandadf = pd.read_csv(csv_path)

        # Derive title from filename
        match = re.search(r'br_(.*?)\.csv', csv_path)
        if not match:
            print(f"  [WARN] Filename does not match 'br_*.csv' pattern – skipping.")
            continue

        plt_title = match.group(1)
        # Use _clean_title only for the output filename, NOT for colour lookup
        # restore ' vs ' separator before translating short codes to full names
        plt_title_display = _clean_title(plt_title.replace('+', ' vs '))

        # Read group names directly from CSV columns — always short codes like
        # 'Ctrl', 'FS' regardless of how the filename was translated.
        group_cols = [c for c in pandadf.columns
                      if c not in ('abbreviation', 'Diff', 'MeanSHAP')]

        # Look up colours using the short codes from the CSV
        drug_colors     = [color_dict.get(name, '#888888') for name in group_cols]
        drug_color_dict = dict(zip(group_cols, drug_colors))

        # Warn if any colour fell back to grey
        # Warn if any colour fell back to grey
        for name, col in drug_color_dict.items():
            if col == '#888888':
                print(f"  [WARN] No colour found for group '{name}' – using grey.")

        # Guard: a valid binary CSV must have exactly 2 group columns
        if len(drug_colors) < 2:
            print(f"  [SKIP] '{os.path.basename(csv_path)}' has only "
                  f"{len(drug_colors)} group column(s) – expected 2. "
                  f"Delete this stale CSV and re-run generate_brainrender_csvs().")
            continue

        region_list  = pandadf['abbreviation']
        region_color = pandadf['Diff'].apply(
            lambda x: drug_colors[0] if x > 0 else drug_colors[1]
        )

        # Identify which colours are "uninformative" (grey-family, i.e. control)
        # so we can skip them in the 3-D render but still show them in the SVG legend.
        def _is_grey(hex_col):
            h = hex_col.lower().lstrip('#')
            if len(h) == 6:
                r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
                return abs(r-g) < 20 and abs(g-b) < 20 and abs(r-b) < 20
            return hex_col.lower() in {'#bbbbbb', '#aaaaaa', '#999999',
                                        '#888888', '#cccccc', '#b0b0b0'}

        # ── 3-D Brainrender scene ─────────────────────────────────────────────
        if not just_labels:
            label_tag = '_labels' if labels_flag else ''

            popup_scene = Scene()
            try:
                popup_scene.background_color = 'white'
            except Exception:
                try:
                    popup_scene.plotter.background('white')
                except Exception:
                    pass
            popup_scene.add_brain_region('root', alpha=0.08, color='white')

            for region, reg_col in zip(region_list, region_color):
                if _is_grey(reg_col):
                    continue   # skip control/grey-coloured regions in 3-D
                actor_obj = popup_scene.add_brain_region(region, color=reg_col, alpha=0.8)
                if labels_flag and actor_obj is not None:
                    popup_scene.add_label(actor_obj, actor_obj.name)

            if out_mode == 'still':
                popup_scene.render(
                    interactive=False,
                    camera=camera,
                    zoom=zoom,
                )
                out_png = os.path.join(output_dir, f'{plt_title_display}{label_tag}.png')
                popup_scene.screenshot(out_png)
                print(f'  Saved PNG → {out_png}')

            elif out_mode == 'vid':
                from brainrender.video import VideoMaker
                set_camera(popup_scene, camera)
                vm = VideoMaker(popup_scene, output_dir, f'{plt_title_display}_spin')
                vm.make_video(azimuth=1.5, duration=15, fps=15)
                print(f'  Saved video → {output_dir}')

            else:
                raise ValueError(f"out_mode must be 'still' or 'vid', got '{out_mode}'")

        # ── SVG text-label panel ──────────────────────────────────────────────
        if not labels_flag:
            fig, ax = plt.subplots()

            # Comparison group names at the top, coloured by their group colour
            for i, drug_text in enumerate(drug_color_dict.keys()):
                ax.text(
                    0, 0.3 + (i * 0.1), drug_text,
                    ha='center', va='center',
                    color=drug_color_dict[drug_text],
                    linespacing=1.5,
                )

            # Region names coloured by SHAP direction
            for i, (color, text) in enumerate(zip(region_color, region_list)):
                ax.text(
                    0, 1.1 - (i * 0.05), text,
                    ha='center', va='center',
                    color=color,
                    linespacing=1.5,
                )

            ax.axis('off')
            plt.tight_layout()

            out_svg = os.path.join(output_dir, f'{plt_title_display}_Text.svg')
            plt.savefig(out_svg, format='svg', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(f'  Saved SVG  → {out_svg}')


# ── Convenience wrapper ───────────────────────────────────────────────────────

if __name__ == '__main__':
    """
    Quick self-test: edit the two paths below and run
        python brainrender_plotRegions.py
    """
    plot_brain_regions(
        csv_dir       = './brainrender_data',   # ← folder with your br_*.csv files
        output_dir    = './brainrender_output',
        out_mode      = 'still',
        labels_flag   = False,
        just_labels   = False,
        camera_preset = 'laptop',
        zoom          = 1.0,
    )
