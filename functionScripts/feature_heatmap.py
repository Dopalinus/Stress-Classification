"""
feature_heatmap.py
==================
Standalone module to generate classifier feature-usage heatmaps for the
stressor cFos pipeline.

Produces one publication-quality heatmap per requested brain hyperstructure
(Isocortex, Thalamus, Hippocampus, Hypothalamus, Amygdala, Striatum/Pallidum,
Olfactory, Midbrain, Hindbrain, Cerebellum, or any custom subset).

Usage (from main.ipynb or any script)
--------------------------------------
    from feature_heatmap import plot_classifier_heatmaps

    plot_classifier_heatmaps(
        lightsheet_data  = lightsheet_data,
        comparisonNames  = comparisonNames,   # list of strings, e.g. ['Ctrl vs FS', ...]
        featureLists     = featureLists,      # nested list from hf.retrieve_dict_data()
        filterByFreq     = filterByFreq,      # minimum selection count threshold
        dirDict          = dirDict,           # for output path
        hyperstructures  = None,              # None → all; or list e.g. ['Isocortex','Thalamus']
        normalize        = True,              # convert counts → % of CV folds
        cv_count         = 100,               # number of CV folds (for % normalisation)
        save_svg         = True,
        save_png         = True,
    )
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import seaborn as sns
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Brain-area assignment
#     Priority order: first matching rule wins.
#     Rules are evaluated top-to-bottom, so more specific patterns go first.
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (hyperstructure_label, list_of_keywords_in_region_name_lower)
# A region is assigned to the FIRST group whose ANY keyword matches.
_AREA_RULES = [
    # ── Cerebellum ───────────────────────────────────────────────────────────
    ("Cerebellum", [
        "cerebell", "lobule", "crus 1", "crus 2", "floccul",
        "copula", "fastigial", "interposed", "arbor vitae",
        "nodulus", "uvula", "declive", "parafloccul",
        "vestibulocerebel", "ansiform", "lingula",
        "dentate nucleus",      # DN = cerebellar deep nucleus
        "inferior olive",       # closely linked to cerebellum
        "lateral reticular nucleus",  # precerebellar
        "nucleus of roller",
        "paramedian reticular",
        "tegmental reticular",
    ]),

    # ── Hippocampus (HPF) ────────────────────────────────────────────────────
    ("Hippocampus", [
        "dentate gyrus", "hippocampal", "hippocampo",
        "field ca", "cornu ammonis",
        "subiculum", "presubiculum", "parasubiculum", "postsubiculum",
        "prosubiculum", "fasciola",
        "induseum griseum", "taenia tecta",
    ]),

    # ── Thalamus ─────────────────────────────────────────────────────────────
    ("Thalamus", [
        "nucleus of thalamus", "of the thalamus",
        "anteroventral nucleus of thalamus",
        "anterodorsal nucleus",           # AD
        "anteromedial nucleus",           # AM
        "anteroventral nucleus",
        "geniculate", "pulvinar",
        "habenula", "habenular",          # LH, MH
        "epithalamic",
        "mediodorsal", "reuniens", "rhomboid", "centromedian",
        "parafascicular", "parataenial",
        "laterodorsal nucleus", "lateral posterior", "posterior complex",
        "ventral anterior", "ventral lateral", "ventral posterior",
        "ventromedial thal", "centrolateral", "central lateral",
        "central medial", "intermediodorsal", "paracentral",
        "peripeduncular", "subparafascicular",
        "ethmoid", "reticular nucleus of the thalamus",
        "submedial nucleus", "suprageniculate",
        "zona incerta",
        "posterior intralaminar", "posterior triangular",
        "xiphoid thalamic",
        "perireunensis",          # PR
        "interanterodorsal", "interanteromedial",
        "nucleus of the posterior commissure",
        "nucleus of the optic tract",
        "olivary pretectal",
        "medial terminal nucleus of the accessory",
        "lateral terminal nucleus of the accessory",
        "dorsal terminal nucleus of the accessory",
    ]),

    # ── Hypothalamus ─────────────────────────────────────────────────────────
    ("Hypothalamus", [
        "hypothalamus", "hypothalamic",
        "mammillary", "supraoptic", "suprachiasmatic",
        "arcuate", "paraventricular hypothal",
        "dorsomedial nucleus", "ventromedial hypothal",
        "lateral hypothal", "posterior hypothal",
        "tuberal", "tuberomammil", "perifornical",
        "anteroventral periventricular", "anterodorsal preoptic",
        "anteroventral preoptic", "parastrial", "subparaventricular",
        "periventricular hypothal", "retrochiasmatic",
        "preparasubthalamic", "subthalamic",
        # preoptic area (classified with hypothalamus in Allen atlas)
        "preoptic area", "preoptic nucleus",
        "lateral preoptic", "medial preoptic",
        "median preoptic", "posterodorsal preoptic",
        "ventrolateral preoptic", "ventromedial preoptic",
        "median eminence",
        "vascular organ of the lamina",
        "subfornical organ",
        "subcommissural organ",
        "fields of forel",
    ]),

    # ── Amygdala ─────────────────────────────────────────────────────────────
    ("Amygdala", [
        "amygdala", "amygdalar",
        "basolateral", "basomedial",
        "central amygdala", "intercalated",
        "piriform-amygdalar",
        "anterior amygdal",
        "hippocampo-amygdalar",
        "cortical amygdal",
        "endopiriform",
        "postpiriform",
    ]),

    # ── Striatum / Pallidum / Septal ──────────────────────────────────────────
    ("Striatum/Pallidum", [
        "striatum", "pallidum",
        "caudoputamen", "nucleus accumbens",
        "globus pallidus", "fundus of striatum",
        "olfactory tubercle",
        "diagonal band",
        "substantia innominata",
        "bed nuclei of the stria terminalis",
        "bed nucleus of the anterior commissure",
        # Septal nuclei (Allen atlas groups under CTX-SP / pallidum)
        "lateral septal nucleus", "medial septal nucleus",
        "septofimbrial", "septohippocampal",
        "triangular nucleus of septum",
        "magnocellular nucleus",   # MA — basal forebrain
    ]),

    # ── Olfactory areas ───────────────────────────────────────────────────────
    ("Olfactory", [
        "olfactory bulb", "olfactory area", "olfactory nerve",
        "anterior olfactory",
        "piriform area", "piriform cortex",
        "entorhinal", "ectorhinal",
        "postrhinal", "perirhinal",
        "accessory olfactory",
        "taenia tecta",
        "nucleus of the lateral olfactory tract",
    ]),

    # ── Midbrain ─────────────────────────────────────────────────────────────
    ("Midbrain", [
        "midbrain",
        "periaqueductal",
        "superior colliculus", "inferior colliculus",
        "substantia nigra", "red nucleus",
        "pedunculopontine", "cuneiform nucleus",
        "edinger-westphal", "oculomotor nucleus", "trochlear nucleus",
        "interpeduncular", "paranigral",
        "ventral tegmental area",    # VTA
        "anterior pretectal", "posterior pretectal", "medial pretectal",
        "parabigeminal",
        "interstitial nucleus of cajal",
        "nucleus of darkschewitsch",
        "nucleus incertus",
        "dorsal raphe", "median raphe",
        "central linear nucleus raphe",
        "rostral linear nucleus raphe",
        "interfascicular nucleus raphe",
        "superior central nucleus raphe",
        "nucleus sagulum",
        "paratrochlear",
        "supraoculomotor",
    ]),

    # ── Hindbrain (Pons + Medulla) ────────────────────────────────────────────
    ("Hindbrain", [
        "pons", "pontine",
        "medulla", "medullary",
        "parabrachial", "locus coer", "locus ceruleus",
        "raphe pallidus", "raphe magnus", "raphe obscurus", "raphe pontis",
        "abducens", "facial motor nucleus", "facial nucleus", "hypoglossal",
        "spinal nucleus", "trigeminal",
        "gigantocellular", "paragigantocellular",
        "nucleus of the solitary", "dorsal motor nucleus",
        "external cuneate",
        "rostral ventrolateral", "sublaterodorsal",
        "subceruleus", "supratrigeminal",
        "tegmental nucleus", "anterior tegmental",
        "vestibular nucleus", "vestibulocochlear",
        "koelliker-fuse",
        "parapyramidal nucleus",
        "parvicellular reticular", "intermediate reticular",
        "nucleus prepositus",
        "nucleus ambiguus",
        "inferior salivatory", "accessory facial",
        "dorsal cochlear", "ventral cochlear",
        "superior olivary",
        "nucleus of the trapezoid body",
        "nucleus of the lateral lemniscus",
        "barrington",
        "paratrigeminal", "peritrigeminal",
        "intertrigeminal",
        "nucleus x", "nucleus y",
        "supragenual nucleus",
    ]),

    # ── Isocortex ─────────────────────────────────────────────────────────────
    # Broad rule last — catches all remaining laminated cortical areas
    ("Isocortex", [
        "layer 1", "layer 2", "layer 2/3", "layer 4",
        "layer 5", "layer 6a", "layer 6b",
        "frontal pole", "temporal association",
        "retrosplenial", "prelimbic", "infralimbic",
        "anterior cingulate", "posterior cingulate",
        "orbital area", "gustatory area", "visceral area",
        "auditory area", "visual area", "somatosensory",
        "somatomotor", "motor area",
        "claustrum", "cortical subplate",
        "area prostriata", "dorsal peduncular area",
    ]),
]

# Display names for the hyperstructures (title in the figure)
_DISPLAY_NAMES = {
    "Isocortex":        "Isocortex",
    "Hippocampus":      "Hippocampus",
    "Thalamus":         "Thalamus",
    "Hypothalamus":     "Hypothalamus",
    "Amygdala":         "Amygdala",
    "Striatum/Pallidum":"Striatum / Pallidum",
    "Olfactory":        "Olfactory Areas",
    "Midbrain":         "Midbrain",
    "Hindbrain":        "Hindbrain",
    "Cerebellum":       "Cerebellum",
}

# Colour map: matches the figure in the paper (dark = high usage)
_CMAP = "YlGnBu"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Assign brain areas to every region in lightsheet_data
# ─────────────────────────────────────────────────────────────────────────────

def assign_brain_areas(lightsheet_data: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of lightsheet_data with 'Brain_Area' populated
    using region-name keyword rules.

    Parameters
    ----------
    lightsheet_data : long-format DataFrame with columns
        'Region_Name' and 'abbreviation' at minimum.

    Returns
    -------
    pd.DataFrame  (copy, does not modify the original)
    """
    df = lightsheet_data.copy()

    # Build mapping once from unique (Region_Name, abbreviation) pairs
    region_map = (
        df[["Region_Name", "abbreviation"]]
        .drop_duplicates()
        .copy()
    )
    region_map["Brain_Area"] = "Other"

    for area_label, keywords in _AREA_RULES:
        mask_unassigned = region_map["Brain_Area"] == "Other"
        name_lower = region_map["Region_Name"].str.lower()
        keyword_match = name_lower.apply(
            lambda n: any(kw in n for kw in keywords)
        )
        region_map.loc[mask_unassigned & keyword_match, "Brain_Area"] = area_label

    # Merge back
    df = df.drop(columns=["Brain_Area"], errors="ignore")
    df = df.merge(
        region_map[["abbreviation", "Brain_Area"]].drop_duplicates(),
        on="abbreviation",
        how="left",
    )
    df["Brain_Area"] = df["Brain_Area"].fillna("Other")

    # Report
    counts = df[["abbreviation", "Brain_Area"]].drop_duplicates()["Brain_Area"].value_counts()
    print("Brain area assignment summary (unique regions):")
    for area, n in counts.items():
        print(f"  {area:<22s}: {n:>4d} regions")
    unassigned = (counts.get("Other", 0))
    if unassigned > 0:
        print(f"  ⚠  {unassigned} regions could not be assigned and are labelled 'Other'.")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Build usage matrix
# ─────────────────────────────────────────────────────────────────────────────

def _build_usage_matrix(
    lightsheet_data: pd.DataFrame,
    comparisonNames: list,
    featureLists: list,
    cv_count: int,
    normalize: bool,
    filterByFreq: float,
) -> pd.DataFrame:
    """
    Build a (region × comparison) DataFrame where each cell is the
    number of times the classifier selected that region, optionally
    normalised to % of CV folds.

    Parameters
    ----------
    lightsheet_data : DataFrame with 'abbreviation' and 'Brain_Area' columns.
    comparisonNames : list of str – column labels, e.g. ['Ctrl vs FS', ...].
    featureLists    : list[list[list[str]]] – featureLists[comp][fold] = [region, ...]
    cv_count        : total number of CV folds used for normalisation.
    normalize       : if True, divide counts by cv_count * 100 (→ %).
    filterByFreq    : minimum raw count; rows below are dropped.

    Returns
    -------
    pd.DataFrame  index=abbreviation, columns=comparisonNames + ['Brain_Area']
    """
    # Flatten each comparison's fold lists into a single list
    flat_lists = [
        [region for fold in comp_folds for region in fold]
        for comp_folds in featureLists
    ]

    # All unique regions that appear in the data
    all_regions = (
        lightsheet_data[["abbreviation", "Brain_Area"]]
        .drop_duplicates()
        .set_index("abbreviation")
    )

    df = pd.DataFrame(0.0, index=all_regions.index, columns=comparisonNames)

    for comp_name, flat in zip(comparisonNames, flat_lists):
        counter = Counter(flat)
        for region, cnt in counter.items():
            if region in df.index:
                df.loc[region, comp_name] = cnt

    # Filter rows below threshold
    df = df[df.sum(axis=1) >= filterByFreq]

    # Normalise
    if normalize and cv_count > 0:
        df = df / cv_count * 100.0

    # Attach Brain_Area
    df["Brain_Area"] = all_regions.reindex(df.index)["Brain_Area"]
    df["Brain_Area"] = df["Brain_Area"].fillna("Other")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Plot a single heatmap for one hyperstructure
# ─────────────────────────────────────────────────────────────────────────────

def _plot_single_heatmap(
    df_area: pd.DataFrame,
    comparisonNames: list,
    area_label: str,
    display_name: str,
    normalize: bool,
    save_svg: bool,
    save_png: bool,
    out_dir: str,
):
    """
    Render and save one heatmap for a single brain hyperstructure.

    Parameters
    ----------
    df_area        : (region × comparison) DataFrame, already filtered to one area.
    comparisonNames: ordered column names.
    area_label     : internal label, used for the filename.
    display_name   : human-readable title shown on the figure.
    normalize      : whether values are in % (affects colour-bar label).
    save_svg / save_png : file-format flags.
    out_dir        : output directory path.
    """
    if df_area.empty:
        print(f"  [skip] {display_name} — no regions passed the frequency filter.")
        return

    # Sort rows by total usage (descending)
    df_area = df_area.loc[df_area.sum(axis=1).sort_values(ascending=False).index]

    matrix = df_area[comparisonNames].values
    yticklabels = df_area.index.tolist()
    xticklabels = comparisonNames

    n_rows = len(yticklabels)
    n_cols = len(xticklabels)

    # Cell size in inches
    cell_w = 0.30
    cell_h = 0.22
    left_margin  = 1.2   # space for y-tick labels
    bottom_margin = 1.6  # space for rotated x-tick labels
    right_margin  = 1.2  # colour-bar space
    top_margin    = 0.5

    fig_w = left_margin + n_cols * cell_w + right_margin
    fig_h = top_margin  + n_rows * cell_h + bottom_margin

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Colour range: 0 → max value in this panel
    vmax = np.nanmax(matrix) if np.nanmax(matrix) > 0 else 1.0

    cbar_label = "Usage in classifier (%)" if normalize else "Selection count"

    hm = sns.heatmap(
        matrix,
        ax=ax,
        cmap=_CMAP,
        vmin=0,
        vmax=vmax,
        linewidths=0.4,
        linecolor="black",
        square=True,
        yticklabels=yticklabels,
        xticklabels=xticklabels,
        cbar=True,
        cbar_kws={"label": cbar_label, "shrink": 0.6},
    )

    # Axis formatting
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=7,
    )
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    ax.tick_params(left=True, bottom=True, width=0.5, length=2)

    # Title
    ax.set_title(display_name, fontsize=9, fontweight="bold", pad=6)

    # Y-axis label
    ax.set_ylabel(display_name, fontsize=8)
    ax.set_xlabel("")

    plt.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"FeatureUsageHeatmap_{area_label.replace('/', '_')}")
    if save_svg:
        fig.savefig(stem + ".svg", format="svg", bbox_inches="tight")
        print(f"  Saved: {stem}.svg")
    if save_png:
        fig.savefig(stem + ".png", format="png", dpi=300, bbox_inches="tight")
        print(f"  Saved: {stem}.png")

    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def plot_classifier_heatmaps(
    lightsheet_data: pd.DataFrame,
    comparisonNames: list,
    featureLists: list,
    filterByFreq: float = 0.0,
    dirDict: dict = None,
    hyperstructures: list = None,
    normalize: bool = True,
    cv_count: int = 100,
    save_svg: bool = True,
    save_png: bool = True,
    reassign_brain_areas: bool = True,
):
    """
    Generate one feature-usage heatmap per brain hyperstructure.

    Parameters
    ----------
    lightsheet_data : long-format DataFrame from lf.load_stressor_data().
    comparisonNames : list of str – comparison labels in the desired display order.
    featureLists    : list[list[list[str]]] – featureLists[i][fold] = [region, ...]
                      as returned by hf.retrieve_dict_data().
    filterByFreq    : minimum raw selection count for a region to appear.
                      Typically 0.75 * cv_count.
    dirDict         : pipeline directory dict; output goes to
                      dirDict['crossComp_figDir'] if provided, else './figures'.
    hyperstructures : list of area labels to plot, e.g. ['Isocortex','Thalamus'].
                      Pass None to plot ALL supported hyperstructures.
    normalize       : if True, cell values are shown as % of cv_count folds.
    cv_count        : number of CV folds (used for normalisation; should match
                      classifyDict['CV_count']).
    save_svg        : save vector SVG file.
    save_png        : save raster PNG file at 300 dpi.
    reassign_brain_areas : if True (default), always re-run the keyword-based
                      brain-area assignment regardless of any existing Brain_Area
                      column (recommended, because the loader sets it to 'Unknown').
    """

    # ── Output directory ─────────────────────────────────────────────────────
    if dirDict is not None and "crossComp_figDir" in dirDict:
        out_dir = dirDict["crossComp_figDir"]
    elif dirDict is not None and "outDir" in dirDict:
        out_dir = os.path.join(dirDict["outDir"], "figures")
    else:
        out_dir = os.path.join(os.getcwd(), "figures")

    # ── Brain area assignment ─────────────────────────────────────────────────
    if reassign_brain_areas or lightsheet_data["Brain_Area"].eq("Unknown").all():
        print("Assigning brain areas from region names …")
        data = assign_brain_areas(lightsheet_data)
    else:
        data = lightsheet_data.copy()

    # ── Which hyperstructures to plot ─────────────────────────────────────────
    all_supported = [label for label, _ in _AREA_RULES]
    # Deduplicate while preserving order
    seen = set()
    all_supported_ordered = []
    for x in all_supported:
        if x not in seen:
            all_supported_ordered.append(x)
            seen.add(x)

    if hyperstructures is None:
        hyperstructures = all_supported_ordered
    else:
        # Validate
        unknown = [h for h in hyperstructures if h not in all_supported_ordered]
        if unknown:
            print(f"  Warning: unrecognised hyperstructure(s): {unknown}")
            print(f"  Supported: {all_supported_ordered}")
        hyperstructures = [h for h in hyperstructures if h in all_supported_ordered]

    print(f"\nHyperstructures to plot: {hyperstructures}")

    # ── Build global usage matrix (all comparisons × all regions) ─────────────
    print("\nBuilding feature-usage matrix …")
    usage_df = _build_usage_matrix(
        lightsheet_data=data,
        comparisonNames=comparisonNames,
        featureLists=featureLists,
        cv_count=cv_count,
        normalize=normalize,
        filterByFreq=filterByFreq,
    )

    print(f"  Usage matrix: {usage_df.shape[0]} regions × {len(comparisonNames)} comparisons")
    print(f"  Value range:  {usage_df[comparisonNames].min().min():.1f} – "
          f"{usage_df[comparisonNames].max().max():.1f}"
          f"{'%' if normalize else ' counts'}")

    # ── Plot one heatmap per hyperstructure ───────────────────────────────────
    print(f"\nGenerating heatmaps → {out_dir}\n")

    for area_label in hyperstructures:
        display_name = _DISPLAY_NAMES.get(area_label, area_label)
        df_area = usage_df[usage_df["Brain_Area"] == area_label][comparisonNames].copy()

        print(f"  Plotting {display_name} ({len(df_area)} regions) …")
        _plot_single_heatmap(
            df_area=df_area,
            comparisonNames=comparisonNames,
            area_label=area_label,
            display_name=display_name,
            normalize=normalize,
            save_svg=save_svg,
            save_png=save_png,
            out_dir=out_dir,
        )

    print("\nDone.")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Optional: combined multi-panel figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_classifier_heatmaps_combined(
    lightsheet_data: pd.DataFrame,
    comparisonNames: list,
    featureLists: list,
    filterByFreq: float = 0.0,
    dirDict: dict = None,
    hyperstructures: list = None,
    normalize: bool = True,
    cv_count: int = 100,
    n_cols: int = 2,
    save_svg: bool = True,
    save_png: bool = True,
    reassign_brain_areas: bool = True,
):
    """
    Same as plot_classifier_heatmaps() but arranges all hyperstructures
    into a single multi-panel figure (n_cols panels per row).
    Useful for a supplementary figure.
    """

    if dirDict is not None and "crossComp_figDir" in dirDict:
        out_dir = dirDict["crossComp_figDir"]
    elif dirDict is not None and "outDir" in dirDict:
        out_dir = os.path.join(dirDict["outDir"], "figures")
    else:
        out_dir = os.path.join(os.getcwd(), "figures")

    if reassign_brain_areas or lightsheet_data["Brain_Area"].eq("Unknown").all():
        print("Assigning brain areas …")
        data = assign_brain_areas(lightsheet_data)
    else:
        data = lightsheet_data.copy()

    all_supported = list(dict.fromkeys([label for label, _ in _AREA_RULES]))
    if hyperstructures is None:
        hyperstructures = all_supported
    else:
        hyperstructures = [h for h in hyperstructures if h in all_supported]

    usage_df = _build_usage_matrix(
        lightsheet_data=data,
        comparisonNames=comparisonNames,
        featureLists=featureLists,
        cv_count=cv_count,
        normalize=normalize,
        filterByFreq=filterByFreq,
    )

    # Filter to only areas that have regions
    valid_areas = [
        a for a in hyperstructures
        if not usage_df[usage_df["Brain_Area"] == a][comparisonNames].empty
    ]

    n_rows_fig = int(np.ceil(len(valid_areas) / n_cols))
    cell_w = 0.30
    cell_h = 0.22
    n_comps = len(comparisonNames)

    # Estimate per-panel height from largest area
    max_regions = max(
        len(usage_df[usage_df["Brain_Area"] == a]) for a in valid_areas
    )

    panel_w = 1.0 + n_comps * cell_w + 1.0   # margins + cells + cbar
    panel_h = 0.5 + max_regions * cell_h + 1.5

    fig_w = n_cols * panel_w
    fig_h = n_rows_fig * panel_h

    fig, axes = plt.subplots(n_rows_fig, n_cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).flatten()

    vmax_global = usage_df[comparisonNames].max().max()
    cbar_label = "Usage (%)" if normalize else "Selection count"

    for idx, area_label in enumerate(valid_areas):
        ax = axes[idx]
        display_name = _DISPLAY_NAMES.get(area_label, area_label)
        df_area = (
            usage_df[usage_df["Brain_Area"] == area_label][comparisonNames]
            .copy()
        )
        df_area = df_area.loc[df_area.sum(axis=1).sort_values(ascending=False).index]

        # Add colour-bar only on rightmost column panels
        show_cbar = (idx % n_cols == n_cols - 1) or (idx == len(valid_areas) - 1)

        sns.heatmap(
            df_area.values,
            ax=ax,
            cmap=_CMAP,
            vmin=0,
            vmax=vmax_global,
            linewidths=0.3,
            linecolor="black",
            square=True,
            yticklabels=df_area.index.tolist(),
            xticklabels=comparisonNames,
            cbar=show_cbar,
            cbar_kws={"label": cbar_label, "shrink": 0.5} if show_cbar else {},
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45, ha="right", rotation_mode="anchor", fontsize=6,
        )
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
        ax.set_title(display_name, fontsize=8, fontweight="bold")
        ax.tick_params(left=True, bottom=True, width=0.4, length=1.5)

    # Hide empty axes
    for idx in range(len(valid_areas), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Classifier Feature Usage by Brain Region", fontsize=10, fontweight="bold", y=1.01)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, "FeatureUsageHeatmap_combined")
    if save_svg:
        fig.savefig(stem + ".svg", format="svg", bbox_inches="tight")
        print(f"Saved: {stem}.svg")
    if save_png:
        fig.savefig(stem + ".png", format="png", dpi=300, bbox_inches="tight")
        print(f"Saved: {stem}.png")

    plt.show()
    plt.close(fig)
    print("Done.")
