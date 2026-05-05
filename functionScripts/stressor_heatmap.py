"""
stressor_heatmap.py
===================
Whole-brain cFos density heatmap for the stressor dataset.

Layout
------
  Y-axis : brain regions grouped by Allen Brain Atlas hyperstructure,
           separated by horizontal dividers with colour-coded labels
  X-axis : Stressor × Timepoint columns, stressors grouped together,
           timepoints separated within each stressor group
           Ctrl: Acute|7D|14D|21D  |  FS: Acute|7D|14D  |  FSW/RS/TS: Acute|7D|14D|21D

Each cell shows the mean cFos density across the 5 animals in that group.
Fiber tracts and ventricles are excluded (marked "Other").

Public API
----------
    plot_density_heatmap(
        csv_path   = "merged_wide_with_acro.csv",
        out_dir    = "Output/",
        scale      = 1e10,          # multiply raw values for display
        vmax_pct   = 99,            # percentile to saturate colormap
    )

    # or call directly:
    python stressor_heatmap.py
"""

import re
import warnings
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

# ── Default paths ──────────────────────────────────────────────────────────────
_DEFAULT_CSV = "merged_wide_with_acro.csv"
_DEFAULT_OUT = "/Output"

# ── Ordered axes ───────────────────────────────────────────────────────────────
STRESSOR_ORDER  = ["Ctrl", "FS", "FSW", "RS", "TS"]
TIMEPOINT_ORDER = ["Acute", "7D", "14D", "21D"]

STRUCTURE_ORDER = [
    "Isocortex",
    "Olfactory",
    "Hippocampus",
    "Striatum & Pallidum",
    "Thalamus",
    "Hypothalamus",
    "Midbrain & Hindbrain",
    "Cerebellum",
]

STRUCTURE_COLORS = {
    "Isocortex":            "#4e79a7",
    "Olfactory":            "#f28e2b",
    "Hippocampus":          "#59a14f",
    "Striatum & Pallidum":  "#b07aa1",
    "Thalamus":             "#9c755f",
    "Hypothalamus":         "#e15759",
    "Midbrain & Hindbrain": "#76b7b2",
    "Cerebellum":           "#edc948",
}

STRESSOR_COLORS = {
    "Ctrl": "#888888",
    "FS":   "#4e79a7",
    "FSW":  "#59a14f",
    "RS":   "#e15759",
    "TS":   "#f28e2b",
}

CMAP = LinearSegmentedColormap.from_list(
    "cfos",
    ["#0d0010", "#3b0030", "#7a0060", "#c0307a",
     "#e8709a", "#f5c0d0", "#fdf0f4", "#ffffff"],
    N=512,
)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Brain-structure assignment
# ═════════════════════════════════════════════════════════════════════════════
def assign_structure(name: str, acro: str) -> str:
    """Rule-based assignment using Region.Name (lower) and acronym."""
    n = name.lower()
    a = str(acro)

    # ── Cerebellum ────────────────────────────────────────────────────────────
    if any(kw in n for kw in [
            "cerebell", "purkinje", "lobule", "ansiform", "paraflocculus",
            "flocculus", "lingula", "culmen", "declive", "folium", "tuber",
            "pyramis", "uvula", "nodulus", "fastigial", "interposed",
            "anterior lobe", "posterior lobe", "flocculonodular"]):
        return "Cerebellum"
    for pfx in ("AN", "CUL4", "CUL5", "DEC", "FLO", "FOL", "LIN", "NOD",
                "PFL", "PRM", "PYR", "SIM", "CENT", "COP", "UVU", "GRN", "ICB"):
        if a.startswith(pfx):
            return "Cerebellum"

    # ── Fiber tracts / ventricles → Other ────────────────────────────────────
    if any(kw in n for kw in [
            "fiber tract", "ventricle", "ventricular", "white matter",
            "commissure", "capsule", "fasciculus", "lemniscus", "peduncle",
            "stria", "tract ", "fornix", "corpus callosum", "fimbria",
            "alveus", "brachium", "internal capsule", "external capsule",
            "optic tract", "optic nerve", "olfactory nerve", "olfactory tract",
            "habenular commissure", "anterior commissure",
            "posterior commissure"]):
        return "Other"
    if a in ("fiber tracts", "root", "VS") or (len(a) >= 2 and a[:2].islower()):
        return "Other"

    # ── Isocortex ─────────────────────────────────────────────────────────────
    _hc_excl = ("entorhinal", "hippocampal", "subicular", "parasubiculum",
                "claustrum", "endopiriform")
    if any(kw in n for kw in [
            "agranular insular", "anterior cingulate", "frontal pole",
            "gustatory", "infralimbic", "prelimbic", "orbital", "perirhinal",
            "postrhinal", "retrosplenial", "temporal association",
            "visceral area", "posterior parietal", "motor area",
            "somatosensory", "auditory area", "visual area", "visual areas",
            "anteromedial area", "anterolateral area",
            "laterointermediate area", "posterolateral area",
            "posteromedial area", "rostrolateral area", "lateromedial area"]):
        if not any(hk in n for hk in _hc_excl):
            return "Isocortex"
    if re.search(r', layer [1-6]', n):
        if not any(hk in n for hk in _hc_excl):
            return "Isocortex"
    for pfx in ("ACA", "AI", "AUDd", "AUDp", "AUDpo", "AUDv", "ECT", "FRP",
                "GU", "ILA", "MOp", "MOs", "ORB", "PERI", "PL", "PTLp",
                "RSP", "SSp", "SSs", "TEa", "VIS", "VISC"):
        if a.startswith(pfx):
            return "Isocortex"

    # ── Olfactory ─────────────────────────────────────────────────────────────
    if any(kw in n for kw in [
            "olfactory bulb", "anterior olfactory", "taenia tecta",
            "dorsal peduncular", "piriform", "postpiriform",
            "cortical amygdaloid", "olfactory tubercle", "main olfactory",
            "accessory olfactory", "endopiriform", "tenia tecta",
            "piriform-amygdalar"]):
        return "Olfactory"
    for pfx in ("AOB", "AON", "COA", "DP", "EPd", "EPv", "NLOT",
                "PAA", "PIR", "TT", "TR"):
        if a.startswith(pfx):
            return "Olfactory"

    # ── Hippocampus ───────────────────────────────────────────────────────────
    if any(kw in n for kw in [
            "hippocampal", "hippocampus", "dentate gyrus", "field ca",
            "cornu ammonis", "entorhinal", "parasubiculum", "postsubiculum",
            "presubiculum", "prosubiculum", "subiculum", "fasciola cinerea",
            "induseum griseum"]):
        return "Hippocampus"
    for pfx in ("CA1", "CA2", "CA3", "DG", "ENT", "FC", "HATA",
                "IG", "PAR", "POST", "PRE", "ProS", "SUB"):
        if a.startswith(pfx):
            return "Hippocampus"

    # ── Striatum & Pallidum ───────────────────────────────────────────────────
    if any(kw in n for kw in [
            "striatum", "caudate putamen", "accumbens", "olfactory tubercle",
            "lateral septal", "septal", "pallidum", "globus pallidus",
            "substantia innominata", "medial septal", "diagonal band",
            "central amygdalar", "intercalated amygdalar",
            "anterior amygdalar", "bed nucleus of the stria terminalis",
            "fundus of the striatum", "islands of calleja",
            "nucleus accumbens", "striatal", "pallidal"]):
        return "Striatum & Pallidum"
    for pfx in ("AAA", "ACB", "BA", "BST", "CEA", "CP", "GPe", "GPi", "GPI",
                "IA", "LA", "LS", "MA", "MS", "NDB", "OT", "PAL",
                "SF", "SI", "STR", "BLA", "BMA", "MEA"):
        if a.startswith(pfx) and len(a) <= len(pfx) + 3:
            return "Striatum & Pallidum"

    # ── Thalamus ──────────────────────────────────────────────────────────────
    if any(kw in n for kw in [
            "thalamus", "thalamic", "geniculate", "habenula", "epithalamus",
            "reticular nucleus of the thalamus"]):
        return "Thalamus"
    for pfx in ("AD", "AM", "ATN", "AV", "CL", "CM", "CLA",
                "Eth", "GENd", "GENv", "IAD", "IAM", "IGL", "ILM", "IMD",
                "IntG", "LD", "LGd", "LGv", "LHb", "LP", "MD", "MED",
                "MG", "MH", "PCN", "PF", "PIL", "PO", "POL", "PT",
                "PVT", "RE", "RH", "RT", "SGN", "SMT", "SPF",
                "SubG", "VAL", "VM", "VPL", "VPLpc", "VPM", "VPMpc", "Xi", "ZI"):
        if a.startswith(pfx):
            return "Thalamus"

    # ── Hypothalamus ──────────────────────────────────────────────────────────
    if any(kw in n for kw in [
            "hypothalamus", "hypothalamic", "arcuate", "dorsomedial hyp",
            "lateral hypothal", "mammillary", "median eminence",
            "medial preoptic", "paraventricular", "periventricular",
            "suprachiasmatic", "supraoptic", "ventromedial hyp",
            "subfornical organ", "tuberomammillary", "zona incerta"]):
        return "Hypothalamus"
    for pfx in ("ADP", "AHN", "ARH", "ASO", "AVPV",
                "DMH", "LHA", "LM", "LPO", "ME",
                "MM", "MBO", "MPN", "MPO",
                "PH", "PMd", "PMv", "PVH", "PVZ",
                "RCH", "SCH", "SFO", "SLD", "SO",
                "STN", "SUM", "TM", "TU", "VMH", "ZI"):
        if a.startswith(pfx):
            return "Hypothalamus"

    # ── Midbrain & Hindbrain ──────────────────────────────────────────────────
    if any(kw in n for kw in [
            "midbrain", "hindbrain", "medulla", "pons", "raphe",
            "periaqueductal", "substantia nigra", "ventral tegmental",
            "superior colliculus", "inferior colliculus", "pretectal",
            "red nucleus", "pedunculopontine", "locus coeruleus",
            "nucleus of the solitary", "dorsal motor nucleus",
            "hypoglossal", "facial motor", "cochlear", "vestibular",
            "olivary", "parabrachial", "reticular", "tegmental",
            "interpeduncular", "dorsal raphe", "median raphe",
            "cuneate", "gracile", "spinal trigeminal", "inferior olivary",
            "area postrema", "nucleus ambiguus", "abducens"]):
        return "Midbrain & Hindbrain"

    # broad fallback on name
    if any(kw in n for kw in ["cortex", "cortical"]):
        return "Isocortex"
    if any(kw in n for kw in ["nucleus", "area"]):
        return "Midbrain & Hindbrain"

    return "Other"


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Data loading
# ═════════════════════════════════════════════════════════════════════════════
def _load_and_prepare(csv_path: str):
    """
    Returns
    -------
    df_wide   : pd.DataFrame  — original wide CSV with added 'structure' column
    mean_data : dict          — (stressor, timepoint) → np.array of region means
    col_keys  : list          — ordered list of (stressor, timepoint) tuples
    """
    df_wide = pd.read_csv(csv_path, sep=";", na_values=["NA", "na", "N/A", ""])

    meta_cols = ["Region.Name", "Region ID", "acronym"]
    data_cols = [c for c in df_wide.columns if c not in meta_cols]

    # Assign brain structures
    df_wide["structure"] = [
        assign_structure(n, a)
        for n, a in zip(df_wide["Region.Name"], df_wide["acronym"])
    ]

    # Build ordered list of individual animal columns: (stressor, timepoint, csv_col)
    parsed = []
    for c in data_cols:
        parts = c.replace("_normalized", "").split("_")
        parsed.append((parts[0], parts[1], c))

    def _sort_key(p):
        s, t, _ = p
        return (STRESSOR_ORDER.index(s) if s in STRESSOR_ORDER else 99,
                TIMEPOINT_ORDER.index(t) if t in TIMEPOINT_ORDER else 99)

    col_keys = sorted(parsed, key=_sort_key)

    return df_wide, col_keys


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Region ordering
# ═════════════════════════════════════════════════════════════════════════════
def _build_region_order(df_wide):
    """Returns (ordered_idx, struct_boundaries)."""
    ordered_idx       = []
    struct_boundaries = {}

    for struct in STRUCTURE_ORDER:
        idx   = df_wide.index[df_wide["structure"] == struct].tolist()
        start = len(ordered_idx)
        ordered_idx.extend(idx)
        struct_boundaries[struct] = (start, start + len(idx))

    return ordered_idx, struct_boundaries


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Matrix assembly
# ═════════════════════════════════════════════════════════════════════════════
def _build_matrix(df_wide, col_keys, ordered_idx):
    """col_keys is list of (stressor, timepoint, csv_col)."""
    mat = np.full((len(ordered_idx), len(col_keys)), np.nan)
    for j, (s, t, csv_col) in enumerate(col_keys):
        vals = df_wide[csv_col].values
        for i, ridx in enumerate(ordered_idx):
            mat[i, j] = vals[ridx]
    return mat


# ═════════════════════════════════════════════════════════════════════════════
# 5.  Plotting
# ═════════════════════════════════════════════════════════════════════════════
def _make_figure(mat, col_keys, ordered_idx, df_wide, struct_boundaries,
                 scale=1e10, vmax_pct=99):
    """
    Build and return the matplotlib Figure.

    Parameters
    ----------
    scale    : float   multiply raw values by this factor for display
    vmax_pct : float   percentile of scaled values used as colormap ceiling
    """
    n_rows, n_cols = mat.shape

    # ── Scale values ──────────────────────────────────────────────────────────
    mat_disp  = mat * scale
    flat_vals = mat_disp[~np.isnan(mat_disp)]
    vmin      = 0.0
    vmax      = float(np.percentile(flat_vals, vmax_pct))
    scale_exp = int(round(np.log10(1.0 / scale)))   # e.g. scale=1e10 → exp=−10
    unit_str  = f"×10⁻¹⁰ mm⁻³" if scale == 1e10 else f"(scaled ×{scale:.0e})"

    # ── Figure geometry ───────────────────────────────────────────────────────
    row_h        = 0.028
    fig_h        = max(14, n_rows * row_h + 3)
    left_lbl_in  = 0.60
    sbar_in      = 0.15
    heat_in      = n_cols * 0.09   # 95 cols × 0.09" ≈ same total width as 19-col mean plot
    gap_in       = 0.20
    cbar_in      = 0.25
    right_pad_in = 0.40
    fig_w        = left_lbl_in + sbar_in + heat_in + gap_in + cbar_in + right_pad_in

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0d0010")

    top_pad    = 0.12
    bot_pad    = 0.05
    heat_h_frac = 1.0 - top_pad - bot_pad

    def _fx(x_in):   return x_in / fig_w
    def _fy(y_frac): return y_frac

    ax_struct = fig.add_axes([_fx(left_lbl_in),
                               bot_pad,
                               _fx(sbar_in),
                               heat_h_frac])

    ax_heat   = fig.add_axes([_fx(left_lbl_in + sbar_in),
                               bot_pad,
                               _fx(heat_in),
                               heat_h_frac])

    ax_cbar   = fig.add_axes([_fx(left_lbl_in + sbar_in + heat_in + gap_in),
                               bot_pad + heat_h_frac * 0.10,
                               _fx(cbar_in),
                               heat_h_frac * 0.50])

    # ── Heatmap ───────────────────────────────────────────────────────────────
    im = ax_heat.imshow(
        mat_disp,
        aspect="auto",
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        origin="upper",
    )

    ax_heat.set_xlim(-0.5, n_cols - 0.5)
    ax_heat.set_ylim(n_rows - 0.5, -0.5)

    # ── X-axis: no per-animal tick labels; timepoint labels centered over groups ─
    ax_heat.set_xticks([])   # animals unlabeled

    # Build group maps
    stressor_groups = defaultdict(list)   # stressor → [col indices]
    tp_groups       = defaultdict(list)   # (stressor, timepoint) → [col indices]
    for j, (s, t, _) in enumerate(col_keys):
        stressor_groups[s].append(j)
        tp_groups[(s, t)].append(j)

    # Timepoint labels centered over each 5-animal block (same as v1 timepoint labels)
    for (s, t), js in tp_groups.items():
        mid = float(np.mean(js))
        ax_heat.annotate(
            t,
            xy=(mid, -0.5), xycoords=("data", "data"),
            xytext=(mid, -0.80), textcoords=("data", "data"),
            ha="center", va="bottom",
            fontsize=4.5, color="white",
            annotation_clip=False,
        )
        # Light separator between timepoint blocks within a stressor
        if min(js) > 0 and col_keys[min(js)][0] == col_keys[min(js) - 1][0]:
            ax_heat.axvline(min(js) - 0.5, color="#333333", lw=0.4, alpha=0.6)

    # Stressor group headers + coloured top bars + thick separators
    for s, js in stressor_groups.items():
        col  = STRESSOR_COLORS.get(s, "white")
        mid  = float(np.mean(js))
        xlo  = min(js) - 0.5
        xhi  = max(js) + 0.5

        ax_heat.annotate(
            s,
            xy=(mid, -0.5), xycoords=("data", "data"),
            xytext=(mid, -1.30), textcoords=("data", "data"),
            ha="center", va="bottom",
            fontsize=6.5, fontweight="bold", color=col,
            annotation_clip=False,
        )
        ax_heat.axvspan(xlo, xhi, ymin=1.0, ymax=1.015,
                        facecolor=col, clip_on=False, alpha=0.9)
        if min(js) > 0:
            ax_heat.axvline(min(js) - 0.5, color="#666666", lw=0.8, alpha=0.9)

    # ── Y-axis: region acronyms ───────────────────────────────────────────────
    acros = df_wide["acronym"].values[ordered_idx]
    ax_heat.set_yticks(range(n_rows))
    ax_heat.set_yticklabels(acros, fontsize=2.8, color="white")
    ax_heat.tick_params(axis="y", left=False, pad=0.5, colors="white", length=0)

    # ── Structure colour bar ──────────────────────────────────────────────────
    color_map_rows = np.zeros((n_rows, 1, 4))
    for struct, (r0, r1) in struct_boundaries.items():
        rgba = mcolors.to_rgba(STRUCTURE_COLORS.get(struct, "#555555"))
        color_map_rows[r0:r1, 0] = rgba

    ax_struct.imshow(
        color_map_rows,
        aspect="auto", interpolation="nearest",
        origin="upper",
        extent=[-0.5, 0.5, n_rows - 0.5, -0.5],
    )
    ax_struct.set_xlim(-0.5, 0.5)
    ax_struct.set_ylim(n_rows - 0.5, -0.5)

    # Structure labels + horizontal dividers
    for struct, (r0, r1) in struct_boundaries.items():
        if r1 <= r0:
            continue
        mid_row = (r0 + r1) / 2.0
        ax_struct.text(
            -1.2, mid_row, struct,
            ha="right", va="center",
            fontsize=5.5, fontweight="bold",
            color=STRUCTURE_COLORS.get(struct, "white"),
        )
        if r0 > 0:
            ax_heat.axhline(r0 - 0.5, color="#444444", lw=0.5, alpha=0.8)

    ax_struct.axis("off")

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cbar = plt.colorbar(im, cax=ax_cbar, orientation="vertical")
    cbar.set_label(
        f"c-Fos+ cell density\n({unit_str})",
        color="white", fontsize=7, labelpad=6,
    )
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=6)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.outline.set_edgecolor("white")
    cbar.outline.set_linewidth(0.5)

    # ── Spine / background style ──────────────────────────────────────────────
    ax_heat.set_facecolor("#0d0010")
    for spine in ax_heat.spines.values():
        spine.set_edgecolor("#444444")
        spine.set_linewidth(0.5)
    ax_heat.tick_params(axis="both", which="both", length=0)

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.995,
        "Whole-Brain cFos Density — Stressor × Timepoint",
        ha="center", va="top",
        fontsize=10, fontweight="bold", color="white",
    )

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 6.  Public entry point
# ═════════════════════════════════════════════════════════════════════════════
def plot_density_heatmap(
    csv_path : str   = _DEFAULT_CSV,
    out_dir  : str   = _DEFAULT_OUT,
    scale    : float = 1e10,
    vmax_pct : float = 99,
    save_svg : bool  = True,
    save_png : bool  = True,
    dpi      : int   = 200,
) -> plt.Figure:
    """
    Generate and save the stressor × timepoint density heatmap.

    Parameters
    ----------
    csv_path : path to merged_wide_with_acro.csv
    out_dir  : directory for saved figures
    scale    : multiply raw density values for display (default 1e10)
    vmax_pct : percentile of scaled values to use as colormap ceiling (default 99)
    save_svg : save SVG file
    save_png : save PNG file
    dpi      : PNG resolution

    Returns
    -------
    matplotlib Figure
    """
    os.makedirs(out_dir, exist_ok=True)

    print("Loading data …")
    df_wide, col_keys = _load_and_prepare(csv_path)

    struct_counts = df_wide["structure"].value_counts()
    print("Structure counts:\n" + struct_counts.to_string())

    print("Building region order …")
    ordered_idx, struct_boundaries = _build_region_order(df_wide)
    n_other = (df_wide["structure"] == "Other").sum()
    print(f"  {len(ordered_idx)} regions plotted  |  {n_other} fiber-tract/other excluded")

    print("Building matrix …")
    mat = _build_matrix(df_wide, col_keys, ordered_idx)
    print(f"  Shape: {mat.shape}  |  NaN: {np.isnan(mat).mean()*100:.1f}%")

    print("Plotting …")
    fig = _make_figure(mat, col_keys, ordered_idx, df_wide,
                       struct_boundaries, scale=scale, vmax_pct=vmax_pct)

    stem = os.path.join(out_dir, "stressor_cfos_heatmap")
    if save_svg:
        fig.savefig(stem + ".svg", format="svg", bbox_inches="tight",
                    facecolor="#0d0010")
        print(f"  Saved → {stem}.svg")
    if save_png:
        fig.savefig(stem + ".png", format="png", bbox_inches="tight",
                    facecolor="#0d0010", dpi=dpi)
        print(f"  Saved → {stem}.png")

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 7.  CLI entry point
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stressor cFos density heatmap")
    parser.add_argument("--csv",     default=_DEFAULT_CSV, help="Path to merged_wide_with_acro.csv")
    parser.add_argument("--out_dir", default=_DEFAULT_OUT,  help="Output directory")
    parser.add_argument("--scale",   default=1e10, type=float, help="Value scale factor (default 1e10)")
    parser.add_argument("--vmax_pct",default=99,   type=float, help="Colormap ceiling percentile (default 99)")
    args = parser.parse_args()

    fig = plot_density_heatmap(
        csv_path = args.csv,
        out_dir  = args.out_dir,
        scale    = args.scale,
        vmax_pct = args.vmax_pct,
    )
    plt.close(fig)
    print("Done.")
