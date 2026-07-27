"""
analyse_data.py
---------------
Reads Dataset_Output_*.xlsx (curvature) and Dataset_Input_*.xlsx (top angle)
and plots histograms for each group, pooled across all sheets (samples).

Dataset groups derived from Summary_of_Datasets.xlsx:
  - All (60–83)  : every dataset 60–83 with data
  - Random       : 60, 62, 64, 66, 661, 68, 70, 701, 82
  - Attractors   : 61, 63, 631, 632, 634, 65, 652, 653, 67, 671, 672, 69, 83
  - All Angles   : 611, 633, 651, 662
"""

# ┌───────────────────────────────────────────────────────────────────────────┐
# │  Imports                                                                  │
# └───────────────────────────────────────────────────────────────────────────┘

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, avoids Tkinter issues on Windows
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore", category=UserWarning)  # suppress openpyxl warnings

# ┌───────────────────────────────────────────────────────────────────────────┐
# │  Configuration                                                            │
# └───────────────────────────────────────────────────────────────────────────┘

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
BASE_DIR     = PROJECT_ROOT / "frustrated-composites-dataset"

# Output file columns
COL_MIN = "Min Curvature Length"
COL_MAX = "Max Curvature Length"

# Input file column
COL_TOP_ANGLE = "Top Angle"

# Histogram bins
N_BINS = 60

# x-axis range for curvature histograms (set to None to use full data range)
CURVATURE_RANGE = (-0.5, 0.5)

# x-axis range for top angle (None = auto)
TOP_ANGLE_RANGE = None

# Output folder for saved figures
OUTPUT_DIR = SCRIPT_DIR / "analysis_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# ┌───────────────────────────────────────────────────────────────────────────┐
# │  Dataset groups                                                           │
# └───────────────────────────────────────────────────────────────────────────┘

GROUPS = {
    "All (60-83)": [
        "60", "61", "611",
        "62", "63", "631", "632", "633", "634",
        "64", "65", "651", "652", "653",
        "66", "661", "662",
        "67", "671", "672",
        "68", "69",
        "70", "701",
        "82", "83",
    ],
    "Random": [
        "60", "62", "64", "66", "661", "68", "70", "701", "82",
    ],
    "Attractors": [
        "61", "63", "631", "632", "634",
        "65", "652", "653",
        "67", "671", "672",
        "69", "83",
    ],
    "All Angles": [
        "611", "633", "651", "662",
    ],
}

# ┌───────────────────────────────────────────────────────────────────────────┐
# │  Generic data loader                                                      │
# └───────────────────────────────────────────────────────────────────────────┘

def load_columns_from_excel(
    file_path: Path,
    columns: list,
) -> tuple:
    """
    Load one or more columns from all sheets of an Excel file,
    pooling values across sheets.

    Returns
    -------
    data       : dict mapping column name -> np.ndarray of float values
    n_samples  : number of sheets successfully read
    """
    if not file_path.exists():
        print(f"  [WARN] File not found, skipping: {file_path.name}")
        return {c: np.array([]) for c in columns}, 0

    try:
        xl = pd.ExcelFile(file_path, engine="openpyxl")
    except Exception as e:
        print(f"  [ERROR] Cannot open {file_path.name}: {e}")
        return {c: np.array([]) for c in columns}, 0

    accumulated = {c: [] for c in columns}
    n_samples = 0
    warned_columns = False

    for sheet in xl.sheet_names:
        try:
            df = xl.parse(sheet_name=sheet)
        except Exception as e:
            print(f"  [WARN] Cannot parse sheet '{sheet}' in {file_path.name}: {e}")
            continue

        missing = [c for c in columns if c not in df.columns]
        if missing:
            if not warned_columns:
                print(f"  [WARN] Missing {missing} in '{file_path.name}' (sheet '{sheet}').")
                print(f"         Available columns: {list(df.columns)}")
                warned_columns = True
            continue

        for col in columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            accumulated[col].append(vals.values)

        n_samples += 1

    if n_samples == 0:
        print(f"  [WARN] No valid data found in {file_path.name}")
        return {c: np.array([]) for c in columns}, 0

    return (
        {c: np.concatenate(accumulated[c]) for c in columns},
        n_samples,
    )


def load_group_data(
    dataset_ids: list,
    base_dir: Path,
    file_prefix: str,
    columns: list,
) -> tuple:
    """
    Load and pool column data for a group of dataset IDs.

    Parameters
    ----------
    file_prefix : "Dataset_Output" or "Dataset_Input"
    columns     : list of column names to load

    Returns
    -------
    pooled_data    : dict column -> pooled np.ndarray
    total_samples  : total sheets across all files
    loaded_ids     : dataset IDs that contributed data
    """
    accumulated = {c: [] for c in columns}
    total_samples = 0
    loaded_ids = []

    for ds_id in dataset_ids:
        file_path = base_dir / f"{file_prefix}_{ds_id}.xlsx"
        data, n = load_columns_from_excel(file_path, columns)
        if n > 0:
            for c in columns:
                if data[c].size > 0:
                    accumulated[c].append(data[c])
            total_samples += n
            loaded_ids.append(ds_id)

    if total_samples == 0:
        return {c: np.array([]) for c in columns}, 0, []

    return (
        {c: np.concatenate(accumulated[c]) if accumulated[c] else np.array([]) for c in columns},
        total_samples,
        loaded_ids,
    )


# ┌───────────────────────────────────────────────────────────────────────────┐
# │  Plotting helpers                                                         │
# └───────────────────────────────────────────────────────────────────────────┘

def _safe(name: str) -> str:
    """Convert a group name to a filesystem-safe string."""
    for ch in [" ", "(", ")", "–", "-"]:
        name = name.replace(ch, "_")
    return name.strip("_")


def _add_histogram_panel(
    ax,
    vals: np.ndarray,
    col_label: str,
    color: str,
    x_range,
    n_bins: int,
    x_label: str,
    absolute: bool = False,
) -> None:
    """Draw a single histogram panel with stats annotation."""
    finite = vals[np.isfinite(vals)]
    if absolute:
        finite = np.abs(finite)
    if finite.size == 0:
        ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(col_label, fontsize=10)
        return

    if x_range is not None:
        plot_vals = finite[(finite >= x_range[0]) & (finite <= x_range[1])]
        pct_shown = 100 * plot_vals.size / finite.size
        hist_range = x_range
    else:
        plot_vals = finite
        pct_shown = 100.0
        hist_range = (float(finite.min()), float(finite.max()))

    ax.hist(
        plot_vals, bins=n_bins, range=hist_range,
        color=color, alpha=0.80, edgecolor="white", linewidth=0.4,
    )
    if x_range is not None:
        ax.set_xlim(x_range)

    ax.set_title(col_label, fontsize=10)
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    stats_lines = [
        f"n (total) = {finite.size:,}",
        f"n (shown) = {plot_vals.size:,} ({pct_shown:.1f}%)",
        f"mean = {finite.mean():.4f}",
        f"std  = {finite.std():.4f}",
        f"min  = {finite.min():.4f}",
        f"max  = {finite.max():.4f}",
    ]
    ax.text(
        0.97, 0.97, "\n".join(stats_lines),
        transform=ax.transAxes, fontsize=7.5,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7, edgecolor="gray"),
    )


def plot_curvature_histogram(
    group_name: str,
    data: dict,
    total_samples: int,
    loaded_ids: list,
    save_dir: Path = OUTPUT_DIR,
    absolute: bool = False,
) -> None:
    """Plot Min/Max Curvature Length side by side and save as SVG."""
    if data[COL_MIN].size == 0 and data[COL_MAX].size == 0:
        print(f"  [SKIP] No curvature data for '{group_name}'")
        return

    abs_tag = " |Absolute|" if absolute else ""
    x_range = (0, 0.5) if absolute else CURVATURE_RANGE

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(
        f"Curvature{abs_tag} — {group_name}  |  Datasets: {', '.join(loaded_ids)}  |  Samples: {total_samples:,}",
        fontsize=11, fontweight="bold", y=1.01,
    )
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    panels = [
        (data[COL_MIN], COL_MIN, "#2196F3"),
        (data[COL_MAX], COL_MAX, "#E91E63"),
    ]
    for idx, (vals, label, color) in enumerate(panels):
        ax = fig.add_subplot(gs[0, idx])
        _add_histogram_panel(ax, vals, label, color, x_range, N_BINS, "Curvature Magnitude", absolute=absolute)

    plt.tight_layout()
    file_suffix = "absolute" if absolute else "normal"
    save_path = save_dir / f"curvature_histogram_{_safe(group_name)}_{file_suffix}.svg"
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    print(f"  Saved → {save_path}")
    plt.close(fig)


def plot_top_angle_histogram(
    group_name: str,
    data: dict,
    total_samples: int,
    loaded_ids: list,
    save_dir: Path = OUTPUT_DIR,
    absolute: bool = False,
) -> None:
    """Plot Top Angle histogram and save as SVG."""
    vals = data.get(COL_TOP_ANGLE, np.array([]))
    if vals.size == 0:
        print(f"  [SKIP] No top angle data for '{group_name}'")
        return

    abs_tag = " |Absolute|" if absolute else ""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        f"Top Angle{abs_tag} — {group_name}  |  Datasets: {', '.join(loaded_ids)}  |  Samples: {total_samples:,}",
        fontsize=11, fontweight="bold", y=1.01,
    )
    _add_histogram_panel(ax, vals, COL_TOP_ANGLE, "#4CAF50", TOP_ANGLE_RANGE, N_BINS, "Top Angle (deg)", absolute=absolute)

    plt.tight_layout()
    file_suffix = "absolute" if absolute else "normal"
    save_path = save_dir / f"top_angle_histogram_{_safe(group_name)}_{file_suffix}.svg"
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    print(f"  Saved → {save_path}")
    plt.close(fig)


# ┌───────────────────────────────────────────────────────────────────────────┐
# │  Main                                                                     │
# └───────────────────────────────────────────────────────────────────────────┘

def main() -> None:
    print(f"\nBase directory : {BASE_DIR}")
    print(f"Output figures : {OUTPUT_DIR}\n")

    if not BASE_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {BASE_DIR}\n"
            "Please update BASE_DIR at the top of this script."
        )

    for group_name, dataset_ids in GROUPS.items():
        print(f"── Group: {group_name} ──")

        # --- Curvature (from output files) ---
        print(f"  Loading curvature (Dataset_Output_*.xlsx)...")
        curv_data, curv_samples, curv_ids = load_group_data(
            dataset_ids, BASE_DIR,
            file_prefix="Dataset_Output",
            columns=[COL_MIN, COL_MAX],
        )
        if curv_samples > 0:
            print(f"  Loaded {curv_samples:,} curvature samples from {len(curv_ids)} datasets")
            plot_curvature_histogram(group_name, curv_data, curv_samples, curv_ids, absolute=False)
            plot_curvature_histogram(group_name, curv_data, curv_samples, curv_ids, absolute=True)
        else:
            print(f"  No curvature data for '{group_name}', skipping.")

        # --- Top Angle (from input files) ---
        print(f"  Loading top angle (Dataset_Input_*.xlsx)...")
        angle_data, angle_samples, angle_ids = load_group_data(
            dataset_ids, BASE_DIR,
            file_prefix="Dataset_Input",
            columns=[COL_TOP_ANGLE],
        )
        if angle_samples > 0:
            print(f"  Loaded {angle_samples:,} top angle samples from {len(angle_ids)} datasets")
            plot_top_angle_histogram(group_name, angle_data, angle_samples, angle_ids, absolute=False)
            plot_top_angle_histogram(group_name, angle_data, angle_samples, angle_ids, absolute=True)
        else:
            print(f"  No top angle data for '{group_name}', skipping.")

        print()

    print("Done.")


if __name__ == "__main__":
    main()