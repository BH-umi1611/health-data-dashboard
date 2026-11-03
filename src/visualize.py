"""Charts: demographics, health status, scatter correlations and a heatmap.

Colours: one blue for single-series charts, blue/orange for male/female,
an ordered blue ramp for ordered categories, and a blue-grey-red scale for
correlations (negative <-> positive).
"""
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw straight to image files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from . import analysis
from .config import (AGE_GROUP_ORDER, BLOOD_ORDER, BMI_ORDER, BP_ORDER, CHOL_ORDER,
                     LABELS, SHORT)

# --- palette -------------------------------------------------------------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GENDER_COLORS = {"Male": BLUE, "Female": ORANGE}
ORDINAL_BLUES = ["#86b6ef", "#5598e7", "#256abf", "#104281"]
DIVERGING = LinearSegmentedColormap.from_list("diverging", ["#256abf", "#f0efec", "#e34948"])

SCATTER_PAIRS = [
    ("age", "systolic_bp"), ("age", "cholesterol"), ("age", "glucose"),
    ("bmi", "systolic_bp"), ("bmi", "cholesterol"), ("systolic_bp", "cholesterol"),
]


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": BASELINE,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False,
    })


def _save(fig, path):
    """Save to a file path or an in-memory buffer (used by the web app)."""
    fig.savefig(path, dpi=150, format="png")
    plt.close(fig)
    return path if hasattr(path, "write") else Path(path)


def _wrap(labels):
    return [textwrap.fill(str(l), 12) for l in labels]


def _ordinal(k):
    idx = np.linspace(0, len(ORDINAL_BLUES) - 1, k).round().astype(int)
    return [ORDINAL_BLUES[i] for i in idx]


# --------------------------------------------------------------- bar charts
def _bar(ax, labels, counts, colors, title, xlabel=None):
    x = np.arange(len(labels))
    bars = ax.bar(x, counts, width=0.62, color=colors, edgecolor=SURFACE, linewidth=1.5)
    ax.set_xticks(x, _wrap(labels))
    ax.bar_label(bars, padding=3, color=INK2, fontsize=9)
    ax.set_ylabel("Number of patients")
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_ylim(0, max(max(counts), 1) * 1.15)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)


def plot_age_histogram(ax, data):
    age = data["age"]
    counts, edges = np.histogram(age, bins=np.arange(15, 95, 5))
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
           color=BLUE, edgecolor=SURFACE, linewidth=1.5)
    mean = age.mean()
    ax.axvline(mean, color=INK2, linewidth=1.5, linestyle="--")
    ax.text(mean + 1, counts.max() * 1.08, f"Mean {mean:.1f} yrs", color=INK2, fontsize=9)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_xlabel(LABELS["age"])
    ax.set_ylabel("Number of patients")
    ax.set_title("Age distribution")
    ax.grid(axis="x", visible=False)


def plot_gender(ax, data):
    labels, counts = analysis.category_counts(data, "gender", ["Male", "Female"])
    _bar(ax, labels, counts, [GENDER_COLORS[l] for l in labels], "Patients by gender")


def plot_age_groups(ax, data):
    labels, counts = analysis.category_counts(data, "age_group", AGE_GROUP_ORDER)
    _bar(ax, labels, counts, BLUE, "Patients by age group", "Age group (years)")


def plot_blood_groups(ax, data):
    labels, counts = analysis.category_counts(data, "blood_group", BLOOD_ORDER)
    _bar(ax, labels, counts, BLUE, "Patients by blood group", "Blood group")


def plot_category(ax, data, column, order, title, xlabel):
    labels, counts = analysis.category_counts(data, column, order)
    _bar(ax, labels, counts, _ordinal(len(labels)), title, xlabel)


# ------------------------------------------------------------- scatter plots
def plot_scatter(ax, data, xcol, ycol, legend=False):
    x, y = data[xcol], data[ycol]
    for g in ("Male", "Female"):
        m = data["gender"] == g
        ax.scatter(x[m], y[m], s=26, color=GENDER_COLORS[g], alpha=0.75,
                   edgecolors=SURFACE, linewidths=0.6, label=g)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, color=INK2, linewidth=2, linestyle="--",
            label="Trend (all patients)")
    r = np.corrcoef(x, y)[0, 1]
    ax.text(0.03, 0.95, f"r = {r:+.2f}", transform=ax.transAxes, va="top",
            color=INK, fontsize=10, fontweight="bold")
    ax.set_xlabel(LABELS[xcol])
    ax.set_ylabel(LABELS[ycol])
    ax.set_title(f"{SHORT[ycol]} vs {SHORT[xcol]}")
    ax.grid(True, axis="both")
    if legend:
        ax.legend(loc="lower right")


# ------------------------------------------------------------- full figures
def demographics_figure(data, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    plot_age_histogram(axes[0, 0], data)
    plot_gender(axes[0, 1], data)
    plot_age_groups(axes[1, 0], data)
    plot_blood_groups(axes[1, 1], data)
    fig.suptitle("Patient demographics", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, path)


def health_status_figure(data, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_category(axes[0], data, "bp_category", BP_ORDER, "Blood pressure category", None)
    plot_category(axes[1], data, "chol_category", CHOL_ORDER, "Cholesterol category", None)
    plot_category(axes[2], data, "bmi_category", BMI_ORDER, "BMI category", None)
    fig.suptitle("Health status of patients", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, path)


def scatter_grid(data, path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    for ax, (x, y) in zip(axes.ravel(), SCATTER_PAIRS):
        plot_scatter(ax, data, x, y)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 0.985))
    fig.suptitle("Relationships between health measurements", x=0.01, y=0.985, ha="left",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, path)


def correlation_heatmap(data, path):
    cols = analysis.CORR_COLS
    corr = analysis.correlation_matrix(data, cols)
    n = len(cols)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(n), [SHORT[c] for c in cols], rotation=35, ha="right")
    ax.set_yticks(range(n), [SHORT[c] for c in cols])
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if abs(corr[i, j]) > 0.55 else INK)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation (r)")
    cbar.outline.set_visible(False)
    ax.set_title("Correlation between measurements")
    fig.tight_layout()
    return _save(fig, path)


def scatter_figure(data, xcol, ycol, path):
    """One scatter plot with trend line, for any pair of measurements."""
    fig, ax = plt.subplots(figsize=(8, 5.4))
    plot_scatter(ax, data, xcol, ycol, legend=True)
    fig.tight_layout()
    return _save(fig, path)


def _single(plot_fn, data, path, *args, figsize=(6.5, 4.6), **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    plot_fn(ax, data, *args, **kwargs)
    fig.tight_layout()
    return _save(fig, path)


def create_all_plots(data, out_dir):
    """Create every chart. Returns the list of image paths."""
    apply_style()
    out_dir = Path(out_dir)
    single_dir = out_dir / "individual"
    single_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        demographics_figure(data, out_dir / "01_demographics.png"),
        health_status_figure(data, out_dir / "02_health_status.png"),
        scatter_grid(data, out_dir / "03_scatter_correlations.png"),
        correlation_heatmap(data, out_dir / "04_correlation_heatmap.png"),
        _single(plot_age_histogram, data, single_dir / "age_histogram.png"),
        _single(plot_gender, data, single_dir / "gender_bar.png"),
        _single(plot_age_groups, data, single_dir / "age_group_bar.png"),
        _single(plot_blood_groups, data, single_dir / "blood_group_bar.png"),
    ]
    for x, y in [("age", "systolic_bp"), ("age", "cholesterol"),
                 ("bmi", "cholesterol"), ("systolic_bp", "cholesterol")]:
        paths.append(_single(plot_scatter, data, single_dir / f"scatter_{x}_vs_{y}.png",
                             x, y, legend=True))
    return paths
