"""
EDA (Exploratory Data Analysis) plotting tools.

Generates comprehensive visualisations during the dataset-profiling phase:
target distribution, numeric histograms, boxplots, correlation heatmap,
pairplot, missing-value heatmap, categorical bar charts, target-vs-feature
plots, and an outlier summary chart.

All functions are pure Python with NO LLM calls.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from agenticml.ml.tools.utils import ensure_dir_exists

# Cap constants to keep plots readable
_MAX_NUMERIC_GRID = 20
_MAX_CATEGORICAL_COLS = 12
_MAX_PAIRPLOT_COLS = 6
_MAX_TOP_CATEGORIES = 15
_MAX_TARGET_FEATURES = 8


def generate_eda_plots(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
    profile: dict,
    output_dir: str,
) -> list[str]:
    """Generate all EDA plots and return a list of saved file paths."""
    ensure_dir_exists(output_dir + "/")
    plots: list[str] = []

    numeric_cols = [c for c in profile.get("numeric_columns", []) if c != target]
    categorical_cols = [c for c in profile.get("categorical_columns", []) if c != target]

    plots.extend(_plot_target_distribution(df, target, problem_type, output_dir))
    plots.extend(_plot_numeric_distributions(df, numeric_cols, output_dir))
    plots.extend(_plot_boxplots(df, numeric_cols, output_dir))
    plots.extend(_plot_correlation_heatmap(df, profile.get("numeric_columns", []), output_dir))
    plots.extend(_plot_pairplot(df, profile.get("numeric_columns", []), target, output_dir))
    plots.extend(_plot_missing_heatmap(df, profile, output_dir))
    plots.extend(_plot_categorical_bars(df, categorical_cols, output_dir))
    plots.extend(_plot_target_vs_features(df, target, problem_type, numeric_cols, output_dir))
    plots.extend(_plot_outlier_summary(df, numeric_cols, output_dir))

    return plots


# ---------------------------------------------------------------------------
# 1. Target distribution
# ---------------------------------------------------------------------------

def _plot_target_distribution(
    df: pd.DataFrame, target: str, problem_type: str, output_dir: str
) -> list[str]:
    if target not in df.columns:
        return []

    path = os.path.join(output_dir, "eda_target_distribution.png")
    fig, ax = plt.subplots(figsize=(9, 5))

    if problem_type == "classification":
        counts = df[target].value_counts()
        counts.plot.bar(ax=ax, color=sns.color_palette("muted", len(counts)))
        ax.set_ylabel("Count")
        for i, v in enumerate(counts):
            ax.text(i, v + max(counts) * 0.01, str(v), ha="center", fontsize=9)
    else:
        sns.histplot(df[target].dropna(), kde=True, ax=ax, bins=40)
        ax.set_ylabel("Frequency")

    ax.set_title(f"Target Distribution — {target}")
    ax.set_xlabel(target)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


# ---------------------------------------------------------------------------
# 2. Numeric feature distributions (grid of histograms)
# ---------------------------------------------------------------------------

def _plot_numeric_distributions(
    df: pd.DataFrame, numeric_cols: list[str], output_dir: str
) -> list[str]:
    cols = numeric_cols[:_MAX_NUMERIC_GRID]
    if not cols:
        return []

    n = len(cols)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(cols):
        ax = axes_flat[i]
        data = df[col].dropna()
        sns.histplot(data, kde=True, ax=ax, bins=30)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Numeric Feature Distributions", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "eda_numeric_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


# ---------------------------------------------------------------------------
# 3. Boxplots (outlier visualisation)
# ---------------------------------------------------------------------------

def _plot_boxplots(
    df: pd.DataFrame, numeric_cols: list[str], output_dir: str
) -> list[str]:
    cols = numeric_cols[:_MAX_NUMERIC_GRID]
    if not cols:
        return []

    n = len(cols)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(cols):
        ax = axes_flat[i]
        data = df[col].dropna()
        sns.boxplot(y=data, ax=ax, flierprops=dict(marker="o", markersize=3, alpha=0.5))
        ax.set_title(col, fontsize=10)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Boxplots — Outlier Visualisation", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "eda_boxplots.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


# ---------------------------------------------------------------------------
# 4. Correlation heatmap
# ---------------------------------------------------------------------------

def _plot_correlation_heatmap(
    df: pd.DataFrame, numeric_cols: list[str], output_dir: str
) -> list[str]:
    if len(numeric_cols) < 2:
        return []

    corr = df[numeric_cols].corr()
    size = max(8, len(numeric_cols) * 0.6)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))

    annot = len(numeric_cols) <= 20
    sns.heatmap(
        corr, annot=annot, fmt=".2f" if annot else "",
        cmap="coolwarm", center=0, ax=ax, square=True,
        linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Heatmap", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "eda_correlation_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


# ---------------------------------------------------------------------------
# 5. Pairplot (top correlated features)
# ---------------------------------------------------------------------------

def _plot_pairplot(
    df: pd.DataFrame, numeric_cols: list[str], target: str, output_dir: str
) -> list[str]:
    if len(numeric_cols) < 2:
        return []

    # Pick columns most correlated with target (or first N if target not numeric)
    cols_to_use: list[str] = []
    if target in numeric_cols:
        corr_with_target = df[numeric_cols].corr()[target].drop(target, errors="ignore").abs()
        cols_to_use = corr_with_target.sort_values(ascending=False).head(_MAX_PAIRPLOT_COLS - 1).index.tolist()
        cols_to_use = [target] + cols_to_use
    else:
        cols_to_use = numeric_cols[:_MAX_PAIRPLOT_COLS]

    sample = df[cols_to_use].dropna()
    if len(sample) > 2000:
        sample = sample.sample(n=2000, random_state=42)

    try:
        g = sns.pairplot(sample, diag_kind="kde", plot_kws={"alpha": 0.4, "s": 12})
        g.figure.suptitle("Pairplot — Top Correlated Features", y=1.02, fontsize=13)
        path = os.path.join(output_dir, "eda_pairplot.png")
        g.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(g.figure)
        return [path]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 6. Missing values heatmap
# ---------------------------------------------------------------------------

def _plot_missing_heatmap(
    df: pd.DataFrame, profile: dict, output_dir: str
) -> list[str]:
    missing_pcts = profile.get("missing_percentages", {})
    cols_with_missing = [c for c, p in missing_pcts.items() if p > 0]
    if not cols_with_missing:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(cols_with_missing) * 0.35)),
                             gridspec_kw={"width_ratios": [2, 1]})

    # Left: heatmap of nullity pattern (sample rows for speed)
    sample = df[cols_with_missing]
    if len(sample) > 500:
        sample = sample.sample(n=500, random_state=42).sort_index()
    axes[0].imshow(sample.isna().values.T, aspect="auto", cmap="Greys", interpolation="none")
    axes[0].set_yticks(range(len(cols_with_missing)))
    axes[0].set_yticklabels(cols_with_missing, fontsize=9)
    axes[0].set_xlabel("Row index (sampled)")
    axes[0].set_title("Missing Data Pattern")

    # Right: bar chart of missing %
    pcts = [missing_pcts[c] for c in cols_with_missing]
    sorted_pairs = sorted(zip(cols_with_missing, pcts), key=lambda x: x[1])
    cols_sorted, pcts_sorted = zip(*sorted_pairs)
    axes[1].barh(cols_sorted, pcts_sorted, color="salmon")
    axes[1].set_xlabel("Missing %")
    axes[1].set_title("Missing Value Percentages")
    for i, v in enumerate(pcts_sorted):
        axes[1].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)

    fig.suptitle("Missing Values Overview", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "eda_missing_values.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


# ---------------------------------------------------------------------------
# 7. Categorical value counts
# ---------------------------------------------------------------------------

def _plot_categorical_bars(
    df: pd.DataFrame, categorical_cols: list[str], output_dir: str
) -> list[str]:
    cols = categorical_cols[:_MAX_CATEGORICAL_COLS]
    if not cols:
        return []

    n = len(cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(cols):
        ax = axes_flat[i]
        counts = df[col].value_counts().head(_MAX_TOP_CATEGORIES)
        counts.plot.barh(ax=ax, color=sns.color_palette("pastel", len(counts)))
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("Count")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Categorical Feature Value Counts", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "eda_categorical_counts.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


# ---------------------------------------------------------------------------
# 8. Target vs features
# ---------------------------------------------------------------------------

def _plot_target_vs_features(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
    numeric_cols: list[str],
    output_dir: str,
) -> list[str]:
    if target not in df.columns or not numeric_cols:
        return []

    # Pick top features by absolute correlation with target
    if pd.api.types.is_numeric_dtype(df[target]):
        corrs = df[numeric_cols].corrwith(df[target]).abs().sort_values(ascending=False)
        top_cols = corrs.head(_MAX_TARGET_FEATURES).index.tolist()
    else:
        top_cols = numeric_cols[:_MAX_TARGET_FEATURES]

    if not top_cols:
        return []

    n = len(top_cols)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    sample = df[[target] + top_cols].dropna()
    if len(sample) > 3000:
        sample = sample.sample(n=3000, random_state=42)

    for i, col in enumerate(top_cols):
        ax = axes_flat[i]
        if problem_type == "regression":
            ax.scatter(sample[col], sample[target], alpha=0.35, s=10)
            ax.set_xlabel(col)
            ax.set_ylabel(target)
        else:
            sns.boxplot(x=sample[target], y=sample[col], ax=ax)
            ax.set_xlabel(target)
            ax.set_ylabel(col)
        ax.set_title(f"{target} vs {col}", fontsize=10)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Target vs Top Features", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "eda_target_vs_features.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


# ---------------------------------------------------------------------------
# 9. Outlier summary
# ---------------------------------------------------------------------------

def _plot_outlier_summary(
    df: pd.DataFrame, numeric_cols: list[str], output_dir: str
) -> list[str]:
    if not numeric_cols:
        return []

    outlier_data: list[dict] = []
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) == 0:
            continue
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = int(((data < lower) | (data > upper)).sum())
        if n_outliers > 0:
            outlier_data.append({
                "column": col,
                "n_outliers": n_outliers,
                "pct": round(n_outliers / len(data) * 100, 2),
            })

    if not outlier_data:
        return []

    outlier_data.sort(key=lambda x: x["pct"])
    cols_sorted = [d["column"] for d in outlier_data]
    pcts_sorted = [d["pct"] for d in outlier_data]
    counts_sorted = [d["n_outliers"] for d in outlier_data]

    fig, ax = plt.subplots(figsize=(10, max(5, len(outlier_data) * 0.4)))
    bars = ax.barh(cols_sorted, pcts_sorted, color="coral")
    ax.set_xlabel("Outlier %")
    ax.set_title("Outlier Summary (IQR Method)", fontsize=13)

    for i, (pct, cnt) in enumerate(zip(pcts_sorted, counts_sorted)):
        ax.text(pct + 0.3, i, f"{pct:.1f}% ({cnt})", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "eda_outlier_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]
