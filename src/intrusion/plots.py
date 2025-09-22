from math import ceil
from pathlib import Path
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _save(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _short_rule_label(label: str) -> str:
    """
    Turn 'R1 packet_size tails ^ R8 browser=Unknown' -> 'R1 ^ R8'.
    For singles, 'R5 failed_logins >= 4' -> 'R5'.
    """
    parts = [p.strip() for p in re.split(r"\s*(?:\^|∧)\s*", label)]
    return " ^ ".join(p.split()[0] for p in parts)


def _numeric_cols(df: pd.DataFrame, include: list[str] | None = None) -> list[str]:
    if include:
        return [c for c in include if c in df.columns]
    return df.select_dtypes(include=[np.number]).columns.tolist()


def make_top_precision_figure(
    df_eval: pd.DataFrame,
    baseline: float,
    *,
    top_n: int = 15,
    short_labels: bool = False,
    rotate: int = 0,
):
    df = (
        df_eval.sort_values(
            ["Precision", "Lift", "Hits"], ascending=[False, False, False]
        )
        .head(top_n)
        .copy()
    )
    labels = [(_short_rule_label(r) if short_labels else r) for r in df["Rule"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, df["Precision"])
    ax.axvline(baseline, linestyle="--")
    ax.set_xlabel("Precision")
    ax.set_ylabel("Rule")
    ax.set_title(f"Top {top_n} Rules by Precision (dashed = baseline)")
    ax.invert_yaxis()
    if rotate:
        ax.set_yticklabels(labels, rotation=rotate, ha="right")
    fig.tight_layout()
    return fig


def plot_top_precision(
    df_eval: pd.DataFrame, baseline: float, out_path: Path, top_n: int = 15
):
    fig = make_top_precision_figure(df_eval, baseline, top_n=top_n)
    _save(fig, out_path)


def make_top_lift_figure(
    df_eval: pd.DataFrame,
    *,
    top_n: int = 15,
    short_labels: bool = False,
    rotate: int = 0,
):
    df = (
        df_eval.sort_values(
            ["Lift", "Precision", "Hits"], ascending=[False, False, False]
        )
        .head(top_n)
        .copy()
    )
    labels = [(_short_rule_label(r) if short_labels else r) for r in df["Rule"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, df["Lift"])
    ax.set_xlabel("Lift")
    ax.set_ylabel("Rule")
    ax.set_title(f"Top {top_n} Rules by Lift")
    ax.invert_yaxis()
    if rotate:
        ax.set_yticklabels(labels, rotation=rotate, ha="right")
    fig.tight_layout()
    return fig


def plot_top_lift(df_eval: pd.DataFrame, out_path: Path, top_n: int = 15):
    fig = make_top_lift_figure(df_eval, top_n=top_n)
    _save(fig, out_path)


def make_cumulative_recall_figure(
    cum_table: pd.DataFrame,
    *,
    short_labels: bool = True,
    rotate: int = 45,
):
    """
    Expects columns: 'Rule', 'Cumulative Recall'
    """
    if cum_table.empty:
        fig, _ = plt.subplots(figsize=(6, 2))
        return fig

    labels = [(_short_rule_label(r) if short_labels else r) for r in cum_table["Rule"]]
    x = list(range(1, len(labels) + 1))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, cum_table["Cumulative Recall"], marker="o")
    ax.set_xlabel("Rules in priority order")
    ax.set_ylabel("Cumulative Recall")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right")

    ax.annotate(
        f"{cum_table['Cumulative Recall'].iloc[-1]:.2f}",
        (x[-1], cum_table["Cumulative Recall"].iloc[-1]),
        xytext=(6, 6),
        textcoords="offset points",
    )

    fig.tight_layout()
    return fig


def plot_cumulative_recall(cum_table: pd.DataFrame, out_path: Path):
    fig = make_cumulative_recall_figure(cum_table)
    _save(fig, out_path)


def make_incremental_precision_figure(
    inc_detail: pd.DataFrame,
    *,
    short_labels: bool = True,
    rotate: int = 45,
):
    """
    Expects columns: 'Rule', 'Incremental Precision'
    """
    if inc_detail.empty:
        fig, _ = plt.subplots(figsize=(6, 2))
        return fig

    labels = [(_short_rule_label(r) if short_labels else r) for r in inc_detail["Rule"]]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, inc_detail["Incremental Precision"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right")
    ax.set_ylabel("Incremental Precision")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Incremental Precision per Step in Priority")
    fig.tight_layout()
    return fig


def plot_incremental_precision(inc_detail: pd.DataFrame, out_path: Path):
    fig = make_incremental_precision_figure(inc_detail)
    _save(fig, out_path)


def make_precision_by_severity_figure(
    by_sev: pd.DataFrame,
    baseline: float,
):
    """
    Expects by_sev with columns: ['rows','precision'] and index of severities.
    """
    if by_sev.empty:
        fig, _ = plt.subplots(figsize=(6, 2))
        return fig

    s = by_sev["precision"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(s.index, s.values)
    ax.axhline(baseline, linestyle="--")
    ax.set_ylabel("Precision")
    ax.set_title("Precision by Severity (dashed = baseline)")
    fig.tight_layout()
    return fig


def plot_precision_by_severity(by_sev: pd.DataFrame, baseline: float, out_path: Path):
    fig = make_precision_by_severity_figure(by_sev, baseline)
    _save(fig, out_path)


def plot_numeric_distributions_grid(
    df: pd.DataFrame,
    out_path: Path,
    *,
    bins: int = 30,
    cols_per_row: int = 2,
    show_kde: bool = True,
    density: bool = True,
    dropna: bool = True,
    height_per_row: float = 4.5,
    include: list[str] | None = None,
):
    fig = make_distributions_grid_figure(
        df,
        bins=bins,
        cols_per_row=cols_per_row,
        show_kde=show_kde,
        density=density,
        dropna=dropna,
        height_per_row=height_per_row,
        include=include,
    )
    _save(fig, out_path)


def make_distributions_grid_figure(
    df: pd.DataFrame,
    *,
    bins: int = 30,
    cols_per_row: int = 2,
    show_kde: bool = True,
    density: bool = True,
    dropna: bool = True,
    height_per_row: float = 4.5,
    include: list[str] | None = None,
):
    numeric_cols = _numeric_cols(df, include)
    if not numeric_cols:
        fig, _ = plt.subplots(figsize=(6, 2))
        return fig

    n = len(numeric_cols)
    rows = ceil(n / cols_per_row)
    fig, axes = plt.subplots(
        rows, cols_per_row, figsize=(15, rows * height_per_row), squeeze=False
    )

    for i, col in enumerate(numeric_cols):
        r, c = divmod(i, cols_per_row)
        ax = axes[r][c]
        series = df[col].dropna() if dropna else df[col]

        ax.hist(series, bins=bins, density=density, edgecolor="black")
        if show_kde and len(series) > 1 and series.nunique() > 1:
            try:
                series.plot(kind="kde", ax=ax)
            except Exception:
                pass

        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Density" if density else "Count")
        ax.grid(True)

    # hide unused axes
    total_axes = rows * cols_per_row
    for j in range(n, total_axes):
        r, c = divmod(j, cols_per_row)
        axes[r][c].set_visible(False)

    fig.tight_layout()
    return fig


def plot_numeric_distributions_individual(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    bins: int = 30,
    show_kde: bool = True,
    density: bool = True,
    dropna: bool = True,
    include: list[str] | None = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    numeric_cols = _numeric_cols(df, include)
    if not numeric_cols:
        return

    for col in numeric_cols:
        series = df[col].dropna() if dropna else df[col]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(series, bins=bins, density=density, edgecolor="black")
        if show_kde and len(series) > 1 and series.nunique() > 1:
            try:
                series.plot(kind="kde", ax=ax)
            except Exception:
                pass

        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Density" if density else "Count")
        ax.grid(True)

        safe = col.replace("/", "_").replace(" ", "_")
        fig.savefig(out_dir / f"dist_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def make_correlation_heatmap_figure(
    df: pd.DataFrame,
    *,
    include: list[str] | None = None,
    method: str = "pearson",  # "pearson" | "spearman" | "kendall"
    mask_upper: bool = True,  # show only lower triangle if True
    annotate: bool = True,  # draw numbers in cells
    figsize: tuple[float, float] | None = None,
):
    numeric_cols = _numeric_cols(df, include)
    if not numeric_cols:
        fig, _ = plt.subplots(figsize=(6, 2))
        return fig

    corr = df[numeric_cols].corr(method=method)  # pyright: ignore[]
    mat = corr.values.copy()
    if mask_upper:
        iu = np.triu_indices_from(mat, k=1)
        mat[iu] = np.nan

    if figsize is None:
        s = max(6, min(1.0 * len(numeric_cols) + 2, 20))
        figsize = (s, s)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.viridis  # pyright: ignore[]
    cmap.set_bad(color="white")
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap=cmap)

    ax.set_xticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_yticklabels(numeric_cols)
    ax.set_title(f"Correlation heatmap ({method})")

    ax.set_xticks(np.arange(-0.5, len(numeric_cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(numeric_cols), 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")

    if annotate:
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                val = mat[i, j]
                if np.isnan(val):
                    continue
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    return fig
