# Code to generate the plots for the quantile and the budget experiments in the paper.
# (Plotting code partly generated using ChatGPT).
import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

METRIC_DISPLAY_NAMES = {
    "bbh": "Average Exact Match",
    "codex": "Pass@10",
    "gsm8k": "Exact Match",
    "tydiqa": "Average F1",
    "mmlu_pro": "Exact Match",
}

BASE_DATASETS = ["bbh", "codex", "gsm8k", "tydiqa"]
LM_EVAL_HARNESS_TASK_NAMES = ["mmlu_pro"]
DATASETS = BASE_DATASETS + LM_EVAL_HARNESS_TASK_NAMES

DISPLAY_NAME = {
    "bbh": "BBH",
    "codex": "Codex",
    "gsm8k": "GSM8K",
    "tydiqa": "TyDiQA",
    "mmlu_pro": "MMLU-Pro",
}

EMBED_METHODS = [
    "Random",
    "RDS+ (RR)",
    "EMBED (RR)",
    "LESS (RR)",
]

LESS_METHODS = [
    "Random",
    "LESS (RR)",
    "LESS (DG)",
    "LESS (KNN-Unif.)",
    "LESS (KNN-KDE)",
    "LESS (UOT)",
]

MODEL_NAME_TO_SIMPLE_NAME = {
    "meta-llama/Llama-2-7b-hf": "llama2-7b",
    "meta-llama/Llama-3.2-3B": "llama3.2-3b",
    "HuggingFaceTB/SmolLM3-3B-Base": "smollm3-3b-base",
    "Qwen/Qwen3-4B-Base": "qwen3-4b-base",
    "allenai/Olmo-3-1025-7B": "olmo3-7b",
}

MODEL_NAME_TO_DISPLAY_NAME = {
    "meta-llama/Llama-2-7b-hf": "Llama 2 7B",
    "meta-llama/Llama-3.2-3B": "Llama 3.2 3B",
    "HuggingFaceTB/SmolLM3-3B-Base": "SmolLM3 3B Base",
    "Qwen/Qwen3-4B-Base": "Qwen3 4B Base",
    "allenai/Olmo-3-1025-7B": "Olmo 3 7B",
}

QUANTILE_METHODS = ["RDS+ (RR)", "EMBED (RR)", "LESS (RR)"]


def _as_hex(c):
    return mcolors.to_hex(c)


def _lighten(color, amount=0.55):
    """Blend color toward white by `amount` in [0,1]. Higher => lighter."""
    r, g, b = mcolors.to_rgb(color)
    r = 1 - (1 - r) * (1 - amount)
    g = 1 - (1 - g) * (1 - amount)
    b = 1 - (1 - b) * (1 - amount)
    return (r, g, b)


def _darken(color, amount=0.25):
    """Blend color toward black by `amount` in [0,1]. Higher => darker."""
    r, g, b = mcolors.to_rgb(color)
    return (r * (1 - amount), g * (1 - amount), b * (1 - amount))


PASTEL = sns.color_palette("Set2", 8)

METHOD_COLORS = {
    "Random": "#B7C7D6",
    "EMBED (DG)": _as_hex(PASTEL[2]),
    "EMBED (UOT)": _as_hex(PASTEL[1]),
    "EMBED (RR)": _as_hex(PASTEL[0]),
    "EMBED (KNN-Unif.)": _as_hex(PASTEL[3]),
    "EMBED (KNN-KDE)": _as_hex(PASTEL[4]),
    "RDS+ (DG)": _as_hex(_lighten(PASTEL[5], 0.15)),
    "RDS+ (UOT)": _as_hex(_lighten(PASTEL[6], 0.15)),
    "RDS+ (RR)": _as_hex(_lighten(PASTEL[1], 0.15)),
    "RDS+ (KNN-Unif.)": _as_hex(_lighten(PASTEL[0], 0.25)),
    "RDS+ (KNN-KDE)": _as_hex(_lighten(PASTEL[2], 0.25)),
    "LESS (DG)": _as_hex(_darken(PASTEL[0], 0.15)),
    "LESS (UOT)": _as_hex(_darken(PASTEL[3], 0.15)),
    "LESS (RR)": _as_hex(_darken(PASTEL[2], 0.15)),
    "LESS (KNN-Unif.)": _as_hex(_darken(PASTEL[5], 0.15)),
    "LESS (KNN-KDE)": _as_hex(_darken(PASTEL[6], 0.15)),
}


def plot_ce_loss_grid_seaborn(
    df: pd.DataFrame,
    zero_shot_ce_df: pd.DataFrame | None,
    output_path: Path,
    *,
    which: str = "dev",
    methods_to_plot: list | None = None,
    model_name: str = "Model",
    fig_w: float = 30.0,
    row_h: float = 5.2,
    n_cols: int = 5,
    wspace: float = 0.32,
    hspace: float = 0.65,
    bottom_margin: float = 0.47,
    font_size: int = 22,
    title_font_size: int = 25,
    legend_font_size: int = 25,
    linewidth: float = 5.0,
    marker_size: float = 15.0,
    title="Loss on Query Set across Subset Budget for Different Data Representations {model_name}",
) -> None:
    if df.empty:
        raise ValueError("No CE-loss data loaded. Check your file paths.")
    if which not in {"dev", "test"}:
        raise ValueError("Argument `which` must be 'dev' or 'test'.")

    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": title_font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": legend_font_size,
        }
    )
    sns.set_theme(
        style="whitegrid",
        rc={
            "axes.titlesize": title_font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": legend_font_size,
        },
    )

    df = df.copy()
    required_cols = {"dataset", "method", "num_samples", which}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    df["num_samples"] = pd.to_numeric(df["num_samples"], errors="coerce")
    df[which] = pd.to_numeric(df[which], errors="coerce")
    df = df.dropna(subset=["dataset", "method", "num_samples", which])

    available_ds = df["dataset"].unique().tolist()
    plot_datasets = [ds for ds in DATASETS if ds in available_ds] or sorted(
        available_ds
    )
    if not plot_datasets:
        raise ValueError(
            "None of the requested datasets are present in the CE-loss data."
        )

    if methods_to_plot is not None:
        present_methods = [m for m in methods_to_plot if m in set(df["method"])]
    else:
        present_methods = sorted(df["method"].unique().tolist())

    palette = {m: METHOD_COLORS[m] for m in present_methods if m in METHOD_COLORS}

    zs_map = {}
    if zero_shot_ce_df is not None and not getattr(zero_shot_ce_df, "empty", True):
        zs_map = zero_shot_ce_df.groupby("dataset")[which].mean().to_dict()

    n_plots = len(plot_datasets)
    n_rows = int(np.ceil(n_plots / n_cols))
    fig_h = row_h * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharey=False)
    axes = np.atleast_1d(axes).flatten()

    fig.suptitle(
        title.format(model_name=model_name),
        fontsize=title_font_size,
        fontweight="bold",
        y=1.10,
    )

    fixed_xticks = np.arange(0, 10001, 2000)
    x_min, x_max = 0, 10000

    for ax, ds in zip(axes, plot_datasets):
        sub = df[df["dataset"] == ds].copy()
        if present_methods:
            sub = sub[sub["method"].isin(present_methods)]

        ax.set_title(DISPLAY_NAME.get(ds, ds))
        ax.set_xlabel("Selected samples")
        ax.set_ylabel("Query loss" if which == "dev" else "Test loss")

        ax.set_xticks(fixed_xticks)
        ax.set_xlim(x_min, x_max)

        ax.tick_params(axis="x", labelrotation=45)
        for lbl in ax.get_xticklabels():
            lbl.set_horizontalalignment("right")

        if sub.empty:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
        else:
            sns.lineplot(
                data=sub,
                x="num_samples",
                y=which,
                hue="method",
                hue_order=present_methods,
                palette=palette,
                marker="o",
                linewidth=linewidth,
                markersize=marker_size,
                estimator="mean",
                errorbar=("se", 1),
                err_kws={"alpha": 0.25, "lw": 0},
                legend=False,
                ax=ax,
            )

        y0 = zs_map.get(ds)
        if y0 is not None and not np.isnan(y0):
            ax.axhline(
                y=float(y0),
                linestyle="--",
                color="gray",
                linewidth=linewidth,
                alpha=0.9,
                zorder=0,
            )

    for ax in axes[len(plot_datasets) :]:
        ax.axis("off")

    handles = []
    if zs_map:
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="--",
                color="gray",
                linewidth=linewidth,
                label="Zero-Shot",
            )
        )
    handles += [
        Line2D(
            [0],
            [0],
            linestyle="-",
            marker="o",
            color=palette[m],
            linewidth=linewidth,
            label=m,
        )
        for m in present_methods
        if m in palette
    ]

    if handles:
        fig.legend(
            handles,
            [h.get_label() for h in handles],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=len(handles),
            frameon=False,
        )

    plt.subplots_adjust(wspace=wspace, hspace=hspace, bottom=bottom_margin)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_true_metric_grid_seaborn(
    df: pd.DataFrame,
    output_path: Path,
    zero_shot_df: pd.DataFrame | None = None,
    methods: list | None = None,
    model_name: str = "Model",
    *,
    fig_w: float = 30.0,
    row_h: float = 5.2,
    n_cols: int = 5,
    wspace: float = 0.32,
    hspace: float = 0.65,
    bottom_margin: float = 0.47,
    font_size: int = 22,
    title_font_size: int = 25,
    legend_font_size: int = 25,
    linewidth: float = 5.0,
    marker_size: float = 15.0,
    title="Performance on Test Set across Subset Budgets for Different Data Representations {model_name}",
) -> None:
    if df.empty:
        raise ValueError("No true-metric data loaded. Check your file paths.")

    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": title_font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": legend_font_size,
        }
    )
    sns.set_theme(
        style="whitegrid",
        rc={
            "axes.titlesize": title_font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": legend_font_size,
        },
    )

    df = df.copy()
    if "num_samples" not in df.columns:
        raise ValueError("Expected column 'num_samples' is missing.")
    df["num_samples"] = pd.to_numeric(df["num_samples"], errors="coerce")
    df = df.dropna(subset=["dataset", "method", "true_metric", "num_samples"])

    available_ds = df["dataset"].unique().tolist()
    plot_datasets = [ds for ds in DATASETS if ds in available_ds] or sorted(
        available_ds
    )
    if not plot_datasets:
        raise ValueError("None of the requested datasets are present in the data.")

    methods_to_use = (
        methods if methods is not None else sorted(df["method"].unique().tolist())
    )
    present_methods = [m for m in methods_to_use if m in set(df["method"])]

    palette = {m: METHOD_COLORS[m] for m in present_methods if m in METHOD_COLORS}

    zs_map = {}
    if zero_shot_df is not None and not getattr(zero_shot_df, "empty", True):
        zs_map = zero_shot_df.groupby("dataset")["true_metric"].mean().to_dict()

    n_plots = len(plot_datasets)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig_h = row_h * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharey=False)
    fig.suptitle(
        title.format(model_name=model_name),
        fontsize=title_font_size,
        fontweight="bold",
        y=1.10,
    )
    axes = np.atleast_1d(axes).flatten()

    fixed_xticks = np.arange(0, 10001, 2000)
    x_min, x_max = 0, 10000

    for ax, ds in zip(axes, plot_datasets):
        sub = df[df["dataset"] == ds].copy()
        if present_methods:
            sub = sub[sub["method"].isin(present_methods)]

        ax.set_title(DISPLAY_NAME.get(ds, ds))
        ax.set_xlabel("Selected samples")
        ax.set_ylabel(METRIC_DISPLAY_NAMES.get(ds, "Score"))

        ax.set_xticks(fixed_xticks)
        ax.set_xlim(x_min, x_max)

        ax.tick_params(axis="x", labelrotation=45)
        for lbl in ax.get_xticklabels():
            lbl.set_horizontalalignment("right")

        if sub.empty:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
        else:
            sns.lineplot(
                data=sub,
                x="num_samples",
                y="true_metric",
                hue="method",
                hue_order=present_methods,
                palette=palette,
                marker="o",
                linewidth=linewidth,
                markersize=marker_size,
                estimator="mean",
                errorbar=("se", 1),
                err_kws={"alpha": 0.25, "lw": 0},
                legend=False,
                ax=ax,
            )

        y0 = zs_map.get(ds)
        if y0 is not None and not np.isnan(y0):
            ax.axhline(
                y=float(y0),
                linestyle="--",
                color="gray",
                linewidth=linewidth,
                alpha=0.9,
                zorder=0,
            )

    for ax in axes[len(plot_datasets) :]:
        ax.axis("off")

    handles = []
    if zs_map:
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="--",
                color="gray",
                linewidth=linewidth,
                label="Zero-Shot",
            )
        )
    handles += [
        Line2D(
            [0],
            [0],
            linestyle="-",
            marker="o",
            color=palette[m],
            linewidth=linewidth,
            label=m,
        )
        for m in present_methods
        if m in palette
    ]

    if handles:
        fig.legend(
            handles,
            [h.get_label() for h in handles],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=len(handles),
            frameon=False,
        )

    plt.subplots_adjust(wspace=wspace, hspace=hspace, bottom=bottom_margin)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ce_loss_grid_and_spearman_heatmap_side_by_side(
    df: pd.DataFrame,
    zero_shot_ce_df: pd.DataFrame,
    output_path: Path,
    model_name: str,
    *,
    fig_w: float = 32.0,
    fig_h: float = 5.2,
    line_block_ratio: float = 6.5,
    heat_block_ratio: float = 1.6,
    gap: float = 0.18,
    linepanel_wspace: float = 0.39,
    bottom_margin: float = 0.37,
    font_size: int = 23,
    title_font_size: int = 27,
    legend_font_size: int = 27,
    linewidth: float = 5.0,
    marker_size: float = 15.0,
    title="Loss on Query Set vs. Subset-Query Distance Quantile and Spearman Correlation ({model_name})",
) -> None:
    if df.empty:
        raise ValueError("No data loaded. Check your file paths.")

    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": title_font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": legend_font_size,
        }
    )
    sns.set_theme(
        style="whitegrid",
        rc={
            "axes.titlesize": title_font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": legend_font_size,
        },
    )

    raw_bins = pd.to_numeric(df["bin"], errors="coerce")
    min_bin = raw_bins.min(skipna=True)
    shift = 1 if min_bin == 0 else 0

    bins = list(range(1, 11))
    n_panels = len(DATASETS)

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(
        title.format(model_name=model_name),
        fontsize=title_font_size,
        fontweight="bold",
        y=1.10,
    )
    outer = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[line_block_ratio, heat_block_ratio],
        wspace=gap,
    )

    left = outer[0].subgridspec(1, n_panels, wspace=linepanel_wspace)
    axes = [fig.add_subplot(left[0, i]) for i in range(n_panels)]

    right = outer[1].subgridspec(1, 2, width_ratios=[1.0, 0.05], wspace=0.10)
    ax_hm = fig.add_subplot(right[0, 0])
    cbar_ax = fig.add_subplot(right[0, 1])

    palette = {m: METHOD_COLORS[m] for m in QUANTILE_METHODS if m in METHOD_COLORS}

    for ax, ds in zip(axes, DATASETS):
        sub = df[(df["dataset"] == ds) & (df["method"].isin(QUANTILE_METHODS))].copy()
        sub["bin_plot"] = pd.to_numeric(sub["bin"], errors="coerce") + shift
        sub["dev"] = pd.to_numeric(sub["dev"], errors="coerce")
        sub = sub.dropna(subset=["bin_plot", "dev"])
        sub = sub[sub["bin_plot"].isin(bins)]

        ax.set_title(DISPLAY_NAME.get(ds, ds))
        ax.set_xlabel("Distance quantile")
        ax.set_ylabel("Query loss")
        ax.set_xticks(bins)
        ax.set_xlim(0.5, 10.5)

        if sub.empty:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            continue

        sns.lineplot(
            data=sub,
            x="bin_plot",
            y="dev",
            hue="method",
            hue_order=QUANTILE_METHODS,
            palette=palette,
            errorbar=None,
            marker="o",
            linewidth=linewidth,
            markersize=marker_size,
            legend=False,
            ax=ax,
        )

        zs_val = zero_shot_ce_df.loc[zero_shot_ce_df["dataset"] == ds, "dev"]
        if not zs_val.empty and pd.notna(zs_val.iloc[0]):
            ax.axhline(
                y=float(zs_val.iloc[0]),
                linestyle="--",
                color="gray",
                linewidth=linewidth,
                alpha=0.9,
                zorder=0,
            )

    handles = [
        Line2D(
            [0],
            [0],
            linestyle="--",
            color="gray",
            linewidth=linewidth,
            label="Zero-Shot",
        )
    ] + [
        Line2D(
            [0],
            [0],
            linestyle="-",
            marker="o",
            color=palette[m],
            linewidth=linewidth,
            label=m,
        )
        for m in QUANTILE_METHODS
        if m in palette
    ]

    left_x0 = min(ax.get_position().x0 for ax in axes)
    left_x1 = max(ax.get_position().x1 for ax in axes)
    left_center_x = 0.5 * (left_x0 + left_x1)

    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="lower center",
        bbox_to_anchor=(left_center_x, 0.02),
        ncol=len(handles),
        frameon=False,
    )

    use = df.copy()
    use = use[use["dev"].notna() & use["bin"].notna()].copy()
    use["bin_plot"] = pd.to_numeric(use["bin"], errors="coerce") + shift
    use["dev"] = pd.to_numeric(use["dev"], errors="coerce")
    use = use.dropna(subset=["bin_plot", "dev"])

    table = pd.DataFrame(index=QUANTILE_METHODS, columns=DATASETS, dtype=float)
    for m in QUANTILE_METHODS:
        for ds in DATASETS:
            sub = use[(use["method"] == m) & (use["dataset"] == ds)]
            rho = (
                sub["bin_plot"].corr(sub["dev"], method="spearman")
                if len(sub) >= 2
                else np.nan
            )
            table.loc[m, ds] = rho

    col_map = {ds: DISPLAY_NAME.get(ds, ds) for ds in DATASETS}
    heatmap_data = table.rename(columns=col_map)

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": font_size},
        cbar=True,
        cbar_ax=cbar_ax,
        cbar_kws={"label": "Spearman corr."},
        ax=ax_hm,
    )

    ax_hm.set_xlabel("")
    ax_hm.set_ylabel("")

    for lbl in ax_hm.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_horizontalalignment("right")

    for lbl in ax_hm.get_yticklabels():
        lbl.set_rotation(45)
        lbl.set_horizontalalignment("right")

    cbar_ax.yaxis.label.set_size(font_size)
    cbar_ax.tick_params(labelsize=font_size)

    fig.subplots_adjust(bottom=bottom_margin)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_true_metric_grid_and_spearman_heatmap_side_by_side(
    df: pd.DataFrame,
    zero_shot_df: pd.DataFrame,
    output_path: Path,
    model_name: str,
    *,
    fig_w: float = 32.0,
    fig_h: float = 5.2,
    line_block_ratio: float = 6.5,
    heat_block_ratio: float = 1.6,
    gap: float = 0.18,
    linepanel_wspace: float = 0.44,
    bottom_margin: float = 0.37,
    font_size: int = 23,
    title_font_size: int = 27,
    legend_font_size: int = 27,
    linewidth: float = 5.0,
    marker_size: float = 15.0,
    title="Performance on Test Set vs. Subset-Query Distance Quantiles and Spearman Correlation ({model_name})",
    sub_quantile: bool = False,
) -> None:
    if df.empty:
        raise ValueError("No true-metric data loaded. Check your file paths.")

    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": title_font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": legend_font_size,
        }
    )
    sns.set_theme(
        style="whitegrid",
        rc={
            "axes.titlesize": title_font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": legend_font_size,
        },
    )

    raw_bins = pd.to_numeric(df["bin"], errors="coerce")
    min_bin = raw_bins.min(skipna=True)
    shift = 1 if min_bin == 0 else 0

    bins = list(range(1, 11))
    n_panels = len(DATASETS)

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(
        title.format(model_name=model_name),
        fontsize=title_font_size,
        fontweight="bold",
        y=1.10,
    )
    outer = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[line_block_ratio, heat_block_ratio],
        wspace=gap,
    )

    left = outer[0].subgridspec(1, n_panels, wspace=linepanel_wspace)
    axes = [fig.add_subplot(left[0, i]) for i in range(n_panels)]

    right = outer[1].subgridspec(1, 2, width_ratios=[1.0, 0.05], wspace=0.10)
    ax_hm = fig.add_subplot(right[0, 0])
    cbar_ax = fig.add_subplot(right[0, 1])

    palette = {m: METHOD_COLORS[m] for m in QUANTILE_METHODS if m in METHOD_COLORS}

    zs_map = {}
    if (
        zero_shot_df is not None
        and not zero_shot_df.empty
        and "true_metric" in zero_shot_df.columns
        and "dataset" in zero_shot_df.columns
    ):
        zs_map = zero_shot_df.groupby("dataset")["true_metric"].mean().to_dict()

    for ax, ds in zip(axes, DATASETS):
        sub = df[(df["dataset"] == ds) & (df["method"].isin(QUANTILE_METHODS))].copy()
        sub["bin_plot"] = pd.to_numeric(sub["bin"], errors="coerce") + shift
        sub["true_metric"] = pd.to_numeric(sub["true_metric"], errors="coerce")
        sub = sub.dropna(subset=["bin_plot", "true_metric"])
        sub = sub[sub["bin_plot"].isin(bins)]

        ax.set_title(DISPLAY_NAME.get(ds, ds))
        ax.set_xlabel("Distance quantile")
        ax.set_ylabel(METRIC_DISPLAY_NAMES.get(ds, "Score"))
        ax.set_xticks(bins)
        ax.set_xlim(0.5, 10.5)

        if sub.empty:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
        else:
            sns.lineplot(
                data=sub,
                x="bin_plot",
                y="true_metric",
                hue="method",
                hue_order=QUANTILE_METHODS,
                palette=palette,
                errorbar=None,
                marker="o",
                linewidth=linewidth,
                markersize=marker_size,
                legend=False,
                ax=ax,
            )

        y0 = zs_map.get(ds)
        if y0 is not None and pd.notna(y0):
            ax.axhline(
                y=float(y0),
                linestyle="--",
                color="gray",
                linewidth=linewidth,
                alpha=0.9,
                zorder=0,
            )

    handles = [
        Line2D(
            [0],
            [0],
            linestyle="--",
            color="gray",
            linewidth=linewidth,
            label="Zero-Shot",
        )
    ] + [
        Line2D(
            [0],
            [0],
            linestyle="-",
            marker="o",
            color=palette[m],
            linewidth=linewidth,
            label=m,
        )
        for m in QUANTILE_METHODS
        if m in palette
    ]

    left_x0 = min(ax.get_position().x0 for ax in axes)
    left_x1 = max(ax.get_position().x1 for ax in axes)
    left_center_x = 0.5 * (left_x0 + left_x1)

    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="lower center",
        bbox_to_anchor=(left_center_x, 0.02),
        ncol=len(handles),
        frameon=False,
    )

    use = df.copy()
    use = use[use["true_metric"].notna() & use["bin"].notna()].copy()
    use["bin_plot"] = pd.to_numeric(use["bin"], errors="coerce") + shift
    use["true_metric"] = pd.to_numeric(use["true_metric"], errors="coerce")
    use = use.dropna(subset=["bin_plot", "true_metric"])

    table = pd.DataFrame(index=QUANTILE_METHODS, columns=DATASETS, dtype=float)
    for m in QUANTILE_METHODS:
        for ds in DATASETS:
            sub = use[(use["method"] == m) & (use["dataset"] == ds)]
            rho = (
                sub["bin_plot"].corr(sub["true_metric"], method="spearman")
                if len(sub) >= 2
                else np.nan
            )
            table.loc[m, ds] = rho

    col_map = {ds: DISPLAY_NAME.get(ds, ds) for ds in DATASETS}
    heatmap_data = table.rename(columns=col_map)

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": font_size},
        cbar=True,
        cbar_ax=cbar_ax,
        cbar_kws={"label": "Spearman corr."},
        ax=ax_hm,
    )

    ax_hm.set_xlabel("")
    ax_hm.set_ylabel("")

    for lbl in ax_hm.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_horizontalalignment("right")

    for lbl in ax_hm.get_yticklabels():
        lbl.set_rotation(45)
        lbl.set_horizontalalignment("right")

    cbar_ax.yaxis.label.set_size(font_size)
    cbar_ax.tick_params(labelsize=font_size)

    fig.subplots_adjust(bottom=bottom_margin)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(args):
    simple_name = MODEL_NAME_TO_SIMPLE_NAME[args.model_name]

    plot_dir = Path(f"files/paper/plots/quantile_budget/{simple_name}/")
    plot_dir.mkdir(parents=True, exist_ok=True)

    csv_dir = Path(f"assets/plot_data/quantile_budget/{simple_name}/")

    quantile_prefix = "sub_bin0_focus" if args.focus_bin0 else "binning"
    base_prefix = "budget"

    def chart_path(prefix: str, *parts: str) -> Path:
        filename = "_".join([prefix, *parts]).replace(" ", "_") + ".pdf"
        return plot_dir / filename

    def csv_path(prefix: str, *parts: str) -> Path:
        filename = "_".join([prefix, *parts]).replace(" ", "_") + ".csv"
        return csv_dir / filename

    def read_csv(prefix: str, *name_parts: str) -> pd.DataFrame:
        path = csv_path(prefix, *name_parts)
        if path.exists():
            print(f"Reading CSV from: {path}")
            return pd.read_csv(path)
        else:
            print(f"CSV not found at: {path}")
            return pd.DataFrame()

    zero_shot_df = read_csv(base_prefix, "true_metric", "zero_shot")
    zero_shot_ce_df = read_csv(base_prefix, "ce_loss", "zero_shot")
    df_ce = read_csv(base_prefix, "ce_loss", "budget")
    df_true = read_csv(base_prefix, "true_metric", "budget")
    df_quantile_ce = read_csv(quantile_prefix, "ce_loss", "quantile")
    df_quantile_true = read_csv(quantile_prefix, "true_metric", "quantile")

    out_true_data_representation = chart_path(
        base_prefix, "true_metric", "data_representation"
    )
    title = "Performance on the Test Set across Subset Budgets for Different Data Representations ({model_name})"
    plot_true_metric_grid_seaborn(
        df_true,
        out_true_data_representation,
        zero_shot_df,
        EMBED_METHODS,
        model_name=MODEL_NAME_TO_DISPLAY_NAME[args.model_name],
        title=title,
    )
    print(f"Saved figure to: {out_true_data_representation}")

    out_ce_data_representation = chart_path(
        base_prefix, "ce_loss_dev", "data_representation"
    )
    title = "Loss on the Query Set across Subset Budgets for Different Data Representations ({model_name})"
    plot_ce_loss_grid_seaborn(
        df_ce,
        zero_shot_ce_df,
        out_ce_data_representation,
        which="dev",
        methods_to_plot=EMBED_METHODS,
        model_name=MODEL_NAME_TO_DISPLAY_NAME[args.model_name],
        title=title,
    )
    print(f"Saved figure to: {out_ce_data_representation}")

    out_true_sel_alg = chart_path(base_prefix, "true_metric", "sel_alg")
    title = "Performance on the Test Set across Subset Budgets for Different Selection Algorithms ({model_name})"
    plot_true_metric_grid_seaborn(
        df_true,
        zero_shot_df=None,
        output_path=out_true_sel_alg,
        methods=LESS_METHODS,
        model_name=MODEL_NAME_TO_DISPLAY_NAME[args.model_name],
        title=title,
    )
    print(f"Saved figure to: {out_true_sel_alg}")

    out_ce_sel_alg = chart_path(base_prefix, "ce_loss_dev", "sel_alg")
    title = "Loss on the Query Set across Subset Budgets for Different Selection Algorithms ({model_name})"
    plot_ce_loss_grid_seaborn(
        df_ce,
        zero_shot_ce_df=None,
        output_path=out_ce_sel_alg,
        which="dev",
        methods_to_plot=LESS_METHODS,
        model_name=MODEL_NAME_TO_DISPLAY_NAME[args.model_name],
        title=title,
    )
    print(f"Saved figure to: {out_ce_sel_alg}")

    out_combined_ce = chart_path(quantile_prefix, "ce_loss_grid_and_spearman_heatmap")
    title = "Loss on the Query Set vs. Subset-Query Distance Quantiles and Spearman Correlation ({model_name})"
    if args.focus_bin0:
        title = "Loss on the Query Set vs. Subset-Query Distance Sub-Quantiles and Spearman Correlation ({model_name})"
    plot_ce_loss_grid_and_spearman_heatmap_side_by_side(
        df_quantile_ce,
        zero_shot_ce_df,
        out_combined_ce,
        MODEL_NAME_TO_DISPLAY_NAME[args.model_name],
        title=title,
    )
    print(f"Saved combined figure to: {out_combined_ce}")

    out_combined_true = chart_path(
        quantile_prefix, "true_metric_grid_and_spearman_heatmap"
    )
    title = "Performance on the Test Set vs. Subset-Query Distance Quantiles and Spearman Correlation ({model_name})"
    if args.focus_bin0:
        title = "Performance on the Test Set vs. Subset-Query Distance Sub-Quantiles and Spearman Correlation ({model_name})"
    plot_true_metric_grid_and_spearman_heatmap_side_by_side(
        df_quantile_true,
        zero_shot_df,
        out_combined_true,
        MODEL_NAME_TO_DISPLAY_NAME[args.model_name],
        title=title,
    )
    print(f"Saved combined figure to: {out_combined_true}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Model name used in the experiment",
    )
    parser.add_argument(
        "--focus_bin0", action="store_true", help="Whether to focus on bin 0 only"
    )
    args = parser.parse_args()
    main(args)
