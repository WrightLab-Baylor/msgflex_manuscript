#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.linewidth": 0.6,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

BASE_DIR = Path(__file__).resolve().parent

METHOD_COLORS = {
    "Percolator": "#3C5488",
    "MS2Rescore": "#8491B4",
    "MSBooster": "#00A087",
    "XPECTRA": "#E64B35",
}

METHOD_ORDER = [
    "Percolator",
    "MS2Rescore",
    "MSBooster",
    "XPECTRA",
]

DATASET_ORDER = [
    "PXD006118",
    "PXD035981",
    "PXD052873",
    "PXD074461",
]

DATASET_LABELS = {
    "PXD006118": "PXD006118",
    "PXD035981": "PXD035981",
    "PXD052873": "PXD052873",
    "PXD074461": "PXD074461",
}

LEVELS = [
    "PSMs",
    "Peptides",
    "Proteins",
]

INPUT_FILES = {
    "PSMs": BASE_DIR / "xpec_comp" / "pdata_psms.tsv",
    "Peptides": BASE_DIR / "xpec_comp" / "pdata_peptides.tsv",
    "Proteins": BASE_DIR / "xpec_comp" / "pdata_proteins.tsv",
}

OUTPUT_PDF = BASE_DIR / "xpectra_main_figure.pdf"
# ----------------------------------------------------------------------
# 3. Helpers
# ----------------------------------------------------------------------
def lighten(hex_color: str, amount: float = 0.55):
    """Blend a hex color toward white."""
    rgb = np.array(
        [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)],
        dtype=float,
    ) / 255.0

    return tuple(rgb + (1.0 - rgb) * amount)


def load_level(level):
    """Load and validate one identification-level table."""
    path = INPUT_FILES[level]

    if not path.exists():
        raise FileNotFoundError(
            f"Input file for '{level}' not found:\n  {path}"
        )

    df = pd.read_csv(path, sep="\t")

    required = {
        "dataset",
        "method",
        "baseline",
        "lost",
        "common",
        "gained",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{level}: missing required columns: {sorted(missing)}"
        )

    numeric_cols = ["baseline", "lost", "common", "gained"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # Basic validity checks.
    if (df["baseline"] <= 0).any():
        bad = df.loc[df["baseline"] <= 0]
        raise ValueError(
            f"{level}: baseline must be > 0. Problematic rows:\n{bad}"
        )

    if (df[numeric_cols[1:]] < 0).any().any():
        raise ValueError(
            f"{level}: lost/common/gained cannot contain negative values."
        )

    # Baseline decomposition:
    # baseline = lost + common
    mismatch = (df["lost"] + df["common"] - df["baseline"]).abs() > 1
    if mismatch.any():
        bad = df.loc[
            mismatch,
            ["dataset", "method", "baseline", "lost", "common"],
        ]
        raise ValueError(
            f"{level}: lost + common does not equal baseline "
            f"within tolerance for {len(bad)} row(s):\n{bad.to_string(index=False)}"
        )

    # Derived percentages.
    df["lost_pct"] = 100.0 * df["lost"] / df["baseline"]
    df["gained_pct"] = 100.0 * df["gained"] / df["baseline"]
    df["net_pct"] = (
        100.0 * (df["gained"] - df["lost"]) / df["baseline"]
    )

    # XPECTRA count relative to the same baseline.
    df["xpectra_total"] = df["common"] + df["gained"]
    df["xpectra_pct_of_baseline"] = (
        100.0 * df["xpectra_total"] / df["baseline"]
    )

    return df

def ordered_rows(df):
    """
    Return rows in the fixed dataset/method order.
    """
    result = df.copy()

    result["dataset"] = pd.Categorical(
        result["dataset"],
        categories=DATASET_ORDER,
        ordered=True,
    )
    result["method"] = pd.Categorical(
        result["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    return result.sort_values(["dataset", "method"])

def collect_panel_limits(data):
    """
    Find shared x limits for all top-row panels.
    """
    lost_values = []
    gained_values = []

    for df in data.values():
        lost_values.extend(df["lost_pct"].tolist())
        gained_values.extend(df["gained_pct"].tolist())

    max_loss = max(lost_values, default=1.0)
    max_gain = max(gained_values, default=1.0)
    max_abs = max(max_loss, max_gain, 1.0)

    limit = max_abs * 1.25

    return -limit, limit

def collect_bottom_limits(data):
    """
    Find shared y limits for all bottom-row panels.
    """
    values = []

    for df in data.values():
        values.extend(df["net_pct"].tolist())

    if not values:
        return -1.0, 1.0

    ymin = min(0.0, min(values))
    ymax = max(0.0, max(values))

    span = max(ymax - ymin, 1.0)
    padding = 0.15 * span

    return ymin - padding, ymax + padding

# 4. Top row: identification turnover
def panel_turnover(
    ax,
    df,
    level_label,
    xlim,
    show_dataset_labels=False,
    show_method_labels=True,
):
    """
    Plot lost vs gained identification percentages.
    """
    df = ordered_rows(df)

    bar_height = 0.62
    group_gap = 1.75

    y_positions = []
    y_labels = []
    dataset_centers = []

    y = 0.0

    for dataset in DATASET_ORDER:
        rows = df[df["dataset"] == dataset]

        block_start = y

        for method in METHOD_ORDER:
            row = rows[rows["method"] == method]

            if row.empty:
                continue

            y_positions.append(y)
            if dataset == DATASET_ORDER[0]:
                y_labels.append(method)
            else:
                y_labels.append("")

            # y_labels.append(method)
            y += 1.0

        if y > block_start:
            dataset_centers.append(
                (dataset, (block_start + y - 1.0) / 2.0)
            )

        y += group_gap

    # Draw bars.
    y = 0.0

    for dataset in DATASET_ORDER:
        rows = df[df["dataset"] == dataset]

        for method in METHOD_ORDER:
            row = rows[rows["method"] == method]

            if row.empty:
                continue

            row = row.iloc[0]
            color = METHOD_COLORS[method]

            lost = -float(row["lost_pct"])
            gained = float(row["gained_pct"])

            # Lost: light version of method color.
            ax.barh(
                y,
                lost,
                height=bar_height,
                color=lighten(color, 0.55),
                edgecolor=color,
                linewidth=0.5,
                zorder=3,
            )

            # Gained: full method color.
            ax.barh(
                y,
                gained,
                height=bar_height,
                color=color,
                edgecolor=color,
                linewidth=0.5,
                zorder=3,
            )

            if gained > 0:
                ax.text(
                    gained + 0.022 * (xlim[1] - xlim[0]),
                    y,
                    f"+{gained:.1f}%",
                    ha="left",
                    va="center",
                    fontsize=5.6,
                    color=color,
                    clip_on=False,
                )

            if lost < 0:
                ax.text(
                    lost - 0.022 * (xlim[1] - xlim[0]),
                    y,
                    f"{lost:.1f}%",
                    ha="right",
                    va="center",
                    fontsize=5.6,
                    color="#666666",
                    clip_on=False,
                )

            y += 1.0

        y += group_gap

    ax.set_xlim(*xlim)
    ax.axvline(
        0,
        color="black",
        linewidth=0.7,
        zorder=4,
    )

    ax.set_yticks(y_positions)
    if show_method_labels:
        ax.set_yticklabels(y_labels)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", left=False)

    ax.invert_yaxis()

    ax.set_xlabel(
        "% of MS-GF+ baseline",
        labelpad=3,
    )

    ax.set_title(
        level_label,
        fontweight="bold",
        pad=8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(
        axis="y",
        length=0,
        pad=3,
    )

    ax.grid(
        axis="x",
        color="#e6e6e6",
        linewidth=0.5,
        zorder=0,
    )

    ax.set_axisbelow(True)

    # Dataset labels are shown only on the first panel.
    if show_dataset_labels:
        for dataset, center in dataset_centers:
            ax.annotate(
                DATASET_LABELS[dataset],
                xy=(0, center),
                xycoords=ax.get_yaxis_transform(),
                xytext=(-14, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=6.0,
                fontweight="bold",
                color="#333333",
                annotation_clip=False,
            )


# 5. Bottom row: net gain
def panel_net_gain(
    ax,
    df,
    level_label,
    ylim,
    show_legend=False,
):
    """Plot net gain/loss relative to the MS-GF+ baseline."""
    df = ordered_rows(df)

    n_methods = len(METHOD_ORDER)
    width = 0.17
    spacing = 1.7

    x = np.arange(len(DATASET_ORDER), dtype=float) * spacing

    # Collect all values first so annotation placement is consistent.
    values_by_method = {}

    for method in METHOD_ORDER:
        values = []

        for dataset in DATASET_ORDER:
            match = df[
                (df["dataset"] == dataset)
                & (df["method"] == method)
            ]["net_pct"]

            values.append(
                float(match.iloc[0]) if not match.empty else 0.0
            )

        values_by_method[method] = values

    # Plot bars.
    for i, method in enumerate(METHOD_ORDER):
        values = values_by_method[method]

        offset = (
            i - (n_methods - 1) / 2.0
        ) * (width + 0.1)

        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.4,
            label=method,
            zorder=3,
        )

        for bar, value in zip(bars, values):
            if abs(value) < 0.5:
                continue

            if value >= 0:
                y_text = value + 0.025 * (ylim[1] - ylim[0])
                va = "bottom"
            else:
                y_text = value - 0.025 * (ylim[1] - ylim[0])
                va = "top"

            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_text,
                f"{value:+.1f}%",
                ha="center",
                va=va,
                fontsize=5.0,
                rotation=90,
                color="#222222",
                clip_on=True,
            )

    ax.axhline(
        0,
        color="black",
        linewidth=0.7,
        zorder=4,
    )

    ax.set_ylim(*ylim)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [DATASET_LABELS[d] for d in DATASET_ORDER],
        rotation=30,
        ha="right",
    )

    ax.set_ylabel(
        "% net gain",
        labelpad=3,
    )

    ax.set_title(
        level_label,
        fontweight="bold",
        pad=8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        color="#e6e6e6",
        linewidth=0.5,
        zorder=0,
    )

    ax.set_axisbelow(True)

    ax.tick_params(
        axis="x",
        length=2.5,
        pad=3,
    )

    if show_legend:
        ax.legend(
            frameon=False,
            ncol=1,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
        )


# ----------------------------------------------------------------------
# 6. Main figure
# ----------------------------------------------------------------------

def main():
    data = {
        level: load_level(level)
        for level in LEVELS
    }

    top_xlim = collect_panel_limits(data)
    bottom_ylim = collect_bottom_limits(data)

    fig = plt.figure(
        figsize=(10.5, 10.5),
    )

    gs = fig.add_gridspec(
        2,
        3,
        hspace=0.30,
        wspace=0.25,
        left=0.10,
        right=0.86,
        top=0.95, 
        bottom=0.10,
        height_ratios=[1.5, 1.0],
    )

    # Panel letters.
    top_letters = ["a)", "b)", "c)"]
    bottom_letters = ["d)", "e)", "f)"]

    # --------------------------------------------------------------
    # Top row
    # --------------------------------------------------------------

    for i, level in enumerate(LEVELS):
        ax = fig.add_subplot(gs[0, i])

        panel_turnover(
            ax,
            data[level],
            level,
            xlim=top_xlim,
            show_dataset_labels=(i == 0),
            show_method_labels=False,
        )

        ax.text(
            -0.20 if i == 0 else -0.10,
            1.06,              # was 1.16 — tighten to reduce top whitespace
            top_letters[i],
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    # --------------------------------------------------------------
    # Bottom row
    # --------------------------------------------------------------

    for i, level in enumerate(LEVELS):
        ax = fig.add_subplot(gs[1, i])

        panel_net_gain(
            ax,
            data[level],
            level,
            ylim=bottom_ylim,
            show_legend=False,
        )

        ax.text(
            -0.20 if i == 0 else -0.10,
            1.03,              # was 1.08 — tighten to reduce top whitespace
            bottom_letters[i],
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    # --------------------------------------------------------------
    # Shared legends
    # --------------------------------------------------------------
    method_handles = [
        Patch(
            facecolor=METHOD_COLORS[m],
            edgecolor="none",
            label=m,
        )
        for m in METHOD_ORDER
    ]

    turnover_handles = [
        Patch(
            facecolor="#777777",
            edgecolor="#777777",
            label="Gained",
        ),
        Patch(
            facecolor=lighten("#777777", 0.55),
            edgecolor="#777777",
            label="Lost",
        ),
    ]

    # Method legend moved to the bottom.
    fig.legend(
        handles=method_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.02),   # was (0.50, 0.975)
        ncol=4,
        frameon=False,
        handlelength=1.0,
        columnspacing=1.2,
        handletextpad=0.4,
        fontsize=6.3,
    )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:")
    print(f"  {OUTPUT_PDF}")

if __name__ == "__main__":
    main()