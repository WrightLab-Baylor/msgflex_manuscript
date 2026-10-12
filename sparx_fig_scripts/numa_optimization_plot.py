"""
Visualize MS-GF+ runtime reduction from NUMA-aware task/thread optimization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("/home/trelangi/Desktop/Metaproteomics_data_II/pdata/PXD017035/metrics_comparison.tsv", sep="\t")

# Convert runtime columns from seconds to minutes
for col in ["auto_task", "numa_opt", "auto_task_bin", "numa_opt_bin"]:
    df[col] = df[col] / 60.0

PAIRS = [
    ("auto_task", "numa_opt", "Conventional pipeline"),
    ("auto_task_bin", "numa_opt_bin", "Binning pipeline"),
]

def paired_stats(a, b):
    """
    Wilcoxon signed-rank (primary) + paired t-test (secondary) + effect size.
    """
    w_stat, w_p = stats.wilcoxon(df[a], df[b])
    t_stat, t_p = stats.ttest_rel(df[a], df[b])
    diff = df[a] - df[b]
    pct_reduction = 100 * diff / df[a]
    # matched-pairs rank-biserial correlation as effect size for Wilcoxon
    r_rb = 1 - (2 * w_stat) / (len(diff) * (len(diff) + 1) / 2)
    return {
        "wilcoxon_p": w_p,
        "ttest_p": t_p,
        "mean_pct_reduction": pct_reduction.mean(),
        "std_pct_reduction": pct_reduction.std(),
        "rank_biserial_r": r_rb,
    }

def p_to_asterisks(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
panel_labels = ["a)", "b)"]

for ax, (col_auto, col_numa, title), label in zip(axes, PAIRS, panel_labels):
    stats_result = paired_stats(col_auto, col_numa)
    print(f"\n{title}: {col_auto} vs {col_numa}")
    print(stats_result)

    data = [df[col_auto].values, df[col_numa].values]
    positions = [1.0, 1.4]

    # Box plot
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.25,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.25),
    )
    for patch, color in zip(bp["boxes"], ["#a6bddb", "#fd8d3c"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Paired points (jittered), no connecting lines
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.05, 0.05, size=len(df))
    for i in range(len(df)):
        x_vals = [positions[0] + jitter[i], positions[1] + jitter[i]]
        y_vals = [df[col_auto].iloc[i], df[col_numa].iloc[i]]
        ax.scatter(x_vals, y_vals, color=["#3182bd", "#e6550d"], s=18, zorder=2)

    ax.set_xticks(positions)
    ax.set_xticklabels(["Auto task\n(default)", "NUMA-optimized"])
    ax.set_ylabel("Runtime (min)")
    ax.set_title(title)
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")

    # Significance bar
    y_max = max(df[col_auto].max(), df[col_numa].max())
    y_bar = y_max * 1.08
    ax.plot([positions[0], positions[0], positions[1], positions[1]], [y_bar, y_bar * 1.02, y_bar * 1.02, y_bar], color="black", linewidth=1)
    sig_label = p_to_asterisks(stats_result["wilcoxon_p"])
    ax.text(
        sum(positions) / 2, y_bar * 1.03,
        f"{sig_label}",
        ha="center", va="bottom", fontsize=9,
    )
    ax.set_ylim(top=y_bar * 1.25)
    ax.set_xlim(positions[0] - 0.3, positions[1] + 0.3)
    
    # Annotate mean % reduction
    ax.text(
        0.5, 0.02,
        f"mean reduction: {stats_result['mean_pct_reduction']:.2f}% "
        f"(± {stats_result['std_pct_reduction']:.2f}%)",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=6, color="grey",
    )
plt.tight_layout()
plt.savefig("numa_runtime_reduction.png", dpi=300)

fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4.5))
pct_reduction_conv = 100 * (df["auto_task"] - df["numa_opt"]) / df["auto_task"]
pct_reduction_bin = 100 * (df["auto_task_bin"] - df["numa_opt_bin"]) / df["auto_task_bin"]

panel_data = [
    (pct_reduction_conv, "Conventional pipeline", "#3182bd"),
    (pct_reduction_bin, "Binning pipeline", "#e6550d"),
]
panel_labels2 = ["a)", "b)"]

for ax, (pct_reduction, title, color), label in zip(axes2, panel_data, panel_labels2):
    x = df["scans"].values
    y = pct_reduction.values

    # Linear fit + 95% CI band
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    # bootstrap CI band for the fit
    rng = np.random.default_rng(1)
    n_boot = 1000
    boot_preds = np.zeros((n_boot, len(x_line)))
    for i in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        s, b, *_ = stats.linregress(x[idx], y[idx])
        boot_preds[i] = s * x_line + b
    ci_low = np.percentile(boot_preds, 2.5, axis=0)
    ci_high = np.percentile(boot_preds, 97.5, axis=0)

    ax.fill_between(x_line, ci_low, ci_high, color=color, alpha=0.15, linewidth=0)
    ax.plot(x_line, y_line, color=color, linewidth=1.5)
    ax.scatter(x, y, color=color, s=45, edgecolor="white", linewidth=0.6, zorder=3)

    ax.set_xlabel("Scan count")
    ax.set_ylabel("Runtime reduction (%)")
    ax.set_title(title)
    ax.text(
        0.03, 0.05,
        f"r={r_value:.2f}, p={p_value:.2f}",
        transform=ax.transAxes, fontsize=9, color="grey",
    )
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")

plt.tight_layout()
plt.savefig("reduction_vs_scans.png", dpi=300)
