"""
Rollup method comparison for QUANTIX
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
METADATA_PATH = Path(
    "sample_metadata.tsv"
)
OUTPUT_DIR = Path(
    "rollup_comparison"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ROLLUP_CROSSTABS = {
    "sum": Path(
        "rollup_matrix/final_01_sum.tsv"
    ),
    "rrollup": Path(
        "rollup_matrix/final_01_rr.tsv"
    ),
    "qrollup": Path(
      "rollup_matrix/final_01_qr.tsv"
    ),
    "zrollup": Path(
        "rollup_matrix/final_01_zr.tsv"
    ),
    "weighted": Path(
        "rollup_matrix/final_01_dwr.tsv"
    ),
}

ID_COL = "Protein"
QVAL_THRESH = 0.05
EXPECTED_LOG2FC_SPIKE = np.log2(2.0)  

COLOR_SPIKE = "#e6550d" # ecoli 
COLOR_BACKGROUND = "#0476c7"  # human

plt.rcParams.update({
    "font.family":"sans-serif",
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 8
})

ORGANISM_TAGS = {
    "_ECOLI":  "ecoli",
    "_HUMAN":  "human",
}

ORGANISM_TAGS_ECOLI = ["_ECOLI", "_ECO", "_ECO57", "_ECOL7", "_ECO24"]
ORGANISM_TAGS_HUMAN = ["_HUMAN"]
ALREADY_TRANSFORMED = {"zrollup"}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def get_organism(pid):
    """Classify a protein ID by organism."""
    pid_upper = str(pid).upper()
    if any(tag in pid_upper for tag in ORGANISM_TAGS_ECOLI):
        return "ecoli"
    if any(tag in pid_upper for tag in ORGANISM_TAGS_HUMAN):
        return "human"
    return "unknown"

def build_organism_map(protein_ids):
    """
    Map raw protein IDs to organism labels.
    """
    mapping = {}
    for pid in protein_ids:
        if not isinstance(pid, str):
            mapping[pid] = "UNKNOWN"
            continue
        match = None
        for tag, label in ORGANISM_TAGS.items():
            if pid.endswith(tag) or tag in pid:
                match = label
                break
        mapping[pid] = match or "UNKNOWN"
    return mapping


def load_and_prepare(work, sample_cols, organism_all, rollup_method):
    if rollup_method in ALREADY_TRANSFORMED:
        return work[sample_cols].copy()

    raw = work[sample_cols].fillna(0)
    human_totals = raw[organism_all.values == "human"].sum(axis=0)

    # Guard against zero human totals (division by zero / inf scale factors)
    scale_factor = human_totals.mean() / human_totals.replace(0, np.nan)
    scale_factor = scale_factor.fillna(scale_factor.mean()).fillna(1.0)

    normalized = raw.multiply(scale_factor, axis=1).replace(0, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log2(normalized.where(normalized > 0))

def benjamini_hochberg(pvalues):
    """BH FDR correction that is NaN-safe and empty-safe."""
    p = np.asarray(pvalues, dtype=float).ravel()
    n_total = p.size
    if n_total == 0:
        return np.array([], dtype=float)

    valid = ~np.isnan(p)
    n_valid = int(valid.sum())
    q = np.full(n_total, np.nan, dtype=float)
    if n_valid == 0:
        return q

    p_valid = p[valid]
    order = np.argsort(p_valid, kind="mergesort")  # stable
    ranks = np.arange(1, n_valid + 1, dtype=float)

    # BH step-up: q_i = p_i * m / rank_i, then enforce monotonicity from the top
    adjusted = p_valid[order] * n_valid / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    q_valid = np.empty(n_valid, dtype=float)
    q_valid[order] = adjusted
    q[valid] = q_valid
    return q

_NAN_PAIR = (np.nan, np.nan)
def safe_ttest_ind(a, b, equal_var=False, alternative="two-sided"):
    """Welch t-test that returns (NaN, NaN) instead of raising."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    if a.size < 2 or b.size < 2:
        return _NAN_PAIR
    if np.nanstd(a) == 0 and np.nanstd(b) == 0:
        # Both groups are constant; t is undefined
        return _NAN_PAIR

    try:
        stat, p = stats.ttest_ind(a, b, equal_var=equal_var,
                                  alternative=alternative, nan_policy="omit")
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return _NAN_PAIR

    if not np.isfinite(stat) or not np.isfinite(p):
        return _NAN_PAIR
    return float(stat), float(p)

def run_validation(
    crosstab_path,
    rollup_method,
    sample_conditions,
    all_meta_samples,
    organism_map,
    output_prefix,
):
    df = pd.read_csv(crosstab_path, sep="\t")

    sample_cols = [s for s in all_meta_samples if s in df.columns]
    missing = [s for s in all_meta_samples if s not in df.columns]
    if missing:
        raise ValueError(
            f"[{rollup_method}] Samples in metadata but not in crosstab: {missing}"
        )

    samples_1x = [s for s in sample_cols if sample_conditions[s] == "1x"]
    samples_2x = [s for s in sample_cols if sample_conditions[s] == "2x"]
    if not samples_1x or not samples_2x:
        raise ValueError(
            f"[{rollup_method}] Missing samples for '1x' or '2x' condition."
        )

    # Dynamic threshold: require at least half of the smaller group, but at least 6
    min_valid_per_group = max(2, min(len(samples_1x), len(samples_2x)) // 2)
    min_valid_per_group = max(min_valid_per_group, 6)  # Ensure at least 6 valid per group

    work = df[[ID_COL] + sample_cols].copy()
    work = work.reset_index(drop=True)

    # Map organism via master lookup; fall back to parsing the ID itself
    organism_all = work[ID_COL].astype(str).map(organism_map).fillna(
        work[ID_COL].apply(get_organism)
    )

    log2_norm = load_and_prepare(work, sample_cols, organism_all, rollup_method)

    valid_1x = log2_norm[samples_1x].notna().sum(axis=1) >= min_valid_per_group
    valid_2x = log2_norm[samples_2x].notna().sum(axis=1) >= min_valid_per_group
    mask = valid_1x & valid_2x

    log2_norm = log2_norm[mask].reset_index(drop=True)
    ids_filt = work.loc[mask, ID_COL].reset_index(drop=True)
    organism_filt = organism_all[mask].reset_index(drop=True)

    n_total = len(work)
    n_pass = len(log2_norm)
    print(f"[{rollup_method}] passing filter: {n_pass} / {n_total}")

    if n_pass == 0:
        print(f"[{rollup_method}] WARNING: 0 proteins passed filter. Returning empty summary.")
        summary = {
            "rollup_method": rollup_method,
            "n_total": n_total,
            "n_pass_filter": 0,
            "n_spike": 0,
            "mean_observed_log2fc": np.nan,
            "mean_error": np.nan,
            "rmse": np.nan,
            "ttest_p_vs_expected": np.nan,
            "n_background": 0,
            "false_positive_rate": np.nan,
            "n_false_positives": 0,
            "n_spike_tested": 0,
            "sensitivity": np.nan,
            "n_detected": 0,
            "auc": 0.5,
        }
        return summary, np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5

    results = []
    for i in range(len(log2_norm)):
        v1 = log2_norm.loc[i, samples_1x].dropna().values
        v2 = log2_norm.loc[i, samples_2x].dropna().values
        
        log2fc = np.mean(v2) - np.mean(v1)
        _, pval = safe_ttest_ind(v2, v1)
        results.append(
            {
                "id": ids_filt.iloc[i],
                "organism": organism_filt.iloc[i],
                "log2fc_observed": log2fc,
                "pvalue": pval,
                "n_1x": len(v1),
                "n_2x": len(v2),
            }
        )

    res = pd.DataFrame(results)
    res["qvalue"] = benjamini_hochberg(res["pvalue"].values)

    spike = res[res["organism"] == "ecoli"].dropna(
        subset=["log2fc_observed", "qvalue"]
    ).copy()
    background = res[res["organism"] == "human"].dropna(
        subset=["log2fc_observed", "qvalue"]
    ).copy()

    print(
        f"[{rollup_method}] annotated: {len(spike)} E. coli | "
        f"{len(background)} Human"
    )

    # --- Accuracy ---
    if len(spike) >= 2:
        errors = spike["log2fc_observed"] - EXPECTED_LOG2FC_SPIKE
        _, p_value = stats.ttest_1samp(
            spike["log2fc_observed"], EXPECTED_LOG2FC_SPIKE
        )
        accuracy = {
            "n_spike": len(spike),
            "mean_observed_log2fc": spike["log2fc_observed"].mean(),
            "mean_error": errors.mean(),
            "rmse": np.sqrt(np.mean(errors ** 2)),
            "ttest_p_vs_expected": p_value,
        }
    elif len(spike) == 1:
        errors = spike["log2fc_observed"] - EXPECTED_LOG2FC_SPIKE
        accuracy = {
            "n_spike": 1,
            "mean_observed_log2fc": spike["log2fc_observed"].iloc[0],
            "mean_error": errors.iloc[0],
            "rmse": np.abs(errors.iloc[0]),
            "ttest_p_vs_expected": np.nan,
        }
    else:
        print(
            f"[{rollup_method}] WARNING: 0 spike proteins found. "
            "Check organism tags / mapping."
        )
        accuracy = {
            "n_spike": 0,
            "mean_observed_log2fc": np.nan,
            "mean_error": np.nan,
            "rmse": np.nan,
            "ttest_p_vs_expected": np.nan,
        }

    # --- Specificity ---
    if len(background) > 0:
        sig_background = background["qvalue"] < QVAL_THRESH
        specificity = {
            "n_background": len(background),
            "false_positive_rate": sig_background.mean(),
            "n_false_positives": sig_background.sum(),
        }
    else:
        specificity = {
            "n_background": 0,
            "false_positive_rate": np.nan,
            "n_false_positives": 0,
        }

    # --- Sensitivity ---
    if len(spike) > 0:
        expected_sign = np.sign(EXPECTED_LOG2FC_SPIKE)
        obs_sign = np.sign(spike["log2fc_observed"])
        correct_direction = obs_sign == expected_sign
        
        sig_spike = (spike["qvalue"] < QVAL_THRESH) & correct_direction
        
        sensitivity = {
            "n_spike_tested": len(spike),
            "sensitivity": sig_spike.mean(),
            "n_detected": int(sig_spike.sum()),
        }
    else:
        sensitivity = {
            "n_spike_tested": 0,
            "sensitivity": np.nan,
            "n_detected": 0,
        }

    # --- ROC + AUC ---
    if len(spike) > 0 and len(background) > 0:
        thresholds = np.logspace(-10, 0, 400)
        tprs = [(spike["qvalue"] < t).mean() for t in thresholds]
        fprs = [(background["qvalue"] < t).mean() for t in thresholds]
        fprs = np.array(fprs)
        tprs = np.array(tprs)
        fprs = np.concatenate([[0.0], fprs, [1.0]])
        tprs = np.concatenate([[0.0], tprs, [1.0]])
        order = np.argsort(fprs)
        auc = np.trapz(tprs[order], fprs[order])
    else:
        fprs = np.array([0.0, 1.0])
        tprs = np.array([0.0, 1.0])
        auc = 0.5

    # --- Per-method diagnostic figure  ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, label in zip(axes, ["a)", "b)"]):
        ax.text(
            -0.14, 1.06, label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom",
        )

    # Volcano
    ax = axes[0]
    q_bg = background["qvalue"].clip(lower=1e-300)
    q_sp = spike["qvalue"].clip(lower=1e-300)
    ax.scatter(
        background["log2fc_observed"], -np.log10(q_bg),
        c="#3182bd", alpha=0.5, s=14, label="Human", edgecolors="none",
    )
    ax.scatter(
        spike["log2fc_observed"], -np.log10(q_sp),
        c=COLOR_SPIKE, alpha=0.75, s=26, label="E. coli",
        edgecolors="white", linewidth=0.3,
    )
    ax.axhline(-np.log10(QVAL_THRESH), color="black", ls="--", lw=1)
    
    # Fix #4: Only plot expected log2FC vertical lines for non-transformed methods
    if rollup_method not in ALREADY_TRANSFORMED:
        ax.axvline(1.0, color="black", ls=":", lw=1.2)
        ax.axvline(-1.0, color="black", ls=":", lw=1.2)
        
    ax.set_xlabel("Observed log2FC" if rollup_method not in ALREADY_TRANSFORMED else "Observed z-diff")
    ax.set_ylabel("-log10(q-value)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # Boxplot
    ax = axes[1]
    groups = [background["log2fc_observed"].values, spike["log2fc_observed"].values]
    bp = ax.boxplot(groups, positions=[1, 2], widths=0.45, patch_artist=True, showfliers=False, medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], [COLOR_BACKGROUND, COLOR_SPIKE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Human", "E. coli"])
    ax.set_ylabel("Observed log2FC" if rollup_method not in ALREADY_TRANSFORMED else "Observed z-diff")

    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    diag_prefix = output_prefix.parent / f"{output_prefix.name}_diagnostics"
    plt.savefig(diag_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save differential results
    diff_path = output_prefix.parent / f"{output_prefix.name}_differential_results.tsv"
    res.to_csv(diff_path, sep="\t", index=False)

    # Flat summary dict
    summary = {
        "rollup_method": rollup_method,
        "n_total": n_total,
        "n_pass_filter": n_pass,
        **accuracy,
        **specificity,
        **sensitivity,
        "auc": auc,
    }
    
    return summary, fprs, tprs, auc

def main():
    meta = pd.read_csv(METADATA_PATH, sep="\t")
    if "sample" not in meta.columns or "condition" not in meta.columns:
        raise ValueError(
            "Metadata file must contain 'sample' and 'condition' columns."
        )

    sample_conditions = dict(zip(meta["sample"], meta["condition"]))
    all_meta_samples = list(meta["sample"].unique())

    ref_method = "sum"
    if ref_method not in ROLLUP_CROSSTABS:
        ref_method = list(ROLLUP_CROSSTABS.keys())[0]
        
    ref_path = ROLLUP_CROSSTABS[ref_method]
    print(f"Building organism map from reference: {ref_method} ({ref_path.name})")
    ref_df = pd.read_csv(ref_path, sep="\t", usecols=[ID_COL])
    organism_map = build_organism_map(ref_df[ID_COL].astype(str).unique())
    print(f"Mapped {len(organism_map)} protein IDs to organism.")

    all_summaries = []
    roc_data = {}
    for method_name, path in ROLLUP_CROSSTABS.items():
        prefix = OUTPUT_DIR / method_name
        s, fprs, tprs, auc = run_validation(
            crosstab_path=path,
            rollup_method=method_name,
            sample_conditions=sample_conditions,
            all_meta_samples=all_meta_samples,
            organism_map=organism_map,
            output_prefix=prefix,
        )
        all_summaries.append(s)
        roc_data[method_name] = {"fprs": fprs, "tprs": tprs, "auc": auc}

    combined = pd.DataFrame(all_summaries)
    combined.to_csv(
        OUTPUT_DIR / "rollup_comparison_summary.tsv", sep="\t", index=False
    )

    # Cross-method comparison table
    metrics_cols = [
        "mean_error",
        "rmse",
        "false_positive_rate",
        "sensitivity",
        "auc",
    ]
    rollup_wide = combined.set_index("rollup_method")[metrics_cols]
    print("\n=== ROLLUP METHOD COMPARISON ===")
    print(rollup_wide.to_string())

    # Comparison figure: one bar panel per metric
    metrics_to_plot = [
        ("rmse", "RMSE (spike log2FC vs. expected)", False),
        ("mean_error", "Mean bias (spike log2FC − expected)", False),
        ("false_positive_rate", "False positive rate (background)", False),
        ("sensitivity", "Sensitivity (spike detection)", True),
        ("auc", "AUC (spike vs. background separability)", True),
    ]

    metrics_filtered = [
        m for m in metrics_to_plot 
        if m[0].lower() != "auc" and "auc" not in m[1].lower()
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes_flat = axes.flatten()

    methods = rollup_wide.index.tolist()
    palette = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    for ax, (metric, title, higher_is_better) in zip(axes_flat, metrics_filtered):
        vals = rollup_wide[metric].values
        bars = ax.bar(
            methods, vals, color=palette, alpha=0.85, edgecolor="black", linewidth=0.5
        )
        valid_vals = [v for v in vals if not np.isnan(v)]
        if valid_vals:
            best_idx = (
                np.argmax(vals) if higher_is_better else np.argmin(vals)
            )
            bars[best_idx].set_edgecolor("black")
            bars[best_idx].set_linewidth(2.0)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=8)
        ax.axhline(0, color="black", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rollup_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved rollup comparison figure: {OUTPUT_DIR / 'rollup_comparison.png'}")

    # --- Combined ROC Plot ---
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Color palette for the 5 methods
    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))
    
    for i, (method_name, data) in enumerate(roc_data.items()):
        ax.plot(
            data["fprs"], data["tprs"], 
            lw=1.25, color=colors[i], 
            label=f"{method_name} (AUC = {data['auc']:.3f})"
        )
        
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate (human background)")
    ax.set_ylabel("True Positive Rate (E.coli spike)")
    ax.set_title("ROC Curve Comparison: Rollup Methods")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "combined_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined ROC curve: {OUTPUT_DIR / 'combined_roc_curve.png'}")

if __name__ == "__main__":
    main()