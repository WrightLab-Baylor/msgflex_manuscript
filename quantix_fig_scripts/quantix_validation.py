#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROTEIN_CROSSTAB_PATH = "rollup_matrix/final_01_wr.tsv"
PEPTIDE_CROSSTAB_PATH = "rollup_matrix/peptide_crosstab_annotated.tsv" 
METADATA_PATH = "rollup_matrix/sample_metadata.tsv"
OUTPUT_DIR = "rollup_validation_output"

QVAL_THRESH = 0.05
EXPECTED_LOG2FC_SPIKE = np.log2(2.0)  #1.0
MIN_VALID_PER_GROUP = 6

COLOR_SPIKE = "#e6550d" # ecoli 
COLOR_BACKGROUND = "#0476c7"  # human
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 10, "axes.labelsize": 8, "figure.dpi": 300,
})

ORGANISM_TAGS_ECOLI = ["_ECOLI", "_ECO", "_ECO57", "_ECOL7", "_ECO24"]
ORGANISM_TAGS_HUMAN = ["_HUMAN"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
})

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def get_organism(pid):
    if not isinstance(pid, str):
        return "unknown"
    pid_upper = pid.upper()
    if any(tag in pid_upper for tag in ORGANISM_TAGS_ECOLI):
        return "ecoli"
    if any(tag in pid_upper for tag in ORGANISM_TAGS_HUMAN):
        return "human"
    return "unknown"

def benjamini_hochberg(pvalues):
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
    order = np.argsort(p_valid, kind="mergesort")
    ranks = np.arange(1, n_valid + 1, dtype=float)

    adjusted = p_valid[order] * n_valid / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    q_valid = np.empty(n_valid, dtype=float)
    q_valid[order] = adjusted
    q[valid] = q_valid
    return q

_NAN_PAIR = (np.nan, np.nan)
def safe_ttest_ind(a, b, equal_var=False, alternative="two-sided"):
    """
    Welch t-test that handles zero-variance without crashing/warnings.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    if a.size < 2 or b.size < 2:
        return _NAN_PAIR
    if np.nanstd(a) == 0 and np.nanstd(b) == 0:
        return _NAN_PAIR

    try:
        stat, p = stats.ttest_ind(a, b, equal_var=equal_var, alternative=alternative, nan_policy="omit")
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return _NAN_PAIR

    if not np.isfinite(stat) or not np.isfinite(p):
        return _NAN_PAIR
    return float(stat), float(p)

def run_validation(crosstab_path, id_col, organism_col, output_prefix, level_label,
                    sample_conditions, all_meta_samples):
    df = pd.read_csv(crosstab_path, sep="\t")

    sample_cols = [s for s in all_meta_samples if s in df.columns]
    missing_in_crosstab = [s for s in all_meta_samples if s not in df.columns]
    if missing_in_crosstab:
        raise ValueError(f"[{level_label}] Samples in metadata but not in crosstab: {missing_in_crosstab}")

    annot_cols = [c for c in df.columns if c not in sample_cols and c != id_col]
    print(f"\n[{level_label}] rows={len(df)}  samples={len(sample_cols)}  annotation cols={annot_cols}")

    samples_1x = [s for s in sample_cols if sample_conditions[s] == "1x"]
    samples_2x = [s for s in sample_cols if sample_conditions[s] == "2x"]
    if not samples_1x or not samples_2x:
        raise ValueError(f"[{level_label}] Missing samples for '1x' or '2x' condition.")

    # --- preprocessing ---
    keep_sample_cols = sample_cols
    work = df[[id_col] + ([organism_col] if organism_col != id_col else []) + keep_sample_cols].copy()
    work = work.reset_index(drop=True)
    organism_all = work[organism_col].apply(get_organism)

    raw = work[keep_sample_cols].fillna(0)
    human_totals = raw[organism_all.values == "human"].sum(axis=0)
    human_totals = human_totals.replace(0, np.nan)
    scale_factor = human_totals.mean() / human_totals
    scale_factor = scale_factor.fillna(1.0) # Fallback to 1.0 if all human totals are 0
    
    normalized = raw.multiply(scale_factor, axis=1)
    normalized = normalized.replace(0, np.nan)
    log2_norm = np.log2(normalized)

    valid_1x = log2_norm[samples_1x].notna().sum(axis=1) >= MIN_VALID_PER_GROUP
    valid_2x = log2_norm[samples_2x].notna().sum(axis=1) >= MIN_VALID_PER_GROUP
    mask = valid_1x & valid_2x

    log2_filt = log2_norm[mask].reset_index(drop=True)
    ids_filt = work.loc[mask, id_col].reset_index(drop=True)
    organism_raw_filt = work.loc[mask, organism_col].reset_index(drop=True)
    print(f"[{level_label}] passing filter: {len(log2_filt)} / {len(log2_norm)}")

    if len(log2_filt) == 0:
        print(f"[{level_label}] WARNING: 0 rows passed filter. Skipping stats and plots.")
        accuracy = {"n_spike": 0, "mean_observed_log2fc": np.nan, "expected_log2fc": EXPECTED_LOG2FC_SPIKE, "mean_error": np.nan, "rmse": np.nan, "ttest_p_vs_expected": np.nan}
        specificity = {"n_background": 0, "false_positive_rate": np.nan, "n_false_positives": 0}
        sensitivity = {"n_spike_tested": 0, "sensitivity": np.nan, "n_detected": 0}
        summary = pd.DataFrame([
            {"metric_category": "accuracy", **accuracy},
            {"metric_category": "specificity", **specificity},
            {"metric_category": "sensitivity", **sensitivity},
        ])
        summary["level"] = level_label
        return summary

    sample_medians = log2_filt.median(axis=0, skipna=True)
    log2_norm = log2_filt.subtract(sample_medians, axis=1)

    # --- differential testing (Welch's t-test + BH-FDR) ---
    results = []
    for i in range(len(log2_norm)):
        v1 = log2_norm.loc[i, samples_1x].dropna().values
        v2 = log2_norm.loc[i, samples_2x].dropna().values
        log2fc = np.mean(v2) - np.mean(v1)
        # Fix 4: Use safe t-test
        _, pval = safe_ttest_ind(v2, v1, equal_var=False)
        results.append({
            "id": ids_filt.iloc[i], "organism_raw": organism_raw_filt.iloc[i],
            "log2fc_observed": log2fc, "pvalue": pval,
            "n_1x": len(v1), "n_2x": len(v2),
        })
    res = pd.DataFrame(results)
    # Fix 1: NaN-safe BH
    res["qvalue"] = benjamini_hochberg(res["pvalue"].values)
    res["organism"] = res["organism_raw"].apply(get_organism)

    spike = res[res["organism"] == "ecoli"].dropna(subset=["log2fc_observed", "qvalue"]).copy()
    background = res[res["organism"] == "human"].dropna(subset=["log2fc_observed", "qvalue"]).copy()
    print(f"[{level_label}] annotated: {len(spike)} ecoli (spike) | {len(background)} human (background)")

    # --- validation metrics ---
    if len(spike) >= 2:
        errors = spike["log2fc_observed"] - EXPECTED_LOG2FC_SPIKE
        _, p_value = stats.ttest_1samp(spike["log2fc_observed"], EXPECTED_LOG2FC_SPIKE)
        accuracy = {
            "n_spike": len(spike), "mean_observed_log2fc": spike["log2fc_observed"].mean(),
            "expected_log2fc": EXPECTED_LOG2FC_SPIKE, "mean_error": errors.mean(),
            "rmse": np.sqrt(np.mean(errors ** 2)), "ttest_p_vs_expected": p_value,
        }
    elif len(spike) == 1:
        errors = spike["log2fc_observed"] - EXPECTED_LOG2FC_SPIKE
        accuracy = {
            "n_spike": 1, "mean_observed_log2fc": spike["log2fc_observed"].iloc[0],
            "expected_log2fc": EXPECTED_LOG2FC_SPIKE, "mean_error": errors.iloc[0],
            "rmse": np.abs(errors.iloc[0]), "ttest_p_vs_expected": np.nan,
        }
    else:
        accuracy = {
            "n_spike": 0, "mean_observed_log2fc": np.nan,
            "expected_log2fc": EXPECTED_LOG2FC_SPIKE, "mean_error": np.nan,
            "rmse": np.nan, "ttest_p_vs_expected": np.nan,
        }

    sig_background = background["qvalue"] < QVAL_THRESH
    specificity = {
        "n_background": len(background), "false_positive_rate": sig_background.mean() if len(background) > 0 else np.nan,
        "n_false_positives": int(sig_background.sum()) if len(background) > 0 else 0,
    }
    
    # Direction-aware sensitivity
    if len(spike) > 0:
        expected_sign = np.sign(EXPECTED_LOG2FC_SPIKE)
        correct_direction = np.sign(spike["log2fc_observed"]) == expected_sign
        sig_spike = (spike["qvalue"] < QVAL_THRESH) & correct_direction
        sensitivity = {
            "n_spike_tested": len(spike), "sensitivity": sig_spike.mean(),
            "n_detected": int(sig_spike.sum()),
        }
    else:
        sensitivity = {
            "n_spike_tested": 0, "sensitivity": np.nan, "n_detected": 0,
        }

    for name, d in [("ACCURACY", accuracy), ("SPECIFICITY", specificity), ("SENSITIVITY", sensitivity)]:
        print(f"\n=== [{level_label}] {name} ===")
        for k, v in d.items():
            print(f"{k:25s}: {v:.4f}" if isinstance(v, float) else f"{k:25s}: {v}")

    # --- plots ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    #volcano
    ax = axes[0]
    if len(background) > 0:
        ax.scatter(background["log2fc_observed"], -np.log10(background["qvalue"].clip(lower=1e-300)),
                   c="#0476c7", alpha=0.5, s=14, label="Human", edgecolors="none")
    if len(spike) > 0:
        ax.scatter(spike["log2fc_observed"], -np.log10(spike["qvalue"].clip(lower=1e-300)),
                   c=COLOR_SPIKE, alpha=0.75, s=26, label="E. coli", edgecolors="white", linewidth=0.3)
    ax.axhline(-np.log10(QVAL_THRESH), color="black", ls="--", lw=1)
    ax.axvline(1.0, color="black", ls=":", lw=1.2)
    ax.axvline(-1.0, color="black", ls=":", lw=1.2)
    ax.set_xlabel("Observed log2FC"); ax.set_ylabel("-log10(q-value)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # Box plot
    ax = axes[1]
    groups = [g for g in [background["log2fc_observed"].values, spike["log2fc_observed"].values] if len(g) > 0]
    labels = [l for l, g in zip(["Human", "E. coli"], [background["log2fc_observed"].values, spike["log2fc_observed"].values]) if len(g) > 0]
    colors = [c for c, g in zip([COLOR_BACKGROUND, COLOR_SPIKE], [background["log2fc_observed"].values, spike["log2fc_observed"].values]) if len(g) > 0]
    
    if groups:
        bp = ax.boxplot(groups, positions=np.arange(1, len(groups)+1), widths=0.45, patch_artist=True,
                         showfliers=False, medianprops=dict(color="black", linewidth=1.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.65)
        ax.set_xticks(np.arange(1, len(groups)+1))
        ax.set_xticklabels(labels)
    ax.set_ylabel("Observed log2FC")
    ax.spines[["top", "right"]].set_visible(False)

    # ROC curve
    ax = axes[2]
    if len(spike) > 0 and len(background) > 0:
        thresholds = np.sort(np.logspace(-10, 0, 400))
        tprs = [(spike["qvalue"] < t).mean() for t in thresholds]
        fprs = [(background["qvalue"] < t).mean() for t in thresholds]
        fprs, tprs = np.array(fprs), np.array(tprs)
        order = np.argsort(fprs)
        auc = np.trapz(tprs[order], fprs[order])
        ax.plot(fprs, tprs, lw=2, color="navy", label=f"ROC (AUC = {auc:.3f})")
    else:
        auc = 0.5
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (No data)")
        
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate (Human)")
    ax.set_ylabel("True Positive Rate (E. coli)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_diagnostics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[{level_label}] AUC = {auc:.4f} | saved {output_prefix}_diagnostics.png")

    # --- save tables ---
    res.to_csv(f"{output_prefix}_differential_results.tsv", sep="\t", index=False)
    summary = pd.DataFrame([
        {"metric_category": "accuracy", **accuracy},
        {"metric_category": "specificity", **specificity},
        {"metric_category": "sensitivity", **sensitivity},
    ])
    summary["level"] = level_label
    summary.to_csv(f"{output_prefix}_validation_summary.tsv", sep="\t", index=False)
    return summary


meta = pd.read_csv(METADATA_PATH, sep="\t")
if "sample" not in meta.columns or "condition" not in meta.columns:
    raise ValueError("Metadata file must contain 'sample' and 'condition' columns.")
sample_conditions = dict(zip(meta["sample"], meta["condition"]))
all_meta_samples = list(meta["sample"].unique())

protein_summary = run_validation(
    crosstab_path=PROTEIN_CROSSTAB_PATH,
    id_col="Protein",
    organism_col="Protein",
    output_prefix=f"{OUTPUT_DIR}/protein_level_quant_validation",
    level_label="Protein",
    sample_conditions=sample_conditions,
    all_meta_samples=all_meta_samples,
)

peptide_summary = run_validation(
    crosstab_path=PEPTIDE_CROSSTAB_PATH,
    id_col="Peptide",
    organism_col="Protein", 
    output_prefix=f"{OUTPUT_DIR}/peptide_level_quant_validation",
    level_label="Peptide",
    sample_conditions=sample_conditions,
    all_meta_samples=all_meta_samples,
)

combined = pd.concat([protein_summary, peptide_summary], ignore_index=True)
combined.to_csv(f"{OUTPUT_DIR}/combined_validation_summary.tsv", sep="\t", index=False)
print(f"\nSaved combined summary: {OUTPUT_DIR}/combined_validation_summary.tsv")