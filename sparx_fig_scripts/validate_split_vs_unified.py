#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.stats import ks_2samp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MASIC_COLS = [
    "PeakMaxIntensity", "PeakArea", "PeakSignalToNoiseRatio",
    "FWHMInScans", "PeakWidthMinutes", "ParentIonIntensity", "StatMomentsArea",
]

# Common suffixes appended by proteomics search pipelines that should be stripped
# when matching files across directories
SEARCH_SUFFIXES = [
    "_fht_PlusSICStats",
    "_fht",
    "_PlusSICStats", 
    "_SICStats",
    "_syn",
    "_fht_syn",
]

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _strip_flanks(peptide: pd.Series) -> pd.Series:
    return (
        peptide.astype(str)
        .str.replace(r"^[A-Z\-]\.|\.[A-Z\-]$", "", regex=True)
        .str.replace(r"[^A-Z]", "", regex=True)
    )

def load_file(path: str | Path, is_rescored: bool = False) -> pd.DataFrame:
    """Load a PSM file. If rescored, use xpec_q as primary QValue."""
    path = Path(path)
    df   = pd.read_csv(path, sep="\t", low_memory=False)
    cols = set(df.columns)

    spec_col = next((c for c in ("MSGFDB_SpecEValue", "SpecEValue") if c in cols), None)
    if spec_col is None:
        raise KeyError(f"No SpecEValue column in {path.name}. Columns: {sorted(cols)}")

    if "Scan" in cols:
        scan = df["Scan"].astype(int)
    elif "ScanNum" in cols:
        scan = df["ScanNum"].astype(int)
    else:
        raise KeyError(f"No scan column in {path.name}. Columns: {sorted(cols)}")

    spec_file = df["SpecFile"].astype(str) if "SpecFile" in cols \
                else pd.Series(path.stem, index=df.index, dtype=str)

    out = pd.DataFrame({
        "SpecFile": spec_file,
        "ScanNum": scan,
        "Peptide_bare": _strip_flanks(df["Peptide"]),
        "Protein": df["Protein"].astype(str),
        "SpecEValue": df[spec_col].astype(float),
        "EValue": df["EValue"].astype(float) if "EValue" in cols else np.nan,
        "DelM_PPM": df["DelM_PPM"].astype(float) if "DelM_PPM" in cols else np.nan,
        "ElutionTime":  df["ElutionTime"].astype(float) if "ElutionTime" in cols else np.nan,
    })

    # QValue handling — rescored files use xpec_q as primary QValue
    if is_rescored:
        if "xpec_q" not in cols:
            raise KeyError(f"Rescored file {path.name} missing xpec_q column. Columns: {sorted(cols)}")
        out["QValue"] = df["xpec_q"].astype(float)
        out["OriginalQValue"] = df["QValue"].astype(float) if "QValue" in cols else np.nan
    else:
        out["QValue"] = df["QValue"].astype(float) if "QValue" in cols else np.nan
        out["OriginalQValue"] = np.nan

    out["PepQValue"] = df["PepQValue"].astype(float) if "PepQValue" in cols else np.nan

    for c in MASIC_COLS:
        out[c] = df[c].astype(float) if c in cols else np.nan

    out["IsDecoy"] = out["Protein"].str.contains(r"^XXX[_:]?", regex=True, na=False)
    out["IsRescored"] = is_rescored

    return out

def filter_fdr(df, fdr):
    """Filter PSMs by FDR threshold. Applies to BOTH targets and decoys."""
    targets = df[~df['IsDecoy']]
    decoys  = df[df['IsDecoy']]
    return pd.concat([
        targets[targets['QValue'] <= fdr], 
        decoys[decoys['QValue'] <= fdr]
    ], ignore_index=True)

def has_masic(df: pd.DataFrame) -> bool:
    if "PeakArea" not in df.columns:
        return False
    pa = df["PeakArea"].dropna()
    return len(pa) > 0 and (pa > 0).any()

# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def match_on_scan(split: pd.DataFrame, unified: pd.DataFrame) -> pd.DataFrame:
    """Inner-join target PSMs on ScanNum. Adds SamePeptide flag."""
    s = split[~split["IsDecoy"]].copy()
    u = unified[~unified["IsDecoy"]].copy()

    join_cols = ["SpecFile", "ScanNum"] \
        if len(set(s["SpecFile"]) & set(u["SpecFile"])) > 0 else ["ScanNum"]

    merged = s.merge(u, on=join_cols, suffixes=("_split", "_uni"), how="inner")
    merged["SamePeptide"] = merged["Peptide_bare_split"] == merged["Peptide_bare_uni"]
    return merged

def count_exclusive_scans(split: pd.DataFrame, unified: pd.DataFrame) -> tuple[int, int]:
    """Count scans exclusive to each dataset using outer join."""
    s = split[~split["IsDecoy"]].copy()
    u = unified[~unified["IsDecoy"]].copy()

    join_cols = ["SpecFile", "ScanNum"] \
        if len(set(s["SpecFile"]) & set(u["SpecFile"])) > 0 else ["ScanNum"]

    merged = s.merge(u, on=join_cols, suffixes=("_split", "_uni"), 
                     how="outer", indicator=True)

    only_split = int((merged["_merge"] == "left_only").sum())
    only_unified = int((merged["_merge"] == "right_only").sum())

    return only_split, only_unified

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _kde_overlay(ax, x_split, x_uni, xlabel, title, log=True):
    for arr, label, color in [
        (x_split, "Bin/Split DB", "#33bbee"),
        (x_uni, "Unified DB", "#b063f3"),
    ]:
        if log:
            v = arr[np.isfinite(arr) & (arr > 0)]
            n_dropped = int((np.isfinite(arr) & (arr <= 0)).sum())
        else:
            v = arr[np.isfinite(arr)]
            n_dropped = 0

        if len(v) < 2:
            continue
        data = np.log10(v) if log else v
        xs = np.linspace(data.min(), data.max(), 400)
        kde = gaussian_kde(data, bw_method="scott")
        label_str = f"{label} (n={len(v):,})"
        if n_dropped > 0 and log:
            label_str += f" [{n_dropped} dropped]"
        ax.plot(xs, kde(xs), lw=1.8, label=label_str, color=color)
        ax.fill_between(xs, kde(xs), alpha=0.12, color=color)

    ax.set_xlabel(f"log10({xlabel})" if log else xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.4)

def _ecdf_overlay(ax, x_split, x_uni, xlabel, title):
    for arr, label, color in [
        (x_split, "Split-merged", "#33bbee"),
        (x_uni,   "Unified DB", "#b063f3"),
    ]:
        v = np.sort(arr[np.isfinite(arr)])
        if len(v) == 0:
            continue
        ecdf = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, ecdf, lw=1.8, label=f"{label} (n={len(v):,})", color=color)

        v_split = x_split[np.isfinite(x_split)]
        v_uni = x_uni[np.isfinite(x_uni)]
        if len(v_split) > 0 and len(v_uni) > 0:
            stat,pval = ks_2samp(v_split, v_uni)
            ax.text(0.98, 0.04, f"KS D={stat:.4f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.4)

def _scatter_log(ax, x, y, xlabel, ylabel, same_mask=None):
    valid = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x_v, y_v = x[valid], y[valid]

    if same_mask is not None:
        sm = same_mask[valid]
        ax.scatter(x_v[sm],  y_v[sm],  s=5, alpha=0.30, color="#0077bb",
                   label=f"Same peptide (n={sm.sum():,})")
        if (~sm).sum() > 0:
            ax.scatter(x_v[~sm], y_v[~sm], s=8, alpha=0.60, color="#ee3377",
                       label=f"Diff peptide (n={(~sm).sum():,})")
        ax.legend(loc="upper right", fontsize=7)
    else:
        ax.scatter(x_v, y_v, s=5, alpha=0.35)

    if len(x_v):
        lo, hi = min(x_v.min(), y_v.min()), max(x_v.max(), y_v.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
        r = np.corrcoef(np.log10(x_v), np.log10(y_v))[0, 1] if len(x_v) >= 2 else float("nan")
        ax.text(0.04, 0.96, f"rₚ = {r:.4f}\nn = {len(x_v):,}",
                transform=ax.transAxes, va="top", ha="left",fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls=":", alpha=0.4)

def _hbar(ax, samples, values, xlabel, title, color):
    order = np.argsort(values)[::-1]
    y_pos = np.arange(len(samples))
    bars  = ax.barh(y_pos, np.array(values)[order], color=color, alpha=0.82)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(np.array(samples)[order], fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="x", ls=":", alpha=0.4)
    for bar, v in zip(bars, np.array(values)[order]):
        ax.text(bar.get_width() + 0.001 * max(abs(ax.get_xlim()[1]), 1e-9),
                bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}" if isinstance(v, float) else f"{v:,}",
                va="center", fontsize=6)

# ---------------------------------------------------------------------------
# Per-sample stats
# ---------------------------------------------------------------------------
def sample_stats(split, unified, matched, name, qval=0.01, 
                 only_split_count=0, only_unified_count=0) -> dict:
    tgt_s = split[~split["IsDecoy"]]
    tgt_u = unified[~unified["IsDecoy"]]

    n_split_q = int((tgt_s["QValue"] <= qval).sum())
    n_uni_q   = int((tgt_u["QValue"] <= qval).sum())

    med_se_s = float(tgt_s["SpecEValue"].median()) if len(tgt_s) else np.nan
    med_se_u = float(tgt_u["SpecEValue"].median()) if len(tgt_u) else np.nan
    se_ratio = med_se_s / med_se_u if (med_se_u and med_se_u > 0) else np.nan

    frac_agree = float(matched["SamePeptide"].mean()) if len(matched) else np.nan

    pa_ratio = np.nan
    if has_masic(split) and has_masic(unified) and "PeakArea_split" in matched.columns:
        pa_s  = matched["PeakArea_split"].to_numpy()
        pa_u  = matched["PeakArea_uni"].to_numpy()
        valid = (pa_s > 0) & (pa_u > 0) & np.isfinite(pa_s) & np.isfinite(pa_u)
        if valid.sum() > 0:
            pa_ratio = float(np.median(pa_s[valid] / pa_u[valid]))

    return {
        "sample": name,
        "n_split_psms": int(len(tgt_s)),
        "n_unified_psms": int(len(tgt_u)),
        f"n_split_q{qval}": n_split_q,
        f"n_uni_q{qval}": n_uni_q,
        "n_matched": int(len(matched)),
        "only_split_scans": only_split_count,
        "only_unified_scans": only_unified_count,
        "frac_same_peptide": frac_agree,
        "specevalue_median_ratio": se_ratio,
        "peakarea_median_ratio": pa_ratio,
        "split_is_rescored": bool(split["IsRescored"].any()),
        "unified_is_rescored": bool(unified["IsRescored"].any()),
    }
# ---------------------------------------------------------------------------
# Identification figure  (3×2)
# ---------------------------------------------------------------------------
def plot_ident(all_split, all_unified, all_matched, per_sample,
               out_path, project_name, qval):

    samples = [s["sample"] for s in per_sample]
    n = len(samples)
    tgt_split   = all_split[~all_split["IsDecoy"]]
    tgt_unified = all_unified[~all_unified["IsDecoy"]]

    same  = all_matched["SamePeptide"].to_numpy()
    n_same = int(same.sum())
    n_diff = int((~same).sum())

    only_sp = sum(s.get("only_split_scans", 0) for s in per_sample)
    only_uni = sum(s.get("only_unified_scans", 0) for s in per_sample)

    overall = float(same.mean()) if len(same) else float("nan")

    fig, axes = plt.subplots(3, 2, figsize=(15, 18))

    _kde_overlay(
        axes[0, 0],
        tgt_split["SpecEValue"].to_numpy(),
        tgt_unified["SpecEValue"].to_numpy(),
        xlabel="SpecEValue",
        title=f"SpecEValue distribution ",
        log=True,
    )

    sv_split = tgt_split["SpecEValue"].dropna().to_numpy()
    sv_uni = tgt_unified["SpecEValue"].dropna().to_numpy()

    #KS Statistic
    if len(sv_split) > 0 and len(sv_uni) > 0:
        stat,pval = ks_2samp(sv_split, sv_uni)
        axes[0,0].text(0.98, 0.04, f"KS D={stat:.3f}",
                transform=axes[0,0].transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    ev_split = tgt_split["EValue"].dropna()
    ev_uni = tgt_unified["EValue"].dropna()
    if len(ev_split) > 0 or len(ev_uni) > 0:
        _kde_overlay(
            axes[0, 1],
            tgt_split["EValue"].to_numpy(),
            tgt_unified["EValue"].to_numpy(),
            xlabel="EValue",
            title=f"EValue distribution ",
            log=True,
        )
    else:
        axes[0, 1].text(0.5, 0.5, "EValue not available", ha="center", va="center",
                       transform=axes[0, 1].transAxes, fontsize=10, color="grey")
        axes[0, 1].axis("off")

    ev_split = tgt_split["EValue"].dropna().to_numpy()
    ev_uni = tgt_unified["EValue"].dropna().to_numpy()
    #KS Statistic
    if len(ev_split) > 0 and len(ev_uni) > 0:
        stat,pval = ks_2samp(ev_split, ev_uni)
        axes[0,1].text(0.98, 0.04, f"KS D={stat:.3f}",
                transform=axes[0,1].transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    _ecdf_overlay(
        axes[1, 0],
        tgt_split["QValue"].to_numpy(),
        tgt_unified["QValue"].to_numpy(),
        xlabel="QValue",
        title=f"QValue ECDF",
    )

    _ecdf_overlay(
        axes[1, 1],
        tgt_split["PepQValue"].to_numpy(),
        tgt_unified["PepQValue"].to_numpy(),
        xlabel="PepQValue",
        title=f"PepQValue ECDF",
    )

    ax = axes[2, 0]
    cats = ["Same\npeptide", "Diff\npeptide", "binned\nonly", "Unified\nonly"]
    vals = [n_same, n_diff, only_sp, only_uni]
    colors = ["#00b909", "#dba5a5", "#6ca7ec", "#883eb6"]
    bars = ax.bar(cats, vals, color=colors, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("# scans / PSMs")
    ax.set_title(f"Peptide assignment agreement ({overall:.2%})")
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    _hbar(
        axes[2, 1], samples,
        [s["frac_same_peptide"] for s in per_sample],
        xlabel="Fraction same peptide",
        title="Per-sample peptide-assignment agreement",
        color="#009988",
    )

    # Panel labels a)–f)
    for ax, label in zip(axes.flat, ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']):
        ax.text(-0.12, 1.02, label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='bottom', ha='right', clip_on=False)

    fig.tight_layout(rect=[0, 0, 1, 0.993])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return overall

# ---------------------------------------------------------------------------
# Quantitation figure  (2×2)
# ---------------------------------------------------------------------------
def plot_quant(all_split, all_unified, all_matched, per_sample,
               out_path, project_name):

    samples = [s["sample"] for s in per_sample]
    tgt_split = all_split[~all_split["IsDecoy"]]
    tgt_unified = all_unified[~all_unified["IsDecoy"]]
    masic_ok = has_masic(all_split) and has_masic(all_unified)
    same = all_matched["SamePeptide"].to_numpy()

    print(f" Quant: has_masic(split)={has_masic(all_split)}, "
          f"has_masic(unified)={has_masic(all_unified)}")

    def _no_data(ax, msg="MASIC columns not present"):
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="grey")
        ax.axis("off")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    if masic_ok and "PeakArea_split" in all_matched.columns:
        _scatter_log(
            axes[0, 0],
            all_matched["PeakArea_split"].to_numpy(),
            all_matched["PeakArea_uni"].to_numpy(),
            xlabel="PeakArea (Bins/Split DB)",
            ylabel="PeakArea (unified DB)",
            same_mask=same,
        )
    else:
        _no_data(axes[0, 0])

    if masic_ok:
        _kde_overlay(
            axes[0, 1],
            tgt_split["PeakArea"].to_numpy(),
            tgt_unified["PeakArea"].to_numpy(),
            xlabel="PeakArea",
            title="PeakArea distribution",
            log=True,
        )
    else:
        _no_data(axes[0, 1])

    pa_ratios = [s["peakarea_median_ratio"] for s in per_sample]
    if masic_ok and any(r is not None and np.isfinite(r) for r in pa_ratios):
        clean = [r if (r is not None and np.isfinite(r)) else 1.0 for r in pa_ratios]
        _hbar(axes[1, 0], samples, clean,
              xlabel="Median PeakArea ratio (bins / unified)",
              title="Per-sample PeakArea ratio)",
              color="#ee7733")
    else:
        _no_data(axes[1, 0])

    if masic_ok:
        _kde_overlay(
            axes[1, 1],
            tgt_split["PeakSignalToNoiseRatio"].to_numpy(),
            tgt_unified["PeakSignalToNoiseRatio"].to_numpy(),
            xlabel="PeakSignalToNoiseRatio",
            title="Peak S/N distribution",
            log=True,
        )
    else:
        _no_data(axes[1, 1])

    for ax, label in zip(axes.flat, ['a)', 'b)', 'c)', 'd)']):
        ax.text(-0.12, 1.02, label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='bottom', ha='right', clip_on=False)

    fig.suptitle(
        f"{project_name + ': ' if project_name else ''}"
        f"Quantitation validation  |  MASIC: {'present' if masic_ok else 'absent'}",
        fontsize=10, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.993])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Rescoring comparison figure — 4-way comparison
# ---------------------------------------------------------------------------
def plot_rescoring_comparison(split_orig, split_rescored, 
                               uni_orig, uni_rescored,
                               per_sample, out_path, project_name, qval):
    """
    2×2 figure comparing original vs rescored for each database strategy.

    EXPECTS dataframes processed by load_file() where:
      - All QValues are in column "QValue" (xpec_q was renamed for rescored files)
      - Decoy flag is in column "IsDecoy" (camelCase)
      - Sample name is in column "_sample"
    """
    samples = [s["sample"] for s in per_sample]
    n = len(samples)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # [0,0] PSM counts: original vs rescored for split DB
    ax = axes[0, 0]
    x = np.arange(n)
    width = 0.35

    split_orig_counts = []
    split_resc_counts = []
    for s in per_sample:
        name = s["sample"]
        so = split_orig[split_orig["_sample"] == name]
        sr = split_rescored[split_rescored["_sample"] == name]
        # Use normalized "QValue" column (xpec_q was renamed by load_file)
        split_orig_counts.append(int((so[~so["IsDecoy"]]["QValue"] <= qval).sum()) if len(so) else 0)
        split_resc_counts.append(int((sr[~sr["IsDecoy"]]["QValue"] <= qval).sum()) if len(sr) else 0)

    ax.bar(x - width/2, split_orig_counts, width, label="Original", color="#33bbee", alpha=0.82)
    ax.bar(x + width/2, split_resc_counts, width, label="Rescored", color="#ee7733", alpha=0.82)
    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(f"PSMs at FDR ≤ {qval}")
    ax.set_title("Split DB: Original vs Rescored")
    ax.legend()
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    # [0,1] PSM counts: original vs rescored for unified DB
    ax = axes[0, 1]
    uni_orig_counts = []
    uni_resc_counts = []
    for s in per_sample:
        name = s["sample"]
        uo = uni_orig[uni_orig["_sample"] == name]
        ur = uni_rescored[uni_rescored["_sample"] == name]
        uni_orig_counts.append(int((uo[~uo["IsDecoy"]]["QValue"] <= qval).sum()) if len(uo) else 0)
        uni_resc_counts.append(int((ur[~ur["IsDecoy"]]["QValue"] <= qval).sum()) if len(ur) else 0)

    ax.bar(x - width/2, uni_orig_counts, width, label="Original", color="#b063f3", alpha=0.82)
    ax.bar(x + width/2, uni_resc_counts, width, label="Rescored", color="#ee7733", alpha=0.82)
    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(f"PSMs at FDR ≤ {qval}")
    ax.set_title("Unified DB: Original vs Rescored")
    ax.legend()
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    # [1,0] QValue ECDF: split original vs rescored
    ax = axes[1, 0]
    tgt_so = split_orig[~split_orig["IsDecoy"]]
    tgt_sr = split_rescored[~split_rescored["IsDecoy"]]

    for arr, label, color in [
        (tgt_so["QValue"].to_numpy(), "Original", "#33bbee"),
        (tgt_sr["QValue"].to_numpy(), "Rescored", "#ee7733"),
    ]:
        v = np.sort(arr[np.isfinite(arr)])
        if len(v) == 0:
            continue
        ecdf = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, ecdf, lw=1.25, label=f"{label} (n={len(v):,})", color=color)
        v_so = tgt_so["QValue"].to_numpy()
        v_so = v_so[np.isfinite(v_so)]
        v_sr = tgt_sr["QValue"].to_numpy()
        v_sr = v_sr[np.isfinite(v_sr)]
        if len(v_so) > 0 and len(v_sr) > 0:
            stat,pval = ks_2samp(v_so, v_sr)
            ax.text(0.98, 0.04, f"KS D={stat:.3f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    ax.set_xlabel("QValue")
    ax.set_xlabel("QValue")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Split DB QValue: Original vs Rescored")
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.4)

    # [1,1] QValue ECDF: unified original vs rescored
    
    ax = axes[1, 1]
    tgt_uo = uni_orig[~uni_orig["IsDecoy"]]
    tgt_ur = uni_rescored[~uni_rescored["IsDecoy"]]

    for arr, label, color in [
        (tgt_uo["QValue"].to_numpy(), "Original", "#b063f3"),
        (tgt_ur["QValue"].to_numpy(), "Rescored", "#ee7733"),
    ]:
        v = np.sort(arr[np.isfinite(arr)])
        if len(v) == 0:
            continue
        ecdf = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, ecdf, lw=1.8, label=f"{label} (n={len(v):,})", color=color)

        v_uo = tgt_uo["QValue"].to_numpy()
        v_uo = v_uo[np.isfinite(v_uo)]
        v_ur = tgt_ur["QValue"].to_numpy()
        v_ur = v_ur[np.isfinite(v_ur)]
        if len(v_uo) > 0 and len(v_ur) > 0:
            stat,pval = ks_2samp(v_uo, v_ur)
            ax.text(0.98, 0.04, f"KS D={stat:.3f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
    ax.set_xlabel("QValue")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Unified DB QValue: Original vs Rescored")
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.4)

    for ax, label in zip(axes.flat, ['a)', 'b)', 'c)', 'd)']):
            ax.text(-0.12, 1.02, label, transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='bottom', ha='right', clip_on=False)

    fig.suptitle(
        f"{project_name + ': ' if project_name else ''}",
        fontsize=10, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.993])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path

# ---------------------------------------------------------------------------
# Sample-pair discovery
# ---------------------------------------------------------------------------
class SamplePair(NamedTuple):
    name: str
    split_path: Path
    unified_path: Path

class RescoredPair(NamedTuple):
    name: str
    split_orig: Path
    split_rescored: Path | None
    unified_orig: Path
    unified_rescored: Path | None

def _normalize_name(filename: str) -> str:
    """
    Extract sample base name, stripping extensions and all pipeline suffixes.
    """
    # Remove extensions
    for ext in [".txt", ".tsv", ".tsv.gz", ".txt.gz"]:
        if filename.lower().endswith(ext):
            filename = filename[:-len(ext)]
            break

    # Remove rescored suffixes
    filename = filename.replace("_rescored", "").replace("_rescore", "")

    for suffix in SEARCH_SUFFIXES:
        if filename.endswith(suffix):
            filename = filename[:-len(suffix)]
            break

    return filename

def _collect_files(directory: Path) -> dict[str, Path]:
    """Collect all data files in a directory, keyed by normalized name."""
    files = {}
    if not directory.exists():
        return files

    for pattern in ["*.txt", "*.tsv", "*.txt.gz", "*.tsv.gz"]:
        for f in sorted(directory.glob(pattern)):
            if f.name.startswith("."):
                continue
            norm = _normalize_name(f.name)
            # Prefer non-rescored if both exist (shouldn't happen with proper dirs)
            if norm not in files or "_rescored" not in f.name:
                files[norm] = f
    return files

def discover_pairs(split_dir: Path, unified_dir: Path) -> list[SamplePair]:
    """Discover original sample pairs."""
    split_files = _collect_files(split_dir)
    unified_files = _collect_files(unified_dir)

    common = sorted(split_files.keys() & unified_files.keys())
    only_split = sorted(split_files.keys() - unified_files.keys())
    only_uni = sorted(unified_files.keys() - split_files.keys())

    if only_split:
        print(f"  WARNING: {len(only_split)} split file(s) with no unified match: "
              f"{only_split[:3]}{'...' if len(only_split) > 3 else ''}")
    if only_uni:
        print(f"  WARNING: {len(only_uni)} unified file(s) with no split match: "
              f"{only_uni[:3]}{'...' if len(only_uni) > 3 else ''}")

    if not common:
        print("No matching file pairs found.")
        return []

    return [SamplePair(name, split_files[name], unified_files[name]) for name in common]

def discover_rescored_pairs(split_dir: Path, unified_dir: Path,
                            split_rescored_dir: Path | None,
                            unified_rescored_dir: Path | None,
                            base_pairs: list[SamplePair]) -> list[RescoredPair]:
    """Match original pairs with their rescored counterparts."""

    split_rescored_files = _collect_files(split_rescored_dir) if split_rescored_dir else {}
    unified_rescored_files = _collect_files(unified_rescored_dir) if unified_rescored_dir else {}

    pairs = []
    for bp in base_pairs:
        sr = split_rescored_files.get(bp.name)
        ur = unified_rescored_files.get(bp.name)
        pairs.append(RescoredPair(
            name=bp.name,
            split_orig=bp.split_path,
            split_rescored=sr,
            unified_orig=bp.unified_path,
            unified_rescored=ur,
        ))

    return pairs

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run(pairs, out_stem: Path, project_name: str, qval: float):
    all_split_f, all_uni_f, all_matched_f = [], [], []
    stats_list = []

    print(f"Loading {len(pairs)} sample pairs...")
    for i, pair in enumerate(pairs, 1):
        print(f"  [{i}/{len(pairs)}] {pair.name}")
        try:
            split   = filter_fdr(load_file(pair.split_path, is_rescored=False), qval)
            unified = filter_fdr(load_file(pair.unified_path, is_rescored=False), qval)
            split["_sample"]   = pair.name
            unified["_sample"] = pair.name
            matched = match_on_scan(split, unified)
            matched["_sample"] = pair.name

            only_sp, only_uni = count_exclusive_scans(split, unified)

        except Exception as e:
            print(f"    ERROR: {e} — skipping")
            continue

        stats_list.append(sample_stats(split, unified, matched, pair.name, qval,
                                      only_split_count=only_sp, 
                                      only_unified_count=only_uni))
        all_split_f.append(split)
        all_uni_f.append(unified)
        all_matched_f.append(matched)

    if not all_split_f:
        raise RuntimeError("No samples loaded successfully.")

    all_split = pd.concat(all_split_f, ignore_index=True)
    all_unified = pd.concat(all_uni_f, ignore_index=True)
    all_matched = pd.concat(all_matched_f, ignore_index=True)

    ident_path = out_stem.with_name(out_stem.stem + "_ident.png")
    quant_path = out_stem.with_name(out_stem.stem + "_quant.png")

    print("Plotting identification figure...")
    overall = plot_ident(all_split, all_unified, all_matched,
                         stats_list, ident_path, project_name, qval)

    print("Plotting quantitation figure...")
    plot_quant(all_split, all_unified, all_matched,
               stats_list, quant_path, project_name)

    summary = {
        "n_samples": len(stats_list),
        "total_split_psms": int(sum(s["n_split_psms"]   for s in stats_list)),
        "total_unified_psms": int(sum(s["n_unified_psms"] for s in stats_list)),
        "total_matched_scans": int(len(all_matched)),
        "overall_frac_same_peptide": overall,
        "per_sample": stats_list,
    }
    return summary, ident_path, quant_path, all_split, all_unified

def run_with_rescoring(rescored_pairs, out_stem: Path, project_name: str, qval: float):
    """Run full analysis including rescored comparison."""

    base_pairs = [SamplePair(p.name, p.split_orig, p.unified_orig) for p in rescored_pairs]
    summary, ident_path, quant_path, all_split_orig, all_uni_orig = \
        run(base_pairs, out_stem, project_name, qval)

    all_split_rescored_f = []
    all_uni_rescored_f = []
    rescored_stats = []

    has_any_rescored = False

    for pair in rescored_pairs:
        print(f"  Loading rescored data for {pair.name}...")

        split_rescored = None
        unified_rescored = None

        if pair.split_rescored and pair.split_rescored.exists():
            try:
                split_rescored = filter_fdr(load_file(pair.split_rescored, is_rescored=True), qval)
                split_rescored["_sample"] = pair.name
                all_split_rescored_f.append(split_rescored)
                has_any_rescored = True
                print(f"  Split rescored: {pair.split_rescored.name}")
            except Exception as e:
                print(f" Could not load split rescored: {e}")
        else:
            print(f" No split rescored file found")

        if pair.unified_rescored and pair.unified_rescored.exists():
            try:
                unified_rescored = filter_fdr(load_file(pair.unified_rescored, is_rescored=True), qval)
                unified_rescored["_sample"] = pair.name
                all_uni_rescored_f.append(unified_rescored)
                has_any_rescored = True
                print(f"    ✓ Unified rescored: {pair.unified_rescored.name}")
            except Exception as e:
                print(f"    ✗ Could not load unified rescored: {e}")
        else:
            print(f"    – No unified rescored file found")

        if split_rescored is not None and unified_rescored is not None:
            matched = match_on_scan(split_rescored, unified_rescored)
            matched["_sample"] = pair.name
            only_sp, only_uni = count_exclusive_scans(split_rescored, unified_rescored)
            rescored_stats.append(sample_stats(
                split_rescored, unified_rescored, matched, pair.name, qval,
                only_split_count=only_sp, only_unified_count=only_uni
            ))

    rescoring_path = None
    if has_any_rescored and all_split_rescored_f and all_uni_rescored_f:
        all_split_rescored = pd.concat(all_split_rescored_f, ignore_index=True)
        all_uni_rescored = pd.concat(all_uni_rescored_f, ignore_index=True)

        rescoring_path = out_stem.with_name(out_stem.stem + "_rescoring.png")
        print("Plotting rescoring comparison figure...")
        plot_rescoring_comparison(
            all_split_orig, all_split_rescored,
            all_uni_orig, all_uni_rescored,
            summary["per_sample"], rescoring_path, project_name, qval
        )

        if len(rescored_stats) > 0:
            all_matched_rescored_f = []
            for pair in rescored_pairs:
                if pair.split_rescored and pair.unified_rescored:
                    try:
                        sr = filter_fdr(load_file(pair.split_rescored, is_rescored=True), qval)
                        ur = filter_fdr(load_file(pair.unified_rescored, is_rescored=True), qval)
                        sr["_sample"] = pair.name
                        ur["_sample"] = pair.name
                        mr = match_on_scan(sr, ur)
                        mr["_sample"] = pair.name
                        all_matched_rescored_f.append(mr)
                    except:
                        pass

            if all_matched_rescored_f:
                all_matched_rescored = pd.concat(all_matched_rescored_f, ignore_index=True)
                rescored_ident_path = out_stem.with_name(out_stem.stem + "_ident_rescored.png")
                print("Plotting rescored identification figure...")
                plot_ident(all_split_rescored, all_uni_rescored, all_matched_rescored,
                          rescored_stats, rescored_ident_path, project_name + " (Rescored)", qval)
    else:
        print("No rescored data loaded successfully — skipping rescoring plots")

    return summary, ident_path, quant_path, rescoring_path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Compare split vs unified proteomic database search strategies. "
                    "Optionally evaluate rescoring impact with separate rescored directories."
    )
    p.add_argument("--split_dir",   required=True, type=Path,
                   help="Directory of original bin/split-DB result files")
    p.add_argument("--unified_dir", required=True, type=Path,
                   help="Directory of original unified-DB result files")
    p.add_argument("--split_rescored_dir", type=Path, default=None,
                   help="Directory of rescored split-DB files (uses xpec_q as QValue)")
    p.add_argument("--unified_rescored_dir", type=Path, default=None,
                   help="Directory of rescored unified-DB files (uses xpec_q as QValue)")
    p.add_argument("--out", required=True, type=Path,
                   help="Output stem — produces <stem>_ident.png, <stem>_quant.png, "
                        "<stem>_rescoring.png, <stem>.stats.tsv")
    p.add_argument("--project", default="", help="Project name for figure titles")
    p.add_argument("--fdr",  type=float, default=0.10,
                   help="FDR threshold — only PSMs with QValue <= this are compared (default 0.10)")
    args = p.parse_args()

    base_pairs = discover_pairs(args.split_dir, args.unified_dir)
    if not base_pairs:
        print("No matching file pairs found in original directories.")
        sys.exit(1)

    print(f"Found {len(base_pairs)} sample pairs in original directories.")

    if args.split_rescored_dir or args.unified_rescored_dir:
        rescored_pairs = discover_rescored_pairs(
            args.split_dir, args.unified_dir,
            args.split_rescored_dir, args.unified_rescored_dir,
            base_pairs
        )

        n_split_r = sum(1 for p in rescored_pairs if p.split_rescored)
        n_uni_r = sum(1 for p in rescored_pairs if p.unified_rescored)
        print(f"Rescored files matched: {n_split_r} split, {n_uni_r} unified")

        summary, ident_path, quant_path, rescoring_path = \
            run_with_rescoring(rescored_pairs, args.out, args.project, args.fdr)
    else:
        summary, ident_path, quant_path, _, _ = \
            run(base_pairs, args.out, args.project, args.fdr)
        rescoring_path = None

    print(f"\nProject summary:")
    print(f"  Samples: {summary['n_samples']}")
    print(f"  Total split PSMs: {summary['total_split_psms']:,}")
    print(f"  Total unified PSMs: {summary['total_unified_psms']:,}")
    print(f"  Total matched scans: {summary['total_matched_scans']:,}")
    print(f"  Overall peptide agreement: {summary['overall_frac_same_peptide']:.2%}")
    print(f"\n  Identification : {ident_path}")
    print(f"  Quantitation : {quant_path}")
    if rescoring_path:
        print(f"  Rescoring Impact : {rescoring_path}")

    stats_path = args.out.with_suffix(".stats.tsv")
    pd.DataFrame(summary["per_sample"]).to_csv(stats_path, sep="\t", index=False)
    print(f"  Per-sample stats: {stats_path}")

if __name__ == "__main__":
    main()