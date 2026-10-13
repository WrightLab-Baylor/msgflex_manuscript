#!/usr/bin/env python3
"""
benchmark_psm_gains.py
"""
import argparse
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

# Config
LABELS = {
    "msgf":  "MS-GF+",
    "perc":  "MS-GF+ + Percolator",
    "ms2r":  "MS-GF+ + MS2Rescore",
    "sparx": "MSGFLEX-SPARX",
    "msb":   "MS-GF+ + MSBooster",
    "xpec":  "SPARX + XPECTRA",
}
COLORS = {
    "msgf": "#888780", "perc": "#1f77b4", "ms2r": "#9467bd",
    "sparx": "#8c564b", "msb": "#17becf", "xpec": "#d62728",
}
TOOL_ORDER = ["msgf", "sparx", "perc", "ms2r", "msb", "xpec"]

# q-value columns each tool is counted on — PSM-level and peptide-level
PSM_Q = {"msgf": "QValue", "perc": "q-value", "ms2r": "qvalue",
         "sparx": "QValue", "msb": "q-value", "xpec": "xpec_q"}
PEP_Q = {"msgf": "PepQValue", "perc": "pep_qvalue", "ms2r": "pep_qvalue",
         "sparx": "PepQValue", "msb": "pep_qvalue", "xpec": "xpec_pepq"}

UNIMOD_MASS = {
    "Oxidation": "+15.9949",
    "Carbamidomethyl": "+57.0215",
    "Phospho": "+79.9663",
    "Deamidated": "+0.9840",
    "Acetyl": "+42.0106",
    "Methylation": "+14.0157",
    "Dimethyl": "+28.0313",
    "Trimethyl": "+42.0470",
    "GlyGly": "+114.0429",
    "Propionamide": "+71.0371",
    "Carbamyl": "+43.0058",
    "Ammonia-loss": "-17.0265",
    "Water-loss": "-18.0106",
    "Pyro-glu":  "-17.0265",
    "Pyro_glu": "-17.0265",
    "Gln->pyro-Glu": "-17.0265",
    "Glu->pyro-Glu": "-18.0106",
    "Dioxidation": "+31.9898",
    "Cys->Trp": "+129.0579",
    "Met->Hse": "-29.9925",
    "TMT6plex": "+229.1629",
    "TMT10plex": "+229.1629",
    "TMT2plex": "+229.1629",
    "TMTpro":"+304.2071",
    "TMTpro16plex": "+304.2071",
    "iTRAQ4plex": "+144.1021",
    "iTRAQ8plex": "+304.2054",
    "TMT6plexNterm": "+229.1629",
    "TMT6plexK": "+229.1629",
    "TMT10plexNterm": "+229.1629",
    "TMT10plexK": "+229.1629",
    "TMTproNterm": "+304.2071",
    "TMTproK": "+304.2071",
}

# Small helpers
def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    try:
        df = pd.read_csv(path, sep=sep, engine='c', on_bad_lines='skip')
    except Exception:
        df = pd.read_csv(path, sep=sep, engine='python', on_bad_lines='skip')
    df.columns = df.columns.str.strip()
    df = df.loc[:, df.columns != ""]
    return df

def get_col(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Cannot find {label}. Tried {candidates}. "
                   f"Present: {list(df.columns)[:30]}")

def first_token(series):
    """
    Vectorized first accession extraction.
    Handles ; , and whitespace delimiters, plus bracket/quote wrappers.
    """
    s = series.astype(str).str.strip().replace("", np.nan)
    s = s.str.strip("[]")
    s = s.str.split(";").str[0]
    s = s.str.split(",").str[0]
    s = s.str.split().str[0]
    s = s.str.strip(";'\"")
    return s.fillna("")

def clean_peptide_series(series):
    """
    Vectorized peptide cleaning (avoids millions of slow Python regex calls).
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(r'/\d+$', '', regex=True)
    s = s.str.lstrip('_')
    mask_front = (s.str.len() > 2) & (s.str[1] == '.')
    s = s.where(~mask_front, s.str[2:])
    mask_back = (s.str.len() > 2) & (s.str[-2] == '.')
    s = s.where(~mask_back, s.str[:-2])
    s = s.str.replace(r'\[UNIMOD:[^\]]+\]', '', regex=True)
    s = s.str.replace(r'\[MOD:[^\]]+\]', '', regex=True)
    s = s.str.replace(r'\[[+\-]?\d+\.?\d*\]', '', regex=True)
    s = s.str.replace(r'[+\-]\d+\.?\d*', '', regex=True)
    s = s.str.replace(r'\([^)]+\)', '', regex=True)
    s = s.str.replace(r'[*#@^~$]', '', regex=True)
    return s.str.strip('-_.:| ')

def extract_scan_nums(series):
    """
    Fast, vectorized scan number extraction tailored for common MS pipelines.
    """
    s = series.astype(str)
    m = s.str.extract(r'msms(\d+)', expand=False)
    m = m.fillna(s.str.extract(r'scan[=:\s](\d+)', flags=re.IGNORECASE, expand=False))
    m = m.fillna(s.str.extract(r'ScanNum[=:\s](\d+)', flags=re.IGNORECASE, expand=False))
    return pd.to_numeric(m, errors='coerce')

# Set-operation helpers (gain / loss / common vs baseline)
def accepted_scans(df, q_col, thr):
    if df.empty or q_col not in df.columns or "ScanNum" not in df.columns:
        return set()
    mask = (df[q_col] <= thr) & (df["is_decoy"] == 0)
    return set(df.loc[mask, "ScanNum"].dropna().astype(int)) - {-1}

def accepted_peptides(df, q_col, thr):
    if df.empty or q_col not in df.columns or "Peptide" not in df.columns:
        return set()
    mask = (df[q_col] <= thr) & (df["is_decoy"] == 0)
    return set(df.loc[mask, "Peptide"].dropna())

def accepted_proteins(df, q_col, thr):
    if df.empty or q_col not in df.columns or "protein" not in df.columns:
        return set()
    mask = (df[q_col] <= thr) & (df["is_decoy"] == 0)
    return set(df.loc[mask, "protein"].replace("", np.nan).dropna().unique())

def count_psms(df, q_col, thr):
    if df.empty or q_col not in df.columns:
        return 0
    return int(((df[q_col] <= thr) & (df["is_decoy"] == 0)).sum())

def count_peptides(df, q_col, thr):
    if df.empty or q_col not in df.columns or "Peptide" not in df.columns:
        return 0
    mask = (df[q_col] <= thr) & (df["is_decoy"] == 0)
    return df.loc[mask, "Peptide"].nunique()

def count_proteins(df, q_col, thr):
    if df.empty or q_col not in df.columns or "protein" not in df.columns:
        return 0
    mask = (df[q_col] <= thr) & (df["is_decoy"] == 0)
    return df.loc[mask, "protein"].replace("", np.nan).dropna().nunique()

# just for ms2rescore as it lacks pepqvalue
def compute_pep_qvalue(psm_df, score_col="score", peptide_col="Peptide"):
    scores = pd.to_numeric(psm_df[score_col], errors="coerce")
    valid_mask = scores.notna() & psm_df[peptide_col].notna()
    if not valid_mask.any():
        return pd.DataFrame(columns=[peptide_col, "is_decoy", "pep_qvalue"])
    df_valid = psm_df[valid_mask].copy()
    df_valid[score_col] = scores[valid_mask]
    best_idx = df_valid.groupby(peptide_col)[score_col].idxmax()
    pep_df = df_valid.loc[best_idx].reset_index(drop=True)
    srt_idx = pep_df[score_col].argsort()[::-1]
    scores_s = pep_df[score_col].values[srt_idx]
    targets_s = (pep_df["is_decoy"].values[srt_idx] == 0).astype(bool)
    cum_targets = targets_s.cumsum()
    cum_decoys = (~targets_s).cumsum()
    with np.errstate(divide="ignore", invalid="ignore"):
        fdr = np.where(cum_targets > 0, cum_decoys / cum_targets, 1.0)
    qvals = np.minimum.accumulate(fdr[::-1])[::-1]
    qvals_out = np.empty_like(qvals)
    qvals_out[srt_idx] = qvals
    pep_df["pep_qvalue"] = qvals_out
    return pep_df

def _protein_col(df):
    """
    XPECTRA/SPARX use 'parent_protein'; MS-GF+/Percolator/MS2Rescore use
    'protein'.
    """
    if "protein" in df.columns:
        return "protein"
    if "parent_protein" in df.columns:
        return "parent_protein"
    return None

def build_consensus_protein_map(samples, args, thresholds):
    dp = args.decoy_prefix
    chunks = []
    for i, sample in enumerate(samples, 1):
        if i % 20 == 0 or i == 1:
            print(f"  [MAP] scanning sample {i}/{len(samples)} ...")

        path = Path(args.msgf) / f"{sample}.tsv"
        if not path.exists():
            path = Path(args.msgf) / f"{sample}.txt"
        if not path.exists():
            continue

        df = read_table(path)
        pep_col  = get_col(df, ["Peptide", "peptide"], "peptide")
        prot_col = get_col(df, ["Protein", "protein"], "protein")

        peptides = clean_peptide_series(df[pep_col])
        prot_lists = df[prot_col].astype(str).str.split(";")

        rows = []
        for pep, prots in zip(peptides, prot_lists):
            for p in prots:
                p = p.strip()
                if p and not p.startswith(dp):
                    rows.append((pep, p))
        if rows:
            chunks.append(pd.DataFrame(rows, columns=["Peptide", "protein"]))

    if not chunks:
        print("  [WARN] No peptide-protein pairs found for consensus map.")
        return {}

    big = pd.concat(chunks, ignore_index=True)
    counts = big.groupby(["Peptide", "protein"]).size().reset_index(name="n")

    counts = counts.sort_values(["Peptide", "n", "protein"], ascending=[True, False, True])
    canonical = (counts.drop_duplicates("Peptide", keep="first")
                        .set_index("Peptide")["protein"].to_dict())

    print(f"  [MAP] Consensus map: {len(canonical):,} peptides -> canonical proteins")
    return canonical

def apply_protein_map(df, protein_map, tool_label=""):
    if df.empty:
        return df
    col = _protein_col(df)
    if col is None or "Peptide" not in df.columns:
        return df

    df = df.copy()
    mapped = df["Peptide"].map(protein_map)

    df[col] = mapped.where(mapped.notna(), df[col])
    return df

# Per-tool loaders - each returns PSM-level DataFrame:
def load_msgf(msgf_dir, sample, decoy_prefix):
    base = Path(msgf_dir)
    path = base / f"{sample}.tsv"
    if not path.exists():
        path = base / f"{sample}.txt"
    if not path.exists():
        print(f"  [WARN] missing {path.name}")
        return pd.DataFrame(), pd.DataFrame()
    df = read_table(path)
    scan_col = get_col(df, ["Scan", "ScanNum"], "scan")
    pep_col  = get_col(df, ["Peptide", "peptide"], "peptide")
    prot_col = get_col(df, ["Protein", "protein"], "protein")
    q_col = get_col(df, ["QValue", "q_value", "qvalue"], "QValue")
    pq_col = get_col(df, ["PepQValue", "pep_q_value", "pepqvalue"], "PepQValue")
    ev_col = next((c for c in ["MSGFDB_SpecEValue", "SpecEValue"] if c in df.columns), None)
    if ev_col:
        df["_rank"] = pd.to_numeric(df[ev_col], errors="coerce").fillna(1.0)
        df = df.sort_values("_rank", kind="stable").drop_duplicates(subset=[scan_col], keep="first")
    else:
        df = df.drop_duplicates(subset=[scan_col], keep="first")
    prot = first_token(df[prot_col])
    ev = pd.to_numeric(df[ev_col], errors="coerce").fillna(1.0) if ev_col else pd.Series(1.0, index=df.index)
    out = pd.DataFrame({
        "ScanNum": pd.to_numeric(df[scan_col], errors="coerce").fillna(-1).astype(int).values,
        "Peptide": clean_peptide_series(df[pep_col]).values,
        "protein": prot.values,
        "is_decoy":prot.str.startswith(decoy_prefix).astype(int).values,
        "QValue":pd.to_numeric(df[q_col],  errors="coerce").fillna(1.0).values,
        "PepQValue": pd.to_numeric(df[pq_col], errors="coerce").fillna(1.0).values,
        "score": (-np.log10(ev.clip(lower=1e-300))).values,
    })
    return out, out

def load_sparx(sparx_dir, sample, decoy_prefix):
    path = Path(sparx_dir) / f"{sample}_fht_PlusSICStats.txt"
    if not path.exists():
        print(f"  [WARN] missing {path.name}")
        return pd.DataFrame(), pd.DataFrame()
    df = read_table(path)
    scan_col = get_col(df, ["Scan", "ScanNum"], "scan")
    pep_col  = get_col(df, ["Peptide", "peptide"], "peptide")
    prot_col = get_col(df, ["Protein", "protein"], "protein")
    q_col = get_col(df, ["QValue", "q_value", "qvalue"], "QValue")
    pq_col = get_col(df, ["PepQValue", "pep_q_value", "pepqvalue"], "PepQValue")
    ev_col = next((c for c in ["MSGFDB_SpecEValue", "SpecEValue"] if c in df.columns), None)
    sort_col = ev_col if ev_col else q_col
    df["_rank"] = pd.to_numeric(df[sort_col], errors="coerce").fillna(1.0)
    df = df.sort_values("_rank", kind="stable").drop_duplicates(subset=[scan_col], keep="first")
    prot = first_token(df[prot_col].astype(str))
    ev = pd.to_numeric(df[ev_col], errors="coerce").fillna(1.0) if ev_col else pd.Series(1.0, index=df.index)
    out = pd.DataFrame({
        "ScanNum": pd.to_numeric(df[scan_col], errors="coerce").fillna(-1).astype(int).values,
        "Peptide": clean_peptide_series(df[pep_col]).values,
        "protein": prot.values,
        "is_decoy": prot.str.startswith(decoy_prefix).astype(int).values,
        "QValue": pd.to_numeric(df[q_col],  errors="coerce").fillna(1.0).values,
        "PepQValue": pd.to_numeric(df[pq_col], errors="coerce").fillna(1.0).values,
        "score": (-np.log10(ev.clip(lower=1e-300))).values,
    })
    return out, out

def load_xpectra(xpectra_dir, sample, decoy_prefix):
    path = Path(xpectra_dir) / f"{sample}_rescored.tsv"
    if not path.exists():
        print(f"  [WARN] missing {path.name}")
        return pd.DataFrame(), pd.DataFrame()
    df = read_table(path)
    pep_col  = get_col(df, ["Peptide", "peptide"], "peptide")
    prot_col = get_col(df, ["Protein", "protein"], "protein")
    q_col = get_col(df, ["xpec_q", "q_value", "QValue"], "xpec_q")
    pq_col = get_col(df, ["xpec_pepq", "pep_q_value", "PepQValue"], "xpec_pepq")
    sc_col = next((c for c in ["xpec_score", "score"] if c in df.columns), None)
    prot = first_token(df[prot_col])
    dec_col = next((c for c in ["is_decoy", "IsDecoy"] if c in df.columns), None)
    is_decoy = (pd.to_numeric(df[dec_col], errors="coerce").fillna(0).astype(int)
                if dec_col else prot.str.startswith(decoy_prefix).astype(int))
    score = (pd.to_numeric(df[sc_col], errors="coerce").fillna(0.0).values
             if sc_col else np.zeros(len(df)))
    if "ScanNum" in df.columns:
        scan_vals = pd.to_numeric(df["ScanNum"], errors="coerce")
    else:
        id_col = next((c for c in ["PSMId", "SpecId", "specId", "psmId", "spectrum_id"] if c in df.columns), None)
        scan_vals = extract_scan_nums(df[id_col]) if id_col is not None else pd.Series(dtype=float)
    out = pd.DataFrame({
        "ScanNum": scan_vals.fillna(-1).astype(int).values,
        "Peptide": clean_peptide_series(df[pep_col]).values,
        "protein": prot.values,
        "is_decoy": is_decoy.values,
        "xpec_q": pd.to_numeric(df[q_col],  errors="coerce").fillna(1.0).values,
        "xpec_pepq": pd.to_numeric(df[pq_col], errors="coerce").fillna(1.0).values,
        "score": score,
    })
    return out, out

def _read_split_pair(tgt_path, dec_path, *, pep_candidates, prot_candidates,
                     q_candidates, out_q_name, score_candidates=None):
    frames = []
    sc_cols_seen = {}
    for path, dflag in [(tgt_path, 0), (dec_path, 1)]:
        side = "target" if dflag == 0 else "decoy"
        if not path.exists():
            print(f"  [WARN] missing {side} file: {path.name}")
            continue
        try:
            d = read_table(path)
            pep_col  = get_col(d, pep_candidates, "peptide")
            prot_col = get_col(d, prot_candidates, "protein")
            q_col = get_col(d, q_candidates, "q-value")
            sc_col = next((c for c in (score_candidates or []) if c in d.columns), None)
            sc_cols_seen[dflag] = sc_col
            raw_prot = (d[prot_col]
                        .astype(str)
                        .str.strip("[]")
                        .str.replace(",", ";", regex=False))
            prot = first_token(raw_prot)
            if "ScanNum" in d.columns:
                scan_nums = pd.to_numeric(d["ScanNum"], errors="coerce").fillna(-1).astype(int).values
            else:
                id_col = next((c for c in ["PSMId", "SpecId", "specId", "psmId"] if c in d.columns), None)
                if id_col is not None:
                    scan_nums = extract_scan_nums(d[id_col]).fillna(-1).astype(int).values
                else:
                    scan_nums = np.full(len(d), -1, dtype=int)
            frames.append(pd.DataFrame({
                "ScanNum": scan_nums,
                "Peptide": clean_peptide_series(d[pep_col]).values,
                "protein": prot.values,
                "is_decoy": dflag,
                out_q_name: pd.to_numeric(d[q_col], errors="coerce").fillna(1.0).values,
                "score": (pd.to_numeric(d[sc_col], errors="coerce").fillna(0.0).values
                              if sc_col else np.zeros(len(d))),
            }))
        except Exception as e:
            print(f"  [WARN] failed to load {side} file {path.name}: {e}")
            continue
    if len(sc_cols_seen) == 2 and sc_cols_seen.get(0) != sc_cols_seen.get(1):
        print(f"  [WARN] {tgt_path.name}: target/decoy score column mismatch")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def load_percolator(perc_dir, sample, decoy_prefix):
    base = Path(perc_dir)
    psm = _read_split_pair(
        base / f"{sample}_psms.txt", base / f"{sample}_decoy_psms.txt",
        pep_candidates=["peptide", "Peptide"],
        prot_candidates=["proteinIds", "Proteins", "protein"],
        q_candidates=["q-value", "q_value", "qvalue", "QValue"],
        out_q_name="q-value", score_candidates=["score", "svm_score"])
    pep = _read_split_pair(
        base / f"{sample}_peptides.txt", base / f"{sample}_decoy_peptides.txt",
        pep_candidates=["peptide", "Peptide"],
        prot_candidates=["proteinIds", "Proteins", "protein"],
        q_candidates=["q-value", "q_value", "qvalue", "QValue"],
        out_q_name="pep_qvalue", score_candidates=["score", "svm_score"])
    return psm, pep

def load_ms2rescore(ms2r_dir, sample, decoy_prefix, fmt="auto"):
    base = Path(ms2r_dir)
    mokapot_psm_path = base / f"{sample}.ms2rescore.psms.tsv"
    perc_psm_path     = base / f"{sample}_psms.txt"
    perc_decoy_path   = base / f"{sample}_decoy_psms.txt"
    resolved_fmt = fmt
    if resolved_fmt == "auto":
        if mokapot_psm_path.exists():
            resolved_fmt = "mokapot"
        elif perc_psm_path.exists() or perc_decoy_path.exists():
            resolved_fmt = "percolator"
        else:
            print(f"  [WARN] no MS2Rescore output found for {sample}")
            return pd.DataFrame(), pd.DataFrame()
    if resolved_fmt == "percolator":
        return _load_ms2rescore_percolator(base, sample)
    return _load_ms2rescore_mokapot(base, sample, decoy_prefix, mokapot_psm_path)

def _load_ms2rescore_mokapot(base, sample, decoy_prefix, psm_path):
    if not psm_path.exists():
        print(f"  [WARN] missing {psm_path.name}")
        return pd.DataFrame(), pd.DataFrame()
    df = read_table(psm_path)
    if "rank" in df.columns:
        df = df[pd.to_numeric(df["rank"], errors="coerce") == 1]
    pep_col  = get_col(df, ["peptidoform", "peptide", "Peptide"], "peptidoform")
    prot_col = get_col(df, ["protein_list", "proteinIds", "protein"], "protein_list")
    q_col = get_col(df, ["qvalue", "q-value", "q_value"], "qvalue")
    sc_col = next((c for c in ["score", "mokapot score"] if c in df.columns), None)
    raw_prot = (df[prot_col]
                .astype(str)
                .str.strip("[]")
                .str.replace(",", ";", regex=False))
    prot = first_token(raw_prot)
    is_decoy = (pd.to_numeric(df["is_decoy"], errors="coerce").fillna(0).astype(int)
                if "is_decoy" in df.columns
                else prot.str.startswith(decoy_prefix).astype(int))
    score = (pd.to_numeric(df[sc_col], errors="coerce").fillna(0.0).values
             if sc_col else np.zeros(len(df)))
    if "ScanNum" in df.columns:
        ms2r_scan = pd.to_numeric(df["ScanNum"], errors="coerce")
    else:
        sid_col = next((c for c in ["spectrum_id", "spectrumID", "SpectrumID"] if c in df.columns), None)
        ms2r_scan = extract_scan_nums(df[sid_col]) if sid_col else pd.Series(dtype=float)
    psm = pd.DataFrame({
        "ScanNum": ms2r_scan,
        "Peptide": clean_peptide_series(df[pep_col]).values,
        "protein": prot.values,
        "is_decoy": is_decoy.values,
        "qvalue": pd.to_numeric(df[q_col], errors="coerce").fillna(1.0).values,
        "score": score,
    })
    pep = compute_pep_qvalue(psm, score_col="score", peptide_col="Peptide")
    return psm, pep

def _load_ms2rescore_percolator(base, sample):
    psm = _read_split_pair(
        base / f"{sample}_psms.txt", base / f"{sample}_decoy_psms.txt",
        pep_candidates=["peptide", "Peptide"],
        prot_candidates=["proteinIds", "Proteins", "protein"],
        q_candidates=["q-value", "q_value", "qvalue", "QValue"],
        out_q_name="qvalue", score_candidates=["score", "svm_score"])
    pep = _read_split_pair(
        base / f"{sample}_peptides.txt", base / f"{sample}_decoy_peptides.txt",
        pep_candidates=["peptide", "Peptide"],
        prot_candidates=["proteinIds", "Proteins", "protein"],
        q_candidates=["q-value", "q_value", "qvalue", "QValue"],
        out_q_name="pep_qvalue", score_candidates=["score", "svm_score"])
    return psm, pep

# Gain / loss computation  (set operations vs MS-GF+ baseline)
def compute_gain_loss(frames, thresholds):
    rows = []
    base_psm_df, base_pep_df = frames.get("msgf", (pd.DataFrame(), pd.DataFrame()))
    for thr in thresholds:
        base_scans = accepted_scans(base_psm_df, PSM_Q["msgf"], thr)
        base_peps = accepted_peptides(base_pep_df, PEP_Q["msgf"], thr)
        base_prots = accepted_proteins(base_psm_df, PSM_Q["msgf"], thr)
        for tk in TOOL_ORDER:
            if tk == "msgf":
                continue
            psm_df, pep_df = frames.get(tk, (pd.DataFrame(), pd.DataFrame()))
            tool_scans = accepted_scans(psm_df, PSM_Q[tk], thr)
            tool_peps = accepted_peptides(pep_df, PEP_Q[tk], thr)
            tool_prots = accepted_proteins(psm_df, PSM_Q[tk], thr)
            t_count = count_psms(psm_df, PSM_Q[tk], thr)
            b_count = count_psms(base_psm_df, PSM_Q["msgf"], thr)
            scan_ops_ok = (bool(tool_scans) and bool(base_scans) and
                           len(tool_scans) >= 0.5 * max(t_count, 1))
            if scan_ops_ok:
                gained_psms = len(tool_scans - base_scans)
                lost_psms = len(base_scans - tool_scans)
                common_psms = len(tool_scans & base_scans)
            else:
                gained_psms = max(0, t_count - b_count)
                lost_psms = max(0, b_count - t_count)
                common_psms = min(t_count, b_count)
            gained_peps = len(tool_peps - base_peps)
            lost_peps = len(base_peps - tool_peps)
            common_peps = len(tool_peps & base_peps)
            gained_prots = len(tool_prots - base_prots)
            lost_prots = len(base_prots - tool_prots)
            common_prots = len(tool_prots & base_prots)
            rows.append({
                "Tool": tk, "FDR": thr,
                "gained_psms": gained_psms,
                "lost_psms": lost_psms,
                "common_psms": common_psms,
                "gained_peptides": gained_peps,
                "lost_peptides": lost_peps,
                "common_peptides": common_peps,
                "gained_proteins": gained_prots,
                "lost_proteins": lost_prots,
                "common_proteins": common_prots,
            })
    return rows

# Plots
def _znorm(a):
    a = np.asarray(a, dtype=float)
    s = a.std()
    return (a - a.mean()) / (s if s > 0 else 1.0)

def _cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else float("nan")

def plot_score_dists(exp_psm, out_dir):
    try:
        from scipy.stats import gaussian_kde
        have_kde = True
    except Exception:
        have_kde = False
    keys = [k for k in TOOL_ORDER if k in exp_psm and not exp_psm[k].empty
            and "score" in exp_psm[k].columns]
    if not keys:
        return
    ncols, nrows = 3, (len(keys) + 2) // 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    # fig.suptitle("Score distributions (z-normalised) — target vs decoy",
    #              fontsize=12, fontweight="bold")
    zr = []
    for k in keys:
        v = pd.to_numeric(exp_psm[k]["score"], errors="coerce").dropna().values
        if len(v) >= 5:
            z = _znorm(v); zr += [np.percentile(z, 1), np.percentile(z, 99)]
    x_lo, x_hi = (max(min(zr), -5), min(max(zr), 5)) if zr else (-5, 5)
    for idx, k in enumerate(keys):
        ax = axes[idx // ncols][idx % ncols]
        df = exp_psm[k]
        raw = pd.to_numeric(df["score"], errors="coerce")
        mu, sd = raw.mean(), (raw.std() or 1.0)
        tgt = ((raw[df["is_decoy"] == 0] - mu) / sd).dropna().values
        dcy = ((raw[df["is_decoy"] == 1] - mu) / sd).dropna().values
        for vals, lbl, c, ls in [(tgt, f"Target (n={len(tgt):,})", COLORS.get(k, "#2ca02c"), "-"),
                                 (dcy, f"Decoy (n={len(dcy):,})", "#ff7f0e", "--")]:
            if len(vals) < 5:
                continue
            if have_kde:
                try:
                    kde = gaussian_kde(vals, bw_method=0.3)
                    xs = np.linspace(x_lo, x_hi, 400)
                    ax.plot(xs, kde(xs), color=c, ls=ls, lw=1.5, label=lbl)
                    ax.fill_between(xs, kde(xs), alpha=0.13, color=c)
                    continue
                except Exception:
                    pass
            ax.hist(vals, bins=50, range=(x_lo, x_hi), density=True,
                    color=c, alpha=0.35, label=lbl)
        d = _cohens_d(tgt, dcy)
        if not np.isnan(d):
            x_pos = 1.05 if d >= 0 else 0.03
            ha = "right" if d >= 0 else "left"
            ax.text(x_pos, 0.1, f"Cohen's d = {d:.2f}", transform=ax.transAxes,
                ha=ha, va="bottom", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.set_title(f"{LABELS[k]}",color=COLORS.get(k, "#333"),fontsize=8,fontweight="bold")
        ax.set_xlabel("z-score"); ax.set_ylabel("Density"); ax.set_xlim(x_lo, x_hi)
        ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    for idx in range(len(keys), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    for ax, label in zip(axes.flat, ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']):
            ax.text(-0.12, 1.00, label, transform=ax.transAxes,
                    fontsize=12, fontweight="bold", va='bottom', ha='right', clip_on=False)
    plt.tight_layout()
    p = os.path.join(out_dir, "score_distributions.png")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  [PLOT] {p}")

def plot_counts(summary, out_dir):
    if summary.empty:
        print("  [WARN] Summary is empty. Skipping identification counts plot.")
        return
    levels = ["PSMs", "Peptides", "Proteins"]
    fdrs   = sorted(summary["FDR"].unique())
    tools  = [t for t in TOOL_ORDER if t in summary["Tool"].values]
    w = 0.8 / len(tools)
    offs = np.linspace(-(len(tools) - 1) / 2 * w, (len(tools) - 1) / 2 * w, len(tools))
    x = np.arange(len(fdrs))
    fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 5))
    fig.suptitle("Identifications at FDR thresholds", fontsize=12, fontweight="bold")
    for ax, lvl in zip(axes, levels):
        for tk, off in zip(tools, offs):
            vals = [summary[(summary.Tool == tk) & (summary.Level == lvl) &
                            (summary.FDR == f)]["Count"].sum() for f in fdrs]
            bars = ax.bar(x + off, vals, w, label=LABELS[tk], color=COLORS[tk], alpha=0.85)
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2,
                        b.get_height() + ax.get_ylim()[1] * 0.005,
                        f"{int(b.get_height()):,}", ha="center", va="bottom", fontsize=6, rotation=90)
        ax.set_title(lvl)
        ax.set_xticks(x); ax.set_xticklabels([f"q≤{f:.0%}" for f in fdrs])
        ax.set_ylabel("Count")
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(tools),
               fontsize=7.5, frameon=True, bbox_to_anchor=(0.5, -0.04))

    for ax, label in zip(axes.flat, ['a)', 'b)', 'c)']):
                ax.text(-0.12, 1.00, label, transform=ax.transAxes,
                        fontsize=12, fontweight="bold", va='bottom', ha='right', clip_on=False)
    plt.tight_layout()
    for ax in axes:
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
    plt.subplots_adjust(bottom=0.15)
    p = os.path.join(out_dir, "identification_counts.png")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  [PLOT] {p}")

def plot_gain(summary, out_dir):
    if summary.empty:
        print("  [WARN] Summary is empty. Skipping gain plot.")
        return
    levels = ["PSMs", "Peptides", "Proteins"]
    fdrs   = sorted(summary["FDR"].unique())
    others = [t for t in TOOL_ORDER if t != "msgf" and t in summary["Tool"].values]
    if not others:
        return
    w = 0.8 / len(others)
    offs = np.linspace(-(len(others) - 1) / 2 * w, (len(others) - 1) / 2 * w, len(others))
    x = np.arange(len(fdrs))
    fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 5))
    fig.suptitle("% gain over MS-GF+ baseline", fontsize=13, fontweight="bold")
    for ax, lvl in zip(axes, levels):
        for tk, off in zip(others, offs):
            gains = []
            for f in fdrs:
                b = summary[(summary.Tool == "msgf") & (summary.Level == lvl) &
                            (summary.FDR == f)]["Count"].sum()
                c = summary[(summary.Tool == tk) & (summary.Level == lvl) &
                            (summary.FDR == f)]["Count"].sum()
                gains.append((c - b) / b * 100 if b > 0 else 0)
            bars = ax.bar(x + off, gains, w, label=LABELS[tk], color=COLORS[tk], alpha=0.85)
            for bb in bars:
                h = bb.get_height()
                if abs(h) > 0.3:
                    ax.text(bb.get_x() + bb.get_width() / 2, h, f"{h:+.1f}%",
                            ha="center", va="bottom" if h >= 0 else "top", fontsize=6)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(lvl)
        ax.set_xticks(x); ax.set_xticklabels([f"q≤{f:.0%}" for f in fdrs])
        ax.set_ylabel("% gain")
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(others),
               fontsize=7.5, frameon=True, bbox_to_anchor=(0.5, -0.04))

    for ax, label in zip(axes.flat, ['a)', 'b)', 'c)']):
                    ax.text(-0.12, 1.00, label, transform=ax.transAxes,
                            fontsize=12, fontweight="bold", va='bottom', ha='right', clip_on=False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    p = os.path.join(out_dir, "gain_over_baseline.png")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  [PLOT] {p}")

def plot_gain_loss_diverging(overlap_df, out_dir):
    if overlap_df.empty:
        print("  [WARN] No overlap data -- skipping gain/loss diverging plot.")
        return

    LOSS_CLR = "#E66101"
    COMMON_CLR = "#C4C4C4"
    GAIN_CLR = "#7244C0"
    BL_CLR = "#252525"
    HATCH_PEP = "/"
    HATCH_PROT = "\\"
    plt.rcParams['hatch.linewidth'] = 0.4

    fdrs    = sorted(overlap_df["FDR"].unique())
    tools   = [t for t in TOOL_ORDER if t != "msgf" and t in overlap_df["Tool"].values]
    n_tools = len(tools)
    if n_tools == 0:
        return

    bar_h  = 0.20
    gap = 0.05
    group_sep = 0.28
    stride = bar_h * 3 + gap * 2 + group_sep

    y_centres = np.arange(n_tools) * stride
    y_centres = y_centres[::-1]
    offset = bar_h + gap
    y_psm = y_centres + offset
    y_pep = y_centres
    y_prot = y_centres - offset

    n_panels = len(fdrs)
    fig_h    = max(5.5, n_tools * 1.5 + 2.5)
    fig, axes = plt.subplots(1, n_panels, figsize=(9.5 * n_panels, fig_h), squeeze=False)
    fig.suptitle("PSM, Peptide, and Protein Overlap vs MS-GF+ Baseline",
                 fontsize=12, fontweight="bold", y=1.02)

    TOOL_LABELS = {
        "perc": "MS-GF+\n+ Percolator",
        "ms2r": "MS-GF+\n+ MS2Rescore",
        "sparx":"MSGFLEX\n- SPARX",
        "msb": "MS-GF+\n+ MSBooster",
        "xpec": "SPARX\n+ XPECTRA",
    }

    for col, thr in enumerate(fdrs):
        ax  = axes[0][col]
        sub = overlap_df[overlap_df["FDR"] == thr]

        # x-range = max total bar across all three levels
        totals = []
        for tk in tools:
            row = sub[sub["Tool"] == tk]
            if row.empty:
                continue
            r = row.iloc[0]
            totals.extend([
                int(r.lost_psms) + int(r.common_psms) + int(r.gained_psms),
                int(r.lost_peptides) + int(r.common_peptides) + int(r.gained_peptides),
                int(r.lost_proteins) + int(r.common_proteins) + int(r.gained_proteins),
            ])
        if not totals:
            ax.set_visible(False)
            continue
        xmax = max(totals) * 1.15

        # MS-GF+ baselines per level (median across tools)
        bl_psm_vals, bl_pep_vals, bl_prot_vals = [], [], []
        for tk in tools:
            row = sub[sub["Tool"] == tk]
            if row.empty:
                continue
            r = row.iloc[0]
            bl_psm_vals.append(int(r.lost_psms) + int(r.common_psms))
            bl_pep_vals.append(int(r.lost_peptides) + int(r.common_peptides))
            bl_prot_vals.append(int(r.lost_proteins) + int(r.common_proteins))
        bl_psm = int(np.median(bl_psm_vals))  if bl_psm_vals  else 0
        bl_pep = int(np.median(bl_pep_vals))  if bl_pep_vals  else 0
        bl_prot = int(np.median(bl_prot_vals)) if bl_prot_vals else 0

        # Alternating background bands
        for i in range(n_tools):
            if i % 2 == 0:
                ax.axhspan(y_centres[i] - stride / 2 + gap / 2,
                           y_centres[i] + stride / 2 - gap / 2,
                           color="#FFFFFF", zorder=0)

        for i, tk in enumerate(tools):
            row = sub[sub["Tool"] == tk]
            if row.empty:
                continue
            r = row.iloc[0]

            # PSM row (solid)
            lp = int(r.lost_psms)
            cp = int(r.common_psms)
            gp = int(r.gained_psms)
            ax.barh(y_psm[i], lp, height=bar_h, left=0,      color=LOSS_CLR,   zorder=3)
            ax.barh(y_psm[i], cp, height=bar_h, left=lp,     color=COMMON_CLR, zorder=3)
            ax.barh(y_psm[i], gp, height=bar_h, left=lp+cp,  color=GAIN_CLR,   zorder=3)

            # Peptide row (hatched)
            lpep = int(r.lost_peptides)
            cpep = int(r.common_peptides)
            gpep = int(r.gained_peptides)
            ax.barh(y_pep[i], lpep, height=bar_h, left=0,           color=LOSS_CLR,
                    hatch=HATCH_PEP, linewidth=0.3, zorder=3)
            ax.barh(y_pep[i], cpep, height=bar_h, left=lpep,        color=COMMON_CLR,
                    hatch=HATCH_PEP, linewidth=0.3, zorder=3)
            ax.barh(y_pep[i], gpep, height=bar_h, left=lpep+cpep,   color=GAIN_CLR,
                    hatch=HATCH_PEP, linewidth=0.3, zorder=3)

            # Protein row (double-hatched)
            lprot = int(r.lost_proteins)
            cprot = int(r.common_proteins)
            gprot = int(r.gained_proteins)
            ax.barh(y_prot[i], lprot, height=bar_h, left=0,  color=LOSS_CLR,
                    hatch=HATCH_PROT, linewidth=0.3, zorder=3)
            ax.barh(y_prot[i], cprot, height=bar_h, left=lprot,         color=COMMON_CLR,
                    hatch=HATCH_PROT, linewidth=0.3, zorder=3)
            ax.barh(y_prot[i], gprot, height=bar_h, left=lprot+cprot, color=GAIN_CLR,
                    hatch=HATCH_PROT, linewidth=0.3, zorder=3)

            # Segment labels
            min_w = xmax * 0.032
            LOSS_TXT = "white"
            COMMON_TXT = "#222222"
            GAIN_TXT = "white"
            OUTSIDE_TXT = "#333333"

            def _seg(y_pos, val, x_left, in_col):
                if val <= 0:
                    return
                if val >= min_w:
                    ax.text(x_left + val / 2, y_pos, f"{val:,}",
                            ha="center", va="center", fontsize=6.2,
                            fontweight="bold", color=in_col, clip_on=True)
                else:
                    ax.text(x_left + val + xmax * 0.006, y_pos, f"{val:,}",
                            ha="left", va="center", fontsize=5.8,
                            fontweight="bold", color=OUTSIDE_TXT, clip_on=True)

            # PSM labels
            _seg(y_psm[i], lp, 0, LOSS_TXT)
            _seg(y_psm[i], cp, lp, COMMON_TXT)
            _seg(y_psm[i], gp, lp+cp, GAIN_TXT)
            # Peptide labels
            _seg(y_pep[i], lpep, 0, LOSS_TXT)
            _seg(y_pep[i], cpep, lpep, COMMON_TXT)
            _seg(y_pep[i], gpep, lpep+cpep, GAIN_TXT)
            # Protein labels
            _seg(y_prot[i], lprot, 0, LOSS_TXT)
            _seg(y_prot[i], cprot, lprot,      COMMON_TXT)
            _seg(y_prot[i], gprot, lprot+cprot, GAIN_TXT)

            # Row tags on right margin
            ax.text(1.005, y_psm[i], "PSM", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=6.0, color=GAIN_CLR, fontweight="bold")
            ax.text(1.005, y_pep[i],  "Pep",  transform=ax.get_yaxis_transform(),
                    va="center", fontsize=6.0, color="#777777", fontweight="bold")
            ax.text(1.005, y_prot[i], "Prot", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=6.0, color="#555555", fontweight="bold")

        # Baseline reference lines
        y_lo = y_centres[-1] - stride / 2
        y_hi = y_centres[0] + stride / 2

        def _draw_baseline(val, name, style):
            if val <= 0:
                return
            ax.plot([val, val], [y_lo, y_hi],
                    color=BL_CLR, linewidth=1.4, linestyle=style, zorder=5, alpha=0.8)
            y_ann = y_centres[-1] - stride * 0.62
            ax.text(val, y_ann, f"MS-GF+\n{name}",
                    ha="center", va="top", fontsize=6.0, color=BL_CLR,
                    fontstyle="italic", linespacing=1.3)

        _draw_baseline(bl_psm,  "baseline", "--")
        if abs(bl_pep - bl_psm) > xmax * 0.04:
            _draw_baseline(bl_pep,  "baseline", ":")
        if abs(bl_prot - bl_psm) > xmax * 0.04 and abs(bl_prot - bl_pep) > xmax * 0.04:
            _draw_baseline(bl_prot, "baseline", "-.")

        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.tick_params(axis="x", labelsize=8)
        ax.set_yticks(y_centres)
        ax.set_yticklabels([TOOL_LABELS.get(t, LABELS.get(t, t)) for t in tools],
                           fontsize=9, va="center")
        ax.tick_params(axis="y", length=0, pad=6)
        ax.set_title(f"FDR <= {thr:.0%}", fontsize=10, fontweight="bold", pad=10)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, linestyle=":", linewidth=0.5, color="#CCCCCC", zorder=1)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.7)
        ax.set_xlabel("Identification count", fontsize=8.5)

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor=LOSS_CLR,   label="Lost  (MS-GF+ only)"),
        mpatches.Patch(facecolor=COMMON_CLR, label="Common (both)"),
        mpatches.Patch(facecolor=GAIN_CLR,   label="Gained (tool only)"),
        mpatches.Patch(facecolor="#DDDDDD",  hatch=HATCH_PEP,
                       label="Peptide rows", edgecolor="#AAAAAA"),
        mpatches.Patch(facecolor="#DDDDDD",  hatch=HATCH_PROT,
                       label="Protein rows", edgecolor="#AAAAAA"),
    ]
    axes[0][-1].legend(handles=legend_patches, loc="lower right",
                       frameon=True, framealpha=0.93, edgecolor="#BBBBBB",
                       fontsize=8.0, ncol=1, handlelength=1.6)

    for ax, label in zip(axes.flat, ['a)', 'b)', 'c)']):
                    ax.text(-0.12, 1.00, label, transform=ax.transAxes,
                            fontsize=12, fontweight="bold", va='bottom', ha='right', clip_on=False)

    fig.tight_layout(rect=[0, 0.04, 1.0, 0.98])
    p = os.path.join(out_dir, "gain_loss_diverging.png")
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {p}")

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--project-path", default=None,
                    help="Project dir containing the standard subfolders: "
                         "results/ (msgf), percolator/, ms2rescore/, SICdir/ (sparx), "
                         "msbooster/, xpectra/rescored/. Individual flags override.")
    ap.add_argument("--msgf", default=None, help="MS-GF+ TSV file or directory")
    ap.add_argument("--perc-dir", default=None)
    ap.add_argument("--ms2r-dir",  default=None)
    ap.add_argument("--ms2r-format", choices=["auto", "mokapot", "percolator"], default="auto",
                    help="MS2Rescore output format. 'auto' (default) detects which files "
                         "are present; 'mokapot' expects {sample}.ms2rescore.psms.tsv; "
                         "'percolator' expects split {sample}_psms.txt / _decoy_psms.txt.")
    ap.add_argument("--sparx-dir", default=None)
    ap.add_argument("--msb-dir", default=None)
    ap.add_argument("--xpectra", default=None)
    ap.add_argument("--out", default="comparison_plots", help="Output directory for plots and summary CSV")
    ap.add_argument("--decoy-prefix", default="XXX_")
    ap.add_argument("--fdr-thresholds", nargs="+", type=float, default=[0.01, 0.02, 0.05])
    args = ap.parse_args()

    if args.project_path:
        root = Path(args.project_path)
        defaults = {
            "msgf": root / "results",
            "perc_dir": root / "percolator",
            "ms2r_dir": root / "ms2rescore",
            "sparx_dir": root / "SICdir",
            "msb_dir": root / "msbooster",
            "xpectra": root / "xpectra" / "rescored",
        }
        for attr, path in defaults.items():
            if getattr(args, attr) is None and path.exists():
                setattr(args, attr, str(path))

    if not args.msgf:
        ap.error("MS-GF+ path required: pass --msgf or -p/--project-path with a results/ subfolder")

    os.makedirs(args.out, exist_ok=True)
    thresholds = sorted(args.fdr_thresholds)
    dp = args.decoy_prefix

    print("[INFO] Discovering samples from MS-GF+ directory filenames...")
    samples = []
    if Path(args.msgf).is_dir():
        files = sorted(Path(args.msgf).glob("*.tsv")) + sorted(Path(args.msgf).glob("*.txt"))
        samples = sorted({f.stem for f in files})
    elif Path(args.msgf).is_file():
        samples = [Path(args.msgf).stem]

    print(f"[INFO] {len(samples)} sample(s) discovered.")
    print("[INFO] Building consensus peptide -> protein map...")
    protein_map = build_consensus_protein_map(samples, args, thresholds)

    # Per-sample loop
    per_sample_rows = []
    overlap_rows    = []
    seen_tool = {tk: False for tk in TOOL_ORDER}
    accum_psm = {tk: [] for tk in TOOL_ORDER}

    import time
    t_start = time.time()
    n_samples = len(samples)

    EMPTY = (pd.DataFrame(), pd.DataFrame())
    for i, sample in enumerate(samples, 1):
        elapsed = time.time() - t_start
        rate = elapsed / i
        eta = rate * (n_samples - i)
        print(f"[{i}/{n_samples}] {sample}  "
              f"(elapsed {elapsed/60:.1f} min, ETA {eta/60:.1f} min)", flush=True)

        frames = {
            "msgf": load_msgf(args.msgf, sample, dp) if args.msgf else EMPTY,
            "perc": load_percolator(args.perc_dir, sample, dp) if args.perc_dir else EMPTY,
            "ms2r": load_ms2rescore(args.ms2r_dir, sample, dp, fmt=args.ms2r_format) if args.ms2r_dir else EMPTY,
            "sparx": load_sparx(args.sparx_dir, sample, dp) if args.sparx_dir else EMPTY,
            "msb": load_percolator(args.msb_dir, sample, dp) if args.msb_dir else EMPTY,
            "xpec": load_xpectra(args.xpectra, sample, dp) if args.xpectra else EMPTY,
        }

        for tk in TOOL_ORDER:
            psm_df, pep_df = frames[tk]
            psm_df = apply_protein_map(psm_df, protein_map, tool_label=tk)
            pep_df = apply_protein_map(pep_df, protein_map, tool_label=tk)
            frames[tk] = (psm_df, pep_df)

            if not psm_df.empty or not pep_df.empty:
                seen_tool[tk] = True
            if not psm_df.empty and "score" in psm_df.columns:
                accum_psm[tk].append(
                    psm_df[["score", "is_decoy"]].astype({"score": "float32", "is_decoy": "int8"})
                )

            for thr in thresholds:
                n_psm = count_psms(psm_df, PSM_Q[tk], thr)
                n_pep = count_peptides(pep_df, PEP_Q[tk], thr)          
                n_prot = count_proteins(psm_df, PSM_Q[tk], thr)
                per_sample_rows.append({"Sample": sample, "Tool": tk, "FDR": thr,
                                        "PSMs": n_psm, "Peptides": n_pep, "Proteins": n_prot})

        for row in compute_gain_loss(frames, thresholds):
            row["Sample"] = sample
            overlap_rows.append(row)

    # Experiment-level = SUM of per-sample distinct counts
    print("\n[INFO] Experiment-level counts (sum of per-sample)")
    ps = pd.DataFrame(per_sample_rows)
    active = [tk for tk in TOOL_ORDER if seen_tool[tk]]

    summary_rows = []
    col_w = 22
    header = f"  {'Metric':<26}" + "".join(f"{LABELS[t]:>{col_w}}" for t in active)
    sep = "-" * len(header)

    for thr in thresholds:
        counts = {}
        for tk in active:
            sub = ps[(ps["Tool"] == tk) & (ps["FDR"] == thr)]
            counts[tk] = (int(sub["PSMs"].sum()),
                          int(sub["Peptides"].sum()),
                          int(sub["Proteins"].sum()))
        print("\n" + sep)
        print(f"  q <= {thr:.0%}")
        print(header)
        print(sep)
        for lvl, idx in [("PSMs", 0), ("Peptides", 1), ("Proteins", 2)]:
            row = "".join(f"{counts[t][idx]:>{col_w},}" for t in active)
            print(f"  {lvl:<26}{row}")
            for t in active:
                summary_rows.append({"FDR": thr, "Tool": t, "Level": lvl,
                                     "Count": counts[t][idx]})
        if "msgf" in counts:
            base = counts["msgf"]
            for lvl, idx in [("PSMs", 0), ("Peptides", 1), ("Proteins", 2)]:
                row = "".join(
                    (f"{'baseline':>{col_w}}" if t == "msgf"
                     else (f"{(counts[t][idx]-base[idx])/base[idx]*100:>+{col_w}.1f}"
                           if base[idx] > 0 else f"{'NA':>{col_w}}"))
                    for t in active)
                print(f"  {'  gain% ' + lvl:<26}{row}")
    print(sep)

    # Save
    summary = pd.DataFrame(summary_rows)
    sp = os.path.join(args.out, "comparison_summary.tsv")
    summary.to_csv(sp, sep="\t", index=False)
    print(f"\n  [OUT] {sp}")

    psp = os.path.join(args.out, "per_sample_summary.tsv")
    pd.DataFrame(per_sample_rows).to_csv(psp, sep="\t", index=False)
    print(f"  [OUT] {psp}")

    # Gain / loss experiment totals
    overlap_df = pd.DataFrame(overlap_rows)
    if not overlap_df.empty:
        overlap_exp = (overlap_df
                       .groupby(["Tool", "FDR"], as_index=False)
                       [["gained_psms", "lost_psms", "common_psms",
                         "gained_peptides", "lost_peptides", "common_peptides",
                         "gained_proteins", "lost_proteins", "common_proteins"]]
                       .sum())

        print("\n[INFO] Gain / Loss vs MS-GF+ baseline (experiment totals)")
        for thr in thresholds:
            sub = overlap_exp[overlap_exp["FDR"] == thr]
            cw  = 14
            hdr = (f"  {'Tool':<24}"
                   + f"{'Gained PSMs':>{cw}}{'Lost PSMs':>{cw}}{'Common PSMs':>{cw}}"
                   + f"{'Gained Peps':>{cw}}{'Lost Peps':>{cw}}{'Common Peps':>{cw}}"
                   + f"{'Gained Prots':>{cw}}{'Lost Prots':>{cw}}{'Common Prots':>{cw}}")
            print(f"\n  q <= {thr:.0%}")
            print("-" * len(hdr))
            print(hdr)
            print("-" * len(hdr))
            for _, r in sub.iterrows():
                print(f"  {LABELS.get(r.Tool, r.Tool):<24}"
                      f"{int(r.gained_psms):>{cw},}{int(r.lost_psms):>{cw},}"
                      f"{int(r.common_psms):>{cw},}"
                      f"{int(r.gained_peptides):>{cw},}{int(r.lost_peptides):>{cw},}"
                      f"{int(r.common_peptides):>{cw},}"
                      f"{int(r.gained_proteins):>{cw},}{int(r.lost_proteins):>{cw},}"
                      f"{int(r.common_proteins):>{cw},}")
            print("-" * len(hdr))

        op = os.path.join(args.out, "gain_loss_vs_baseline.tsv")
        overlap_exp.to_csv(op, sep="\t", index=False)
        print(f"\n  [OUT] {op}")
    else:
        overlap_exp = pd.DataFrame()

    # Plots
    print("\n[INFO] Plots")
    plot_counts(summary, args.out)
    plot_gain(summary, args.out)
    if not overlap_exp.empty:
        plot_gain_loss_diverging(overlap_exp, args.out)
    exp_psm = {tk: (pd.concat(frs, ignore_index=True) if frs else pd.DataFrame())
               for tk, frs in accum_psm.items()}
    plot_score_dists(exp_psm, args.out)
    print("\n[INFO] Done.")

if __name__ == "__main__":
    main()
