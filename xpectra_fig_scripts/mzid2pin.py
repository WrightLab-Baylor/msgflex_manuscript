#!/usr/bin/env python3
"""
mzid_to_pin.py
--------------
"""

import argparse
import csv
import math
import os
import sys
import glob
from collections import defaultdict
import xml.etree.ElementTree as stdET
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from pyteomics import mzid
    from lxml import etree
except ImportError:
    sys.exit("[ERROR] Run: pip install pyteomics psims lxml")

# ---------------------------------------------------------------------------
# PIN columns matching MSBooster's expected feature names
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "ExpMass",
    "retentiontime",
    "rank",
    "abs_ppm",
    "isotope_errors",
    "hyperscore",
    "delta_hyperscore",
    "matched_ion_num",
    "complementary_ions",
    "ion_series",
    "weighted_average_abs_fragment_ppm",
    "length_7",
    "length_8",
    "length_9_30",
    "length_31",
    "ntt",
    "nmc",
    "charge_1",
    "charge_2",
    "charge_3",
    "charge_4",
    "charge_5",
    "charge_6",
    "charge_7_or_more",
    "group_1",
    "group_2",
    "group_3",
    "group_other",
    "-17.0265nQ",
    "-18.0106nE",
    "119.0041C",
    "15.9949M",
]

HEADER = ["SpecId", "Label", "ScanNr"] + FEATURE_COLS + ["Peptide", "Proteins"]

MOD_COLUMNS = {
    -17.0265: "-17.0265nQ",
    -18.0106: "-18.0106nE",
    57.0215: "119.0041C",
     15.9949: "15.9949M",
}
MOD_TOL = 0.002

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_decoy(proteins, prefix) :
    if not proteins:
        return False
    return all(p.startswith(prefix) for p in proteins)

def check_existing(out_path) :
    return not os.path.isfile(out_path)

def safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def parse_scan_number(spectrum_id):
    for token in spectrum_id.split():
        if token.startswith("scan="):
            return int(token.split("=")[1])
        if token.startswith("index="):
            return int(token.split("=")[1]) + 1
        if token.startswith("spectrum="):
            return int(token.split("=")[1])
    parts = spectrum_id.replace("=", " ").split()
    for p in reversed(parts):
        if p.isdigit():
            return int(p)
    return 0

def load_rt_map(mzml_dir, raw_basename):
    """
    Extract scan_number -> retention_time (minutes) using raw iterparse.
    """
    rt_map = {}
    candidates = [
        os.path.join(mzml_dir, raw_basename + ".mzML"),
        os.path.join(mzml_dir, raw_basename + ".mzml"),
    ]
    mzml_path = next((c for c in candidates if os.path.isfile(c)), None)
    if mzml_path is None:
        print(f"  [WARNING] No mzML found for {raw_basename}, retentiontime=0",
              file=sys.stderr)
        return rt_map

    print(f"  [INFO] Loading RT from {os.path.basename(mzml_path)}", file=sys.stderr)
    try:
        scan_nr = None
        for event, elem in stdET.iterparse(mzml_path, events=("start", "end")):
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

            if event == "start" and tag == "spectrum":
                sid = elem.get("id", "")
                scan_nr = parse_scan_number(sid)

            elif event == "start" and tag == "cvParam" and scan_nr is not None:
                if elem.get("name", "") == "scan start time":
                    rt_map[scan_nr] = float(elem.get("value", 0.0))

            elif event == "end" and tag == "spectrum":
                elem.clear()
                scan_nr = None

    except Exception as e:
        print(f"  [WARNING] mzML read error: {e}", file=sys.stderr)

    print(f"  [INFO] Loaded {len(rt_map)} RT values", file=sys.stderr)
    return rt_map

def build_peptide_string(peptide_seq, mods):
    mod_map = defaultdict(list)
    for m in mods:
        pos  = m.get("location", 0)
        mass = safe_float(m.get("monoisotopicMassDelta", 0))
        mod_map[pos].append(mass)

    pep_str   = ""
    mod_flags = {c: 0 for c in MOD_COLUMNS.values()}

    for i, aa in enumerate(peptide_seq, start=1):
        pep_str += aa
        if i in mod_map:
            for dm in mod_map[i]:
                pep_str += f"[{dm:+.4f}]"
                for ref_mass, col in MOD_COLUMNS.items():
                    if abs(dm - ref_mass) <= MOD_TOL:
                        mod_flags[col] = 1

    return pep_str, mod_flags

def peptide_length_flags(seq):
    n = len(seq)
    return {
        "length_7": 1 if n == 7 else 0,
        "length_8": 1 if n == 8 else 0,
        "length_9_30": 1 if 9 <= n <= 30 else 0,
        "length_31":   1 if n >= 31 else 0,
    }

def charge_flags(charge):
    return {
        "charge_1": 1 if charge == 1 else 0,
        "charge_2": 1 if charge == 2 else 0,
        "charge_3": 1 if charge == 3 else 0,
        "charge_4": 1 if charge == 4 else 0,
        "charge_5": 1 if charge == 5 else 0,
        "charge_6": 1 if charge == 6 else 0,
        "charge_7_or_more": 1 if charge >= 7 else 0,
    }

def calc_ntt(prev_aa, next_aa, seq):
    """Count tryptic termini: KR|not-P at each end."""
    n = 0
    if prev_aa in ("-", "K", "R"):
        n += 1
    if next_aa in ("-", "*") or seq[-1] in ("K", "R"):
        n += 1
    return n

# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------
def extract_psms(mzid_path, prefix, rt_map):
    psms = []
    raw_basename = os.path.splitext(os.path.basename(mzid_path))[0]

    # use_index=False skips building the byte-offset index — major speedup
    with mzid.MzIdentML(mzid_path, use_index=True) as reader:
        for spectrum_result in reader:
            spectrum_id = spectrum_result.get("spectrumID", "")
            scan_nr = parse_scan_number(spectrum_id)
            rt = rt_map.get(scan_nr, 0.0)

            items = spectrum_result.get("SpectrumIdentificationItem", [])
            # Top hit only -- MS-GF+ TDA mode gives one PSM per spectrum
            for rank, sii in enumerate(items[:1], start=1):
                peptide_seq  = sii.get("PeptideSequence", "")
                mods = sii.get("Modification", [])
                charge = int(sii.get("chargeState", 2))
                exp_mz = safe_float(sii.get("experimentalMassToCharge", 0))
                calc_mz = safe_float(sii.get("calculatedMassToCharge", exp_mz))

                spec_evalue  = safe_float(sii.get("MS-GF:SpecEValue",  1.0))
                raw_score = safe_float(sii.get("MS-GF:RawScore",     0))
                denovo_score = safe_float(sii.get("MS-GF:DeNovoScore",  0))

                log10_evalue = -math.log10(spec_evalue + 1e-300)
                delta_hyper = denovo_score - raw_score
                exp_mass = exp_mz * charge - charge * 1.007276
                abs_ppm = abs(exp_mz - calc_mz) / calc_mz * 1e6 if calc_mz else 0.0

                pep_core, mod_flags = build_peptide_string(peptide_seq, mods)

                prev_aa = next_aa = "-"
                proteins = []
                nmc = 0
                for ev in sii.get("PeptideEvidenceRef", []):
                    acc = ev.get("accession", ev.get("protein description", ""))
                    if acc:
                        proteins.append(acc)
                    if prev_aa == "-":
                        prev_aa = ev.get("pre",  "-") or "-"
                        next_aa = ev.get("post", "-") or "-"
                    nmc = int(ev.get("missedCleavages", 0) or 0)

                if not proteins:
                    proteins = [sii.get("accession", "unknown")]

                label  = -1 if is_decoy(proteins, prefix) else 1
                spec_id = f"{raw_basename}_msms{scan_nr}.{scan_nr}.{scan_nr}.{charge}_{rank}"
                peptide = f"{prev_aa}.{pep_core}.{next_aa}"
                prot_str= ";".join(proteins) + ";"

                row = {
                    "SpecId": spec_id,
                    "Label": label,
                    "ScanNr": scan_nr,
                    "ExpMass": round(exp_mass, 7),
                    "retentiontime": round(rt, 7),
                    "rank": rank,
                    "abs_ppm": abs_ppm,
                    "isotope_errors": 0,
                    "hyperscore": log10_evalue,
                    "delta_hyperscore": delta_hyper,
                    "matched_ion_num": 0,
                    "complementary_ions": 0,
                    "ion_series":  0,
                    "weighted_average_abs_fragment_ppm": abs_ppm,
                    "ntt": calc_ntt(prev_aa, next_aa, peptide_seq),
                    "nmc": nmc,
                    "group_1":  0,
                    "group_2":  0,
                    "group_3":  0,
                    "group_other": 0,
                    "Peptide":  peptide,
                    "Proteins": prot_str,
                }
                row.update(mod_flags)
                row.update(peptide_length_flags(peptide_seq))
                row.update(charge_flags(charge))
                psms.append(row)

    return psms

def process_parallel(args_tuple):
    mzid_path, out_path, prefix, rt_map = args_tuple
    raw_basename = os.path.splitext(os.path.basename(mzid_path))[0]
    print(f"  [INFO] Processing {os.path.basename(mzid_path)}", file=sys.stderr)
    rt_map = load_rt_map(rt_map, raw_basename)
    psms = extract_psms(mzid_path, prefix, rt_map)
    return raw_basename, psms, out_path

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert a directory of MS-GF+ .mzid files (TDA mode) "
                    "to MSBooster-compatible PIN files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input-dir",  required=True,
                        help="Directory containing .mzid files.")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Directory for output .pin files (created if absent).")
    parser.add_argument("--mzml-dir", default=None,
                        help="Directory containing .mzML files for RT lookup "
                             "(defaults to --input-dir).")
    parser.add_argument("--decoy-prefix", default="XXX_",
                        metavar="PREFIX",
                        help="Decoy protein prefix used by MS-GF+ -tda 1 "
                             "(e.g. rev_, DECOY_, REV_, XXX_). Default: XXX_")
    parser.add_argument("-t", "--threads", type=int, default=1,
                        help="Number of parallel workers for processing files.")
    
    args = parser.parse_args()

    mzid_files = sorted(glob.glob(os.path.join(args.input_dir, "*.mzid")))
    if not mzid_files:
        sys.exit(f"[ERROR] No .mzid files found in {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    mzml_dir = args.mzml_dir or args.input_dir
    prefix = args.decoy_prefix
    n_workers = max(1, args.threads)

    print(f"[INFO] Decoy prefix: {prefix}", file=sys.stderr)
    print(f"[INFO] Files to process: {len(mzid_files)}", file=sys.stderr)
    print(f"[INFO] Parallel workers: {n_workers}", file=sys.stderr)

    tasks = []
    for mzid_path in mzid_files:
        raw_basename = os.path.splitext(os.path.basename(mzid_path))[0]
        out_path  = os.path.join(args.output_dir, raw_basename + ".pin")
        if not check_existing(out_path):
            print(f"  [INFO] Skipping existing {out_path}", file=sys.stderr)
            continue
        tasks.append((mzid_path, out_path, prefix, mzml_dir))
        
    total_psms = 0
    files_processed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_parallel, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            raw_basename, psms, out_path = fut.result()
            files_processed += 1
            total_psms += len(psms)

            if not psms:
                print(f"  [WARNING] No PSMs extracted for {raw_basename}, skipping.", file=sys.stderr)
                continue

            with open(out_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=HEADER, delimiter="\t",
                                        extrasaction="ignore")
                writer.writeheader()
                writer.writerows(psms)

            targets = sum(1 for p in psms if p["Label"] ==  1)
            decoys  = sum(1 for p in psms if p["Label"] == -1)
            print(f"  [INFO] {len(psms)} PSMs => {out_path}", file=sys.stderr)
            print(f"  [INFO] Targets: {targets}  Decoys: {decoys}", file=sys.stderr)

    print(f"[INFO] Processed {files_processed}/{len(mzid_files)} files.", file=sys.stderr)
   
    print("\n[INFO] Done.", file=sys.stderr)

if __name__ == "__main__":
    main()