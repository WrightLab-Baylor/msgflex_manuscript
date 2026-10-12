#!/usr/bin/env python3
"""
mzid_to_pepxml.py
-----------------
Convert a directory of MS-GF+ mzIdentML files to pepXML format
compatible with MSBooster.
"""

import argparse
import glob
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from pyteomics import mzid
    from lxml import etree
except ImportError:
    sys.exit("[ERROR] Run: pip install pyteomics psims lxml")
import xml.etree.ElementTree as stdET


# XML namespace

PEPXML_NS  = "http://regis-web.systemsbiology.net/pepXML"
PEPXML_XSD = "http://sashimi.sourceforge.net/schema_revision/pepXML/pepXML_v120.xsd"


# Helpers
def safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def is_decoy(proteins, prefix):
    """True if ALL proteins start with the decoy prefix."""
    if not proteins:
        return False
    return all(p.startswith(prefix) for p in proteins)

def check_existing(out_path):
    """Return True if the output file does **not** already exist."""
    return not os.path.isfile(out_path)

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
    rt_map = {}
    candidates = [
        os.path.join(mzml_dir, raw_basename + ".mzML"),
        os.path.join(mzml_dir, raw_basename + ".mzml"),
    ]
    mzml_path = next((c for c in candidates if os.path.isfile(c)), None)
    if mzml_path is None:
        print(f"  [WARNING] No mzML found for {raw_basename}, RT=0", file=sys.stderr)
        return rt_map
    print(f"  [INFO] Loading RT from {os.path.basename(mzml_path)}", file=sys.stderr)
    try:
        scan_nr = None
        for event, elem in stdET.iterparse(mzml_path, events=("start", "end")):
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if event == "start" and tag == "spectrum":
                scan_nr = parse_scan_number(elem.get("id", ""))
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

def get_db_path(mzid_path) -> str:
    try:
        for event, elem in stdET.iterparse(mzid_path, events=("start",)):
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag == "SearchDatabase":
                loc = elem.get("location", "")
                if loc:
                    return loc.replace("file://", "")
            elem.clear()
    except Exception:
        pass
    return "unknown.fasta"

def build_modified_peptide(peptide_seq, mods) -> tuple:
    mod_map = defaultdict(list)
    for m in mods:
        pos  = m.get("location", 0)
        mass = safe_float(m.get("monoisotopicMassDelta", 0))
        mod_map[pos].append(mass)

    pep_str = ""
    aa_mods = []
    aa_masses = {
        "A": 71.03711,  "R": 156.10111, "N": 114.04293, "D": 115.02694,
        "C": 103.00919, "E": 129.04259, "Q": 128.05858, "G":  57.02146,
        "H": 137.05891, "I": 113.08406, "L": 113.08406, "K": 128.09496,
        "M": 131.04049, "F": 147.06841, "P":  97.05276, "S":  87.03203,
        "T": 101.04768, "W": 186.07931, "Y": 163.06333, "V":  99.06841,
    }

    for i, aa in enumerate(peptide_seq, start=1):
        pep_str += aa
        if i in mod_map:
            for dm in mod_map[i]:
                pep_str += f"[{dm:.4f}]"
                base = aa_masses.get(aa, 0.0)
                aa_mods.append({
                    "position":  i,
                    "aminoacid": aa,
                    "massdiff":  f"{dm:+.4f}",
                    "mass":      round(base + dm, 5),
                })

    return pep_str, aa_mods

# ---------------------------------------------------------------------------
# XML construction
# ---------------------------------------------------------------------------
def make_run_element(raw_basename, db_path) -> tuple:
    XSI = "http://www.w3.org/2001/XMLSchema-instance"
    nsmap = {None: PEPXML_NS, "xsi": XSI}
    root = etree.Element("msms_pipeline_analysis", nsmap=nsmap)
    root.set("date",        datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    root.set("summary_xml", f"{raw_basename}.pepXML")
    root.set(f"{{{XSI}}}schemaLocation", f"{PEPXML_NS} {PEPXML_XSD}")

    run = etree.SubElement(root, "msms_run_summary")
    run.set("base_name",    raw_basename)
    run.set("raw_data_type","raw")
    run.set("raw_data",     ".mzML")
    run.set("search_engine","MS-GF+")

    enz  = etree.SubElement(run, "sample_enzyme")
    enz.set("name", "trypsin")
    spec = etree.SubElement(enz, "specificity")
    spec.set("cut",    "KR")
    spec.set("no_cut", "P")
    spec.set("sense",  "C")

    ss = etree.SubElement(run, "search_summary")
    ss.set("base_name",             raw_basename)
    ss.set("search_engine",         "MS-GF+")
    ss.set("search_engine_version", "unknown")
    ss.set("precursor_mass_type",   "monoisotopic")
    ss.set("fragment_mass_type",    "monoisotopic")
    ss.set("search_id",             "1")

    # Single database entry — fixes multi-db idconvert error
    db_elem = etree.SubElement(ss, "search_database")
    db_elem.set("local_path", db_path)
    db_elem.set("type",       "AA")

    return root, run

def add_spectrum_query(run_elem, scan_nr, charge, exp_mz, rt,
                       sii, proteins, prefix):

    peptide_seq  = sii.get("PeptideSequence", "")
    mods = sii.get("Modification",    [])
    spec_evalue  = safe_float(sii.get("MS-GF:SpecEValue",  1.0))
    evalue = safe_float(sii.get("MS-GF:EValue",      1.0))
    raw_score = safe_float(sii.get("MS-GF:RawScore",     0))
    denovo_score = safe_float(sii.get("MS-GF:DeNovoScore",  0))
    calc_mz = safe_float(sii.get("calculatedMassToCharge", exp_mz))

    neutral_mass = exp_mz  * charge - charge * 1.007276
    calc_neutral = calc_mz * charge - charge * 1.007276
    mass_diff = neutral_mass - calc_neutral

    mod_peptide, aa_mods = build_modified_peptide(peptide_seq, mods)

    prev_aa = next_aa = "-"
    nmc = 0
    for ev in sii.get("PeptideEvidenceRef", []):
        prev_aa = ev.get("pre",  "-") or "-"
        next_aa = ev.get("post", "-") or "-"
        nmc     = int(sii.get("missedCleavages", 0) or 0)
        break

    raw_basename = run_elem.get("base_name", "unknown")

    sq = etree.SubElement(run_elem, "spectrum_query")
    sq.set("spectrum", f"{raw_basename}.{scan_nr}.{scan_nr}.{charge}")
    sq.set("start_scan", str(scan_nr))
    sq.set("end_scan", str(scan_nr))
    sq.set("precursor_neutral_mass", f"{neutral_mass:.6f}")
    sq.set("assumed_charge", str(charge))
    sq.set("index", str(scan_nr))
    sq.set("retention_time_min", f"{rt:.4f}")  # mzML minutes

    sr = etree.SubElement(sq, "search_result")

    sh = etree.SubElement(sr, "search_hit")
    sh.set("hit_rank", "1")
    sh.set("peptide", peptide_seq)
    sh.set("peptide_prev_aa", prev_aa)
    sh.set("peptide_next_aa", next_aa)
    sh.set("protein", proteins[0] if proteins else "unknown")
    sh.set("num_tot_proteins", str(len(proteins)))
    sh.set("num_matched_ions", "0")
    sh.set("tot_num_ions", "0")
    sh.set("calc_neutral_pep_mass", f"{calc_neutral:.6f}")
    sh.set("massdiff", f"{mass_diff:.6f}")
    sh.set("num_tol_term", "2")
    sh.set("num_missed_cleavages", str(nmc))
    sh.set("is_rejected", "0")
    sh.set("modified_peptide", mod_peptide)

    for alt in proteins[1:]:
        ap = etree.SubElement(sh, "alternative_protein")
        ap.set("protein", alt)
        ap.set("num_tol_term", "2")

    if aa_mods:
        mi = etree.SubElement(sh, "modification_info")
        mi.set("modified_peptide", mod_peptide)
        for m in aa_mods:
            am = etree.SubElement(mi, "mod_aminoacid_mass")
            am.set("position", str(m["position"]))
            am.set("mass",     str(m["mass"]))

    def add_score(name, val):
        s = etree.SubElement(sh, "search_score")
        s.set("name",  name)
        s.set("value", str(val))

    add_score("MS-GF:RawScore", raw_score)
    add_score("MS-GF:DeNovoScore", denovo_score)
    add_score("MS-GF:SpecEValue", spec_evalue)
    add_score("MS-GF:EValue", evalue)
    add_score("MS-GF:nlog_SpecEValue", -math.log10(spec_evalue + 1e-300))

    label = -1 if is_decoy(proteins, prefix) else 1
    ar = etree.SubElement(sh, "analysis_result")
    ar.set("analysis", "peptideprophet")
    pp = etree.SubElement(ar, "peptideprophet_result")
    pp.set("probability",  "0.0" if label == -1 else "1.0")
    pp.set("all_ntt_prob", "(0,0,0)" if label == -1 else "(0,0,1)")
    pi = etree.SubElement(pp, "search_score_summary")
    p = etree.SubElement(pi, "parameter")
    p.set("name",  "decoy")
    p.set("value", "1" if label == -1 else "0")


# Per-file conversion
def convert_one(mzid_path, out_path, prefix, mzml_dir):
    raw_basename = os.path.splitext(os.path.basename(mzid_path))[0]
    db_path  = get_db_path(mzid_path)
    rt_map = load_rt_map(mzml_dir, raw_basename)

    root, run_elem = make_run_element(raw_basename, db_path)

    count = 0
    with mzid.MzIdentML(mzid_path, use_index=True) as reader:
        for spectrum_result in reader:
            spectrum_id = spectrum_result.get("spectrumID", "")
            scan_nr     = parse_scan_number(spectrum_id)
            rt          = rt_map.get(scan_nr, 0.0)

            items = spectrum_result.get("SpectrumIdentificationItem", [])
            # Top hit only — MS-GF+ TDA mode gives one PSM per spectrum
            for sii in items[:1]:
                charge  = int(sii.get("chargeState", 2))
                exp_mz  = safe_float(sii.get("experimentalMassToCharge", 0))

                proteins = []
                for ev in sii.get("PeptideEvidenceRef", []):
                    acc = ev.get("accession", ev.get("protein description", ""))
                    if acc:
                        proteins.append(acc)
                if not proteins:
                    proteins = [sii.get("accession", "unknown")]

                add_spectrum_query(run_elem, scan_nr, charge, exp_mz,
                                   rt, sii, proteins, prefix)
                count += 1

    tree = etree.ElementTree(root)
    with open(out_path, "wb") as fh:
        tree.write(fh, pretty_print=True,
                   xml_declaration=True, encoding="UTF-8")
    return count

def process_parallel(args_tuple):
    """Wrapper for ThreadPoolExecutor.map/submit."""
    mzid_path, out_path, prefix, mzml_dir = args_tuple
    raw_basename = os.path.splitext(os.path.basename(mzid_path))[0]
    print(f"  [INFO] Processing {os.path.basename(mzid_path)}", file=sys.stderr)
    try:
        n = convert_one(mzid_path, out_path, prefix, mzml_dir)
        return raw_basename, n, out_path, None
    except Exception as e:
        return raw_basename, 0, out_path, str(e)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert a directory of MS-GF+ .mzid files to "
                    "MSBooster-compatible pepXML files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input-dir",  required=True,
                        help="Directory containing .mzid files.")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Directory for output .pepXML files.")
    parser.add_argument("--mzml-dir", default=None,
                        help="Directory containing .mzML files for RT "
                             "(defaults to --input-dir).")
    parser.add_argument("--decoy-prefix", default="rev_",
                        metavar="PREFIX",
                        help="Decoy protein prefix used by MS-GF+ -tda 1 "
                             "(e.g. rev_, DECOY_, REV_, XXX_). Default: rev_")
    parser.add_argument("-t", "--threads", type=int, default=4,
                        help="Number of parallel workers (default: 4).")
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

    # Prepare arguments for each worker
    tasks = []
    for mzid_path in mzid_files:
        raw_basename = os.path.splitext(os.path.basename(mzid_path))[0]
        out_path = os.path.join(args.output_dir, raw_basename + ".pepXML")
        if not check_existing(out_path):
            print(f"  [INFO] Skipping existing {out_path}", file=sys.stderr)
            continue
        tasks.append((mzid_path, out_path, prefix, mzml_dir))

    total_psms = 0
    failed = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_parallel, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            mzid_path = futures[fut]
            try:
                raw_basename, n, out_path, err = fut.result()
            except Exception as e:
                print(f"  [ERROR] Error processing {mzid_path}: {e}", file=sys.stderr)
                failed.append(mzid_path)
                continue
            if err:
                print(f"  [ERROR] Failed to process {mzid_path}: {err}", file=sys.stderr)
                failed.append(mzid_path)
                continue
            total_psms += n
            print(f"  [INFO] {n} spectrum queries → {out_path}", file=sys.stderr)

    print(f"\n[INFO] Done. Total PSMs written: {total_psms}", file=sys.stderr)
    if failed:
        print(f"[WARNING] Failed to process {len(failed)} files:", file=sys.stderr)
        for f in failed:
            print(f" {f}", file=sys.stderr)

if __name__ == "__main__":
    main()