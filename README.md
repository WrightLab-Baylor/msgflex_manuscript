This repository contains the scripts used in analyzing the data associated with
manuscript "MSGFLEX: An optimized end-to-end pipeline for metaproteomics analysis built on MS-GF+ with a dedicated rescoring model, XPECTRA"

# Description/Usage of each script

`fasta_splitter.py` - This Python script is useful for splitting large protein databases into smaller bins for `sparx-binning` pipeline

usage:

```
usage: fasta_splitter.py [-h] -i INPUT [-n NUM_PARTS] [-seq SEQ_PER_PART] [-r] [-o OUTPUT]

Stream-split a FASTA into parts without loading all sequences.

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input FASTA file
  -n NUM_PARTS, --num_parts NUM_PARTS
                        Number of parts
  -seq SEQ_PER_PART, --seq_per_part SEQ_PER_PART
                        Number of sequences per part (e.g. 50000)
  -r, --random          Randomize sequence allocation (two-pass, index-based)
  -o OUTPUT, --output OUTPUT
                        Output directory (default: <input_basename>_split)
```

## SPARX module data-analysis

`numa_optimization_plot.py` - Python script used to visualize MS-GF+ runtime reduction from NUMA-aware task/thread optimization.

Statistical tests used:
 - Wilcoxon signed-rank
 - paired t-test

`validate_split_vs_unified.py` - Python script used to validate the bias produced by splitting/binning of large metaproteomic databases.

## XPECTRA module data-analysis

MSBooster and Percolator needs '.pin' & '.pepXML' format files for rescoring. So we used _in-house_ written scripts to convert MS_GF+ resulted '.mzid' files to respetive formats. 

`mzid2pin.py` - Python script used to convert '.mzid' files directly from directory to '.pin' format using '.mzML' to retrieve retention time information.

script usage:

```
usage: mzid2pin.py [-h] -i INPUT_DIR -o OUTPUT_DIR [--mzml-dir MZML_DIR] [--decoy-prefix PREFIX] [-t THREADS]

Convert a directory of MS-GF+ .mzid files (TDA mode) to MSBooster-compatible PIN files.

options:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input-dir INPUT_DIR
                        Directory containing .mzid files. (default: None)
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory for output .pin files (created if absent). (default: None)
  --mzml-dir MZML_DIR   Directory containing .mzML files for RT lookup (defaults to --input-dir). (default: None)
  --decoy-prefix PREFIX
                        Decoy protein prefix used by MS-GF+ -tda 1 (e.g. rev_, DECOY_, REV_, XXX_). Default: XXX_ (default: XXX_)
  -t THREADS, --threads THREADS
                        Number of parallel workers for processing files. (default: 1)
```

`mzid2pepxml.py` - Python script used to convert '.mzid' files directly from directory to '.pepXML' format using '.mzML' to retrieve retention time information.

script usage:

```
usage: mzid2pepxml.py [-h] -i INPUT_DIR -o OUTPUT_DIR [--mzml-dir MZML_DIR] [--decoy-prefix PREFIX] [-t THREADS]

Convert a directory of MS-GF+ .mzid files to MSBooster-compatible pepXML files.

options:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input-dir INPUT_DIR
                        Directory containing .mzid files. (default: None)
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory for output .pepXML files. (default: None)
  --mzml-dir MZML_DIR   Directory containing .mzML files for RT (defaults to --input-dir). (default: None)
  --decoy-prefix PREFIX
                        Decoy protein prefix used by MS-GF+ -tda 1 (e.g. rev_, DECOY_, REV_, XXX_). Default: rev_ (default: rev_)
  -t THREADS, --threads THREADS
                        Number of parallel workers (default: 4). (default: 4)
```





