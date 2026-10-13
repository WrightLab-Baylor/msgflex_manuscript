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





