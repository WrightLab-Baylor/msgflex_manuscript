#!/usr/bin/env python3

import argparse
import os
import sys
import math
import random
from Bio import SeqIO

"""
split_fasta_stream_fixed.py

Stream-splitting FASTA into parts without loading all sequences into memory.
Guarantees exact part counts when -n is used by computing explicit sizes per part.

Requires: biopython (SeqIO)
"""

def compute_part_sizes(total, num_parts=None, seq_per_part=None):
    """
    Return (num_parts, seq_per_part, sizes_list)
    sizes_list is a list of integers with length == num_parts summing to total.
    """
    if seq_per_part and num_parts:
        # both specified -> prefer seq_per_part (explicit)
        num_parts = (total + seq_per_part - 1) // seq_per_part
    elif seq_per_part:
        num_parts = (total + seq_per_part - 1) // seq_per_part
    elif num_parts:
        # distribute as evenly as possible across exactly num_parts
        base = total // num_parts
        rem = total % num_parts
        sizes = [base + 1 if i < rem else base for i in range(num_parts)]
        # if some parts would be zero (num_parts > total), shrink to total parts of size 1
        if any(s == 0 for s in sizes):
            # make min(total, num_parts) parts: first `total` parts get 1, others removed
            sizes = [1] * total
            num_parts = total
        seq_per_part = None
        return num_parts, seq_per_part, sizes

    else:
        raise ValueError("Either seq_per_part (-seq) or num_parts (-n) must be provided.")

    # seq_per_part path: build sizes so that last part may be smaller
    sizes = []
    remaining = total
    while remaining > 0:
        take = min(seq_per_part, remaining)
        sizes.append(take)
        remaining -= take
    num_parts = len(sizes)
    return num_parts, seq_per_part, sizes

def split_fasta_stream(input_fasta, num_parts=None, seq_per_part=None, randomize=False, output_dir=None):
    base_name = os.path.splitext(os.path.basename(input_fasta))[0]
    if output_dir is None:
        output_dir = f"{base_name}_split"
    os.makedirs(output_dir, exist_ok=True)

    # Pass 1: count total records
    print(f"Counting sequences in: {input_fasta}", file=sys.stderr)
    total = sum(1 for _ in SeqIO.parse(input_fasta, "fasta"))
    print(f"Total sequences: {total}", file=sys.stderr)

    if total == 0:
        raise SystemExit("No sequences found in input.")

    # Handle inconsistent input: if both provided, prefer seq_per_part but print a message
    if num_parts and seq_per_part:
        print("Warning: both -n (num_parts) and -seq (seq_per_part) provided. "
              "Script will prioritize -seq and compute num_parts accordingly.", file=sys.stderr)

    num_parts, seq_per_part, sizes = compute_part_sizes(total, num_parts=num_parts, seq_per_part=seq_per_part)

    print(f"Planned output: {num_parts} part(s).", file=sys.stderr)
    print(f"Part sizes (first 10 shown): {sizes[:10]}{'...' if len(sizes)>10 else ''}", file=sys.stderr)

    if randomize:
        # Random mode: two-pass but only store integer indices (lightweight)
        indices = list(range(total))
        random.shuffle(indices)
        # create mapping index -> part_id based on sizes
        part_assign = [None] * total
        idx = 0
        for p, s in enumerate(sizes):
            for _ in range(s):
                if idx >= total:
                    break
                part_assign[indices[idx]] = p  # assign original-record-index -> part
                idx += 1

        # sanity check
        if any(x is None for x in part_assign):
            raise RuntimeError("Internal error building randomized assignment.")

        # open handles
        handles = [open(os.path.join(output_dir, f"{base_name}_part{p+1}.fasta"), "w") for p in range(num_parts)]

        for rec_idx, record in enumerate(SeqIO.parse(input_fasta, "fasta")):
            p = part_assign[rec_idx]
            SeqIO.write(record, handles[p], "fasta")

        for h in handles:
            h.close()

    else:
        # Sequential streaming mode: write exact sizes
        handles = []
        for p, s in enumerate(sizes):
            out_path = os.path.join(output_dir, f"{base_name}_part{p+1}.fasta")
            handles.append(open(out_path, "w"))

        current_part = 0
        written_in_part = 0
        for record in SeqIO.parse(input_fasta, "fasta"):
            # If current part is already full, advance
            while current_part < len(sizes) and written_in_part >= sizes[current_part]:
                current_part += 1
                written_in_part = 0
            if current_part >= len(sizes):
                # shouldn't happen, but safe guard
                break
            SeqIO.write(record, handles[current_part], "fasta")
            written_in_part += 1

        for h in handles:
            h.close()

    print("Done splitting.", file=sys.stderr)
    print(f"Wrote parts to: {output_dir}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Stream-split a FASTA into parts without loading all sequences.")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file")
    parser.add_argument("-n", "--num_parts", type=int, help="Number of parts")
    parser.add_argument("-seq", "--seq_per_part", type=int, help="Number of sequences per part (e.g. 50000)")
    parser.add_argument("-r", "--random", action="store_true", help="Randomize sequence allocation (two-pass, index-based)")
    parser.add_argument("-o", "--output", help="Output directory (default: <input_basename>_split)")
    args = parser.parse_args()

    split_fasta_stream(args.input, num_parts=args.num_parts, seq_per_part=args.seq_per_part,
                       randomize=args.random, output_dir=args.output)

if __name__ == "__main__":
    main()