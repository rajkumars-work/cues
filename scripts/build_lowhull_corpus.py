#!/usr/bin/env python3
"""Derive the A5 corpora from the already-fetched wide training CSV.

Filters ./data/corpus/big_d15_autolabel_binrho_k7.csv (the same file the C2/C3
architecture ablation uses — don't refetch it) by `e_above_hull`, this
project's own "lowhull" convention (<= 0.1; "onhull" would be <= 0). Also
writes a size-matched random subsample of the wide corpus, row-count-equal to
the filtered file, which is the *primary* A5 comparison (architecture_design's
A5 doc: "same architecture, size, and epoch budget -- only stability-filtering
varies"). The untouched full wide CSV remains available as the secondary,
full-size arm.

Usage:
    python3 scripts/build_lowhull_corpus.py \
        --wide ./data/corpus/big_d15_autolabel_binrho_k7.csv \
        --out-dir ./data/corpus

Writes, under --out-dir:
    big_d15_lowhull.csv          -- e_above_hull <= --threshold (default 0.1)
    big_d15_wide_matched.csv     -- random subsample of the wide CSV, same
                                     row count as the lowhull file, seeded
                                     deterministically (--seed, default 0)
"""
import argparse
import csv
import random
import sys

EAH_COL = "e_above_hull"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wide", default="./data/corpus/big_d15_autolabel_binrho_k7.csv")
    ap.add_argument("--out-dir", default="./data/corpus")
    ap.add_argument("--threshold", type=float, default=0.1,
                     help="e_above_hull cutoff, inclusive (0.1 = this project's 'lowhull' tier; 0 = 'onhull')")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.wide, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if EAH_COL not in fieldnames:
            sys.exit(f"'{EAH_COL}' not found in {args.wide}'s header: {fieldnames}")
        rows = list(reader)

    kept, dropped_missing = [], 0
    for row in rows:
        v = row.get(EAH_COL, "")
        try:
            fv = float(v)
        except (TypeError, ValueError):
            dropped_missing += 1
            continue
        if fv <= args.threshold:
            kept.append(row)

    lowhull_path = f"{args.out_dir}/big_d15_lowhull.csv"
    with open(lowhull_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept)

    rng = random.Random(args.seed)
    usable = [r for r in rows if r.get(EAH_COL, "") not in ("", None)]
    matched = rng.sample(usable, len(kept)) if len(kept) <= len(usable) else usable

    matched_path = f"{args.out_dir}/big_d15_wide_matched.csv"
    with open(matched_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(matched)

    print(f"wide corpus:        {len(rows)} rows ({dropped_missing} missing/unparseable {EAH_COL}, excluded from both outputs below)")
    print(f"lowhull (<= {args.threshold}):  {len(kept)} rows -> {lowhull_path}")
    print(f"wide, size-matched:  {len(matched)} rows (seed={args.seed}) -> {matched_path}")
    print("Full untouched wide CSV remains at the --wide path for the secondary, full-size arm.")


if __name__ == "__main__":
    main()
