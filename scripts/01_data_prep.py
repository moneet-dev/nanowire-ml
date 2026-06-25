"""
01_data_prep.py — Phase 1: build the merged, labeled nanowire dataset.

Reads the three source FASTAs (NCBI + UniProt positives, RecA negatives),
caps negatives at 999 (matching the official repo's `if i >= 999: break`),
assigns clean collision-free ids, removes exact-duplicate sequences, and writes:

  - data/raw/merged.fasta            (tracked)
  - data/raw/labels.csv              (tracked)
  - results/metrics/data_sanity.csv  (tracked sanity report)

Run from the repo root:  python scripts/01_data_prep.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import data_utils as du  # noqa: E402

RAW = os.path.join(REPO, "data", "raw")
METRICS = os.path.join(REPO, "results", "metrics")

POSITIVE_FILES = [
    os.path.join(RAW, "NCBI_doconno.fasta"),
    os.path.join(RAW, "Uniprot_doconno.fasta"),
]
NEGATIVE_FILES = [os.path.join(RAW, "Negative.fasta")]


def main() -> None:
    missing = [f for f in POSITIVE_FILES + NEGATIVE_FILES if not os.path.exists(f)]
    if missing:
        print("ERROR: missing source FASTA file(s):")
        for m in missing:
            print(f"  - {m}")
        print(
            "\nObtain them from the official repo "
            "(bicbioeng/nanowire-protein-prediction, 'Prediction dataset/') or "
            "Mendeley doi:10.17632/vdvdrj3k2p.1, and place them in data/raw/."
        )
        sys.exit(1)

    # Paper-faithful (redundant) variant: cap negatives at 999 and keep redundant
    # sequences (the original study removed only full-record duplicates). This
    # ~999/999 set is the basis for the Phase 3 reproduction of Table 2.
    df_red = du.build_dataset(POSITIVE_FILES, NEGATIVE_FILES, negative_cap=999, dedup=False)
    du.write_fasta(df_red, os.path.join(RAW, "merged_redundant.fasta"))
    du.write_labels(df_red, os.path.join(RAW, "labels_redundant.csv"))

    # Primary non-redundant variant: drop exact-duplicate sequences, removing the
    # train/test leakage that inflates the published performance estimates.
    df = du.build_dataset(POSITIVE_FILES, NEGATIVE_FILES, negative_cap=999, dedup=True)
    merged_fasta = os.path.join(RAW, "merged.fasta")
    labels_csv = os.path.join(RAW, "labels.csv")
    du.write_fasta(df, merged_fasta)
    du.write_labels(df, labels_csv)

    rep = du.sanity_report(df)
    noncanon = du.noncanonical_residues(df)
    rep["n_seqs_with_noncanonical_residues"] = len(noncanon)
    rep["redundant_n_total"] = int(len(df_red))
    rep["redundant_n_positive"] = int((df_red.label == 1).sum())
    rep["redundant_n_negative"] = int((df_red.label == 0).sum())

    os.makedirs(METRICS, exist_ok=True)
    sanity_csv = os.path.join(METRICS, "data_sanity.csv")
    pd.Series(rep, name="value").rename_axis("metric").to_csv(sanity_csv)

    print("Phase 1 - data preparation complete\n")
    print(f"  non-redundant : {os.path.relpath(merged_fasta, REPO)}, "
          f"{os.path.relpath(labels_csv, REPO)}")
    print("  redundant     : data/raw/merged_redundant.fasta, data/raw/labels_redundant.csv")
    print(f"  sanity CSV    : {os.path.relpath(sanity_csv, REPO)}\n")
    width = max(len(k) for k in rep)
    for k, v in rep.items():
        print(f"  {k:<{width}} : {v}")
    if noncanon:
        examples = list(noncanon.items())[:5]
        print(
            f"\n  note: {len(noncanon)} sequence(s) contain a non-canonical residue "
            f"(X/B/Z/U/O/*), e.g. {examples}"
        )


if __name__ == "__main__":
    main()
