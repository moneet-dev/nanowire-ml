"""
02_feature_extraction.py — Phase 2: iFeature descriptor extraction.

For a dataset variant, extract the 22 iFeature descriptor groups, concatenate
them into a master feature matrix, and write:

  - data/features/<variant>/<DESC>.csv         (per-descriptor; gitignored)
  - data/features/master_features[_redundant].csv  (gitignored, large)
  - results/metrics/feature_manifest[_redundant].csv  (tracked: descriptor sizes)

Usage (from repo root):
  python scripts/02_feature_extraction.py            # non-redundant (default)
  python scripts/02_feature_extraction.py redundant  # paper-faithful variant
  python scripts/02_feature_extraction.py nr redundant
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import feature_utils as fu  # noqa: E402

RAW = os.path.join(REPO, "data", "raw")
FEAT = os.path.join(REPO, "data", "features")
METRICS = os.path.join(REPO, "results", "metrics")

# variant -> (fasta, labels, master_csv, manifest_csv)
VARIANTS = {
    "nr": ("merged.fasta", "labels.csv",
           "master_features.csv", "feature_manifest.csv"),
    "redundant": ("merged_redundant.fasta", "labels_redundant.csv",
                  "master_features_redundant.csv", "feature_manifest_redundant.csv"),
}


def run_variant(name: str, fasta_name: str, labels_name: str,
                master_name: str, manifest_name: str) -> None:
    fasta = os.path.join(RAW, fasta_name)
    labels = os.path.join(RAW, labels_name)
    out_dir = os.path.join(FEAT, name)

    print(f"\n=== variant '{name}': {fasta_name} ===", flush=True)
    t0 = time.time()
    feats = fu.extract_all_features(fasta, out_dir)
    master = fu.build_master_matrix(feats, labels, os.path.join(FEAT, master_name))
    dt = time.time() - t0

    # Per-descriptor feature counts (as extracted, before dedup/constant-drop).
    prefixes = pd.Series([c.split("::", 1)[0] for c in feats.columns])
    manifest = prefixes.value_counts().rename_axis("descriptor").rename("n_features")
    manifest.sort_index().to_csv(os.path.join(METRICS, manifest_name))

    failed = feats.attrs.get("descriptors_failed", {})
    print(f"\n  extracted in {dt:.1f}s")
    print(f"  raw concat shape : {feats.shape}")
    print(f"  master shape     : {master.shape}  (features={master.shape[1] - 1}, +label)")
    print(f"  NaNs filled      : {master.attrs.get('n_nan_filled', 0)}")
    print(f"  descriptors ok   : {len(feats.attrs.get('descriptors_ok', []))}/22")
    if failed:
        print(f"  FAILED           : {failed}")
    print(f"  master  -> {os.path.relpath(os.path.join(FEAT, master_name), REPO)}")
    print(f"  manifest-> {os.path.relpath(os.path.join(METRICS, manifest_name), REPO)}")


def main() -> None:
    os.makedirs(METRICS, exist_ok=True)
    requested = sys.argv[1:] or ["nr"]
    for name in requested:
        if name not in VARIANTS:
            print(f"unknown variant '{name}'; choices: {list(VARIANTS)}")
            sys.exit(1)
        run_variant(name, *VARIANTS[name])


if __name__ == "__main__":
    main()
