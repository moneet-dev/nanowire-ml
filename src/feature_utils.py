"""
feature_utils.py — iFeature descriptor extraction wrappers for NanowireML.

Wraps ``iFeatureOmegaCLI`` to extract the 22 descriptor groups used by
Raya et al. 2025 and concatenate them into a single master feature matrix.
Fully parameterized on ``(fasta_path, output_dir)`` so P2 can reuse it to build
node features for CREID proteins.
"""
from __future__ import annotations

import os

import pandas as pd

#: The 22 descriptor groups from Raya et al. 2025 (iFeatureOmegaCLI names).
DESCRIPTORS: list[str] = [
    "AAC", "GAAC", "CKSAAP type 1", "DPC type 1", "DPC type 2",
    "TPC type 1", "TPC type 2", "CTDC", "CTDT", "CTDD",
    "CTriad", "KNN", "Geary", "Moran", "NMBroto", "AC", "CC",
    "SOCNumber", "QSOrder", "PAAC", "ZScale", "AAIndex",
]

#: Filename-safe aliases for descriptor names that contain spaces.
ALIASES: dict[str, str] = {
    "CKSAAP type 1": "CKSAAP",
    "DPC type 1": "DPC1",
    "DPC type 2": "DPC2",
    "TPC type 1": "TPC1",
    "TPC type 2": "TPC2",
}


def alias(desc: str) -> str:
    """Return a filename-safe short name for a descriptor."""
    return ALIASES.get(desc, desc.replace(" ", "_"))


def extract_all_features(
    fasta_path: str,
    output_dir: str,
    descriptors: list[str] = DESCRIPTORS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run iFeatureOmega for each descriptor and return the concatenated matrix.

    Each descriptor is written to ``{output_dir}/{alias}.csv`` and its columns
    are namespaced ``{alias}::{col}`` to keep them unique across groups. A failing
    descriptor is logged and skipped rather than aborting the whole run; the list
    of successes/failures is stored in ``df.attrs``.
    """
    import iFeatureOmegaCLI

    os.makedirs(output_dir, exist_ok=True)
    protein = iFeatureOmegaCLI.iProtein(fasta_path)

    frames: list[pd.DataFrame] = []
    ok: list[str] = []
    failed: dict[str, str] = {}

    for desc in descriptors:
        try:
            protein.get_descriptor(desc)
            csv_path = os.path.join(output_dir, f"{alias(desc)}.csv")
            protein.to_csv(csv_path, index=True, header=True)
            df = pd.read_csv(csv_path, index_col=0)
            df.columns = [f"{alias(desc)}::{c}" for c in df.columns]
            frames.append(df)
            ok.append(desc)
            if verbose:
                print(f"[ok]   {desc:16s} -> {df.shape[1]:>5d} features")
        except Exception as e:  # noqa: BLE001 - one bad descriptor must not abort
            failed[desc] = str(e)
            if verbose:
                print(f"[FAIL] {desc:16s} -> {e}")

    if not frames:
        raise RuntimeError("No descriptors extracted successfully.")

    out = pd.concat(frames, axis=1)
    out.attrs["descriptors_ok"] = ok
    out.attrs["descriptors_failed"] = failed
    return out


def build_master_matrix(
    feature_df: pd.DataFrame,
    labels_csv: str,
    out_path: str,
    drop_constant: bool = True,
) -> pd.DataFrame:
    """Join the feature matrix to labels on ``seq_id`` and clean it.

    Steps: coerce to numeric, fill NaNs with 0, optionally drop constant columns,
    drop duplicate columns, append the ``label`` column, and write ``out_path``.
    Returns the saved DataFrame (indexed by ``seq_id``).
    """
    labels = pd.read_csv(labels_csv).set_index("seq_id")["label"]

    X = feature_df.copy()
    common = X.index.intersection(labels.index)
    X = X.loc[common]
    y = labels.loc[common]

    X = X.apply(pd.to_numeric, errors="coerce")
    n_nan = int(X.isna().sum().sum())
    X = X.fillna(0.0)

    if drop_constant:
        X = X.loc[:, X.nunique() > 1]
    X = X.loc[:, ~X.T.duplicated()]  # drop duplicate columns

    master = X.copy()
    master["label"] = y.values
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    master.to_csv(out_path)

    master.attrs["n_nan_filled"] = n_nan
    master.attrs["n_features"] = X.shape[1]
    return master
