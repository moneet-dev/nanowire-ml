"""
feature_utils.py — iFeature descriptor extraction wrappers for NanowireML.

Wraps ``iFeatureOmegaCLI`` to extract the 22 descriptor groups used by Raya et
al. 2025 and concatenate them into a single master feature matrix. Fully
parameterized on ``(fasta_path, output_dir)`` so P2 can reuse it to build node
features for CREID proteins.

Performance note: iFeatureOmega's Tripeptide-Composition normalization recomputes
``sum(tmpCode)`` inside an 8000-element comprehension (O(8000^2) per sequence),
which makes ``TPC type 1`` take minutes on ~1700 sequences. :func:`_fast_tpc`
patches it with a mathematically identical O(L) version (and tolerates
non-canonical residues by skipping their tripeptides).
"""
from __future__ import annotations

import os

import numpy as np
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

#: Minimum expected feature counts per descriptor — used to detect stale
#: encodings when get_descriptor() silently fails on some iFeatureOmegaCLI
#: versions (returns no error but leaves the previous descriptor's data).
_MIN_FEATURES: dict[str, int] = {
    "AAC": 20, "GAAC": 5, "CKSAAP type 1": 1600, "DPC type 1": 400,
    "DPC type 2": 400, "TPC type 1": 8000, "TPC type 2": 8000,
    "CTDC": 39, "CTDT": 39, "CTDD": 195, "CTriad": 343, "KNN": 240,
    "Geary": 24, "Moran": 24, "NMBroto": 24, "AC": 24, "CC": 168,
    "SOCNumber": 6, "QSOrder": 40, "PAAC": 23, "ZScale": 23, "AAIndex": 23,
}

_AA = "ACDEFGHIKLMNPQRSTVWY"
_PATCHED = False


def alias(desc: str) -> str:
    """Return a filename-safe short name for a descriptor."""
    return ALIASES.get(desc, desc.replace(" ", "_"))


def _fast_tpc(self, normalized: bool = True) -> bool:
    """Drop-in replacement for ``iProtein._TPC`` — identical output, O(L) per
    sequence instead of O(8000^2). Tolerates non-canonical residues by skipping
    any tripeptide that contains one."""
    import re

    aaidx = {aa: i for i, aa in enumerate(_AA)}
    columns = [f"TPC_{a}{b}{c}" for a in _AA for b in _AA for c in _AA]
    rows, index = [], []
    for entry in self.fasta_list:
        name, sequence = entry[0], re.sub("-", "", entry[1])
        tmp = [0] * 8000
        total = 0
        for j in range(len(sequence) - 2):
            a = aaidx.get(sequence[j])
            b = aaidx.get(sequence[j + 1])
            c = aaidx.get(sequence[j + 2])
            if a is None or b is None or c is None:
                continue
            tmp[a * 400 + b * 20 + c] += 1
            total += 1
        if normalized and total:
            inv = 1.0 / total
            tmp = [v * inv for v in tmp]
        rows.append(tmp)
        index.append(name)
    self.encodings = pd.DataFrame(np.asarray(rows, dtype=float), columns=columns, index=index)
    return True


def _install_speed_patches() -> None:
    """Monkeypatch iFeatureOmega's pathologically slow TPC normalization."""
    global _PATCHED
    if _PATCHED:
        return
    import iFeatureOmegaCLI

    iFeatureOmegaCLI.iProtein._TPC = _fast_tpc
    _PATCHED = True


def _get_available_descriptors(protein) -> set[str]:
    """Return the set of descriptor names this iFeatureOmegaCLI version supports."""
    # Access the private __cmd_dict through name-mangling
    for attr in ("_iProtein__cmd_dict", "__cmd_dict"):
        d = getattr(protein, attr, None)
        if d is not None:
            return set(d.keys())
    return set()


def _resolve_descriptors(
    protein, requested: list[str], verbose: bool = True,
) -> list[str]:
    """Check which requested descriptors are actually available in this
    iFeatureOmegaCLI version and warn about missing ones. Returns the
    filtered list of available descriptors."""
    available = _get_available_descriptors(protein)
    if not available:
        if verbose:
            print("[warn] could not inspect available descriptors — "
                  "trying all requested names", flush=True)
        return requested
    resolved: list[str] = []
    for desc in requested:
        if desc in available:
            resolved.append(desc)
        else:
            if verbose:
                print(f"[skip] {desc:16s} — not in this iFeatureOmegaCLI "
                      f"version's descriptor list", flush=True)
    if verbose and len(resolved) < len(requested):
        missing = set(requested) - set(resolved)
        print(f"[info] {len(resolved)}/{len(requested)} descriptors available; "
              f"missing: {sorted(missing)}", flush=True)
        print(f"[info] available descriptors: {sorted(available)}", flush=True)
    return resolved


def extract_all_features(
    fasta_path: str,
    output_dir: str,
    descriptors: list[str] = DESCRIPTORS,
    verbose: bool = True,
    write_per_descriptor: bool = True,
) -> pd.DataFrame:
    """Run iFeatureOmega for each descriptor and return the concatenated matrix.

    Reads each descriptor straight from the in-memory ``protein.encodings`` (no
    CSV round-trip). Columns are namespaced ``{alias}::{col}`` to stay unique
    across groups. A failing descriptor is logged and skipped; the success/failure
    lists are stored in ``df.attrs``.
    """
    import iFeatureOmegaCLI

    _install_speed_patches()
    os.makedirs(output_dir, exist_ok=True)
    protein = iFeatureOmegaCLI.iProtein(fasta_path)

    descriptors = _resolve_descriptors(protein, list(descriptors), verbose)

    frames: list[pd.DataFrame] = []
    ok: list[str] = []
    failed: dict[str, str] = {}

    for desc in descriptors:
        try:
            # Null out encodings so we can detect silent failures —
            # iFeatureOmega's get_descriptor() prints "The descriptor type
            # does not exist" but doesn't raise or set encodings to None.
            protein.encodings = None
            protein.get_descriptor(desc)
            enc = protein.encodings
            if enc is None:
                raise RuntimeError("descriptor not recognized by this iFeatureOmegaCLI version")
            n_feat = enc.shape[1]
            min_expected = _MIN_FEATURES.get(desc, 1)
            if n_feat < min_expected:
                raise RuntimeError(
                    f"got {n_feat} features, expected >= {min_expected} "
                    f"(stale encoding from previous descriptor?)")
            df = enc.copy()
            df.columns = [f"{alias(desc)}::{c}" for c in df.columns]
            if write_per_descriptor:
                df.to_csv(os.path.join(output_dir, f"{alias(desc)}.csv"))
            frames.append(df)
            ok.append(desc)
            if verbose:
                print(f"[ok]   {desc:16s} -> {df.shape[1]:>5d} features", flush=True)
        except Exception as e:  # noqa: BLE001 - one bad descriptor must not abort
            failed[desc] = str(e)
            if verbose:
                print(f"[FAIL] {desc:16s} -> {e}", flush=True)

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
    drop_duplicate_cols: bool = True,
) -> pd.DataFrame:
    """Join the feature matrix to labels on ``seq_id`` and clean it.

    Steps: align on index, fill any NaNs with 0, drop constant columns (zero
    variance) and exact-duplicate columns, append ``label``, and write
    ``out_path``. Returns the saved DataFrame (indexed by ``seq_id``).
    """
    labels = pd.read_csv(labels_csv).set_index("seq_id")["label"]

    X = feature_df
    common = X.index.intersection(labels.index)
    X = X.loc[common]
    y = labels.loc[common]

    # iFeature encodings are already float; coerce only if something slipped through.
    if not all(str(t).startswith(("float", "int")) for t in X.dtypes):
        X = X.apply(pd.to_numeric, errors="coerce")
    n_nan = int(X.isna().sum().sum())
    if n_nan:
        X = X.fillna(0.0)

    if drop_constant:
        X = X.loc[:, X.std(axis=0) > 0]
    if drop_duplicate_cols:
        X = X.loc[:, ~X.T.duplicated()]

    master = X.copy()
    master["label"] = y.values
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    master.to_csv(out_path)

    master.attrs["n_nan_filled"] = n_nan
    master.attrs["n_features"] = X.shape[1]
    return master
