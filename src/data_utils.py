"""
data_utils.py — FASTA loading, labeling, and dataset assembly for NanowireML.

Designed to be generic (P2-reusable): every function accepts arbitrary FASTA
paths and makes no nanowire-specific assumption beyond the default labeling
convention (positives -> 1, negatives -> 0).
"""
from __future__ import annotations

import os
from collections import Counter

import pandas as pd
from Bio import SeqIO

#: The 20 canonical amino acids, in a fixed order.
CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
#: Aromatic residues — relevant to the e-pili "9% aromatic" rule.
AROMATIC_AA = "FYW"


def load_fasta(path: str) -> list[tuple[str, str, str]]:
    """Return a list of ``(record_id, description, sequence)`` for a FASTA file.

    Sequences are upper-cased. Gaps/whitespace are stripped by BioPython.
    """
    out: list[tuple[str, str, str]] = []
    for rec in SeqIO.parse(path, "fasta"):
        out.append((rec.id, rec.description, str(rec.seq).upper()))
    return out


def build_dataset(
    positive_files: list[str],
    negative_files: list[str],
    negative_cap: int | None = 999,
    dedup: bool = True,
) -> pd.DataFrame:
    """Assemble a labeled protein dataset.

    Positives (label 1) come from ``positive_files``; negatives (label 0) from
    ``negative_files``, truncated to the first ``negative_cap`` records in file
    order (matching the official repo's ``if i >= 999: break``). Clean,
    collision-free ids (``pos_0001`` / ``neg_0001``) are assigned so the feature
    matrix can be joined back to labels unambiguously.

    Returns a DataFrame with columns:
    ``seq_id, source, label, length, sequence, orig_id``.
    The number of exact-duplicate sequences removed is stored in
    ``df.attrs['n_dropped_duplicates']``.
    """
    rows: list[dict] = []

    pidx = 0
    for f in positive_files:
        src = os.path.splitext(os.path.basename(f))[0]
        for rid, _desc, seq in load_fasta(f):
            rows.append(
                dict(seq_id=f"pos_{pidx:04d}", source=src, label=1,
                     length=len(seq), sequence=seq, orig_id=rid)
            )
            pidx += 1

    nidx = 0
    for f in negative_files:
        src = os.path.splitext(os.path.basename(f))[0]
        for rid, _desc, seq in load_fasta(f):
            if negative_cap is not None and nidx >= negative_cap:
                break
            rows.append(
                dict(seq_id=f"neg_{nidx:04d}", source=src, label=0,
                     length=len(seq), sequence=seq, orig_id=rid)
            )
            nidx += 1

    df = pd.DataFrame(rows)
    n_dropped = 0
    if dedup:
        before = len(df)
        df = df.drop_duplicates(subset="sequence").reset_index(drop=True)
        n_dropped = before - len(df)
    df.attrs["n_dropped_duplicates"] = n_dropped
    return df


def write_fasta(df: pd.DataFrame, path: str) -> None:
    """Write the merged dataset to FASTA using the clean ``seq_id`` as header."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        for _, r in df.iterrows():
            fh.write(f">{r.seq_id}\n{r.sequence}\n")


def write_labels(df: pd.DataFrame, path: str) -> None:
    """Write the label table (everything except the raw sequence)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df[["seq_id", "source", "label", "length", "orig_id"]].to_csv(path, index=False)


def aa_composition(seq: str) -> dict[str, float]:
    """Fractional amino-acid composition over the 20 canonical residues."""
    n = len(seq)
    if n == 0:
        return {aa: 0.0 for aa in CANONICAL_AA}
    c = Counter(seq)
    return {aa: c.get(aa, 0) / n for aa in CANONICAL_AA}


def aromatic_fraction(seq: str) -> float:
    """Fraction of aromatic residues (F/Y/W) — the e-pili conductivity signature."""
    if not seq:
        return 0.0
    return sum(seq.count(a) for a in AROMATIC_AA) / len(seq)


def noncanonical_residues(df: pd.DataFrame) -> dict[str, str]:
    """Map ``seq_id`` -> non-standard residues present (X/B/Z/U/O/*), for records
    that contain any. Useful because some iFeature descriptors expect the 20
    canonical AAs and may emit NaNs otherwise."""
    bad: dict[str, str] = {}
    canon = set(CANONICAL_AA)
    for _, r in df.iterrows():
        extra = sorted(set(r.sequence) - canon)
        if extra:
            bad[r.seq_id] = "".join(extra)
    return bad


def sanity_report(df: pd.DataFrame) -> dict:
    """Compute dataset-level sanity statistics (counts, length distribution,
    duplicate count, mean aromatic fraction per class)."""
    rep: dict = {
        "n_total": int(len(df)),
        "n_positive": int((df.label == 1).sum()),
        "n_negative": int((df.label == 0).sum()),
        "n_duplicate_sequences_dropped": int(df.attrs.get("n_dropped_duplicates", 0)),
        "n_unique_sequences": int(df.sequence.nunique()),
    }
    for lbl, name in [(1, "positive"), (0, "negative")]:
        L = df.loc[df.label == lbl, "length"]
        if len(L):
            rep[f"len_{name}_min"] = int(L.min())
            rep[f"len_{name}_median"] = float(L.median())
            rep[f"len_{name}_max"] = int(L.max())
            rep[f"len_{name}_mean"] = round(float(L.mean()), 1)
        arom = df.loc[df.label == lbl, "sequence"].map(aromatic_fraction)
        rep[f"aromatic_frac_{name}_mean"] = round(float(arom.mean()), 4)
    return rep
