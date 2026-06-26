"""
04b_esm2_pipeline.py — Phase 5: ESM-2 embedding pipeline.

Generates per-protein embeddings from Meta's ESM-2 protein language model
(esm2_t33_650M_UR50D, 1280-dim) and runs the same classifier battery as Phase 4
(RF, XGBoost, LR, calibrated XGBoost, stacking) on ESM-2 features instead of
iFeature descriptors. Produces a head-to-head comparison table.

**Requires T4 GPU runtime.** ESM-2 inference on ~1664 sequences takes ~5 min
on T4; classifiers are fast on 1280 features.

Writes:
  - data/features/esm2_embeddings.npy           (gitignored)
  - results/metrics/esm2_table.csv               (tracked)
  - results/metrics/esm2_vs_ifeature.csv          (tracked)
  - results/figures/esm2_roc_curves.png           (gitignored)

Usage: python scripts/04b_esm2_pipeline.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import eval_utils as eu  # noqa: E402

RAW = os.path.join(REPO, "data", "raw")
FEAT = os.path.join(REPO, "data", "features")
METRICS = os.path.join(REPO, "results", "metrics")
FIGURES = os.path.join(REPO, "results", "figures")


def generate_esm2_embeddings(fasta_path: str, out_path: str) -> np.ndarray:
    """Generate mean-pooled ESM-2 embeddings (1280-dim) for each sequence."""
    import esm

    print("Loading ESM-2 model (esm2_t33_650M_UR50D)...", flush=True)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  device: {device}", flush=True)

    from Bio import SeqIO
    records = list(SeqIO.parse(fasta_path, "fasta"))
    print(f"  {len(records)} sequences to embed", flush=True)

    # ESM-2 has a max token limit — truncate long sequences to 1022 residues
    MAX_LEN = 1022
    embeddings = []
    batch_size = 4  # small batches to avoid OOM on T4 (16 GB)
    t0 = time.time()

    for start in range(0, len(records), batch_size):
        batch_recs = records[start:start + batch_size]
        data = [(r.id, str(r.seq)[:MAX_LEN]) for r in batch_recs]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = model(tokens, repr_layers=[33], return_contacts=False)
        # Mean-pool over residue positions (exclude BOS/EOS tokens)
        for i, r in enumerate(batch_recs):
            seq_len = min(len(str(r.seq)), MAX_LEN)
            rep = results["representations"][33][i, 1:seq_len + 1, :]
            embeddings.append(rep.mean(dim=0).cpu().numpy())

        if (start // batch_size) % 25 == 0:
            elapsed = time.time() - t0
            done = start + len(batch_recs)
            eta = elapsed / done * (len(records) - done) if done else 0
            print(f"  embedded {done}/{len(records)} ({elapsed:.0f}s, ETA {eta:.0f}s)",
                  flush=True)

    embeddings_np = np.stack(embeddings)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, embeddings_np)
    dt = time.time() - t0
    print(f"  done: shape={embeddings_np.shape}, saved to {out_path} [{dt:.0f}s]", flush=True)
    return embeddings_np


def run_classifiers(X, y, Xtr, Xte, ytr, yte, tag: str) -> list[dict]:
    """Run the classifier battery and return evaluation rows."""
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    models = {
        "RF":              RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost":         XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                         random_state=42, eval_metric="logloss",
                                         tree_method="hist", n_jobs=-1),
        "LR":              LogisticRegression(C=1.0, max_iter=2000, solver="saga"),
        "XGBoost+Isotonic": CalibratedClassifierCV(
                               XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                             random_state=42, eval_metric="logloss",
                                             tree_method="hist", n_jobs=-1),
                               method="isotonic", cv=5),
        "Stacking":        StackingClassifier(
                               estimators=[
                                   ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
                                   ("xgb", XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                                         random_state=42, eval_metric="logloss",
                                                         tree_method="hist", n_jobs=-1)),
                               ],
                               final_estimator=LogisticRegression(max_iter=2000),
                               cv=5, passthrough=False, n_jobs=-1),
    }

    rows = []
    probas = {}
    for name, m in models.items():
        t0 = time.time()
        m.fit(Xtr_s, ytr)
        proba = m.predict_proba(Xte_s)[:, 1]
        pred = (proba >= 0.5).astype(int)
        ev = eu.full_eval(yte, pred, proba)
        roc_ci = eu.bootstrap_metric_ci(yte, proba, "roc_auc")
        pr_ci = eu.bootstrap_metric_ci(yte, proba, "pr_auc")
        dt = time.time() - t0

        rows.append({
            "features": tag, "model": name,
            "accuracy": round(ev["accuracy"], 4),
            "roc_auc": round(ev["roc_auc"], 4),
            "roc_auc_ci": f"[{roc_ci[1]:.4f}, {roc_ci[2]:.4f}]",
            "pr_auc": round(ev["pr_auc"], 4),
            "pr_auc_ci": f"[{pr_ci[1]:.4f}, {pr_ci[2]:.4f}]",
            "ece": round(ev["ece"], 4),
        })
        probas[name] = proba
        print(f"  {tag:10s} {name:20s} roc={ev['roc_auc']:.4f} "
              f"pr={ev['pr_auc']:.4f} ece={ev['ece']:.4f} [{dt:.0f}s]", flush=True)

    return rows, probas


def main() -> None:
    # ── 1. Generate or load ESM-2 embeddings ──────────────────────────────────
    emb_path = os.path.join(FEAT, "esm2_embeddings.npy")
    fasta_path = os.path.join(RAW, "merged.fasta")
    labels_path = os.path.join(RAW, "labels.csv")

    if not torch.cuda.is_available():
        print("WARNING: No GPU detected. ESM-2 inference will be very slow on CPU.", flush=True)

    if os.path.exists(emb_path):
        print(f"ESM-2 embeddings found at {emb_path} — loading.", flush=True)
        X_esm = np.load(emb_path)
    else:
        X_esm = generate_esm2_embeddings(fasta_path, emb_path)

    labels = pd.read_csv(labels_path)
    y = labels["label"].values
    print(f"\nESM-2: X={X_esm.shape}, pos={int(y.sum())}, neg={int((y == 0).sum())}\n",
          flush=True)

    # ── 2. Train/test split (same seed as all other phases) ───────────────────
    Xtr, Xte, ytr, yte = train_test_split(
        X_esm, y, test_size=0.30, random_state=6, stratify=y)

    # ── 3. Run classifiers on ESM-2 ──────────────────────────────────────────
    print("=== ESM-2 classifiers ===", flush=True)
    esm_rows, esm_probas = run_classifiers(X_esm, y, Xtr, Xte, ytr, yte, "ESM-2")

    os.makedirs(METRICS, exist_ok=True)
    esm_df = pd.DataFrame(esm_rows)
    esm_df.to_csv(os.path.join(METRICS, "esm2_table.csv"), index=False)

    # ── 4. Load iFeature results for comparison ──────────────────────────────
    ifeature_path = os.path.join(METRICS, "extended_table.csv")
    if os.path.exists(ifeature_path):
        ifeat_df = pd.read_csv(ifeature_path)
        ifeat_df["features"] = "iFeature"

        # Build comparison table
        compare = pd.concat([
            ifeat_df[["features", "model", "roc_auc", "pr_auc", "ece"]],
            esm_df[["features", "model", "roc_auc", "pr_auc", "ece"]],
        ], ignore_index=True)

        pivot = compare.pivot_table(
            index="model", columns="features",
            values=["roc_auc", "pr_auc", "ece"], aggfunc="first")
        pivot.to_csv(os.path.join(METRICS, "esm2_vs_ifeature.csv"))
        print("\n=== ESM-2 vs iFeature comparison ===", flush=True)
        print(pivot.to_string(), flush=True)
    else:
        print("\niFeature extended_table.csv not found — run Phase 4 first for comparison.",
              flush=True)

    # ── 5. ROC + PR curves ───────────────────────────────────────────────────
    os.makedirs(FIGURES, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for name, proba in esm_probas.items():
        RocCurveDisplay.from_predictions(yte, proba, name=name, ax=ax1)
        PrecisionRecallDisplay.from_predictions(yte, proba, name=name, ax=ax2)
    ax1.set_title("ROC — ESM-2 Features")
    ax1.legend(loc="lower right", fontsize=8)
    ax2.set_title("PR — ESM-2 Features")
    ax2.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "esm2_roc_pr_curves.png"), dpi=150)
    plt.close(fig)

    print(f"\nfigures -> results/figures/esm2_roc_pr_curves.png", flush=True)
    print("Phase 5 (ESM-2) complete.", flush=True)


if __name__ == "__main__":
    main()
