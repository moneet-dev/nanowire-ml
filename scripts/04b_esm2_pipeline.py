"""
04b_esm2_pipeline.py — Phase 2 (PRIMARY): ESM-2 protein language model pipeline.

ESM-2 is the PRIMARY feature extraction method for this study. iFeature (Phase 3)
serves as the traditional-baseline comparison to Raya et al. 2025.

Key contributions over 04b v1:
  1. 5-fold stratified CV — aligned with ablation study; more robust than 70/30 split
  2. Layer-wise probing — identifies which transformer layer captures nanowire signal
     (single forward pass extracts layers 6/12/18/24/30/33 simultaneously)
  3. UMAP visualization — confirms linear separability in ESM-2 embedding space
  4. Saves OOF predictions (esm2_cv_preds.csv) for McNemar's test in Phase 4c
  5. Supports nr and redundant variants via CLI arg

ESM-2 model: esm2_t33_650M_UR50D (650 M params, 33 transformer layers, 1280-dim)
Mean-pool over residue positions at layer 33; layer probing covers 6/12/18/24/30/33.

Runtime estimate on T4: ~3 min embedding, ~4 min CV battery, ~1 min layer probing,
~3 min UMAP = ~11 min total.

Outputs:
  data/features/esm2_embeddings[_redundant].npy      — (N×1280) float32, gitignored
  data/features/esm2_layer_embs[_redundant].npz      — per-layer arrays, gitignored
  results/metrics/esm2_cv_table[_redundant].csv      — classifier battery (5-fold CV)
  results/metrics/esm2_layer_table[_redundant].csv   — per-layer probing (LR)
  results/metrics/esm2_cv_preds[_redundant].csv      — OOF predictions for McNemar test
  results/figures/esm2_umap[_redundant].png          — UMAP coloured by class
  results/figures/esm2_layer_probing[_redundant].png — ROC-AUC / ECE vs layer depth
  results/figures/esm2_cv_roc_pr[_redundant].png     — ROC + PR curves (OOF)

Usage: python scripts/04b_esm2_pipeline.py [nr|redundant]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import eval_utils as eu  # noqa: E402

RAW  = REPO / "data" / "raw"
FEAT = REPO / "data" / "features"
MET  = REPO / "results" / "metrics"
FIGS = REPO / "results" / "figures"

PROBE_LAYERS = [6, 12, 18, 24, 30, 33]
CV           = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
BATCH_SIZE   = 4   # sequences per ESM-2 forward pass; lower if OOM on smaller GPU
MAX_LEN      = 1022  # ESM-2 token limit (excluding BOS/EOS)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_sequences(fasta_path: Path) -> tuple[list[str], list[str]]:
    from Bio import SeqIO
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    return [r.id for r in records], [str(r.seq) for r in records]


def align_labels(y_series: pd.Series, seq_ids: list[str]) -> np.ndarray:
    """Return label array aligned to seq_ids order."""
    missing = [sid for sid in seq_ids if sid not in y_series.index]
    if missing:
        raise KeyError(f"{len(missing)} seq_ids not in labels CSV: {missing[:5]}")
    return np.array([y_series[sid] for sid in seq_ids])


# ── ESM-2 embedding ───────────────────────────────────────────────────────────

def generate_esm2_embeddings(
    fasta_path: Path,
    emb_path: Path,
    layer_path: Path,
    seq_ids: list[str],
    sequences: list[str],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Extract mean-pooled ESM-2 embeddings at PROBE_LAYERS in a single forward pass."""
    import torch
    import esm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading ESM-2 model (esm2_t33_650M_UR50D) on {device}...", flush=True)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval().to(device)

    layer_acc: dict[int, list] = {L: [] for L in PROBE_LAYERS}
    t0 = time.time()
    n = len(sequences)

    for start in range(0, n, BATCH_SIZE):
        batch_ids  = seq_ids[start : start + BATCH_SIZE]
        batch_seqs = sequences[start : start + BATCH_SIZE]
        data = [(sid, seq[:MAX_LEN]) for sid, seq in zip(batch_ids, batch_seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with torch.no_grad():
            out = model(tokens, repr_layers=PROBE_LAYERS, return_contacts=False)

        for i, seq in enumerate(batch_seqs):
            seq_len = min(len(seq), MAX_LEN)
            for L in PROBE_LAYERS:
                rep = out["representations"][L][i, 1 : seq_len + 1, :]
                layer_acc[L].append(rep.mean(dim=0).cpu().numpy())

        done = start + len(batch_seqs)
        if (start // BATCH_SIZE) % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / done * (n - done) if done else 0
            print(f"  {done}/{n} [{elapsed:.0f}s, ETA {eta:.0f}s]", flush=True)

    layer_arrays = {L: np.stack(v, axis=0) for L, v in layer_acc.items()}
    X_l33 = layer_arrays[33]

    FEAT.mkdir(parents=True, exist_ok=True)
    np.save(str(emb_path), X_l33)
    np.savez_compressed(
        str(layer_path),
        **{f"layer_{L}": layer_arrays[L] for L in PROBE_LAYERS},
    )
    dt = time.time() - t0
    print(
        f"  saved {emb_path.name} {X_l33.shape}  +  "
        f"{layer_path.name} ({len(PROBE_LAYERS)} layers)  [{dt:.0f}s]",
        flush=True,
    )
    return X_l33, layer_arrays


# ── classifier battery ────────────────────────────────────────────────────────

def _build_models() -> dict[str, Pipeline]:
    return {
        "LR": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, solver="saga")),
        ]),
        "RF": Pipeline([
            ("sc", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ]),
        "XGBoost": Pipeline([
            ("sc", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                random_state=42, eval_metric="logloss",
                tree_method="hist", n_jobs=-1,
            )),
        ]),
        "XGB+Isotonic": Pipeline([
            ("sc", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                XGBClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.1,
                    random_state=42, eval_metric="logloss",
                    tree_method="hist", n_jobs=-1,
                ),
                method="isotonic", cv=3,
            )),
        ]),
        "Stacking": Pipeline([
            ("sc", StandardScaler()),
            ("clf", StackingClassifier(
                estimators=[
                    ("rf",  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
                    ("xgb", XGBClassifier(
                        n_estimators=100, max_depth=5, learning_rate=0.1,
                        random_state=42, eval_metric="logloss",
                        tree_method="hist", n_jobs=-1,
                    )),
                ],
                final_estimator=LogisticRegression(max_iter=2000),
                cv=3, n_jobs=-1,
            )),
        ]),
    }


def run_classifier_battery(
    X: np.ndarray, y: np.ndarray
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """5-fold CV classifier battery. Returns metric rows and OOF probability arrays."""
    rows: list[dict] = []
    oof_probas: dict[str, np.ndarray] = {}

    for name, pipe in _build_models().items():
        t0 = time.time()
        oof = cross_val_predict(pipe, X, y, cv=CV, method="predict_proba")[:, 1]
        pred = (oof >= 0.5).astype(int)
        ev = eu.full_eval(y, pred, oof)
        roc_ci = eu.bootstrap_metric_ci(y, oof, "roc_auc")
        pr_ci  = eu.bootstrap_metric_ci(y, oof, "pr_auc")
        dt = time.time() - t0

        rows.append({
            "model":      name,
            "n_features": X.shape[1],
            "accuracy":   round(ev["accuracy"],  4),
            "roc_auc":    round(ev["roc_auc"],   4),
            "roc_auc_ci": f"[{roc_ci[1]:.4f}, {roc_ci[2]:.4f}]",
            "pr_auc":     round(ev["pr_auc"],    4),
            "pr_auc_ci":  f"[{pr_ci[1]:.4f}, {pr_ci[2]:.4f}]",
            "ece":        round(ev["ece"],        4),
            "seconds":    round(dt, 1),
        })
        oof_probas[name] = oof
        print(
            f"  {name:15s}  roc={ev['roc_auc']:.4f}  "
            f"pr={ev['pr_auc']:.4f}  ece={ev['ece']:.4f}  [{dt:.0f}s]",
            flush=True,
        )

    return rows, oof_probas


# ── layer probing ─────────────────────────────────────────────────────────────

def probe_layers(
    layer_arrays: dict[int, np.ndarray], y: np.ndarray
) -> list[dict]:
    """Fit LR on mean-pooled embeddings from each layer. Identifies which layer
    captures the most classification-relevant information."""
    rows: list[dict] = []
    for L in PROBE_LAYERS:
        X_L = layer_arrays[L]
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, solver="saga")),
        ])
        t0 = time.time()
        oof = cross_val_predict(pipe, X_L, y, cv=CV, method="predict_proba")[:, 1]
        roc = float(roc_auc_score(y, oof))
        pr  = float(average_precision_score(y, oof))
        ece = float(eu.compute_ece(y, oof))
        dt  = time.time() - t0
        rows.append({"layer": L, "roc_auc": round(roc, 4),
                     "pr_auc": round(pr, 4), "ece": round(ece, 4)})
        print(
            f"  layer {L:2d}  roc={roc:.4f}  pr={pr:.4f}  ece={ece:.4f}  [{dt:.0f}s]",
            flush=True,
        )
    return rows


# ── figures ───────────────────────────────────────────────────────────────────

def plot_umap(X: np.ndarray, y: np.ndarray, out: Path) -> None:
    try:
        from umap import UMAP
    except ImportError:
        print("  [umap] umap-learn not installed — skipping UMAP plot", flush=True)
        return

    print("  fitting UMAP (n_neighbors=15, min_dist=0.1, random_state=42)...", flush=True)
    t0 = time.time()
    X_sc = StandardScaler().fit_transform(X)
    X2   = UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_jobs=1).fit_transform(X_sc)
    print(f"  done [{time.time()-t0:.0f}s]", flush=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = {1: ("#2196F3", "Nanowire (+)"), 0: ("#FF9800", "RecA (−)")}
    for cls in [1, 0]:
        mask = y == cls
        c, label = palette[cls]
        ax.scatter(X2[mask, 0], X2[mask, 1], c=c,
                   label=f"{label}  (n={int(mask.sum())})",
                   s=16, alpha=0.75, linewidths=0)
    ax.set_title("UMAP projection — ESM-2 layer-33 embeddings", fontsize=13)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  UMAP -> {out.name}", flush=True)


def plot_layer_probing(rows: list[dict], out: Path) -> None:
    layers = [r["layer"]   for r in rows]
    rocs   = [r["roc_auc"] for r in rows]
    prs    = [r["pr_auc"]  for r in rows]
    eces   = [r["ece"]     for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(layers, rocs, "o-", color="#2196F3", lw=2, ms=7, label="ROC-AUC")
    ax1.plot(layers, prs,  "s--", color="#4CAF50", lw=2, ms=7, label="PR-AUC")
    ax1.set_xlabel("ESM-2 Transformer Layer", fontsize=11)
    ax1.set_ylabel("5-Fold CV Score", fontsize=11)
    ax1.set_title("Layer Probing — LR Classifier")
    ax1.set_xticks(layers)
    ax1.set_ylim(max(0.9, min(rocs + prs) - 0.01), 1.002)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    ax2.plot(layers, eces, "D-", color="#F44336", lw=2, ms=7, label="ECE")
    ax2.set_xlabel("ESM-2 Transformer Layer", fontsize=11)
    ax2.set_ylabel("Expected Calibration Error (ECE)", fontsize=11)
    ax2.set_title("Layer Probing — Calibration (ECE ↓)")
    ax2.set_xticks(layers)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    for ax in (ax1, ax2):
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  layer probing figure -> {out.name}", flush=True)


def plot_roc_pr(y: np.ndarray, oof_probas: dict[str, np.ndarray], out: Path) -> None:
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for name, proba in oof_probas.items():
        RocCurveDisplay.from_predictions(y, proba, name=name, ax=ax1)
        PrecisionRecallDisplay.from_predictions(y, proba, name=name, ax=ax2)
    ax1.set_title("ROC — ESM-2 (5-Fold CV OOF)")
    ax1.legend(loc="lower right", fontsize=8)
    ax2.set_title("PR — ESM-2 (5-Fold CV OOF)")
    ax2.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  ROC/PR figure -> {out.name}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    variant = (sys.argv[1:] or ["nr"])[0]
    if variant not in ("nr", "redundant"):
        print(f"Unknown variant '{variant}'. Use 'nr' or 'redundant'.", flush=True)
        sys.exit(1)

    suffix     = "" if variant == "nr" else "_redundant"
    fasta_name = "merged.fasta"            if variant == "nr" else "merged_redundant.fasta"
    label_name = "labels.csv"              if variant == "nr" else "labels_redundant.csv"

    fasta_path = RAW / fasta_name
    labels_df  = pd.read_csv(str(RAW / label_name))
    y_series   = labels_df.set_index("seq_id")["label"]

    emb_path   = FEAT / f"esm2_embeddings{suffix}.npy"
    layer_path = FEAT / f"esm2_layer_embs{suffix}.npz"

    # ── 1. Embeddings ─────────────────────────────────────────────────────────
    seq_ids, sequences = load_sequences(fasta_path)

    if emb_path.exists() and layer_path.exists():
        print(f"Loading cached embeddings: {emb_path.name}  {layer_path.name}", flush=True)
        X_esm       = np.load(str(emb_path))
        npz         = np.load(str(layer_path))
        layer_arrays = {int(k.split("_")[1]): npz[k] for k in npz.files}
    else:
        import torch
        if not torch.cuda.is_available():
            print("WARNING: No GPU detected — ESM-2 will be very slow on CPU.", flush=True)
        print(f"Generating ESM-2 embeddings for variant='{variant}' ...", flush=True)
        X_esm, layer_arrays = generate_esm2_embeddings(
            fasta_path, emb_path, layer_path, seq_ids, sequences)

    y = align_labels(y_series, seq_ids)
    print(
        f"\nESM-2: X={X_esm.shape}  pos={int(y.sum())}  neg={int((y == 0).sum())}\n",
        flush=True,
    )

    MET.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    # ── 2. Classifier battery (5-fold CV) ─────────────────────────────────────
    print("=== Classifier Battery (5-Fold Stratified CV) ===", flush=True)
    rows, oof_probas = run_classifier_battery(X_esm, y)
    cv_df = pd.DataFrame(rows)
    cv_path = MET / f"esm2_cv_table{suffix}.csv"
    cv_df.to_csv(str(cv_path), index=False)
    print(f"\nwrote {cv_path.relative_to(REPO)}", flush=True)

    # Save OOF predictions for best-ECE model (for McNemar's test in 04c)
    best_row  = min(rows, key=lambda r: r["ece"])
    best_name = best_row["model"]
    best_oof  = oof_probas[best_name]
    preds_df  = pd.DataFrame({
        "seq_id":     seq_ids,
        "true_label": y,
        "esm2_pred":  (best_oof >= 0.5).astype(int),
        "esm2_proba": best_oof,
        "best_model": best_name,
    })
    preds_path = MET / f"esm2_cv_preds{suffix}.csv"
    preds_df.to_csv(str(preds_path), index=False)
    print(f"OOF predictions (best ECE model: {best_name}) -> {preds_path.name}", flush=True)

    # ── 3. Layer probing ──────────────────────────────────────────────────────
    print("\n=== Layer Probing (LR, 5-Fold CV) ===", flush=True)
    layer_rows = probe_layers(layer_arrays, y)
    layer_df   = pd.DataFrame(layer_rows)
    layer_path_csv = MET / f"esm2_layer_table{suffix}.csv"
    layer_df.to_csv(str(layer_path_csv), index=False)
    best_layer = layer_rows[int(np.argmax([r["roc_auc"] for r in layer_rows]))]["layer"]
    print(f"Best layer by ROC-AUC: {best_layer}", flush=True)

    # ── 4. Figures ────────────────────────────────────────────────────────────
    print("\n=== Figures ===", flush=True)
    plot_umap(X_esm, y, FIGS / f"esm2_umap{suffix}.png")
    plot_layer_probing(layer_rows, FIGS / f"esm2_layer_probing{suffix}.png")
    plot_roc_pr(y, oof_probas, FIGS / f"esm2_cv_roc_pr{suffix}.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"ESM-2 pipeline complete  (variant={variant})", flush=True)
    print(cv_df[["model", "roc_auc", "pr_auc", "ece"]].to_string(index=False), flush=True)
    print(f"\nBest ECE model: {best_name}  ECE={best_row['ece']:.4f}", flush=True)
    print(f"Best ROC-AUC layer: {best_layer}", flush=True)


if __name__ == "__main__":
    main()
