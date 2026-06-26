"""
03_reproduce_baseline.py — Phase 3: reproduce Raya et al. 2025 Table 2.

Trains SVM, RF, XGBoost, LR, and MLP on the iFeature master matrix using a
70/30 stratified hold-out split (random_state=6, matching the official repo) and
reports test-set metrics alongside the paper's published numbers. Adds PR-AUC and
ECE (both omitted by the paper). Writes
results/metrics/reproduction_table[_<variant>].csv.

Usage (from repo root):
  python scripts/03_reproduce_baseline.py            # paper-faithful redundant set
  python scripts/03_reproduce_baseline.py nr         # non-redundant set
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import eval_utils as eu  # noqa: E402

FEAT = os.path.join(REPO, "data", "features")
METRICS = os.path.join(REPO, "results", "metrics")

MASTER = {"redundant": "master_features_redundant.csv", "nr": "master_features.csv"}

# Paper Table 2 (Raya et al. 2025): (accuracy, ROC-AUC).
PAPER = {
    "SVM":     (0.9487, 0.9696),
    "RF":      (0.9669, 0.9826),
    "XGBoost": (0.9665, 0.9857),
    "LR":      (0.9605, 0.9903),
    "MLP":     (0.9613, 0.9920),
}


def build_models() -> dict:
    return {
        "SVM":     SVC(probability=True, random_state=42),
        "RF":      RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss",
                                 tree_method="hist", n_jobs=-1),
        "LR":      LogisticRegression(max_iter=2000),
        "MLP":     MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    }


def main() -> None:
    variant = (sys.argv[1:] or ["redundant"])[0]
    if variant not in MASTER:
        print(f"unknown variant '{variant}'; choices: {list(MASTER)}")
        sys.exit(1)
    master_path = os.path.join(FEAT, MASTER[variant])
    if not os.path.exists(master_path):
        print(f"ERROR: {master_path} not found. Run "
              f"`python scripts/02_feature_extraction.py {variant}` first.")
        sys.exit(1)

    df = pd.read_csv(master_path, index_col=0)
    y = df["label"].values
    X = df.drop(columns=["label"]).values
    print(f"variant '{variant}': X={X.shape}, "
          f"pos={int(y.sum())}, neg={int((y == 0).sum())}\n", flush=True)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.30, random_state=6, stratify=y)

    rows = []
    for name, clf in build_models().items():
        t0 = time.time()
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = eu.full_eval(yte, pred, proba)
        dt = time.time() - t0

        p_acc, p_roc = PAPER[name]
        rows.append({
            "model": name,
            "paper_acc": p_acc, "test_acc": round(m["accuracy"], 4),
            "test_precision": round(m["precision"], 4),
            "test_recall": round(m["recall"], 4), "test_f1": round(m["f1"], 4),
            "paper_roc_auc": p_roc, "test_roc_auc": round(m["roc_auc"], 4),
            "test_pr_auc": round(m["pr_auc"], 4), "test_ece": round(m["ece"], 4),
            "fit_seconds": round(dt, 1),
        })
        print(f"  {name:8s} acc={m['accuracy']:.4f} (paper {p_acc})  "
              f"roc_auc={m['roc_auc']:.4f} (paper {p_roc})  "
              f"pr_auc={m['pr_auc']:.4f}  ece={m['ece']:.4f}  [{dt:.0f}s]", flush=True)

    out = pd.DataFrame(rows)
    suffix = "" if variant == "redundant" else f"_{variant}"
    out_path = os.path.join(METRICS, f"reproduction_table{suffix}.csv")
    out.to_csv(out_path, index=False)
    print(f"\nwrote {os.path.relpath(out_path, REPO)}")


if __name__ == "__main__":
    main()
