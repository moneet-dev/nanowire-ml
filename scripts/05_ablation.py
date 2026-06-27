"""
05_ablation.py — Phase 6: feature group ablation study.

Systematically evaluates classifier performance on individual and combined
descriptor groups to determine which iFeature subsets drive performance.
This addresses Gap 4 (incomplete feature ablation) in the paper.

Tests these configurations with the best classifier (XGBoost, tuned params):
  - Each descriptor group alone (22 runs)
  - Biologically motivated combinations (DPC, CTD, DPC+CTD)
  - Top-100 SHAP features (from Phase 4)
  - All features (baseline)

Writes results/metrics/ablation_table.csv.

Usage: python scripts/05_ablation.py [nr|redundant]  (default: nr)
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import eval_utils as eu  # noqa: E402

FEAT = os.path.join(REPO, "data", "features")
METRICS = os.path.join(REPO, "results", "metrics")
FIGURES = os.path.join(REPO, "results", "figures")
MASTER = {"nr": "master_features.csv", "redundant": "master_features_redundant.csv"}

DESCRIPTOR_GROUPS = {
    "AAC": "AAC::", "GAAC": "GAAC::", "CKSAAP": "CKSAAP::",
    "DPC1": "DPC1::", "DPC2": "DPC2::", "TPC1": "TPC1::", "TPC2": "TPC2::",
    "CTDC": "CTDC::", "CTDT": "CTDT::", "CTDD": "CTDD::",
    "CTriad": "CTriad::", "KNN": "KNN::", "Geary": "Geary::",
    "Moran": "Moran::", "NMBroto": "NMBroto::", "AC": "AC::", "CC": "CC::",
    "SOCNumber": "SOCNumber::", "QSOrder": "QSOrder::", "PAAC": "PAAC::",
    "ZScale": "ZScale::", "AAIndex": "AAIndex::",
}

COMBO_CONFIGS = {
    "DPC_only":    ["DPC1", "DPC2"],
    "CTD_only":    ["CTDC", "CTDT", "CTDD"],
    "DPC+CTD":     ["DPC1", "DPC2", "CTDC", "CTDT", "CTDD"],
    "AAC+DPC+CTD": ["AAC", "DPC1", "DPC2", "CTDC", "CTDT", "CTDD"],
    "all":         list(DESCRIPTOR_GROUPS.keys()),
}


def select_columns(df: pd.DataFrame, groups: list[str]) -> np.ndarray:
    prefixes = [DESCRIPTOR_GROUPS[g] for g in groups]
    cols = [c for c in df.columns if c != "label" and any(c.startswith(p) for p in prefixes)]
    return df[cols].values, len(cols)


def get_xgb(variant: str) -> XGBClassifier:
    params_path = os.path.join(METRICS, f"best_params{'_' + variant if variant != 'nr' else ''}.json")
    params = {}
    if os.path.exists(params_path):
        with open(params_path) as f:
            params = json.load(f).get("XGBoost", {})
    return XGBClassifier(**params, random_state=42, eval_metric="logloss",
                         tree_method="hist", n_jobs=-1)


def evaluate_config(name: str, X: np.ndarray, y: np.ndarray, n_feat: int,
                    clf, cv) -> dict:
    t0 = time.time()
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    ev = eu.full_eval(y, pred, proba)
    roc_ci = eu.bootstrap_metric_ci(y, proba, "roc_auc")
    pr_ci = eu.bootstrap_metric_ci(y, proba, "pr_auc")
    dt = time.time() - t0
    print(f"  {name:20s} n={n_feat:>6d}  roc={ev['roc_auc']:.4f}  "
          f"pr={ev['pr_auc']:.4f}  ece={ev['ece']:.4f}  [{dt:.0f}s]", flush=True)
    return {
        "config": name, "n_features": n_feat,
        "accuracy": round(ev["accuracy"], 4),
        "roc_auc": round(ev["roc_auc"], 4),
        "roc_auc_ci": f"[{roc_ci[1]:.4f}, {roc_ci[2]:.4f}]",
        "pr_auc": round(ev["pr_auc"], 4),
        "pr_auc_ci": f"[{pr_ci[1]:.4f}, {pr_ci[2]:.4f}]",
        "ece": round(ev["ece"], 4),
    }


def main() -> None:
    variant = (sys.argv[1:] or ["nr"])[0]
    path = os.path.join(FEAT, MASTER[variant])
    df = pd.read_csv(path, index_col=0)
    y = df["label"].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgb = get_xgb(variant)

    suffix = "" if variant == "nr" else f"_{variant}"
    rows = []

    # Single descriptors
    print("=== Single descriptor groups ===", flush=True)
    for name, prefix in DESCRIPTOR_GROUPS.items():
        X_sub, n = select_columns(df, [name])
        if n == 0:
            continue
        rows.append(evaluate_config(name, X_sub, y, n, get_xgb(variant), cv))

    # Combinations
    print("\n=== Combinations ===", flush=True)
    for name, groups in COMBO_CONFIGS.items():
        X_sub, n = select_columns(df, groups)
        if n == 0:
            print(f"  {name:20s} SKIPPED — no matching features in matrix", flush=True)
            continue
        rows.append(evaluate_config(name, X_sub, y, n, get_xgb(variant), cv))

    # Top-100 SHAP features (if Phase 4 ran)
    shap_path = os.path.join(METRICS, f"shap_top100{suffix}.csv")
    if os.path.exists(shap_path):
        print("\n=== SHAP-selected features ===", flush=True)
        top_feats = pd.read_csv(shap_path)["feature"].tolist()
        avail = [f for f in top_feats if f in df.columns]
        if avail:
            X_sub = df[avail].values
            rows.append(evaluate_config("SHAP_top100", X_sub, y, len(avail),
                                        get_xgb(variant), cv))
            X_sub50 = df[avail[:50]].values
            rows.append(evaluate_config("SHAP_top50", X_sub50, y, len(avail[:50]),
                                        get_xgb(variant), cv))

    os.makedirs(METRICS, exist_ok=True)
    out = pd.DataFrame(rows)
    out_path = os.path.join(METRICS, f"ablation_table{suffix}.csv")
    out.to_csv(out_path, index=False)
    print(f"\nwrote {os.path.relpath(out_path, REPO)}", flush=True)

    # Bar chart
    os.makedirs(FIGURES, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    out_sorted = out.sort_values("roc_auc", ascending=True)
    colors = ["#2196F3" if n < 1000 else "#FF9800" if n < 5000 else "#4CAF50"
              for n in out_sorted["n_features"]]
    ax.barh(out_sorted["config"], out_sorted["roc_auc"], color=colors, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("5-Fold CV ROC-AUC")
    ax.set_title("Feature Group Ablation Study")
    ax.set_xlim(0.8, 1.005)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, f"ablation_bar{suffix}.png"), dpi=150)
    plt.close(fig)
    print(f"figure -> results/figures/ablation_bar{suffix}.png", flush=True)
    print("\nPhase 6 complete.", flush=True)


if __name__ == "__main__":
    main()
