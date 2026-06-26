"""
04_extended_pipeline.py — Phase 4: extended classifier pipeline.

Novel contributions beyond reproduction:
  1. Hyperparameter tuning (RandomizedSearchCV, 50 iterations, stratified 5-fold)
  2. Calibration (Platt scaling + isotonic regression) with ECE / reliability
  3. Stacking ensemble (RF + XGBoost + LR + SVM → LogisticRegression meta-learner)
  4. SHAP feature attribution (TreeExplainer on best XGBoost)
  5. Bootstrap confidence intervals on ROC-AUC and PR-AUC

Writes results/metrics/extended_table.csv and results/figures/*.

Usage: python scripts/04_extended_pipeline.py [nr|redundant]  (default: nr)
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
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import eval_utils as eu  # noqa: E402

FEAT = os.path.join(REPO, "data", "features")
METRICS = os.path.join(REPO, "results", "metrics")
FIGURES = os.path.join(REPO, "results", "figures")
MASTER = {"nr": "master_features.csv", "redundant": "master_features_redundant.csv"}


def load_data(variant: str):
    path = os.path.join(FEAT, MASTER[variant])
    df = pd.read_csv(path, index_col=0)
    y = df["label"].values
    X = df.drop(columns=["label"])
    feature_names = list(X.columns)
    return X.values, y, feature_names


def tune_model(name, estimator, param_dist, Xtr, ytr, cv):
    print(f"  tuning {name} (50 iter)...", end="", flush=True)
    t0 = time.time()
    search = RandomizedSearchCV(
        estimator, param_dist, n_iter=50, cv=cv, scoring="roc_auc",
        random_state=42, n_jobs=-1, error_score="raise",
    )
    search.fit(Xtr, ytr)
    dt = time.time() - t0
    print(f" best={search.best_score_:.4f} [{dt:.0f}s]", flush=True)
    return search.best_estimator_, search.best_params_


def main() -> None:
    variant = (sys.argv[1:] or ["nr"])[0]
    X, y, feature_names = load_data(variant)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.30, random_state=6, stratify=y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    print(f"variant '{variant}': train={Xtr.shape}, test={Xte.shape}\n", flush=True)

    # ── 1. Hyperparameter tuning ──────────────────────────────────────────────
    print("=== Hyperparameter tuning ===", flush=True)

    best_rf, rf_params = tune_model("RF", RandomForestClassifier(random_state=42, n_jobs=-1), {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2", 0.3],
    }, Xtr_s, ytr, cv)

    best_xgb, xgb_params = tune_model("XGBoost", XGBClassifier(
        random_state=42, eval_metric="logloss", tree_method="hist", n_jobs=-1), {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
    }, Xtr_s, ytr, cv)

    best_lr, lr_params = tune_model("LR", LogisticRegression(max_iter=2000), {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["saga"],
    }, Xtr_s, ytr, cv)

    # ── 2. Calibration ───────────────────────────────────────────────────────
    print("\n=== Calibration ===", flush=True)
    cal_platt = CalibratedClassifierCV(best_xgb, method="sigmoid", cv=5)
    cal_iso   = CalibratedClassifierCV(best_xgb, method="isotonic", cv=5)
    cal_platt.fit(Xtr_s, ytr)
    cal_iso.fit(Xtr_s, ytr)

    # ── 3. Stacking ensemble ─────────────────────────────────────────────────
    print("  building stacking ensemble...", end="", flush=True)
    t0 = time.time()
    stack = StackingClassifier(
        estimators=[
            ("rf",  best_rf),
            ("xgb", best_xgb),
            ("lr",  best_lr),
        ],
        final_estimator=LogisticRegression(C=1.0, max_iter=2000),
        cv=5, passthrough=False, n_jobs=-1,
    )
    stack.fit(Xtr_s, ytr)
    dt = time.time() - t0
    print(f" [{dt:.0f}s]", flush=True)

    # ── Evaluate all models ──────────────────────────────────────────────────
    print("\n=== Evaluation ===", flush=True)
    models = {
        "RF (tuned)":            best_rf,
        "XGBoost (tuned)":       best_xgb,
        "LR (tuned)":            best_lr,
        "XGBoost+Platt":         cal_platt,
        "XGBoost+Isotonic":      cal_iso,
        "Stacking":              stack,
    }

    rows = []
    probas = {}
    for name, m in models.items():
        m_fitted = m
        if not hasattr(m, "classes_"):
            m_fitted = m.fit(Xtr_s, ytr)
        proba = m_fitted.predict_proba(Xte_s)[:, 1]
        pred = (proba >= 0.5).astype(int)
        ev = eu.full_eval(yte, pred, proba)

        roc_ci = eu.bootstrap_metric_ci(yte, proba, "roc_auc")
        pr_ci  = eu.bootstrap_metric_ci(yte, proba, "pr_auc")

        rows.append({
            "model": name,
            "accuracy": round(ev["accuracy"], 4),
            "precision": round(ev["precision"], 4),
            "recall": round(ev["recall"], 4),
            "f1": round(ev["f1"], 4),
            "roc_auc": round(ev["roc_auc"], 4),
            "roc_auc_ci": f"[{roc_ci[1]:.4f}, {roc_ci[2]:.4f}]",
            "pr_auc": round(ev["pr_auc"], 4),
            "pr_auc_ci": f"[{pr_ci[1]:.4f}, {pr_ci[2]:.4f}]",
            "ece": round(ev["ece"], 4),
        })
        probas[name] = proba
        print(f"  {name:22s} acc={ev['accuracy']:.4f} roc={ev['roc_auc']:.4f} "
              f"pr={ev['pr_auc']:.4f} ece={ev['ece']:.4f}", flush=True)

    os.makedirs(METRICS, exist_ok=True)
    suffix = "" if variant == "nr" else f"_{variant}"
    out_df = pd.DataFrame(rows)
    out_path = os.path.join(METRICS, f"extended_table{suffix}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nwrote {os.path.relpath(out_path, REPO)}", flush=True)

    # Save best params
    params_path = os.path.join(METRICS, f"best_params{suffix}.json")
    with open(params_path, "w") as f:
        json.dump({"RF": rf_params, "XGBoost": xgb_params, "LR": lr_params}, f, indent=2, default=str)

    # ── 4. Figures ────────────────────────────────────────────────────────────
    os.makedirs(FIGURES, exist_ok=True)

    # ROC curves
    from sklearn.metrics import RocCurveDisplay
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in probas.items():
        RocCurveDisplay.from_predictions(yte, proba, name=name, ax=ax)
    ax.set_title("ROC Curves — Extended Pipeline")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, f"roc_curves{suffix}.png"), dpi=150)
    plt.close(fig)

    # PR curves
    from sklearn.metrics import PrecisionRecallDisplay
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in probas.items():
        PrecisionRecallDisplay.from_predictions(yte, proba, name=name, ax=ax)
    ax.set_title("Precision–Recall Curves — Extended Pipeline")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, f"pr_curves{suffix}.png"), dpi=150)
    plt.close(fig)

    # Calibration / reliability diagrams
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, (name, proba) in zip(axes.flat, probas.items()):
        conf, acc, cnt = eu.reliability_curve(yte, proba)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.bar(conf, acc, width=0.08, alpha=0.5, edgecolor="k", linewidth=0.5)
        ax.set_title(name, fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    fig.suptitle("Calibration Reliability Diagrams", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, f"calibration_plots{suffix}.png"), dpi=150)
    plt.close(fig)

    # ── 5. SHAP ──────────────────────────────────────────────────────────────
    print("\n=== SHAP (XGBoost tuned) ===", flush=True)
    explainer = shap.TreeExplainer(best_xgb)
    shap_values = explainer.shap_values(Xte_s)
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, Xte_s, feature_names=feature_names,
                      max_display=20, show=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, f"shap_summary{suffix}.png"), dpi=150)
    plt.close(fig)

    # Save top-100 SHAP feature names
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:100]
    top_features = [feature_names[i] for i in top_idx]
    pd.DataFrame({"rank": range(1, 101), "feature": top_features,
                   "mean_abs_shap": mean_abs_shap[top_idx]}
                 ).to_csv(os.path.join(METRICS, f"shap_top100{suffix}.csv"), index=False)

    print(f"\nfigures -> results/figures/", flush=True)
    print(f"top-100 SHAP features -> results/metrics/shap_top100{suffix}.csv", flush=True)
    print("\nPhase 4 complete.", flush=True)


if __name__ == "__main__":
    main()
