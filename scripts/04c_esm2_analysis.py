"""
04c_esm2_analysis.py — Phase 4: Statistical comparison of ESM-2 vs iFeature.

Runs both feature sets through IDENTICAL 5-fold CV folds so results are directly
comparable. Addresses methodological alignment gap from previous pipeline.

Outputs:
  1. Aligned comparison table (same CV folds, same classifiers)
  2. McNemar's test: is ESM-2 LR significantly better than iFeature LR?
  3. Bootstrap 95% CI on AUC difference
  4. Combined experiment: ESM-2 + top-50 SHAP iFeature features
  5. Publication-ready comparison figure

Requires:
  - Phase 2 (04b_esm2_pipeline.py) to have run: esm2_embeddings.npy + esm2_cv_preds.csv
  - Phase 3 (02_feature_extraction.py + 04_extended_pipeline.py) for iFeature matrix + SHAP

Usage: python scripts/04c_esm2_analysis.py [nr|redundant]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import eval_utils as eu  # noqa: E402

FEAT = REPO / "data" / "features"
MET  = REPO / "results" / "metrics"
FIGS = REPO / "results" / "figures"
CV   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ── statistical tests ─────────────────────────────────────────────────────────

def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    name_a: str = "A",
    name_b: str = "B",
) -> dict:
    """McNemar's test with continuity correction (Edwards 1948).

    b = classifier A correct, B wrong
    c = classifier A wrong, B correct
    Null: b == c (same error rate). Chi-squared approx with df=1.
    """
    b = int(((y_pred_a == y_true) & (y_pred_b != y_true)).sum())
    c = int(((y_pred_a != y_true) & (y_pred_b == y_true)).sum())

    if b + c < 10:
        # Exact binomial for small disagreement counts
        from scipy.stats import binom_test  # type: ignore[attr-defined]
        try:
            # scipy >= 1.9 uses binomtest; fall back to binom_test for older
            from scipy.stats import binomtest
            p = float(binomtest(b, b + c, 0.5).pvalue)
        except ImportError:
            p = float(binom_test(b, b + c, 0.5))
        stat = float("nan")
    else:
        from scipy.stats import chi2  # type: ignore[attr-defined]
        stat = float((abs(b - c) - 1) ** 2 / (b + c))
        p    = float(1 - chi2.cdf(stat, df=1))

    if b > c and p < 0.05:
        interpretation = f"{name_a} significantly better (p={p:.4f})"
    elif c > b and p < 0.05:
        interpretation = f"{name_b} significantly better (p={p:.4f})"
    else:
        interpretation = f"No significant difference (p={p:.4f})"

    return {
        "classifier_a": name_a,
        "classifier_b": name_b,
        "b_a_right_b_wrong": b,
        "c_a_wrong_b_right": c,
        "mcnemar_stat": round(stat, 4) if not np.isnan(stat) else "exact",
        "p_value": round(p, 6),
        "significant_at_0.05": bool(p < 0.05),
        "interpretation": interpretation,
    }


def bootstrap_auc_diff(
    y: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Bootstrap 95% CI for (ROC-AUC_A - ROC-AUC_B) and its p-value."""
    from sklearn.metrics import roc_auc_score

    rng  = np.random.default_rng(seed)
    n    = len(y)
    diffs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(
            float(roc_auc_score(y[idx], proba_a[idx]))
            - float(roc_auc_score(y[idx], proba_b[idx]))
        )
    d = np.array(diffs)
    obs = float(roc_auc_score(y, proba_a)) - float(roc_auc_score(y, proba_b))
    # one-sided p-value: proportion of bootstrap samples where B >= A
    p_one_sided = float((d <= 0).mean())
    return {
        "roc_auc_diff_obs": round(obs, 6),
        "ci_low_95":  round(float(np.percentile(d, 2.5)),  6),
        "ci_high_95": round(float(np.percentile(d, 97.5)), 6),
        "p_one_sided_B_ge_A": round(p_one_sided, 6),
    }


# ── aligned CV evaluation ─────────────────────────────────────────────────────

def aligned_cv(
    X: np.ndarray, y: np.ndarray, name: str, clf_name: str = "LR"
) -> tuple[dict, np.ndarray, np.ndarray]:
    """One classifier × one feature set, 5-fold CV. Returns metrics, OOF proba, OOF pred."""
    if clf_name == "LR":
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, solver="saga")),
        ])
    elif clf_name == "XGBoost":
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                random_state=42, eval_metric="logloss",
                tree_method="hist", n_jobs=-1,
            )),
        ])
    else:
        raise ValueError(f"Unknown classifier: {clf_name}")

    oof   = cross_val_predict(pipe, X, y, cv=CV, method="predict_proba")[:, 1]
    pred  = (oof >= 0.5).astype(int)
    ev    = eu.full_eval(y, pred, oof)
    roc_ci = eu.bootstrap_metric_ci(y, oof, "roc_auc")
    pr_ci  = eu.bootstrap_metric_ci(y, oof, "pr_auc")

    row = {
        "method":     name,
        "classifier": clf_name,
        "n_features": X.shape[1],
        "roc_auc":    round(ev["roc_auc"],  4),
        "roc_auc_ci": f"[{roc_ci[1]:.4f}, {roc_ci[2]:.4f}]",
        "pr_auc":     round(ev["pr_auc"],   4),
        "pr_auc_ci":  f"[{pr_ci[1]:.4f}, {pr_ci[2]:.4f}]",
        "ece":        round(ev["ece"],       4),
        "accuracy":   round(ev["accuracy"],  4),
    }
    print(
        f"  {name:25s} [{clf_name:8s}]  n={X.shape[1]:>6d}  "
        f"roc={ev['roc_auc']:.4f}  pr={ev['pr_auc']:.4f}  ece={ev['ece']:.4f}",
        flush=True,
    )
    return row, oof, pred


# ── figure ────────────────────────────────────────────────────────────────────

def plot_comparison(cmp_df: pd.DataFrame, out: Path) -> None:
    metrics = [("roc_auc", "ROC-AUC ↑"), ("pr_auc", "PR-AUC ↑"), ("ece", "ECE ↓")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    clfs = cmp_df["classifier"].unique()
    methods = cmp_df["method"].unique()
    palette = {
        ("ESM-2",    "LR"):      "#2196F3",
        ("ESM-2",    "XGBoost"): "#1565C0",
        ("iFeature", "LR"):      "#FF9800",
        ("iFeature", "XGBoost"): "#E65100",
        ("ESM-2+SHAP50", "LR"):  "#4CAF50",
    }

    for ax, (metric, title) in zip(axes, metrics):
        labels_plot: list[str] = []
        vals: list[float] = []
        colors: list[str] = []

        for method in methods:
            for clf in clfs:
                row = cmp_df[(cmp_df["method"] == method) & (cmp_df["classifier"] == clf)]
                if row.empty:
                    continue
                labels_plot.append(f"{method}\n{clf}")
                vals.append(float(row[metric].iloc[0]))
                colors.append(palette.get((method, clf), "#9E9E9E"))

        x_pos = np.arange(len(labels_plot))
        bars  = ax.bar(x_pos, vals, color=colors, edgecolor="k", linewidth=0.4, width=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_plot, fontsize=7.5)
        ax.set_title(title, fontsize=11)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7,
            )
        if metric in ("roc_auc", "pr_auc"):
            ax.set_ylim(max(0.97, min(vals) - 0.01), 1.003)
        else:
            ax.set_ylim(0, max(vals) * 1.35)

    # Legend patches
    from matplotlib.patches import Patch
    legend_items = [
        Patch(color="#2196F3", label="ESM-2 + LR"),
        Patch(color="#1565C0", label="ESM-2 + XGBoost"),
        Patch(color="#FF9800", label="iFeature + LR"),
        Patch(color="#E65100", label="iFeature + XGBoost"),
        Patch(color="#4CAF50", label="ESM-2 + SHAP50 + LR"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("ESM-2 vs iFeature — Aligned 5-Fold CV", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {out.name}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    variant = (sys.argv[1:] or ["nr"])[0]
    suffix  = "" if variant == "nr" else "_redundant"

    # ── Load ESM-2 embeddings ─────────────────────────────────────────────────
    emb_path   = FEAT / f"esm2_embeddings{suffix}.npy"
    preds_path = MET  / f"esm2_cv_preds{suffix}.csv"
    if not emb_path.exists():
        print(f"ERROR: {emb_path.name} not found. Run Phase 2 (04b) first.", flush=True)
        sys.exit(1)

    X_esm    = np.load(str(emb_path))
    preds_df = pd.read_csv(str(preds_path))
    seq_ids  = preds_df["seq_id"].tolist()
    y        = preds_df["true_label"].values

    # ── Load iFeature matrix (reindexed to ESM-2 sequence order) ─────────────
    master_name = "master_features.csv" if variant == "nr" else "master_features_redundant.csv"
    master_path = FEAT / master_name
    if not master_path.exists():
        print(
            f"ERROR: {master_path.name} not found. "
            "Run Phase 3 (02_feature_extraction.py) first.",
            flush=True,
        )
        sys.exit(1)

    df_feat  = pd.read_csv(str(master_path), index_col=0)
    # Reindex iFeature rows to match ESM-2 sequence order
    missing = [sid for sid in seq_ids if sid not in df_feat.index]
    if missing:
        print(
            f"WARNING: {len(missing)} sequences from ESM-2 not in iFeature matrix "
            "(possibly different dataset variants). Using intersection.",
            flush=True,
        )
        seq_ids = [sid for sid in seq_ids if sid in df_feat.index]
        keep    = [preds_df["seq_id"].tolist().index(sid) for sid in seq_ids]
        y       = y[keep]
        X_esm   = X_esm[keep]

    df_feat  = df_feat.loc[seq_ids]
    y_check  = df_feat["label"].values
    assert np.array_equal(y, y_check), "Label mismatch between ESM-2 and iFeature"
    feat_cols = [c for c in df_feat.columns if c != "label"]
    X_feat   = df_feat[feat_cols].values

    print(
        f"Variant: {variant}  N={len(y)}  pos={int(y.sum())}  neg={int((y==0).sum())}\n"
        f"ESM-2: {X_esm.shape}   iFeature: {X_feat.shape}",
        flush=True,
    )

    MET.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    oof_probas: dict[str, np.ndarray] = {}
    oof_preds:  dict[str, np.ndarray] = {}

    # ── Aligned comparison ────────────────────────────────────────────────────
    print("\n=== Aligned 5-Fold CV Comparison ===", flush=True)
    for feature_set, X_m in [("ESM-2", X_esm), ("iFeature", X_feat)]:
        for clf in ["LR", "XGBoost"]:
            row, proba, pred = aligned_cv(X_m, y, feature_set, clf)
            key = f"{feature_set}+{clf}"
            rows.append(row)
            oof_probas[key] = proba
            oof_preds[key]  = pred

    # ── ESM-2 + top-50 SHAP iFeature combination ──────────────────────────────
    shap_path = MET / f"shap_top100{suffix}.csv"
    if shap_path.exists():
        print("\n=== ESM-2 + top-50 SHAP iFeature (combined) ===", flush=True)
        top_feats = pd.read_csv(str(shap_path))["feature"].tolist()[:50]
        avail = [f for f in top_feats if f in feat_cols]
        if avail:
            idx_combo = [feat_cols.index(f) for f in avail]
            X_combo   = np.hstack([X_esm, X_feat[:, idx_combo]])
            row, proba, pred = aligned_cv(X_combo, y, "ESM-2+SHAP50", "LR")
            rows.append(row)
            oof_probas["ESM-2+SHAP50+LR"] = proba
            oof_preds["ESM-2+SHAP50+LR"]  = pred
        else:
            print("  No SHAP features found in iFeature matrix — skipping combo.", flush=True)
    else:
        print(
            f"  {shap_path.name} not found — run Phase 3 extended pipeline first "
            "for the combined experiment.",
            flush=True,
        )

    # Save comparison table
    cmp_df = pd.DataFrame(rows)
    cmp_path = MET / f"comparison_table{suffix}.csv"
    cmp_df.to_csv(str(cmp_path), index=False)
    print(f"\nwrote {cmp_path.relative_to(REPO)}", flush=True)

    # ── McNemar's test ────────────────────────────────────────────────────────
    print("\n=== McNemar's Test ===", flush=True)
    mc_rows: list[dict] = []
    for clf in ["LR", "XGBoost"]:
        key_esm = f"ESM-2+{clf}"
        key_if  = f"iFeature+{clf}"
        if key_esm in oof_preds and key_if in oof_preds:
            mc = mcnemar_test(y, oof_preds[key_esm], oof_preds[key_if],
                              key_esm, key_if)
            mc_rows.append(mc)
            print(
                f"  {clf:8s}:  b={mc['b_a_right_b_wrong']}  c={mc['c_a_wrong_b_right']}"
                f"  p={mc['p_value']:.6f}  → {mc['interpretation']}",
                flush=True,
            )

    mc_df = pd.DataFrame(mc_rows)
    mc_path = MET / f"mcnemar_result{suffix}.csv"
    mc_df.to_csv(str(mc_path), index=False)

    # ── Bootstrap AUC difference ──────────────────────────────────────────────
    print("\n=== Bootstrap 95% CI on ROC-AUC Difference (ESM-2 LR − iFeature LR) ===",
          flush=True)
    if "ESM-2+LR" in oof_probas and "iFeature+LR" in oof_probas:
        diff = bootstrap_auc_diff(y, oof_probas["ESM-2+LR"], oof_probas["iFeature+LR"])
        print(
            f"  Observed diff: {diff['roc_auc_diff_obs']:+.6f}  "
            f"95% CI: [{diff['ci_low_95']:+.6f}, {diff['ci_high_95']:+.6f}]  "
            f"p(B≥A): {diff['p_one_sided_B_ge_A']:.6f}",
            flush=True,
        )
        pd.DataFrame([{**diff, "comparison": "ESM-2_LR_minus_iFeature_LR"}]).to_csv(
            str(MET / f"auc_diff_ci{suffix}.csv"), index=False)

    # ── Figure ────────────────────────────────────────────────────────────────
    print("\n=== Generating comparison figure ===", flush=True)
    plot_comparison(cmp_df, FIGS / f"esm2_vs_ifeature_comparison{suffix}.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("ESM-2 vs iFeature — aligned comparison complete", flush=True)
    print(
        cmp_df[["method", "classifier", "n_features", "roc_auc", "pr_auc", "ece"]]
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
