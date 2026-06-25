"""
eval_utils.py — evaluation metrics for NanowireML.

Covers the metrics the original paper omits (Gaps 2 & 3 in P1_architecture.md):
PR-AUC, Expected Calibration Error (ECE), reliability curves, and bootstrap
confidence intervals for uncertainty quantification.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_ece(y_true, y_prob, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width probability bins.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() * abs(acc - conf)
    return float(ece / n)


def full_eval(y_true, y_pred, y_prob) -> dict:
    """All headline metrics for a binary classifier given labels, hard
    predictions, and positive-class probabilities."""
    y_true = np.asarray(y_true)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "ece": compute_ece(y_true, y_prob),
    }


def reliability_curve(y_true, y_prob, n_bins: int = 10):
    """Return ``(bin_confidence, bin_accuracy, bin_count)`` arrays for plotting a
    reliability diagram. Empty bins yield NaN accuracy and zero count."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf, acc, cnt = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            conf.append((lo + hi) / 2)
            acc.append(np.nan)
            cnt.append(0)
        else:
            conf.append(float(y_prob[mask].mean()))
            acc.append(float(y_true[mask].mean()))
            cnt.append(int(mask.sum()))
    return np.array(conf), np.array(acc), np.array(cnt)


def bootstrap_metric_ci(
    y_true,
    y_prob,
    metric: str = "roc_auc",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
):
    """Percentile bootstrap CI for a probability-based metric (Gap 3:
    uncertainty quantification the original paper omits).

    Returns ``(point_estimate, ci_low, ci_high)``.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    fn = roc_auc_score if metric == "roc_auc" else average_precision_score
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:  # need both classes for AUC
            continue
        stats.append(fn(y_true[idx], y_prob[idx]))
    stats_arr = np.asarray(stats)
    return (
        float(stats_arr.mean()),
        float(np.percentile(stats_arr, 100 * alpha / 2)),
        float(np.percentile(stats_arr, 100 * (1 - alpha / 2))),
    )
