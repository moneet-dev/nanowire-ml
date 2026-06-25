"""
classify.py — autoresearch target file for NanowireML.

Edit ONLY the section between ### BEGIN EXPERIMENT ### and ### END EXPERIMENT ###.
Do not modify data loading, the CV strategy, or metric reporting.

Run:   python src/classify.py
Emits: PR_AUC=..., ROC_AUC=..., ECE=...  on stdout (parsed by the autoresearch loop).
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_REPO, "data", "features", "master_features.csv")

# ── Fixed: data loading ───────────────────────────────────────────────────────
df = pd.read_csv(_DATA, index_col=0)
y = df["label"].values
X = df.drop(columns=["label"]).values

# ── Fixed: evaluation strategy ────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

### BEGIN EXPERIMENT ###
# Agent edits here: model class, hyperparameters, feature selection, calibration.
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
### END EXPERIMENT ###

# ── Fixed: benchmark reporting (do not edit) ──────────────────────────────────
pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

roc_auc = roc_auc_score(y, proba)
pr_auc = average_precision_score(y, proba)


def _ece(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        e += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return e / n


ece = _ece(y, proba)

print(f"PR_AUC={pr_auc:.6f}")
print(f"ROC_AUC={roc_auc:.6f}")
print(f"ECE={ece:.6f}")
