"""
06_autoresearch_loop.py — Phase 7: autoresearch-style experiment loop.

Implements the keep/revert pattern from Karpathy's autoresearch framework:
  1. Read current classify.py and its benchmark score
  2. Propose a modification (via a list of predefined experiments)
  3. Run classify.py, extract PR_AUC from stdout
  4. If improved: git commit; else: git revert
  5. Log the result and repeat

This script runs PREDEFINED experiments (not AI-generated). Each experiment
replaces the ### BEGIN EXPERIMENT ### block in src/classify.py with a new model
configuration. The full autonomous AI-agent variant would use the Claude API to
propose modifications, but that requires an API key and is out of scope for the
local-first implementation.

Usage: python scripts/06_autoresearch_loop.py
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSIFY = os.path.join(REPO, "src", "classify.py")
LOG_FILE = os.path.join(REPO, "results", "metrics", "autoresearch_log.csv")
BACKUP_DIR = os.path.join(REPO, "results", "_backups")

BEGIN_MARKER = "### BEGIN EXPERIMENT ###"
END_MARKER = "### END EXPERIMENT ###"

EXPERIMENTS = [
    {
        "name": "RF_100",
        "hypothesis": "Baseline: RF with 100 trees",
        "code": """from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)""",
    },
    {
        "name": "RF_500",
        "hypothesis": "More trees may reduce variance",
        "code": """from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)""",
    },
    {
        "name": "XGB_default",
        "hypothesis": "XGBoost with default params",
        "code": """from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss",
                      tree_method="hist", n_jobs=-1)""",
    },
    {
        "name": "XGB_tuned",
        "hypothesis": "XGBoost with deeper trees and lower learning rate",
        "code": """from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.05,
                      subsample=0.9, colsample_bytree=0.9,
                      random_state=42, eval_metric="logloss",
                      tree_method="hist", n_jobs=-1)""",
    },
    {
        "name": "LR_l2",
        "hypothesis": "Logistic regression — linear baseline, good calibration",
        "code": """from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0, max_iter=2000, solver="saga", penalty="l2")""",
    },
    {
        "name": "LR_l1_C10",
        "hypothesis": "L1 regularization for built-in feature selection",
        "code": """from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=10, max_iter=2000, solver="saga", penalty="l1")""",
    },
    {
        "name": "calibrated_XGB",
        "hypothesis": "Isotonic calibration should improve ECE without hurting AUC",
        "code": """from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
base = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                     random_state=42, eval_metric="logloss",
                     tree_method="hist", n_jobs=-1)
model = CalibratedClassifierCV(base, method="isotonic", cv=5)""",
    },
    {
        "name": "stacking_RF_XGB_LR",
        "hypothesis": "Stacking ensemble may capture complementary signals",
        "code": """from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
stack = StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ("xgb", XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                              random_state=42, eval_metric="logloss",
                              tree_method="hist", n_jobs=-1)),
    ],
    final_estimator=LogisticRegression(max_iter=2000),
    cv=5, passthrough=False, n_jobs=-1,
)
model = stack""",
    },
    {
        "name": "XGB_DPC_only",
        "hypothesis": "DPC features alone (Cys-His motif) may suffice",
        "code": """from xgboost import XGBClassifier
import pandas as pd, os, numpy as np
_r = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_d = pd.read_csv(os.path.join(_r, "data", "features", "master_features.csv"), index_col=0)
_cols = [c for c in _d.columns if c.startswith("DPC1::") or c.startswith("DPC2::")]
# Monkey-patch: replace X with DPC-only before the pipeline runs
import sys
_mod = sys.modules[__name__]
X = _d[_cols].values  # noqa: F811
y = _d["label"].values  # noqa: F811
model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                      random_state=42, eval_metric="logloss",
                      tree_method="hist", n_jobs=-1)""",
    },
    {
        "name": "XGB_top100_variance",
        "hypothesis": "Top-100 highest-variance features (data-driven selection)",
        "code": """from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline as InnerPipeline
selector = SelectKBest(f_classif, k=100)
inner = InnerPipeline([("select", selector),
                       ("clf", XGBClassifier(n_estimators=200, max_depth=5,
                                            learning_rate=0.1, random_state=42,
                                            eval_metric="logloss",
                                            tree_method="hist", n_jobs=-1))])
model = inner""",
    },
]


def read_classify():
    with open(CLASSIFY, "r") as f:
        return f.read()


def write_experiment(code: str):
    text = read_classify()
    start = text.index(BEGIN_MARKER) + len(BEGIN_MARKER)
    end = text.index(END_MARKER)
    new = text[:start] + "\n" + code + "\n" + text[end:]
    with open(CLASSIFY, "w") as f:
        f.write(new)


def run_classify(timeout: int = 300) -> dict[str, float] | None:
    try:
        result = subprocess.run(
            [sys.executable, CLASSIFY],
            capture_output=True, text=True, timeout=timeout, cwd=REPO,
        )
    except subprocess.TimeoutExpired:
        return None
    metrics: dict[str, float] = {}
    for line in result.stdout.strip().split("\n"):
        if "=" in line:
            key, val = line.split("=", 1)
            try:
                metrics[key.strip()] = float(val.strip())
            except ValueError:
                pass
    if result.returncode != 0:
        print(f"    stderr: {result.stderr[:200]}", flush=True)
    return metrics if metrics else None


def git_commit(msg: str):
    subprocess.run(["git", "add", CLASSIFY], cwd=REPO, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=REPO, capture_output=True)


def git_revert():
    subprocess.run(["git", "checkout", "--", CLASSIFY], cwd=REPO, capture_output=True)


def main() -> None:
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)

    original = read_classify()

    best_pr_auc = 0.0
    log_existed = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as logf:
        writer = csv.writer(logf)
        if not log_existed:
            writer.writerow(["iteration", "name", "hypothesis", "pr_auc",
                             "roc_auc", "ece", "kept", "seconds"])

        for i, exp in enumerate(EXPERIMENTS):
            print(f"\n[{i+1}/{len(EXPERIMENTS)}] {exp['name']}: {exp['hypothesis']}", flush=True)

            shutil.copy(CLASSIFY, os.path.join(BACKUP_DIR, f"classify_{i:03d}.py"))
            write_experiment(exp["code"])

            t0 = time.time()
            metrics = run_classify()
            dt = time.time() - t0

            if metrics is None:
                print(f"    FAILED (no metrics returned) [{dt:.0f}s]", flush=True)
                git_revert()
                writer.writerow([i, exp["name"], exp["hypothesis"],
                                 "", "", "", False, round(dt, 1)])
                logf.flush()
                continue

            pr = metrics.get("PR_AUC", 0.0)
            roc = metrics.get("ROC_AUC", 0.0)
            ece = metrics.get("ECE", 0.0)

            if pr > best_pr_auc:
                best_pr_auc = pr
                git_commit(f"exp#{i}: {exp['name']} PR_AUC={pr:.6f}")
                kept = True
                mark = "KEPT (new best)"
            else:
                git_revert()
                kept = False
                mark = "reverted"

            print(f"    PR_AUC={pr:.6f} ROC_AUC={roc:.6f} ECE={ece:.6f} "
                  f"[{dt:.0f}s] → {mark}", flush=True)
            writer.writerow([i, exp["name"], exp["hypothesis"],
                             round(pr, 6), round(roc, 6), round(ece, 6),
                             kept, round(dt, 1)])
            logf.flush()

    # Restore original classify.py (the best version is in git history)
    with open(CLASSIFY, "w") as f:
        f.write(original)

    print(f"\n=== Autoresearch loop complete ===")
    print(f"Best PR_AUC: {best_pr_auc:.6f}")
    print(f"Log: {os.path.relpath(LOG_FILE, REPO)}")
    print(f"Best classify.py is in the latest 'exp#' git commit.")


if __name__ == "__main__":
    main()
