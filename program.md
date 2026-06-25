# NanowireML Autoresearch Program

## Goal
Maximize **PR_AUC** on stratified 5-fold CV of the nanowire protein binary
classification task. Secondary metric: ROC_AUC. Tertiary: keep ECE < 0.05
(well-calibrated probabilities).

## Context
- Dataset: ~997 nanowire proteins (positive) vs 999 RecA proteins (negative)
- Features: hand-crafted iFeature descriptors (`data/features/master_features.csv`)
- Published baseline (Raya et al. 2025): RF ROC_AUC=0.9826, XGBoost ROC_AUC=0.9857
- Current best: [updated each iteration by the loop]

## Rules
- Edit ONLY the section between `### BEGIN EXPERIMENT ###` and `### END EXPERIMENT ###`
- Do not change data loading, CV strategy, or metric printing
- Each experiment must complete in < 5 minutes on CPU
- Prefer interpretable models over black-box when PR_AUC is within 0.005

## Hypothesis log
[The loop appends what was tried and why, with the resulting PR_AUC.]

## Promising directions
- Cys-His dipeptide features (DPC) dominate SHAP — try DPC-only models
- CTD descriptors (CTDC, CTDT, CTDD) strong individually — test combinations
- XGBoost with top-100 SHAP features may beat the full-feature RF
- Calibrated XGBoost likely improves ECE significantly
