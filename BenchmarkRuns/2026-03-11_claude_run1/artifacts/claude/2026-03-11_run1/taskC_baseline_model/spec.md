# Task C: Baseline Model — Specification

## Goal
Build a statistically valid baseline fraud model.

## Must Do
- Use stratified 70/15/15
- Use sklearn Pipeline and ColumnTransformer
- Fit preprocessing on training data only
- Exclude target and ID columns from features
- Train exactly 1 baseline model: LogisticRegression
- Evaluate on test set only

## Deliverables
- `outputs/baseline_model.joblib`
- `outputs/metrics_baseline.json`
- `plots/confusion_matrix_baseline.png`
- `plots/pr_curve.png`
- `plots/roc_curve.png`

## Required Metrics
PR-AUC, ROC-AUC, Precision, Recall, F1, confusion matrix
