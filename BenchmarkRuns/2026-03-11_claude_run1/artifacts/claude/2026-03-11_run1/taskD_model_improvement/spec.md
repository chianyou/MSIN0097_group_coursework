# Task D: Model Improvement — Specification

## Goal
Improve performance without violating statistical validity.

## Rules
- Test at least 3 candidate models (from: LogisticRegression, RandomForestClassifier, HistGradientBoostingClassifier)
- Use at least 2 improvement methods from: feature engineering, hyperparameter tuning, model change, threshold tuning
- Use train/valid for model selection; test only once for final evaluation

## Improvement Methods Used
1. **Feature Engineering**: cyclic hour encoding (sin/cos), IP risk × login volume interaction, failed attempts × IP risk, transaction amount ratio
2. **Hyperparameter Tuning**: tuned C/solver for LR, n_estimators/max_depth for RF, learning_rate/max_depth for HGBC
3. **Threshold Tuning**: selected optimal decision threshold by maximising F1 on validation set

## Deliverables
- `outputs/metrics_improved.json`
- `plots/confusion_matrix_improved.png`
- `plots/pr_curve_improved.png`
- `plots/roc_curve_improved.png`
- `notes.md` with comparison table and candidate results
