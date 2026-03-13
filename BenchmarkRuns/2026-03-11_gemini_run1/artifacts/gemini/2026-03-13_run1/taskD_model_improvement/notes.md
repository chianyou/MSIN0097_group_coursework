# Task D Notes
## Tested Candidate Models on Validation Set
- LogisticRegression: PR-AUC = 0.0659
- RandomForestClassifier: PR-AUC = 0.0668
- HistGradientBoostingClassifier: PR-AUC = 0.0578

Selected Model: RandomForestClassifier (Threshold tuned on validation set: 0.1000)

## Baseline vs Improved Comparison
| Metric | Baseline | Improved |
|---|---|---|
| PR-AUC | (see Task C) | 0.0679 |
| F1 | (see Task C) | 0.0931 |

## Improvements Applied
1. Model change: Tested 3 candidate models and selected the best.
2. Threshold tuning: Optimized decision threshold on the validation set for maximum F1 score.
