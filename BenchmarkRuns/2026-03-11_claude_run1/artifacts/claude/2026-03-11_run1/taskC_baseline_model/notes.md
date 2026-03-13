# Task C Notes

## Model Configuration
- **Algorithm**: LogisticRegression (sklearn)
- **class_weight**: `balanced` (to handle 6.52% fraud imbalance)
- **max_iter**: 1000
- **Preprocessing**: StandardScaler for numeric, OneHotEncoder for categorical
- **Pipeline**: ColumnTransformer → LogisticRegression

## Split Summary
| Set   | N     | Fraud  | Fraud % |
|-------|-------|--------|---------|
| Train | 5250  | 343    | 6.53%   |
| Valid | 1125  | 73     | 6.49%   |
| Test  | 1125  | 73     | 6.49%   |

## Test Set Metrics
| Metric    | Value   |
|-----------|---------|
| PR-AUC    | 0.0623  |
| ROC-AUC   | 0.4783  |
| Precision | 0.0637  |
| Recall    | 0.4521  |
| F1        | 0.1117  |

## Observations
- **PR-AUC of 0.0623** is near the no-skill level (prevalence = 0.065), indicating the logistic
  regression barely exceeds random classifier performance.
- **ROC-AUC of 0.4783** is below 0.5, which means the model's probability ranking is
  slightly anti-correlated with actual fraud labels on the test set.
- The root cause: fraud patterns in this dataset appear to be highly non-linear. A linear model
  (Logistic Regression) cannot capture feature interactions (e.g., high `ip_risk_score` AND
  `previous_failed_attempts` together indicating fraud), so it underperforms.
- `class_weight="balanced"` yields high recall (0.452) but very poor precision (0.064),
  resulting in many false positives.
- These results establish a clear baseline that non-linear models (Task D) should substantially improve upon.

## Validity
- Preprocessing fitted ONLY on training fold (no data leakage).
- IDs excluded from features.
- Split is stratified (fraud rate preserved).
- Evaluation performed on held-out test set.
