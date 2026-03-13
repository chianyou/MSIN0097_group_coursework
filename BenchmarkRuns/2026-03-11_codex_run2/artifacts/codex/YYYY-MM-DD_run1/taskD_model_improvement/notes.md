# Task D Notes

## Candidate Validation Results

| Model | FE | Val PR-AUC | Val F1 | Selected Threshold |
|---|---:|---:|---:|---:|
| LogisticRegression_baseline_plus_fe | True | 0.0648 | 0.0000 | 0.50 |
| RandomForestClassifier_shallow | True | 0.0689 | 0.1220 | 0.25 |
| HistGradientBoostingClassifier_default | False | 0.0578 | 0.0388 | 0.20 |

## Baseline vs Improved

| Metric | Baseline | Improved |
|---|---:|---:|
| PR-AUC | 0.0711 | 0.1044 |
| ROC-AUC | 0.5135 | 0.5312 |
| Precision | 0.0000 | 0.0659 |
| Recall | 0.0000 | 1.0000 |
| F1 | 0.0000 | 0.1236 |

## Changes Applied

- Model change across three approved candidates.
- Hyperparameter tuning through candidate-specific settings.
- Feature engineering: amount-to-average ratio, high-risk-hour flag, and amount-risk interaction.
- Threshold tuning on the validation split only.
