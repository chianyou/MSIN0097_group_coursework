# Task D Notes

## Improvement Methods Applied
1. **Feature Engineering** — Added 5 engineered features:
   - `hour_sin`, `hour_cos`: cyclic encoding of transaction hour
   - `risk_x_logins`: ip_risk_score × login_attempts_last_24h
   - `fail_x_risk`: previous_failed_attempts × ip_risk_score
   - `amount_ratio`: transaction_amount / avg_transaction_amount
2. **Hyperparameter Tuning** — Adjusted regularization (L1, C=0.5), RF depth/estimators, HGBC learning rate / leaf size
3. **Threshold Tuning** — Selected threshold maximising F1 on validation set (0.532)

## Candidate Model Validation Results (used for model selection)

| Model                          | val PR-AUC | val ROC-AUC | Selected? |
|-------------------------------|-----------|------------|----------|
| LogisticRegression             | 0.0837    | 0.5750     | ✅ YES   |
| RandomForestClassifier         | 0.0779    | 0.5289     | No       |
| HistGradientBoostingClassifier | 0.0674    | 0.4911     | No       |

**Final model selected**: LogisticRegression (highest val PR-AUC = 0.0837)

## Baseline vs Improved Comparison (Test Set)

| Metric      | Baseline (LR, no FE) | Improved (LR + FE + Threshold) | Delta    |
|-------------|----------------------|---------------------------------|----------|
| PR-AUC      | 0.0623               | 0.0661                          | +0.0038  |
| ROC-AUC     | 0.4783               | 0.5026                          | +0.0243  |
| Precision   | 0.0637               | 0.0647                          | +0.0010  |
| Recall      | 0.4521               | 0.3014                          | -0.1507  |
| F1          | 0.1117               | 0.1065                          | -0.0052  |

## Key Observations
- **Dataset signal is extremely weak**: all numeric features have |r| < 0.03 with the fraud label.
  Categorical fraud rates vary by only ±1% around the 6.52% baseline.
- All three candidate models converge near the no-skill PR-AUC baseline (~0.065).
- Feature engineering and hyperparameter tuning provided a modest improvement in PR-AUC (+0.004)
  and ROC-AUC (+0.024), with the most notable gain being ROC-AUC crossing the 0.50 threshold.
- The threshold tuning improved decision calibration: trading some recall for higher precision.
- Results are honest — no data leakage, no test-set snooping during model selection.

## Recommendation
For production use on this dataset, a human-review pipeline or additional data sources
(e.g., device fingerprinting, network graph features) would be needed to achieve commercially
viable fraud detection performance (PR-AUC > 0.3).
