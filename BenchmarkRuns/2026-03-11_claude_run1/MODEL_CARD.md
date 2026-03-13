# Model Card — Digital Payment Fraud Detection

**Run ID**: `2026-03-11_run1`
**Tool**: `claude`
**Date**: 2026-03-12

---

## Model Details

### Baseline Model (Task C)
- **Architecture**: `sklearn.linear_model.LogisticRegression`
- **Preprocessing**: `StandardScaler` (numeric) + `OneHotEncoder` (categorical) via `ColumnTransformer`
- **Wrapped in**: `sklearn.Pipeline`
- **Hyperparameters**: `max_iter=1000`, `class_weight="balanced"`, `random_state=42`
- **Artifact**: `artifacts/claude/2026-03-11_run1/taskC_baseline_model/outputs/baseline_model.joblib`

### Improved Model (Task D)
- **Architecture**: `sklearn.linear_model.LogisticRegression` (selected via validation PR-AUC)
- **Additional features**: 5 engineered features (cyclic hour, interaction terms, amount ratio)
- **Hyperparameters**: `C=0.5`, `max_iter=2000`, `solver="saga"`, `l1_ratio=1.0`, `class_weight="balanced"`, `random_state=42`
- **Decision threshold**: 0.5319 (tuned on validation set by maximising F1)
- **Artifact**: `artifacts/claude/2026-03-11_run1/taskD_model_improvement/outputs/improved_model.joblib`

---

## Intended Use
Binary fraud classification for digital payment transactions. This is a benchmark exercise;
the model should not be deployed in production without further validation on live data.

---

## Training Data

| Property | Value |
|----------|-------|
| Source | `Digital_Payment_Fraud_Detection_Dataset.csv` |
| Rows | 7,500 |
| Features (baseline) | 12 (8 numeric, 4 categorical) |
| Features (improved) | 17 (13 numeric, 4 categorical) |
| Target | `fraud_label` (binary: 0=legitimate, 1=fraud) |
| Fraud rate | 6.52% (489/7500) |
| ID columns (excluded) | `transaction_id`, `user_id` |

### Data Split
| Set | Rows | Fraud Count | Fraud % |
|-----|------|-------------|---------|
| Train | 5250 | 343 | 6.53% |
| Validation | 1125 | 73 | 6.49% |
| Test | 1125 | 73 | 6.49% |

---

## Evaluation Metrics (Test Set)

| Metric | Baseline LR | Improved LR | Delta |
|--------|-------------|-------------|-------|
| **PR-AUC** | 0.0623 | 0.0661 | +0.004 |
| ROC-AUC | 0.4783 | 0.5026 | +0.024 |
| Precision | 0.0637 | 0.0647 | +0.001 |
| Recall | 0.4521 | 0.3014 | -0.151 |
| F1 | 0.1117 | 0.1065 | -0.005 |

---

## Limitations and Caveats

1. **Very weak signal**: All numeric features have |Pearson r| < 0.03 with `fraud_label`.
   Categorical fraud rates differ by ≤ 1% from the population rate. The dataset is synthetic
   with intentionally low discriminability.

2. **Near no-skill performance**: PR-AUC ≈ 0.066 is barely above the no-skill baseline of
   0.065 (= fraud prevalence). These models cannot be relied upon to detect fraud reliably.

3. **Class imbalance**: 6.52% fraud rate. `class_weight="balanced"` was used but cannot
   compensate for absent feature signal.

4. **Not production-ready**: These models are benchmark artifacts. Real-world deployment
   requires additional data sources and thorough fairness/bias evaluation.

---

## Ethical Considerations
- Fraud detection systems can produce false positives that block legitimate transactions.
- Any deployment must include human review of flagged transactions.
- Model decisions should not be the sole basis for blocking users.

---

## Candidate Models Considered (Task D)

| Model | Validation PR-AUC | Validation ROC-AUC |
|-------|-------------------|-------------------|
| LogisticRegression | **0.0837** ← selected | 0.5750 |
| RandomForestClassifier | 0.0779 | 0.5289 |
| HistGradientBoostingClassifier | 0.0674 | 0.4911 |
