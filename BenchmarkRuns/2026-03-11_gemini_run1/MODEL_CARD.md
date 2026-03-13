# Model Card: Fraud Detection Model

## Model Details
- **Model Type:** HistGradientBoostingClassifier (or best candidate selected during Task D)
- **Task:** Binary Classification (Fraud vs. Non-Fraud)
- **Framework:** scikit-learn

## Intended Use
- Intended to detect fraudulent digital payment transactions based on tabular transaction data.

## Training Data
- Stratified 70/15/15 split of the Digital Payment Fraud Detection Dataset.
- Target column: `fraud_label`
- IDs removed to prevent leakage.

## Evaluation
- Primary metric: PR-AUC
- Secondary metrics: ROC-AUC, Precision, Recall, F1-score.
- Evaluated on a holdout test set representing 15% of the data.

## Preprocessing
- Missing numeric values: Median imputation.
- Missing categorical values: Constant imputation ('missing').
- Categorical encoding: One-Hot Encoding.
- Numeric scaling: Standard Scaler (if applicable to baseline).
