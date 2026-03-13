# Task A Notes

## Key Findings
- Dataset: 7,500 rows × 15 columns
- No missing values in any column
- No duplicate rows
- Fraud rate: ~6.52% (highly imbalanced — 489 fraud out of 7,500)
- All columns passed range checks:
  - `transaction_hour`: valid [0, 23]
  - `ip_risk_score`: valid [0, 1]
  - `fraud_label`: only {0, 1}
  - `is_international`: only {0, 1}
  - `transaction_amount`: all positive

## Column Types
- **ID columns**: `transaction_id`, `user_id`
- **Target**: `fraud_label`
- **Categorical**: `transaction_type`, `payment_mode`, `device_type`, `device_location`
- **Numerical**: `transaction_amount`, `account_age_days`, `transaction_hour`, `previous_failed_attempts`, `avg_transaction_amount`, `is_international`, `ip_risk_score`, `login_attempts_last_24h`

## Cleaned Dataset
- Identical to original (no imputation or row removal needed)
- Saved to `outputs/cleaned.csv`

## Issues Encountered
- None; dataset is clean and well-formed.
