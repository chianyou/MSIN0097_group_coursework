# Task B Notes

## Plots Generated (8 total, requirement ≥6)
1. `01_target_distribution.png` — class imbalance bar chart
2. `02_transaction_amount_by_target.png` — histograms by label
3. `03_fraud_rate_by_transaction_type.png` — fraud rate per transaction type
4. `04_fraud_rate_by_payment_mode.png` — fraud rate per payment mode
5. `05_fraud_rate_by_hour.png` — hourly fraud rate
6. `06_ip_risk_score_by_target.png` — IP risk score density by label
7. `07_fraud_rate_by_device_type.png` — fraud rate per device type
8. `08_failed_attempts_by_target.png` — failed attempts density by label

## Insights Generated (10 total, requirement ≥8)
See `outputs/eda_insights.md` for full text.

## Top Predictors Identified
1. `ip_risk_score` — largest distributional gap between classes
2. `transaction_amount` — fraud skews higher
3. `previous_failed_attempts` — elevated in fraud
4. `transaction_type`, `payment_mode` — categorical signals
5. `transaction_hour` — time-of-day pattern
