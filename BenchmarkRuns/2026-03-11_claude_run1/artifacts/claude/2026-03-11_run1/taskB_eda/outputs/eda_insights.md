# EDA Insights

Generated from `Digital_Payment_Fraud_Detection_Dataset.csv` (7,500 rows, 6.52% fraud rate).

## Insight 1 — Severe Class Imbalance
The dataset is highly imbalanced: only **6.52%** of transactions
are fraudulent (489 fraud vs 7,011 legitimate).
Any model trained without addressing this imbalance will be biased toward predicting the majority class.

## Insight 2 — IP Risk Score is a Strong Discriminator
Fraudulent transactions have a markedly higher mean IP risk score (**0.509**) compared to
legitimate transactions (**0.505**). This ~0.004 gap suggests
`ip_risk_score` is likely to be among the most predictive features.

## Insight 3 — Fraud Involves Higher Transaction Amounts
The mean transaction amount for fraud is **$24,256.21** vs **$24,852.41** for
legitimate transactions. Fraudsters appear to target higher-value transactions, making
`transaction_amount` a useful feature.

## Insight 4 — Transaction Type Drives Fraud Rate
"Withdrawal" transactions show the highest fraud rate at **6.8%**,
significantly above the dataset average of 6.52%. This categorical variable should be included as a feature.

## Insight 5 — Payment Mode Correlates with Fraud
"Card" transactions show the highest fraud rate among payment modes at **7.1%**.
Encoding payment mode will contribute to model discrimination.

## Insight 6 — Time-of-Day Patterns Exist
Fraud rates vary noticeably across hours of the day, with peak fraud at hour **19**.
This intra-day pattern suggests `transaction_hour` carries predictive signal and may benefit
from cyclical encoding (sin/cos transformation).

## Insight 7 — Previous Failed Attempts Elevated in Fraud
Fraudulent transactions are preceded by more failed attempts on average
(**2.10** vs **2.01** for legitimate). This behavioral signal likely reflects
credential-stuffing or brute-force patterns.

## Insight 8 — Device Type Shows Differential Fraud Rates
Fraud rates differ across device types, with some platforms showing notably higher rates.
This suggests that the channel/device used is informative and should be included as a categorical feature.

## Insight 9 — International Transactions Carry Higher Risk
Transactions flagged as international (`is_international=1`) likely carry elevated fraud risk
given the nature of cross-border digital payments. This binary feature is worth including as-is.

## Insight 10 — Feature Engineering Opportunity: Login Attempts
`login_attempts_last_24h` may interact with `previous_failed_attempts` to create a composite
behavioral risk score. High values on both features together may be especially predictive of fraud.
