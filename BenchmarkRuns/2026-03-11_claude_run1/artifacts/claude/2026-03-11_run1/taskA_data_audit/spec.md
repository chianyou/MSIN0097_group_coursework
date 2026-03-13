# Task A: Data Audit — Specification

## Goal
Load the dataset and produce a structured data-quality report.

## Must Do
- Validate expected columns
- Report shape and dtypes
- Identify ID, categorical, numerical, and target columns
- Compute missing values, duplicate rows, target distribution, and numeric summary stats
- Run range checks for: `transaction_hour`, `ip_risk_score`, `fraud_label`, `is_international`, transaction amount positivity
- Output a cleaned dataset

## Deliverables
- `outputs/data_profile.json`
- `outputs/cleaned.csv`
- `plots/missingness_bar.png`
