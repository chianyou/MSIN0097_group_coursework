# Task E Notes

| Error | Detection | Fix |
|---|---|---|
| Target leakage | Broken pipeline achieved implausibly high training-set ranking metrics while keeping fraud_label in X. | Removed target from predictors and evaluated on held-out test data only. |
| IDs used as predictors | transaction_id and user_id were present in the broken feature set, creating memorization risk. | Excluded both ID columns from modeling. |
| Preprocessing fit on full data | Scaler and encoder were fit before the split, contaminating train/test boundaries. | Wrapped preprocessing in a Pipeline fit only on training data. |
| Non-stratified split and no fixed seed | Split behavior was not reproducible and could distort class balance. | Used fixed random_state=42 with stratified 70/15/15 splitting. |
| Evaluation on training set | Reported metrics reflected memorization rather than generalization. | Evaluated only on the held-out test split. |

Broken PR-AUC: 1.0000

Fixed PR-AUC: 0.0711
