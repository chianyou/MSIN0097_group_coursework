# Task E Notes
## Bugs Detected and Fixed
1. **IDs used as predictors**: The broken pipeline used `transaction_id` and `user_id` as features. Fixed by explicitly dropping these columns before training.
2. **Non-stratified split, no fixed seed**: Used `train_test_split` without `stratify=y` or `random_state`. Fixed by adding `stratify=y` and `random_state=42`.
3. **Preprocessing fit on full data**: Handled missing values (`X.fillna`) on the entire dataset prior to splitting. Fixed by using a `ColumnTransformer` pipeline that fits imputers only on the training set.
4. **Evaluation on training set**: The model generated predictions and metrics using `X_train_num`. Fixed by evaluating on `X_test`.
