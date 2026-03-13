# Task E Notes — Bug and Leakage Debugging

## Broken Pipeline Results
| Metric    | Broken  | Fixed   |
|-----------|---------|---------|
| PR-AUC    | 1.0000  | 0.0623  |
| ROC-AUC   | 1.0000  | 0.4783  |
| Precision | 1.0000  | 0.0637  |
| Recall    | 1.0000  | 0.4521  |
| F1        | 1.0000  | 0.1117  |

**Suspicious behavior**: All metrics = 1.0. A perfect classifier on a real fraud dataset
is implausible and is a reliable signal of target leakage or other data contamination.

---

## Error → Detection → Fix Mapping

### BUG-1: Target Leakage
| | Detail |
|---|---|
| **Error** | `fraud_label` was not removed from the feature matrix `X`. The model received the target column as an input feature. |
| **Detection** | All metrics equal exactly 1.0. The model could trivially predict the label because it directly observed the label during training and inference. |
| **Fix** | `X = df.drop(columns=[TARGET] + ID_COLS)` — explicitly drop `fraud_label` before constructing features. |

### BUG-2: ID Columns as Predictors
| | Detail |
|---|---|
| **Error** | `transaction_id` (string T1, T2…) and `user_id` (U1234…) were kept in the feature matrix and label-encoded. Each ID value is unique, acting as a near-perfect row index. |
| **Detection** | Unique ID features allow a model to memorise train rows and retrieve predictions at inference, inflating metrics. Checking feature importance or noticing non-numeric columns being encoded naively is a signal. |
| **Fix** | Explicitly list `ID_COLS = ["transaction_id", "user_id"]` and exclude from `X`. |

### BUG-3: Preprocessing Fitted on Full Data
| | Detail |
|---|---|
| **Error** | `StandardScaler().fit_transform(X)` was applied to the entire dataset (7,500 rows) before any train/test split. This means the scaler's mean and standard deviation were computed from test-set rows, leaking test distribution statistics into the training process. |
| **Detection** | StandardScaler fitting should only ever see training rows. Applying it before splitting is a well-known form of data leakage. |
| **Fix** | Wrap scaler inside a `sklearn.Pipeline` with `ColumnTransformer`. `pipeline.fit(X_train, y_train)` ensures the scaler is fitted ONLY on `X_train`. |

### BUG-4: Non-stratified Split + No Fixed Seed
| | Detail |
|---|---|
| **Error** | `train_test_split(…, stratify=None, random_state=None)` was used. Without stratification on a 6.52% fraud class, some splits may have significantly different class ratios (or even zero fraud cases). Without a fixed seed, results are not reproducible. |
| **Detection** | Running the broken pipeline multiple times gives different metrics each time. Inspecting train/test class distributions reveals imbalanced fraud rates. |
| **Fix** | `train_test_split(…, stratify=y, random_state=42)` — stratify on the target and fix the seed. |
