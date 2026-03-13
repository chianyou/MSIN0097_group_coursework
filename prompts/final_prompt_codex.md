# Final Benchmark Prompt for Codex

You are participating in a controlled benchmark for agent tooling in data science.

Follow the benchmark specification exactly.

## Fixed Configuration

- `TOOL_NAME = "codex"`
- `RUN_ID = "YYYY-MM-DD_run1"`
- `DATA_PATH = "./data/Digital_Payment_Fraud_Detection_Dataset.csv"`
- `TARGET_COL = "fraud_label"`
- `ID_COLS = ["transaction_id", "user_id"]`
- `RANDOM_SEED = 42`

## Hard Rules

1. Do not fabricate any results, runs, plots, logs, or citations.
2. Use a stratified `70/15/15` train/valid/test split.
3. Use `PR-AUC` as the primary metric.
4. Preserve a full audit trail for every task.
5. If you cannot run something, label it clearly as `NOT RUN`, but still generate runnable code and exact commands.
6. Do not change the benchmark rules or dataset.
7. Minimize human intervention; proceed using the stated defaults.

## Allowed Python Stack

Default stack only unless strictly necessary:
- pandas
- numpy
- scikit-learn
- matplotlib
- joblib

If you use any extra dependency, add it to `requirements.txt` and make sure the project remains rerunnable.

## Required Output Structure

Create this structure:

```text
README.md
requirements.txt
MODEL_CARD.md
reports/
  tool_run_summary.md
scripts/
  run_all.sh
src/
artifacts/
  codex/
    <RUN_ID>/
      taskA_data_audit/
      taskB_eda/
      taskC_baseline_model/
      taskD_model_improvement/
      taskE_bug_leakage_debug/
      taskF_reproducibility_reporting/
```

Inside each task folder include:
- `spec.md`
- `commands.txt`
- `logs.txt`
- `notes.md`
- `outputs/`
- `plots/`

## Tasks

### Task A: Data Audit

Goal:
- Load the dataset and produce a structured data-quality report.

Must do:
- Validate expected columns
- Report shape and dtypes
- Identify ID, categorical, numerical, and target columns
- Compute missing values, duplicate rows, target distribution, and numeric summary stats
- Run range checks for:
  - `transaction_hour`
  - `ip_risk_score`
  - `fraud_label`
  - `is_international`
  - transaction amount positivity
- Output a cleaned dataset

Deliverables:
- `outputs/data_profile.json`
- `outputs/cleaned.csv`
- `plots/missingness_bar.png`

### Task B: EDA and Insight Generation

Goal:
- Produce evidence-based EDA of fraud patterns.

Must do:
- Generate at least 6 plots including:
  - target distribution
  - transaction amount distribution by target
  - fraud rate by `transaction_type`
  - fraud rate by `payment_mode`
  - fraud rate by `transaction_hour`
  - `ip_risk_score` distribution by target
- Write at least 8 evidence-based insights

Deliverables:
- `plots/*.png`
- `outputs/eda_insights.md`
- `outputs/eda_support_stats.json`

### Task C: Baseline Model

Goal:
- Build a statistically valid baseline fraud model.

Must do:
- Use stratified `70/15/15`
- Use sklearn `Pipeline` and `ColumnTransformer`
- Fit preprocessing on training data only
- Exclude target and ID columns from features unless explicitly justified
- Train exactly `1` baseline model
- That baseline model must be `LogisticRegression`
- Evaluate on test set only

Deliverables:
- `outputs/baseline_model.joblib`
- `outputs/metrics_baseline.json`
- `plots/confusion_matrix_baseline.png`
- `plots/pr_curve.png`
- `plots/roc_curve.png`

Required metrics:
- PR-AUC
- ROC-AUC
- Precision
- Recall
- F1
- confusion matrix

### Task D: Model Improvement

Goal:
- Improve performance without violating statistical validity.

You must test at least `3` candidate models.

Candidate models are restricted to this set only:
- `LogisticRegression`
- `RandomForestClassifier`
- `HistGradientBoostingClassifier`

You may select the final best model, but it must come from this approved set.

You must also do at least `2` improvement methods from:
- feature engineering
- hyperparameter tuning
- model change
- threshold tuning on validation set

Rules:
- Use train/valid for model selection
- Use test only once for final evaluation
- Explain what changed and why

Deliverables:
- `outputs/metrics_improved.json`
- `plots/confusion_matrix_improved.png`
- `plots/pr_curve_improved.png`
- `plots/roc_curve_improved.png`
- `notes.md` with a baseline-vs-improved comparison table
- `notes.md` must also list all tested candidate models and the validation results used to select the final model

### Task E: Bug and Leakage Debugging

Goal:
- Detect and fix a deliberately flawed pipeline.

Broken version must contain at least 3 of:
- target leakage
- IDs used as predictors
- preprocessing fit on full data
- non-stratified split
- no fixed seed
- evaluation on training set

Must do:
1. Run the broken pipeline
2. Record the suspicious behavior or failure
3. Explain why the behavior is invalid
4. Produce a corrected pipeline
5. Re-run and report fixed metrics

Deliverables:
- `src/broken_pipeline.py`
- `src/fixed_pipeline.py`
- `outputs/metrics_broken.json`
- `outputs/metrics_fixed.json`
- `notes.md` mapping error -> detection -> fix

### Task F: Reproducibility and Reporting

Goal:
- Package the full workflow into a reproducible, auditable project.

Must do:
- Create `README.md`
- Create `requirements.txt`
- Create `scripts/run_all.sh`
- Create `MODEL_CARD.md`
- Create `reports/tool_run_summary.md`
- Ensure each task folder has commands, logs, notes, outputs, and plots

`tool_run_summary.md` must include:
- task completion status
- key metrics per task
- top 3 failures encountered
- estimated time spent
- iteration count
- strengths and weaknesses of the agent on this benchmark

## Execution Instructions

Start with Task A and proceed sequentially through Task F.

Do not stop after planning. Produce the full project structure and all required artifacts.
