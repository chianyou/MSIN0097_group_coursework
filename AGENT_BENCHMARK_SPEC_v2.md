# Agent Benchmark Specification v2

## Purpose

This benchmark is designed to compare coding/data-science agents objectively on the same fraud-detection workflow. All agents must operate under the same task definitions, success criteria, constraints, and scoring rubric.

The benchmark is intended to support coursework requirements for:
- practical exploration and benchmarking
- comparative analysis of agent tooling
- reflection on failure modes, reproducibility, and best practices

## Benchmark Principles

1. Same data
- Use the same dataset file: `./data/Digital_Payment_Fraud_Detection_Dataset.csv`

2. Same task instructions
- Every agent receives the same task specification for each task.
- Only tool-specific identifiers may differ, such as `TOOL_NAME` and `RUN_ID`.

3. Same execution budget
- Same time budget per task
- Same iteration cap per task
- Same number of clarification attempts

4. Same evaluation criteria
- All outputs are scored with the same rubric.

5. Auditability
- All commands, logs, outputs, plots, and errors must be preserved.

6. Minimal human intervention
- The human evaluator may only:
  - provide the task prompt
  - run the agent
  - answer environment questions consistently across agents
  - execute the submitted commands if the benchmark setup requires it
- The human evaluator must not manually edit the agent's code during task execution.

## Global Experimental Rules

### Fixed configuration

- `DATA_PATH = "./data/Digital_Payment_Fraud_Detection_Dataset.csv"`
- `TARGET_COL = "fraud_label"`
- `ID_COLS = ["transaction_id", "user_id"]`
- `RANDOM_SEED = 42`
- Split must be `train/valid/test = 70/15/15`
- Split must be stratified by target
- Primary metric must be `PR-AUC`
- Secondary metrics:
  - ROC-AUC
  - Precision
  - Recall
  - F1
  - Confusion matrix

### Allowed libraries

Default stack:
- pandas
- numpy
- scikit-learn
- matplotlib
- joblib

If an agent wants to use anything else, that must be:
- explicitly justified
- added to `requirements.txt`
- successfully rerunnable

### Human response policy

If an agent asks a clarifying question that is already answered in this specification, the evaluator should reply:

`Use the benchmark defaults in the specification.`

If an agent asks to relax a rule, the evaluator should reply:

`Do not change the benchmark rules. Continue with the stated constraints.`

## Standard Output Structure

Each agent must produce the following structure:

```text
<tool_root>/
  README.md
  requirements.txt
  MODEL_CARD.md
  reports/
    tool_run_summary.md
  scripts/
    run_all.sh
  src/
  artifacts/
    <tool_name>/
      <run_id>/
        taskA_data_audit/
        taskB_eda/
        taskC_baseline_model/
        taskD_model_improvement/
        taskE_bug_leakage_debug/
        taskF_reproducibility_reporting/
```

Inside each task folder:

- `spec.md`
- `commands.txt`
- `logs.txt`
- `notes.md`
- `outputs/`
- `plots/`

## Scoring Framework

Each task is scored on 5 dimensions.

### Per-dimension scale

- `0 = failed / absent`
- `1 = weak`
- `2 = partial`
- `3 = good`
- `4 = strong`

### Dimensions

1. Correctness
- Does the solution run?
- Does it satisfy the stated task spec?
- Are outputs complete?

2. Statistical Validity
- Are splits correct?
- Is leakage avoided?
- Are metrics computed on the correct split?
- Are conclusions methodologically defensible?

3. Reproducibility
- Can another team rerun it?
- Are commands, requirements, seeds, and artifacts complete?

4. Output Quality
- Code clarity
- Plot quality
- Documentation quality
- Interpretability of findings

5. Efficiency
- Number of iterations
- Time to acceptable output
- Amount of unnecessary rework

### Task weights

| Task | Weight |
|---|---:|
| Task A: Data Audit | 10 |
| Task B: EDA | 15 |
| Task C: Baseline Model | 20 |
| Task D: Model Improvement | 20 |
| Task E: Bug/Leakage Debug | 20 |
| Task F: Reproducibility/Reporting | 15 |

Total benchmark score = 100.

## Benchmark Tasks

## Task A: Data Audit

### Objective

Load the dataset and produce a structured data quality report.

### Required work

- Read the dataset from `DATA_PATH`
- Validate expected columns
- Report shape and dtypes
- Identify:
  - ID columns
  - categorical columns
  - numeric columns
  - target column
- Compute:
  - missing values by column
  - duplicate row count
  - target distribution
  - basic numeric summary stats
- Perform range checks for at least:
  - `transaction_hour` in `[0, 23]`
  - `ip_risk_score` in a plausible range
  - `fraud_label` binary
  - `is_international` binary
  - transaction amount non-negative or positive
- Produce a cleaned dataset

### Deliverables

- `outputs/data_profile.json`
- `outputs/cleaned.csv`
- `plots/missingness_bar.png`
- `notes.md` describing any anomalies

### Success criteria

- Findings are explicit even if there are no issues.
- Cleaning logic is documented.

## Task B: EDA and Insight Generation

### Objective

Produce evidence-based exploratory analysis of fraud patterns.

### Required work

Generate at least 6 plots, including:
- target distribution
- transaction amount distribution by target
- fraud rate by `transaction_type`
- fraud rate by `payment_mode`
- fraud rate by `transaction_hour`
- `ip_risk_score` distribution by target

At least one additional plot is recommended:
- fraud rate by `is_international`
- device type
- device location

Write at least 8 insights that:
- reference plots or numeric evidence
- avoid unsupported claims

### Deliverables

- `plots/*.png`
- `outputs/eda_insights.md`
- `outputs/eda_support_stats.json`

### Success criteria

- Plots are readable and correctly labeled.
- Insights are traceable to evidence.

## Task C: Baseline Model

### Objective

Build a statistically valid baseline fraud model and evaluation harness.

### Required work

- Use a stratified 70/15/15 split
- Use sklearn `Pipeline` and `ColumnTransformer`
- Fit preprocessing on training data only
- Exclude target and ID columns from features unless explicitly justified
- Baseline task must train exactly 1 baseline model
- That baseline model must be `LogisticRegression`
- Evaluate on test set only

### Required outputs

- `outputs/baseline_model.joblib`
- `outputs/metrics_baseline.json`
- `plots/confusion_matrix_baseline.png`
- `plots/pr_curve.png`
- `plots/roc_curve.png`

### Required metrics

- PR-AUC
- ROC-AUC
- Precision
- Recall
- F1
- confusion matrix

### Success criteria

- One rerun reproduces materially identical results.
- Metric discipline is correct.

## Task D: Model Improvement

### Objective

Improve model performance without violating statistical validity.

### Required work

You must evaluate at least 3 candidate models.

The candidate model pool is restricted to:
- `LogisticRegression`
- `RandomForestClassifier`
- `HistGradientBoostingClassifier`

The agent may choose the final best model, but only from this approved set.

You must also use at least 2 of the following improvement methods:
- Feature engineering
- Hyperparameter tuning
- Model change
- Threshold tuning on validation set

Rules:
- Model selection must use train/valid only
- Test set may be used only once for final evaluation
- Improvement must be explained, not only reported

### Deliverables

- `outputs/metrics_improved.json`
- `plots/confusion_matrix_improved.png`
- `plots/pr_curve_improved.png`
- `plots/roc_curve_improved.png`
- `notes.md` with a baseline vs improved comparison table

### Success criteria

- Improvement is measurable or honestly reported as not achieved.
- Method changes are clearly justified.
- The submitted report clearly shows which candidate models were tested and why the final one was selected.

## Task E: Bug and Leakage Debugging

### Objective

Detect and fix a deliberately flawed pipeline.

### Required broken-pipeline issues

The broken version must contain at least 3 of:
- target leakage
- IDs used as predictors
- preprocessing fit on full dataset
- non-stratified split
- no fixed seed
- evaluation on training set

### Required work

1. Run the broken pipeline
2. Record the suspicious behavior or failure
3. Explain why the behavior is invalid
4. Produce a corrected pipeline
5. Re-run and report fixed metrics

### Deliverables

- `src/broken_pipeline.py`
- `src/fixed_pipeline.py`
- `outputs/metrics_broken.json`
- `outputs/metrics_fixed.json`
- `notes.md` mapping:
  - error
  - detection
  - fix

### Success criteria

- The fixed version follows the same discipline as Task C.
- Explanations are methodologically correct.

## Task F: Reproducibility and Reporting

### Objective

Package the workflow into a reproducible and auditable project.

### Required work

- Create a `README.md` with setup and rerun instructions
- Create `requirements.txt`
- Create `scripts/run_all.sh`
- Create `MODEL_CARD.md`
- Create `reports/tool_run_summary.md`
- Ensure each task folder contains commands, logs, notes, outputs, and plots

### Required summary content

`tool_run_summary.md` must include:
- task completion status
- key metrics per task
- top 3 failures encountered
- estimated time spent
- iteration count
- notable strengths and weaknesses of the agent on this benchmark

### Success criteria

- A third party can understand and rerun the whole benchmark.

## Cross-Agent Comparison Protocol

After all agents complete Tasks A-F, compare them using:

1. Benchmark scores by task
2. Aggregate weighted score
3. Result table with key metrics
4. Failure-mode table
5. Qualitative observations

### Required comparison table

At minimum:

| Agent | Task A | Task B | Task C | Task D | Task E | Task F | Total |
|---|---:|---:|---:|---:|---:|---:|---:|

And one metrics-focused table:

| Agent | Baseline PR-AUC | Improved PR-AUC | Recall | Precision | Reproducible? | Major Failure |
|---|---:|---:|---:|---:|---|---|

## Recommended Benchmark Execution Settings

To improve fairness, use:

- Time budget per task: `20 minutes`
- Iteration cap per task: `8 rounds`
- Total benchmark cap per agent: `120 minutes`

If an agent finishes early, do not grant extra rounds beyond the cap unless all agents receive the same extension.

## Standard Prompt Template

Use this for each agent with only `TOOL_NAME` and `RUN_ID` changed.

```text
You are participating in a benchmark for agent tooling in data science.

Follow the benchmark specification exactly.

TOOL_NAME = "<tool_name>"
RUN_ID = "<run_id>"
DATA_PATH = "./data/Digital_Payment_Fraud_Detection_Dataset.csv"

Rules:
- Do not fabricate any results, plots, logs, or citations.
- Keep a complete audit trail.
- Use RANDOM_SEED = 42.
- Use stratified 70/15/15 train/valid/test split.
- Use PR-AUC as the primary metric.
- Complete Tasks A-F exactly as defined in the benchmark.
- If you cannot run something, label it clearly as NOT RUN and still generate runnable code and exact commands.

Required output structure:
- README.md
- requirements.txt
- MODEL_CARD.md
- reports/tool_run_summary.md
- scripts/run_all.sh
- src/
- artifacts/<tool_name>/<run_id>/taskA...taskF...

Start with Task A and proceed sequentially.
```

## What This Benchmark Still Does Not Cover

This benchmark is strong for:
- correctness
- leakage awareness
- reproducibility
- practical DS workflow support

It is weaker for:
- long-horizon autonomous planning across multiple days
- advanced statistical modeling beyond standard ML
- UI-based workflows
- external retrieval quality

These limitations should be acknowledged in the final report.

## Recommended Coursework Mapping

This benchmark supports the coursework as follows:

- Practical exploration and benchmarking:
  - Tasks A-F
- Comparative analysis:
  - weighted scoring + metrics tables + failure tables
- Reflection:
  - notes, failures, reproducibility observations

This benchmark does not replace:
- literature review
- final written synthesis
- bibliography
- group collaboration appendix

Those must still be written separately.
