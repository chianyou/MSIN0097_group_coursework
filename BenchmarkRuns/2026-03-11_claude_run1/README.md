# Digital Payment Fraud Detection — Benchmark Run

**Tool**: `claude`
**Run ID**: `2026-03-11_run1`
**Date**: 2026-03-12
**Dataset**: `Digital_Payment_Fraud_Detection_Dataset.csv` (7,500 rows, 15 columns)
**Primary Metric**: PR-AUC

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (Tasks A–F)
bash scripts/run_all.sh
```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── MODEL_CARD.md
├── data/
│   └── Digital_Payment_Fraud_Detection_Dataset.csv
├── src/
│   ├── task_a_data_audit.py
│   ├── task_b_eda.py
│   ├── task_c_baseline.py
│   ├── task_d_improvement.py
│   ├── broken_pipeline.py
│   └── fixed_pipeline.py
├── scripts/
│   └── run_all.sh
├── reports/
│   └── tool_run_summary.md
└── artifacts/
    └── claude/
        └── 2026-03-11_run1/
            ├── taskA_data_audit/
            ├── taskB_eda/
            ├── taskC_baseline_model/
            ├── taskD_model_improvement/
            ├── taskE_bug_leakage_debug/
            └── taskF_reproducibility_reporting/
```

## Configuration

| Parameter      | Value                                           |
|----------------|-------------------------------------------------|
| `TOOL_NAME`    | `claude`                                        |
| `RUN_ID`       | `2026-03-11_run1`                               |
| `DATA_PATH`    | `./data/Digital_Payment_Fraud_Detection_Dataset.csv` |
| `TARGET_COL`   | `fraud_label`                                   |
| `ID_COLS`      | `["transaction_id", "user_id"]`                 |
| `RANDOM_SEED`  | `42`                                            |
| `SPLIT`        | `70/15/15` stratified                           |

## Results Summary

| Task | Outcome | Key Metric |
|------|---------|-----------|
| A: Data Audit | ✅ Complete | 0 missing, 0 duplicates, 6.52% fraud rate |
| B: EDA | ✅ Complete | 8 plots, 10 insights |
| C: Baseline | ✅ Complete | PR-AUC = 0.0623 (LogisticRegression) |
| D: Improvement | ✅ Complete | PR-AUC = 0.0661 (+6.1% over baseline) |
| E: Bug Debug | ✅ Complete | Broken PR-AUC=1.0 → Fixed PR-AUC=0.0623 |
| F: Reporting | ✅ Complete | Full audit trail created |

## Important Note on Model Performance

All models converge near the no-skill baseline (PR-AUC ≈ 0.065). Investigation revealed that
all numeric features have |r| < 0.03 with `fraud_label`, and categorical fraud rates vary by
only ±1% from the base rate. This dataset appears to be synthetic with intentionally weak
fraud signals. Results are honest and reproducible — no data leakage was introduced.

## Reproducibility

All scripts use `RANDOM_SEED = 42`. Running `bash scripts/run_all.sh` from the project root
will regenerate all artifacts deterministically.
