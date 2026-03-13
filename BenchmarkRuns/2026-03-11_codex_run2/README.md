# Digital Payment Fraud Detection Benchmark

This project implements the full Codex benchmark workflow for fraud detection using the dataset at `/Users/bettylin/Downloads/Predictive Group/BenchmarkRuns/2026-03-11_codex_run2/data/Digital_Payment_Fraud_Detection_Dataset.csv`.

## Run

```bash
bash scripts/run_all.sh
```

## Benchmark defaults

- Tool: `codex`
- Run ID: `YYYY-MM-DD_run1`
- Target column: `fraud_label`
- ID columns: `['transaction_id', 'user_id']`
- Random seed: `42`
- Primary metric: `PR-AUC`

## Structure

- `src/benchmark_runner.py`: orchestrates Tasks A-F
- `src/broken_pipeline.py`: intentionally invalid pipeline for Task E
- `src/fixed_pipeline.py`: corrected Task E pipeline
- `artifacts/codex/YYYY-MM-DD_run1/...`: task-level audit trail, outputs, plots, logs, and notes
