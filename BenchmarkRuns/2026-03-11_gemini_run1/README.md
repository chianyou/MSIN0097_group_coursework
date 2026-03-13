# Digital Payment Fraud Detection Benchmark

This repository contains the results of the agent tooling benchmark for the Digital Payment Fraud Detection dataset.

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the pipeline:
   ```bash
   sh scripts/run_all.sh
   ```

## Directory Structure

- `src/`: Python source code for each task.
- `scripts/`: Shell scripts to execute the pipeline.
- `reports/`: Markdown reports including the tool run summary.
- `artifacts/`: Output directory containing models, plots, metrics, and logs organized by task.
