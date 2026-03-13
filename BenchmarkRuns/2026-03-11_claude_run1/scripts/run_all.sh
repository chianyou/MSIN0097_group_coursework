#!/usr/bin/env bash
# run_all.sh — Full benchmark pipeline
# Usage: bash scripts/run_all.sh
# Requires: python3 with packages in requirements.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================="
echo " Digital Payment Fraud Detection Benchmark"
echo " TOOL=claude  RUN_ID=2026-03-11_run1"
echo " $(date)"
echo "============================================="

cd "$ROOT"

echo ""
echo "[1/6] Task A: Data Audit"
python3 src/task_a_data_audit.py

echo ""
echo "[2/6] Task B: EDA and Insight Generation"
python3 src/task_b_eda.py

echo ""
echo "[3/6] Task C: Baseline Model (LogisticRegression)"
python3 src/task_c_baseline.py

echo ""
echo "[4/6] Task D: Model Improvement"
python3 src/task_d_improvement.py

echo ""
echo "[5/6] Task E: Bug & Leakage Debugging"
python3 src/broken_pipeline.py
python3 src/fixed_pipeline.py

echo ""
echo "[6/6] Task F: Reproducibility artifacts already created."
echo "      See: README.md, requirements.txt, MODEL_CARD.md, reports/"

echo ""
echo "============================================="
echo " All tasks complete."
echo "============================================="
