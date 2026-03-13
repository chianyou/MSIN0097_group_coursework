#!/bin/bash
set -e

echo "Starting Benchmark Pipeline"

echo "Running Task A: Data Audit..."
python3 src/taskA.py

echo "Running Task B: EDA..."
python3 src/taskB.py

echo "Running Task C: Baseline Model..."
python3 src/taskC.py

echo "Running Task D: Model Improvement..."
python3 src/taskD.py

echo "Running Task E: Bug and Leakage Debugging..."
python3 src/broken_pipeline.py || true
python3 src/fixed_pipeline.py

echo "Task F artifacts generated directly via agent."

echo "Pipeline complete."
