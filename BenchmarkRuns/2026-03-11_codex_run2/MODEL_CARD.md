# Model Card

## Overview

The final selected fraud model was `RandomForestClassifier_shallow` chosen using validation PR-AUC under a stratified 70/15/15 split.

## Intended use

Binary fraud screening support for digital payment transactions.

## Training data

Source file: `/Users/bettylin/Downloads/Predictive Group/BenchmarkRuns/2026-03-11_codex_run2/data/Digital_Payment_Fraud_Detection_Dataset.csv` with target `fraud_label`.

## Metrics

- Baseline PR-AUC: 0.0711
- Improved PR-AUC: 0.1044
- Improved ROC-AUC: 0.5312
- Improved precision: 0.0659
- Improved recall: 1.0000
- Improved F1: 0.1236

## Limitations

- Single dataset benchmark only.
- Threshold selected on one validation split rather than cross-validation.
- Potential real-world drift and fairness issues were not evaluated in this benchmark.
