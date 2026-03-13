# Task E: Bug and Leakage Debugging — Specification

## Goal
Detect and fix a deliberately flawed pipeline.

## Required Bugs (≥3 of)
- target leakage ✅
- IDs used as predictors ✅
- preprocessing fit on full data ✅
- non-stratified split ✅
- no fixed seed (included in BUG-4)

## Must Do
1. Run the broken pipeline
2. Record the suspicious behavior
3. Explain why the behavior is invalid
4. Produce a corrected pipeline
5. Re-run and report fixed metrics

## Deliverables
- `src/broken_pipeline.py`
- `src/fixed_pipeline.py`
- `outputs/metrics_broken.json`
- `outputs/metrics_fixed.json`
- `notes.md` mapping error → detection → fix
