# Final Model Governance Checklist (Consolidated)

All three evaluator versions agreed on the model governance assessment. This document reflects the unanimous findings.

## Approved Rules (from Benchmark Spec v2)

| Rule | Requirement |
|---|---|
| Baseline model count | Exactly 1 |
| Baseline model type | LogisticRegression |
| Candidate models tested | At least 3 |
| Approved model pool | LogisticRegression, RandomForestClassifier, HistGradientBoostingClassifier |
| Final model must be from approved pool | Yes |
| Minimum improvement methods | At least 2 |
| Selection evidence | Must use validation set (not test set) |

---

## Final Governance Checklist

| Agent | Run ID | Baseline count | Baseline type | Baseline correct? | Candidates tested | Candidate names | Approved pool only? | Methods count | Methods applied | Final model | Val-based selection? | Pass / Fail | Notes |
|---|---|---:|---|---|---:|---|---|---:|---|---|---|---|---|
| Claude | 2026-03-11_run1 | 1 | LogisticRegression | ✅ Yes | 3 | LogisticRegression, RandomForestClassifier, HistGradientBoostingClassifier | ✅ Yes | 3 | Feature engineering (cyclic hour, interaction terms, amount ratio); Hyperparameter tuning (L1, C=0.5, solver=saga); Threshold tuning (max-F1 on val, threshold=0.532) | LogisticRegression | ✅ Yes — val PR-AUC: LR=0.0837 > RF=0.0779 > HGBC=0.0674 | ✅ **PASS** | Full governance compliance. All rules met with margin. |
| Codex | YYYY-MM-DD_run1 | 1 | LogisticRegression | ✅ Yes | 3 | LogisticRegression_baseline_plus_fe, RandomForestClassifier_shallow, HistGradientBoostingClassifier_default | ✅ Yes | 4 | Model change; Hyperparameter tuning (candidate-specific); Feature engineering (amount ratio, high-risk-hour flag, amount-risk interaction); Threshold tuning (val split, threshold=0.25) | RandomForestClassifier_shallow | ✅ Yes — val PR-AUC: RF=0.0689 > LR=0.0648 > HGBC=0.0578 | ✅ **PASS** | Full governance compliance. Note: run ID placeholder "YYYY-MM-DD_run1" never substituted. Final threshold (0.25) produces degenerate recall=1.0 — rules-compliant but practically concerning. |
| Gemini | 2026-03-13_run1 | 1 | LogisticRegression | ✅ Yes | 3 | LogisticRegression, RandomForestClassifier, HistGradientBoostingClassifier | ✅ Yes | 2 | Model change (across 3 candidates); Threshold tuning (val split, threshold=0.10) | RandomForestClassifier | ✅ Yes — val PR-AUC: RF=0.0668 > LR=0.0659 > HGBC=0.0578 | ✅ **PASS** (minimum) | Minimum compliance only — exactly 2 methods, no feature engineering. Improved test PR-AUC (0.0679) lower than baseline (0.0711) — rules-compliant but goal not achieved. Model card inconsistency noted (v3). |

---

## Evidence Sources Verified

| Agent | metrics_baseline.json | metrics_improved.json | Task D notes.md | tool_run_summary.md | Training scripts |
|---|---|---|---|---|---|
| Claude | ✅ PR-AUC=0.0623, model=LogisticRegression | ✅ PR-AUC=0.0661, 3 candidates with val results, 3 methods listed | ✅ Full comparison table, candidate val PR-AUC, all methods explained | ✅ Comprehensive | ✅ task_c_baseline.py, task_d_improvement.py |
| Codex | ✅ PR-AUC=0.0711, model=LogisticRegression | ✅ PR-AUC=0.1044, 3 candidates with val results, 4 methods listed | ✅ Comparison table with val PR-AUC | ⚠️ Present but brief | ✅ benchmark_runner.py |
| Gemini | ✅ PR-AUC=0.0711 | ✅ PR-AUC=0.0679, 3 candidates in notes | ⚠️ Minimal — baseline value shown as "(see Task C)"; regression undisclosed | ⚠️ Generic/brief | ✅ taskC.py, taskD.py |

---

## Summary

| Agent | Governance Status | Improvement Effective? | Key Flag |
|---|---|---|---|
| Claude | ✅ **PASS — full compliance** | Yes (+6.1% PR-AUC) | None |
| Codex | ✅ **PASS — full compliance** | Yes (numerically), but degenerate | Threshold=0.25 yields recall=1.0; run ID placeholder |
| Gemini | ✅ **PASS — minimum compliance** | No (regression) | Improved PR-AUC < baseline; undisclosed |
