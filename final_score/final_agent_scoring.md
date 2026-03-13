# Final Agent Benchmark Scoring (Consolidated)

## Run Metadata

- Date: 2026-03-13
- Evaluators: Three independent evaluations — Chienyu (v1), Stephen (v2), Chiying (v3)
- Dataset: Digital_Payment_Fraud_Detection_Dataset.csv (7,500 rows × 15 columns)
- Benchmark spec version: v2
- Consolidation method: Arithmetic mean of each scoring dimension across all three versions

## Agents

- Agent 1: Claude (claude-sonnet-4-6, Run ID: 2026-03-11_run1)
- Agent 2: Codex (Run ID: YYYY-MM-DD_run1)
- Agent 3: Gemini CLI (Run ID: 2026-03-13_run1)

## Scoring Scale

- 0 = failed / absent
- 1 = weak
- 2 = partial
- 3 = good
- 4 = strong

## Task Weights

| Task | Weight |
|---|---:|
| Task A: Data Audit | 10 |
| Task B: EDA | 15 |
| Task C: Baseline Model | 20 |
| Task D: Model Improvement | 20 |
| Task E: Bug/Leakage Debug | 20 |
| Task F: Reproducibility/Reporting | 15 |

---

## Per-Agent Rubric Sheet (Final Averaged Scores)

> Each dimension score = mean of the three evaluator scores (rounded to 2 d.p.)
> Avg Task Score = mean of the 5 dimensions
> Weighted Score = Avg Task Score × Weight (max per task = 4 × weight; max total = 400)

---

### Agent 1: Claude

| Task | Correctness | Statistical Validity | Reproducibility | Output Quality | Efficiency | Avg Task Score | Weight | Weighted Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Task A | 3.67 | 3.67 | 4.00 | 3.67 | 3.33 | 3.67 | 10 | 36.70 |
| Task B | 4.00 | 3.67 | 4.00 | 4.00 | 3.67 | 3.87 | 15 | 58.00 |
| Task C | 4.00 | 4.00 | 4.00 | 3.67 | 3.67 | 3.87 | 20 | 77.33 |
| Task D | 4.00 | 4.00 | 4.00 | 4.00 | 3.00 | 3.80 | 20 | 76.00 |
| Task E | 4.00 | 4.00 | 4.00 | 4.00 | 3.67 | 3.93 | 20 | 78.67 |
| Task F | 4.00 | 4.00 | 4.00 | 4.00 | 3.33 | 3.87 | 15 | 58.00 |
| **Total** | | | | | | | **100** | **384.70** |

**Normalised score: 96.2 / 100**

#### Per-Version Breakdown (Claude)

| Task | v1 (Chienyu) | v2 (Stephen) | v3 (Chiying) | Final Avg |
|---|---:|---:|---:|---:|
| Task A | 4.00 | 3.20 | 3.80 | 3.67 |
| Task B | 4.00 | 3.80 | 3.80 | 3.87 |
| Task C | 3.80 | 4.00 | 3.80 | 3.87 |
| Task D | 3.80 | 3.80 | 3.80 | 3.80 |
| Task E | 4.00 | 4.00 | 3.80 | 3.93 |
| Task F | 4.00 | 3.80 | 3.80 | 3.87 |
| **Total (/400)** | 392.00 | 382.00 | 380.00 | **384.70** |

#### Notes

- Task A: Outputs fully correct and complete; minor deduction for a range-check counting bug in binary column reporting (v2) and slightly below-par efficiency noted by v2/v3.
- Task B: Met plot (8) and insight (10) requirements with numeric evidence; partial deduction for a handful of statements not perfectly aligned with computed stats.
- Task C: Most rigorous baseline — sklearn Pipeline, stratified split, no leakage. ROC-AUC below 0.5 honestly explained. Small output quality deduction for near-random absolute performance.
- Task D: 3 candidates from approved pool, 3 improvement methods (FE + HP tuning + threshold tuning). Improvement modest (+6.1% PR-AUC) due to weak dataset signal. Efficiency −1 for 3 iterations.
- Task E: Unanimous full or near-full marks. 4 bugs found and fixed; detailed error-detection-fix mapping; broken PR-AUC=1.0 conclusively demonstrated.
- Task F: All required files present (README, requirements.txt, MODEL_CARD, run_all.sh, tool_run_summary). Comprehensive reproducibility checklist. Minor efficiency deduction in v2/v3.

#### Major Failure Modes

1. Weak dataset signal — all models converge near no-skill PR-AUC ≈ 0.065; improvement marginal.
2. Baseline ROC-AUC below 0.5 — requires explanation for reviewers unfamiliar with the dataset.
3. No automated hyperparameter search; manual selection required 3 iterations in Task D.

#### Major Strengths

1. End-to-end completion; all 6 tasks fully compliant with spec.
2. Honest reporting — documented dataset limitations rather than inflating results.
3. Best documentation depth — notes, tool_run_summary, and MODEL_CARD all comprehensive.

---

### Agent 2: Codex

| Task | Correctness | Statistical Validity | Reproducibility | Output Quality | Efficiency | Avg Task Score | Weight | Weighted Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Task A | 3.67 | 3.67 | 3.00 | 3.00 | 4.00 | 3.47 | 10 | 34.67 |
| Task B | 3.33 | 2.67 | 3.33 | 2.67 | 4.00 | 3.20 | 15 | 48.00 |
| Task C | 3.67 | 3.67 | 3.67 | 2.67 | 4.00 | 3.53 | 20 | 70.67 |
| Task D | 3.67 | 3.00 | 3.67 | 3.00 | 4.00 | 3.47 | 20 | 69.33 |
| Task E | 4.00 | 4.00 | 3.67 | 3.33 | 4.00 | 3.80 | 20 | 76.00 |
| Task F | 3.00 | 3.00 | 3.00 | 2.33 | 4.00 | 3.07 | 15 | 46.00 |
| **Total** | | | | | | | **100** | **344.67** |

**Normalised score: 86.2 / 100**

#### Per-Version Breakdown (Codex)

| Task | v1 (Chienyu) | v2 (Stephen) | v3 (Chiying) | Final Avg |
|---|---:|---:|---:|---:|
| Task A | 3.60 | 3.60 | 3.20 | 3.47 |
| Task B | 2.80 | 3.60 | 3.20 | 3.20 |
| Task C | 3.40 | 3.60 | 3.60 | 3.53 |
| Task D | 3.40 | 3.60 | 3.40 | 3.47 |
| Task E | 4.00 | 3.80 | 3.60 | 3.80 |
| Task F | 3.00 | 3.00 | 3.20 | 3.07 |
| **Total (/400)** | 339.00 | 355.00 | 340.00 | **344.67** |

#### Notes

- Task A: Core audit correct; data_profile.json contains non-portable absolute paths. Run ID placeholder "YYYY-MM-DD_run1" never substituted — affects traceability.
- Task B: Exactly meets the 8-insight minimum; two directional errors reduce statistical validity: Insight 2 claims fraudulent transactions have a higher mean amount (24,256.21) than non-fraudulent (24,852.41) — direction reversed, fraud mean is lower; Insight 3 claims median IP risk score is higher for fraud cases (0.485) than non-fraud (0.503) — direction reversed, fraud median is lower. Statistical validity docked across all 3 evaluators.
- Task C: Valid LR baseline and pipeline. PR-AUC=0.0711. Precision=Recall=F1=0 at default threshold=0.5 (model predicts all-negative) — anomaly flagged but not explained in notes.
- Task D: Largest absolute PR-AUC improvement (0.0711→0.1044). However selected threshold=0.25 yields recall=1.0 (essentially flags every transaction as fraud), which is a degenerate classifier. Not discussed in notes.
- Task E: Best-in-class — detected 5 bugs including eval-on-training-set missed by other agents. Clear error-detection-fix table; small deduction in v1/v3 for brevity.
- Task F: Files present but tool_run_summary is brief; model card has limited depth; run ID placeholder throughout.

#### Major Failure Modes

1. EDA Insights 2 and 3 contain directional errors: Insight 2 reverses the fraud vs non-fraud mean transaction amount (fraud mean 24,256.21 is actually lower, not higher, than non-fraud mean 24,852.41); Insight 3 reverses the median IP risk score direction (fraud median 0.485 is actually lower, not higher, than non-fraud median 0.503).
2. Task D threshold=0.25 produces degenerate recall=1.0; not flagged or explained.
3. Run ID placeholder "YYYY-MM-DD_run1" never substituted — incomplete benchmark setup.

#### Major Strengths

1. Highest raw improved PR-AUC (0.1044 vs 0.0661 / 0.0679).
2. Best Task E score — detected 5 bugs; clean error-detection-fix mapping.
3. Fastest end-to-end execution with consistent 1-iteration completion.

---

### Agent 3: Gemini CLI

| Task | Correctness | Statistical Validity | Reproducibility | Output Quality | Efficiency | Avg Task Score | Weight | Weighted Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Task A | 3.33 | 3.33 | 3.33 | 2.00 | 4.00 | 3.20 | 10 | 32.00 |
| Task B | 2.67 | 2.00 | 3.33 | 2.00 | 4.00 | 2.80 | 15 | 42.00 |
| Task C | 3.33 | 3.00 | 3.33 | 2.33 | 3.67 | 3.13 | 20 | 62.67 |
| Task D | 2.67 | 2.33 | 3.00 | 2.33 | 4.00 | 2.87 | 20 | 57.33 |
| Task E | 3.00 | 3.00 | 2.67 | 2.33 | 3.67 | 2.93 | 20 | 58.67 |
| Task F | 2.33 | 2.33 | 2.67 | 1.67 | 4.00 | 2.60 | 15 | 39.00 |
| **Total** | | | | | | | **100** | **291.67** |

**Normalised score: 72.9 / 100**

#### Per-Version Breakdown (Gemini)

| Task | v1 (Chienyu) | v2 (Stephen) | v3 (Chiying) | Final Avg |
|---|---:|---:|---:|---:|
| Task A | 3.00 | 3.60 | 3.00 | 3.20 |
| Task B | 2.20 | 3.00 | 3.20 | 2.80 |
| Task C | 2.60 | 3.60 | 3.20 | 3.13 |
| Task D | 2.80 | 2.60 | 3.20 | 2.87 |
| Task E | 3.00 | 2.20 | 3.60 | 2.93 |
| Task F | 2.20 | 2.80 | 2.80 | 2.60 |
| **Total (/400)** | 264.00 | 291.00 | 320.00 | **291.67** |

#### Notes

- Task A: Core stats correct; notes are only two generic sentences; cleaned.csv has a malformed row (T2 missing a field); range_checks output format non-standard.
- Task B: 8 insights meet the minimum, but Insight 7 incorrectly claims missing values exist. Remaining insights are vague with no numeric citations. EDA support stats cover only 4 dimensions.
- Task C: LR baseline runs and metrics file is present. Precision=Recall=F1=0 same as Codex but notes are a single generic line with no analysis.
- Task D: Improved PR-AUC (0.0679) is lower than baseline (0.0711) — a regression. Comparison table uses "(see Task C)" instead of the actual baseline value, obscuring the failure. Only 2 improvement methods applied (no feature engineering).
- Task E: 4 bugs identified; however metrics_broken.json only shows Accuracy_Train, not PR-AUC=1.0 — the key leakage evidence is absent, reducing auditability.
- Task F: tool_run_summary is a generic template document with minimal specifics. Task F notes are one vague sentence. Model card in v3 noted to be inconsistent with the final selected model.

#### Major Failure Modes

1. Unacknowledged PR-AUC regression in Task D (improved=0.0679 < baseline=0.0711; not disclosed).
2. Pervasive documentation quality gap — notes across all tasks are 1–2 generic sentences.
3. metrics_broken.json missing PR-AUC — key leakage evidence absent from Task E.

#### Major Strengths

1. Complete structural compliance — all 6 task folders with required subdirectory layout.
2. Model governance rules met — 3 approved candidates, validated selection, ≥2 methods.
3. Fast single-pass execution with no iteration failures reported.

---

## Cross-Agent Comparison

| Agent | Task A | Task B | Task C | Task D | Task E | Task F | **Total (/400)** | **Score (/100)** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Claude | 36.70 | 58.00 | 77.33 | 76.00 | 78.67 | 58.00 | **384.70** | **96.2** |
| Codex | 34.67 | 48.00 | 70.67 | 69.33 | 76.00 | 46.00 | **344.67** | **86.2** |
| Gemini | 32.00 | 42.00 | 62.67 | 57.33 | 58.67 | 39.00 | **291.67** | **72.9** |

*Weighted Score = Avg Task Score × Task Weight. Max per task = 4 × weight. Max total = 400.*

## Metrics Comparison Table

| Agent | Baseline PR-AUC | Improved PR-AUC | Δ PR-AUC | Precision | Recall | Reproducible? | Major Failure |
|---|---:|---:|---:|---:|---:|---|---|
| Claude | 0.0623 | 0.0661 | +0.0038 | 0.0647 | 0.3014 | Yes | Weak dataset signal limits improvement |
| Codex | 0.0711 | 0.1044 | +0.0333 | 0.0659 | 1.0000 | Partial | Degenerate threshold (recall=1.0); EDA directional errors in Insights 2 & 3; placeholder run ID |
| Gemini | 0.0711 | 0.0679 | −0.0032 | 0.0569 | 0.2568 | Partial | PR-AUC regression undisclosed; minimal documentation; incomplete broken metrics |

## Model Governance Checks

| Agent | Baseline model count | Baseline type correct? | Candidate models tested | Approved pool only? | Improvement methods count | Pass / Fail |
|---|---:|---|---:|---|---|---|
| Claude | 1 | Yes (LogisticRegression) | 3 (LR, RF, HGBC) | Yes | 3 (FE + HP tuning + threshold) | **PASS** |
| Codex | 1 | Yes (LogisticRegression) | 3 (LR+FE, RF_shallow, HGBC_default) | Yes | 4 (model + FE + HP + threshold) | **PASS** |
| Gemini | 1 | Yes (LogisticRegression) | 3 (LR, RF, HGBC) | Yes | 2 (model change + threshold) | **PASS** (minimum) |

## Final Ranking

1. **Claude** — 384.70 / 400 (96.2 / 100)
2. **Codex** — 344.67 / 400 (86.2 / 100)
3. **Gemini** — 291.67 / 400 (72.9 / 100)

## Final Summary

| Category | Winner | Runner-up |
|---|---|---|
| Most reliable agent | **Claude** | Codex |
| Best statistical discipline | **Claude** | Codex |
| Best reproducibility | **Claude** | Codex |
| Highest PR-AUC improvement | **Codex** (+0.0333) | Claude (+0.0038) |
| Fastest to acceptable output | **Gemini / Codex** | — |
| Most common failure pattern | EDA factual errors: Gemini Insight 7 (false missing-value claim); Codex Insights 2 & 3 (directional reversals on amount and IP risk score) + all-zero Precision/Recall/F1 at default threshold (Codex & Gemini) | — |

**Recommendation for future use:**
- **Claude** — production benchmark use; best for rigour, auditability, and honest reporting.
- **Codex** — fast prototyping where documentation depth is secondary; add explicit threshold review step and run-ID substitution check before finalising.
- **Gemini** — requires additional prompting for documentation standards, factual EDA verification, and PR-AUC comparison checks before results can be trusted.
