# Benchmark Final Summary — Digital Payment Fraud Detection

**Dataset:** Digital_Payment_Fraud_Detection_Dataset.csv (7,500 rows, 15 columns, 6.52% fraud rate)
**Consolidation date:** 2026-03-13
**Evaluations merged:** v1 (Chienyu), v2 (Stephen), v3 (Chiying)
**Method:** Arithmetic mean of per-dimension scores across all three evaluations

---

## Final Scores

| Rank | Agent | Weighted Score (/400) | Normalised (/100) |
|---|---|---:|---:|
| 🥇 1 | **Claude** | 384.70 | **96.2** |
| 🥈 2 | **Codex** | 344.67 | **86.2** |
| 🥉 3 | **Gemini** | 291.67 | **72.9** |

---

## Per-Task Final Scores

| Task (Weight) | Claude | Codex | Gemini |
|---|---:|---:|---:|
| A — Data Audit (×10) | **36.70** | 34.67 | 32.00 |
| B — EDA (×15) | **58.00** | 48.00 | 42.00 |
| C — Baseline Model (×20) | **77.33** | 70.67 | 62.67 |
| D — Model Improvement (×20) | **76.00** | 69.33 | 57.33 |
| E — Bug / Leakage Debug (×20) | **78.67** | **76.00** | 58.67 |
| F — Reproducibility (×15) | **58.00** | 46.00 | 39.00 |
| **Total (/400)** | **384.70** | **344.67** | **291.67** |

---

## Version-by-Version Score Comparison

| Agent | v1 (Chienyu) | v2 (Stephen) | v3 (Chiying) | **Final Avg** | Spread |
|---|---:|---:|---:|---:|---:|
| Claude | 392 | 382 | 380 | **384.7** | ±6 |
| Codex | 339 | 355 | 340 | **344.7** | ±8 |
| Gemini | 264 | 291 | 320 | **291.7** | ±28 |

> All v2 scores normalised from /100 to /400 for comparison.
> Gemini shows the highest inter-evaluator variance (±28), reflecting genuine ambiguity in how its sparse output was assessed.

---

## Key Metrics

| Agent | Baseline PR-AUC | Improved PR-AUC | Δ PR-AUC | Recall | Model Governance |
|---|---:|---:|---:|---:|---|
| Claude | 0.0623 | 0.0661 | **+0.0038** | 0.301 | ✅ PASS |
| Codex | 0.0711 | **0.1044** | **+0.0333** | 1.000* | ✅ PASS |
| Gemini | 0.0711 | 0.0679 | **−0.0032** | 0.257 | ✅ PASS (min) |

*Codex recall=1.0 is degenerate: threshold=0.25 flags virtually all transactions as fraud.

---

## Critical Findings by Agent

### Claude ✅
- Strongest documentation and reproducibility across all tasks.
- Honest reporting of weak dataset signal — did not inflate metrics.
- Minor deductions: near-random absolute performance (dataset limitation), no automated hyperparameter search.

### Codex ⚠️
- Highest raw PR-AUC improvement, but the improved model is practically degenerate (recall=1.0).
- EDA contains directional errors: Insight 2 incorrectly states fraud has a higher mean transaction amount than non-fraud (fraud mean 24,256.21 is actually lower than non-fraud mean 24,852.41); Insight 3 incorrectly states fraud has a higher median IP risk score (fraud median 0.485 is actually lower than non-fraud median 0.503).
- Run ID placeholder "YYYY-MM-DD_run1" was never replaced — reduces traceability.
- Best Task E score: detected 5 bugs (most of any agent).

### Gemini ⚠️
- Only agent where the "improved" model performs worse than the baseline on the primary metric.
- Documentation quality is the weakest: notes are generic 1–2 sentence summaries throughout.
- EDA Insight 7 incorrectly claims the dataset contains missing values (0 missing values confirmed); a distinct error from Codex's.
- metrics_broken.json missing PR-AUC — the key leakage evidence is absent.

---

## Recommendations

| Use Case | Recommended Agent |
|---|---|
| Production benchmark / auditable submission | **Claude** |
| Fast prototype + high PR-AUC (with threshold review) | **Codex** |
| Quick structural draft (with mandatory manual review) | **Gemini** |

---

## Files in This Directory

| File | Contents |
|---|---|
| `final_agent_scoring.md` | Full rubric with per-version breakdown, notes, and narrative for all 3 agents |
| `final_agent_scoring.csv` | Machine-readable per-task scores with all three evaluator values and averages |
| `final_model_governance_checklist.md` | Consolidated model governance compliance table with evidence sources |
| `final_model_governance_checklist.csv` | Machine-readable governance checklist |
| `final_summary.md` | This file — executive summary of final rankings and key findings |
