# MSIN0097 Predictive Analytics — Group Coursework
## AI Agent Benchmark: Digital Payment Fraud Detection

**Module:** MSIN0097 Predictive Analytics 2025–26
**Dataset:** Digital_Payment_Fraud_Detection_Dataset.csv (7,500 rows × 15 columns, 6.52% fraud rate)
**Benchmark date:** 2026-03-11 / 2026-03-13
**Agents evaluated:** Claude (Anthropic), Codex (OpenAI), Gemini CLI (Google)

---

## 📄 Coursework Submission

| Deliverable | Path |
|---|---|
| **Executive Report (Word)** | `MSIN0097_Group_Coursework.docx` |
| **Coursework Brief (PDF)** | `MSIN0097_ Predictive Analytics 25-26 Group Coursework.pdf` |

> **Start here for grading:** Open `MSIN0097_Group_Coursework.docx` — the 2,000-word executive report covers literature review, benchmark methodology, results, comparative analysis, failure modes, and recommendations.

---

## 🗂 Project Structure

```
Predictive Group/
│
├── README.md                              ← This file
├── MSIN0097_Group_Coursework.docx         ← ★ Final submission (Word report)
├── AGENT_BENCHMARK_SPEC_v2.md             ← Benchmark rules, rubric & task specs
├── Digital_Payment_Fraud_Detection_Dataset.csv  ← Original dataset (root copy)
│
├── prompts/                               ← Prompts fed to each agent
│   ├── final_prompt_claude.md
│   ├── final_prompt_codex.md
│   └── final_prompt_gemini.md
│
├── BenchmarkRuns/                         ← Raw agent outputs (one folder per agent)
│   ├── 2026-03-11_claude_run1/            ← Claude run
│   ├── 2026-03-11_codex_run2/             ← Codex run
│   └── 2026-03-11_gemini_run1/            ← Gemini CLI run
│
└── final_score/                           ← ★ Final consolidated evaluation
    ├── final_summary.md                   ← Executive score summary & rankings
    ├── final_agent_scoring.md             ← Full rubric with per-evaluator breakdown
    ├── final_agent_scoring.csv            ← Machine-readable scores (all dimensions)
    ├── final_model_governance_checklist.md← Governance compliance table
    └── final_model_governance_checklist.csv
```

---

## 🏆 Final Results (Quick Reference)

| Rank | Agent | Score (/100) | Weighted (/400) |
|---|---|---:|---:|
| 🥇 1 | **Claude** | **96.2** | 384.70 |
| 🥈 2 | **Codex** | **86.2** | 344.67 |
| 🥉 3 | **Gemini CLI** | **72.9** | 291.67 |

Primary metric: **PR-AUC** (Precision-Recall Area Under Curve)

| Agent | Baseline PR-AUC | Improved PR-AUC | Δ |
|---|---:|---:|---:|
| Claude | 0.0623 | 0.0661 | +0.0038 |
| Codex | 0.0711 | 0.1044 | +0.0333 |
| Gemini | 0.0711 | 0.0679 | −0.0032 |

---

## 📐 Benchmark Design

### Task Pipeline (6 tasks, identical prompt for all agents)

| Task | Name | Weight | Key Deliverables |
|---|---|---:|---|
| **A** | Data Audit | 10 | `data_profile.json`, `cleaned.csv` |
| **B** | Exploratory Data Analysis | 15 | `eda_insights.md` (≥10), `eda_support_stats.json`, 8 plots |
| **C** | Baseline Model | 20 | `metrics_baseline.json`, `baseline_model.joblib`, PR/ROC curves |
| **D** | Model Improvement | 20 | `metrics_improved.json`, `improved_model.joblib`, ≥2 methods |
| **E** | Bug & Leakage Debug | 20 | `metrics_broken.json`, `metrics_fixed.json`, bug report |
| **F** | Reproducibility & Reporting | 15 | `README`, `requirements.txt`, `MODEL_CARD`, `run_all.sh`, `tool_run_summary` |

### Scoring Rubric

Each task is scored across **5 dimensions** on a **0–4 scale**:

| Dimension | What it measures |
|---|---|
| Correctness | Factual accuracy of outputs against ground truth |
| Statistical Validity | Correct computation, no directional errors |
| Reproducibility | Can results be reproduced from provided code/config? |
| Output Quality | Depth, completeness, and clarity of deliverables |
| Efficiency | Single-pass completion; minimal unnecessary iterations |

> **Weighted Score = Avg Task Score × Task Weight**
> Maximum total = 4 × 100 = **400 points**

### Evaluators (3 independent, scores averaged)

| Version | Evaluator | Method |
|---|---|---|
| v1 | Chienyu | Human review of all outputs |
| v2 | Stephen | Human review of all outputs |
| v3 | Chiying | Human review of all outputs |

Final score = arithmetic mean of v1 + v2 + v3 per dimension per task.

---

## 📁 Agent Run Folder Structure

Each agent run follows this layout (Claude shown as example):

```
BenchmarkRuns/2026-03-11_claude_run1/
│
├── README.md                    ← Agent-level run summary
├── MODEL_CARD.md                ← Model card (bias, limitations, governance)
├── requirements.txt             ← Python dependencies
├── data/
│   └── Digital_Payment_Fraud_Detection_Dataset.csv
├── src/
│   ├── task_a_data_audit.py
│   ├── task_b_eda.py
│   ├── task_c_baseline.py
│   ├── task_d_improvement.py
│   ├── broken_pipeline.py       ← Intentionally buggy pipeline (Task E input)
│   └── fixed_pipeline.py        ← Agent's corrected pipeline (Task E output)
├── scripts/
│   └── run_all.sh               ← End-to-end reproduction script
├── reports/
│   └── tool_run_summary.md      ← Full execution log & narrative
└── artifacts/claude/2026-03-11_run1/
    ├── taskA_data_audit/
    │   ├── outputs/             ← data_profile.json, cleaned.csv
    │   ├── plots/               ← missingness_bar.png
    │   ├── notes.md
    │   ├── commands.txt
    │   └── logs.txt
    ├── taskB_eda/
    │   ├── outputs/             ← eda_insights.md, eda_support_stats.json
    │   ├── plots/               ← 8 EDA plots
    │   └── ...
    ├── taskC_baseline_model/
    │   ├── outputs/             ← metrics_baseline.json, baseline_model.joblib
    │   ├── plots/               ← pr_curve.png, roc_curve.png, confusion_matrix
    │   └── ...
    ├── taskD_model_improvement/
    │   ├── outputs/             ← metrics_improved.json, improved_model.joblib
    │   ├── plots/               ← pr_curve_improved.png, ...
    │   └── ...
    ├── taskE_bug_leakage_debug/
    │   ├── outputs/             ← metrics_broken.json, metrics_fixed.json
    │   └── ...
    └── taskF_reproducibility_reporting/
        ├── outputs/
        └── ...
```

> **Codex run path:** `BenchmarkRuns/2026-03-11_codex_run2/artifacts/codex/YYYY-MM-DD_run1/`
> **Gemini run path:** `BenchmarkRuns/2026-03-11_gemini_run1/artifacts/gemini/2026-03-13_run1/`

---

## 📊 Key Output Files by Task (All Agents)

### Task A — Data Audit

| Agent | data_profile.json | cleaned.csv |
|---|---|---|
| Claude | `BenchmarkRuns/2026-03-11_claude_run1/artifacts/claude/2026-03-11_run1/taskA_data_audit/outputs/data_profile.json` | `.../outputs/cleaned.csv` |
| Codex | `BenchmarkRuns/2026-03-11_codex_run2/artifacts/codex/YYYY-MM-DD_run1/taskA_data_audit/outputs/data_profile.json` | `.../outputs/cleaned.csv` |
| Gemini | `BenchmarkRuns/2026-03-11_gemini_run1/artifacts/gemini/2026-03-13_run1/taskA_data_audit/outputs/data_profile.json` | `.../outputs/cleaned.csv` |

### Task B — EDA

| Agent | EDA Insights | Support Stats |
|---|---|---|
| Claude | `.../taskB_eda/outputs/eda_insights.md` | `.../outputs/eda_support_stats.json` |
| Codex | `.../taskB_eda/outputs/eda_insights.md` | `.../outputs/eda_support_stats.json` |
| Gemini | `.../taskB_eda/outputs/eda_insights.md` | `.../outputs/eda_support_stats.json` |

### Task C — Baseline Model

| Agent | Metrics | Model File |
|---|---|---|
| Claude | PR-AUC=0.0623 | `.../taskC_baseline_model/outputs/metrics_baseline.json` |
| Codex | PR-AUC=0.0711 | `.../taskC_baseline_model/outputs/metrics_baseline.json` |
| Gemini | PR-AUC=0.0711 | `.../taskC_baseline_model/outputs/metrics_baseline.json` |

### Task D — Model Improvement

| Agent | Metrics | Key Note |
|---|---|---|
| Claude | PR-AUC=0.0661 (+0.0038) | `.../taskD_model_improvement/outputs/metrics_improved.json` |
| Codex | PR-AUC=0.1044 (+0.0333) | Degenerate: threshold=0.25, recall=1.0 |
| Gemini | PR-AUC=0.0679 (−0.0032) | Regression vs baseline; not disclosed |

### Task E — Bug & Leakage Debug

| Agent | Broken PR-AUC | Bugs Found | Fixed Metrics |
|---|---|---|---|
| Claude | 1.0 (leakage) | 4 | `.../taskE_bug_leakage_debug/outputs/metrics_fixed.json` |
| Codex | 1.0 (leakage) | 5 | `.../taskE_bug_leakage_debug/outputs/metrics_fixed.json` |
| Gemini | N/A (missing from JSON) | 4 | `.../taskE_bug_leakage_debug/outputs/metrics_fixed.json` |

---

## 📋 Evaluation Files (final_score/)

| File | Purpose | Best for |
|---|---|---|
| `final_summary.md` | 1-page rankings, key metrics, critical findings per agent | **Quick overview** |
| `final_agent_scoring.md` | Full rubric — per-task scores, all 3 evaluator versions, notes, failure modes | **Detailed grading review** |
| `final_agent_scoring.csv` | Same data as above in spreadsheet-importable format | **Data analysis** |
| `final_model_governance_checklist.md` | Governance compliance: approved model pool, method count, bias checks | **Governance review** |
| `final_model_governance_checklist.csv` | Machine-readable governance checklist | **Data analysis** |

---

## 🔬 Benchmark Spec

Full benchmark rules, task definitions, success criteria, and rubric:
→ `AGENT_BENCHMARK_SPEC_v2.md`

---

*MSIN0097 Group Coursework · UCL School of Management · 2025–26*
