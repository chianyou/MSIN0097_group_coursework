# Task F Notes

## Artifacts Created
All project-level reproducibility files were created:

| File | Description |
|------|-------------|
| `README.md` | Project overview, quick start, config table, results summary |
| `requirements.txt` | Python dependencies (pandas, numpy, sklearn, matplotlib, joblib) |
| `scripts/run_all.sh` | Executable shell script to re-run all 6 tasks end-to-end |
| `MODEL_CARD.md` | Model details, training data, metrics, limitations, ethical notes |
| `reports/tool_run_summary.md` | Full task status, metrics, failures, strengths/weaknesses |

## Reproducibility Checklist
- [x] Fixed random seed (42) used consistently across all scripts
- [x] Stratified splits preserve fraud rate in all sets
- [x] Preprocessing fitted on training data only (sklearn Pipeline)
- [x] No test-set snooping during model selection (validation set used)
- [x] All scripts runnable from project root via `bash scripts/run_all.sh`
- [x] No external dependencies beyond default stack (pandas, numpy, sklearn, matplotlib, joblib)
- [x] All task folders contain spec, commands, logs, notes, outputs/, plots/
