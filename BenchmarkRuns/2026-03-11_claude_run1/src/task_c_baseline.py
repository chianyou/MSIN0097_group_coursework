"""
Task C: Baseline Model
Goal: Build a statistically valid baseline fraud model using LogisticRegression.
Split: 70/15/15 stratified. Evaluate on test set only.
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CLEANED = ROOT / "artifacts" / "claude" / "2026-03-11_run1" / "taskA_data_audit" / "outputs" / "cleaned.csv"
OUT_DIR = ROOT / "artifacts" / "claude" / "2026-03-11_run1" / "taskC_baseline_model"
OUTPUTS = OUT_DIR / "outputs"
PLOTS = OUT_DIR / "plots"
OUTPUTS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TARGET = "fraud_label"
ID_COLS = ["transaction_id", "user_id"]

# ── Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(CLEANED)
print(f"Loaded {df.shape}")

# ── Feature/target split ───────────────────────────────────────────────────
drop_cols = ID_COLS + [TARGET]
X = df.drop(columns=drop_cols)
y = df[TARGET]

cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print(f"Categorical features : {cat_cols}")
print(f"Numerical features   : {num_cols}")

# ── Stratified 70 / 15 / 15 split ─────────────────────────────────────────
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.15 / 0.85,
    random_state=RANDOM_SEED, stratify=y_train_full
)

print(f"Train  : {X_train.shape}  fraud={y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"Valid  : {X_valid.shape}  fraud={y_valid.sum()} ({y_valid.mean()*100:.2f}%)")
print(f"Test   : {X_test.shape}   fraud={y_test.sum()} ({y_test.mean()*100:.2f}%)")

# ── Preprocessing pipeline ─────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
        class_weight="balanced",  # handle imbalance
    )),
])

# ── Fit on training data only ──────────────────────────────────────────────
print("Fitting baseline LogisticRegression …")
pipeline.fit(X_train, y_train)
print("Fit complete.")

# ── Evaluate on TEST set only ──────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

pr_auc = average_precision_score(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

metrics = {
    "model": "LogisticRegression",
    "split": {"train": len(y_train), "valid": len(y_valid), "test": len(y_test)},
    "PR_AUC": round(pr_auc, 6),
    "ROC_AUC": round(roc_auc, 6),
    "Precision": round(precision, 6),
    "Recall": round(recall, 6),
    "F1": round(f1, 6),
    "confusion_matrix": cm.tolist(),
}

print("\n=== Baseline Metrics (TEST) ===")
for k, v in metrics.items():
    if k not in ("confusion_matrix", "split"):
        print(f"  {k:12s} = {v}")

with open(OUTPUTS / "metrics_baseline.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  Saved: {OUTPUTS / 'metrics_baseline.json'}")

# ── Save model ─────────────────────────────────────────────────────────────
joblib.dump(pipeline, OUTPUTS / "baseline_model.joblib")
print(f"  Saved: {OUTPUTS / 'baseline_model.joblib'}")

# ── Plot: Confusion Matrix ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Fraud"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — Baseline LogisticRegression (Test Set)")
plt.tight_layout()
fig.savefig(PLOTS / "confusion_matrix_baseline.png", dpi=150)
plt.close(fig)
print(f"  Saved: confusion_matrix_baseline.png")

# ── Plot: PR Curve ─────────────────────────────────────────────────────────
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(recall_curve, precision_curve, color="darkorange", lw=2, label=f"PR-AUC = {pr_auc:.4f}")
ax.axhline(y=y_test.mean(), color="navy", linestyle="--", label=f"Baseline (prevalence={y_test.mean():.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve — Baseline (Test Set)")
ax.legend()
plt.tight_layout()
fig.savefig(PLOTS / "pr_curve.png", dpi=150)
plt.close(fig)
print(f"  Saved: pr_curve.png")

# ── Plot: ROC Curve ────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC-AUC = {roc_auc:.4f}")
ax.plot([0, 1], [0, 1], "k--", label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Baseline (Test Set)")
ax.legend()
plt.tight_layout()
fig.savefig(PLOTS / "roc_curve.png", dpi=150)
plt.close(fig)
print(f"  Saved: roc_curve.png")

print("Task C complete.")
