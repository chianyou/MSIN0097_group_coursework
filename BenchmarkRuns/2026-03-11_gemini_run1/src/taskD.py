import pandas as pd
import numpy as np
import json
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve, ConfusionMatrixDisplay

RUN_ID = "2026-03-13_run1"
BASE_OUT_DIR = f"artifacts/gemini/{RUN_ID}/taskD_model_improvement"
CLEAN_DATA = f"artifacts/gemini/{RUN_ID}/taskA_data_audit/outputs/cleaned.csv"
TARGET_COL = "fraud_label"
ID_COLS = ["transaction_id", "user_id"]
RANDOM_SEED = 42

os.makedirs(f"{BASE_OUT_DIR}/outputs", exist_ok=True)
os.makedirs(f"{BASE_OUT_DIR}/plots", exist_ok=True)

df = pd.read_csv(CLEAN_DATA)

X = df.drop(columns=[TARGET_COL] + [c for c in ID_COLS if c in df.columns])
y = df[TARGET_COL]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp)

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), cat_cols)
])

models = {
    'LogisticRegression': LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, class_weight='balanced'),
    'RandomForestClassifier': RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=100),
    'HistGradientBoostingClassifier': HistGradientBoostingClassifier(random_state=RANDOM_SEED)
}

val_results = {}
best_model_name = None
best_pr_auc = -1
best_pipeline = None

for name, clf in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    pipeline.fit(X_train, y_train)
    y_val_proba = pipeline.predict_proba(X_valid)[:, 1]
    pr_auc = average_precision_score(y_valid, y_val_proba)
    val_results[name] = pr_auc
    if pr_auc > best_pr_auc:
        best_pr_auc = pr_auc
        best_model_name = name
        best_pipeline = pipeline

y_val_proba_best = best_pipeline.predict_proba(X_valid)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_valid, y_val_proba_best)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_threshold = thresholds[np.argmax(f1_scores)]

y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= best_threshold).astype(int)

metrics = {
    'PR-AUC': average_precision_score(y_test, y_test_proba),
    'ROC-AUC': roc_auc_score(y_test, y_test_proba),
    'Precision': precision_score(y_test, y_test_pred),
    'Recall': recall_score(y_test, y_test_pred),
    'F1': f1_score(y_test, y_test_pred),
    'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist()
}

with open(f"{BASE_OUT_DIR}/outputs/metrics_improved.json", "w") as f:
    json.dump(metrics, f, indent=4)

cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title(f"Confusion Matrix - Improved ({best_model_name})")
plt.savefig(f"{BASE_OUT_DIR}/plots/confusion_matrix_improved.png")
plt.close()

precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
plt.figure()
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'PR Curve - Improved ({best_model_name})')
plt.savefig(f"{BASE_OUT_DIR}/plots/pr_curve_improved.png")
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_test_proba)
plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title(f'ROC Curve - Improved ({best_model_name})')
plt.savefig(f"{BASE_OUT_DIR}/plots/roc_curve_improved.png")
plt.close()

with open(f"{BASE_OUT_DIR}/notes.md", "w") as f:
    f.write("# Task D Notes\n")
    f.write("## Tested Candidate Models on Validation Set\n")
    for name, pr_auc in val_results.items():
        f.write(f"- {name}: PR-AUC = {pr_auc:.4f}\n")
    f.write(f"\nSelected Model: {best_model_name} (Threshold tuned on validation set: {best_threshold:.4f})\n\n")
    f.write("## Baseline vs Improved Comparison\n")
    f.write("| Metric | Baseline | Improved |\n")
    f.write("|---|---|---|\n")
    f.write(f"| PR-AUC | (see Task C) | {metrics['PR-AUC']:.4f} |\n")
    f.write(f"| F1 | (see Task C) | {metrics['F1']:.4f} |\n")
    f.write("\n## Improvements Applied\n")
    f.write("1. Model change: Tested 3 candidate models and selected the best.\n")
    f.write("2. Threshold tuning: Optimized decision threshold on the validation set for maximum F1 score.\n")

with open(f"{BASE_OUT_DIR}/spec.md", "w") as f:
    f.write("Task D: Model Improvement Specification.\nGoal: Improve performance without violating statistical validity.")
with open(f"{BASE_OUT_DIR}/commands.txt", "w") as f:
    f.write("python src/taskD.py\n")
with open(f"{BASE_OUT_DIR}/logs.txt", "w") as f:
    f.write("Started Task D.\nTested 3 candidate models.\nTuned threshold on validation set.\nEvaluated on test set.\nSaved plots and metrics.\nCompleted Task D.\n")
