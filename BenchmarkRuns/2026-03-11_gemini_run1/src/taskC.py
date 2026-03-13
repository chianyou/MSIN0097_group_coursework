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
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve, ConfusionMatrixDisplay

RUN_ID = "2026-03-13_run1"
BASE_OUT_DIR = f"artifacts/gemini/{RUN_ID}/taskC_baseline_model"
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

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=RANDOM_SEED, max_iter=1000))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    'PR-AUC': average_precision_score(y_test, y_proba),
    'ROC-AUC': roc_auc_score(y_test, y_proba),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1': f1_score(y_test, y_pred),
    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
}

with open(f"{BASE_OUT_DIR}/outputs/metrics_baseline.json", "w") as f:
    json.dump(metrics, f, indent=4)

joblib.dump(model, f"{BASE_OUT_DIR}/outputs/baseline_model.joblib")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Baseline")
plt.savefig(f"{BASE_OUT_DIR}/plots/confusion_matrix_baseline.png")
plt.close()

precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure()
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve - Baseline')
plt.savefig(f"{BASE_OUT_DIR}/plots/pr_curve.png")
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve - Baseline')
plt.savefig(f"{BASE_OUT_DIR}/plots/roc_curve.png")
plt.close()

with open(f"{BASE_OUT_DIR}/spec.md", "w") as f:
    f.write("Task C: Baseline Model Specification.\nGoal: Build a statistically valid baseline fraud model using LogisticRegression.")
with open(f"{BASE_OUT_DIR}/commands.txt", "w") as f:
    f.write("python src/taskC.py\n")
with open(f"{BASE_OUT_DIR}/logs.txt", "w") as f:
    f.write("Started Task C.\nSplit data (70/15/15).\nCreated and fitted pipeline.\nEvaluated model on test set.\nSaved model and metrics.\nSaved plots.\nCompleted Task C.\n")
with open(f"{BASE_OUT_DIR}/notes.md", "w") as f:
    f.write("# Task C Notes\nBaseline model successfully trained and evaluated on test set.")
