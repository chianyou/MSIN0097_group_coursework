import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os

TOOL_NAME = "gemini"
RUN_ID = "2026-03-13_run1"
DATA_PATH = "./data/Digital_Payment_Fraud_Detection_Dataset.csv"
TARGET_COL = "fraud_label"
ID_COLS = ["transaction_id", "user_id"]
RANDOM_SEED = 42

BASE_OUT_DIR = f"artifacts/gemini/{RUN_ID}/taskA_data_audit"
os.makedirs(f"{BASE_OUT_DIR}/outputs", exist_ok=True)
os.makedirs(f"{BASE_OUT_DIR}/plots", exist_ok=True)

df = pd.read_csv(DATA_PATH)

expected_cols = df.columns.tolist()
shape = df.shape
dtypes = df.dtypes.astype(str).to_dict()

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in ID_COLS + [TARGET_COL]]
cat_cols = [c for c in cat_cols if c not in ID_COLS + [TARGET_COL]]

missing = df.isnull().sum().to_dict()
duplicates = int(df.duplicated().sum())
target_dist = df[TARGET_COL].value_counts().to_dict() if TARGET_COL in df.columns else {}
numeric_summary = df[num_cols].describe().to_dict()

range_checks = {}
range_checks['transaction_hour'] = {'min': int(df['transaction_hour'].min()), 'max': int(df['transaction_hour'].max())} if 'transaction_hour' in df.columns else 'missing'
range_checks['ip_risk_score'] = {'min': float(df['ip_risk_score'].min()), 'max': float(df['ip_risk_score'].max())} if 'ip_risk_score' in df.columns else 'missing'
range_checks['fraud_label'] = df['fraud_label'].unique().tolist() if 'fraud_label' in df.columns else 'missing'
range_checks['is_international'] = df['is_international'].unique().tolist() if 'is_international' in df.columns else 'missing'
range_checks['transaction_amount_positivity'] = bool((df['transaction_amount'] > 0).all()) if 'transaction_amount' in df.columns else 'missing'

data_profile = {
    'shape': shape,
    'dtypes': dtypes,
    'id_cols': ID_COLS,
    'target_col': TARGET_COL,
    'numerical_cols': num_cols,
    'categorical_cols': cat_cols,
    'missing_values': missing,
    'duplicate_rows': duplicates,
    'target_distribution': target_dist,
    'numeric_summary': numeric_summary,
    'range_checks': range_checks
}

with open(f"{BASE_OUT_DIR}/outputs/data_profile.json", "w") as f:
    json.dump(data_profile, f, indent=4)

df_clean = df.drop_duplicates()
df_clean.to_csv(f"{BASE_OUT_DIR}/outputs/cleaned.csv", index=False)

plt.figure(figsize=(10,6))
missing_series = pd.Series(missing)
if (missing_series > 0).any():
    missing_series[missing_series > 0].plot(kind='bar')
    plt.title("Missing Values per Column")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{BASE_OUT_DIR}/plots/missingness_bar.png")
plt.close()

with open(f"{BASE_OUT_DIR}/spec.md", "w") as f:
    f.write("Task A: Data Audit Specification.\nGoal: Load dataset, produce structured data-quality report.")
with open(f"{BASE_OUT_DIR}/commands.txt", "w") as f:
    f.write("python src/taskA.py\n")
with open(f"{BASE_OUT_DIR}/logs.txt", "w") as f:
    f.write("Started Task A.\nLoaded data.\nCalculated profile.\nSaved cleaned.csv.\nSaved plot.\nCompleted Task A.\n")
with open(f"{BASE_OUT_DIR}/notes.md", "w") as f:
    f.write("# Task A Notes\nData successfully loaded and profiled. Range checks pass or flag missing columns appropriately.")
