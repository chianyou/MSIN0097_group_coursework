"""
Task A: Data Audit
Goal: Load the dataset and produce a structured data-quality report.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Digital_Payment_Fraud_Detection_Dataset.csv"
OUT_DIR = ROOT / "artifacts" / "claude" / "2026-03-11_run1" / "taskA_data_audit"
OUTPUTS = OUT_DIR / "outputs"
PLOTS = OUT_DIR / "plots"
OUTPUTS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
TARGET_COL = "fraud_label"
ID_COLS = ["transaction_id", "user_id"]

# ── Load ───────────────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

# ── Column classification ──────────────────────────────────────────────────
all_cols = df.columns.tolist()
id_cols = [c for c in ID_COLS if c in all_cols]
target_cols = [TARGET_COL]
cat_cols = [c for c in df.select_dtypes(include="object").columns if c not in id_cols + target_cols]
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in id_cols + target_cols]

print(f"  ID cols       : {id_cols}")
print(f"  Target col    : {target_cols}")
print(f"  Categorical   : {cat_cols}")
print(f"  Numerical     : {num_cols}")

# ── Missing values ─────────────────────────────────────────────────────────
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(4)

# ── Duplicate rows ─────────────────────────────────────────────────────────
dup_rows = int(df.duplicated().sum())
print(f"  Duplicate rows: {dup_rows}")

# ── Target distribution ────────────────────────────────────────────────────
target_dist = df[TARGET_COL].value_counts().to_dict()
target_pct = (df[TARGET_COL].value_counts(normalize=True) * 100).round(4).to_dict()

# ── Numeric summary ────────────────────────────────────────────────────────
num_summary = df[num_cols].describe().round(4).to_dict()

# ── Range checks ──────────────────────────────────────────────────────────
range_checks = {}

# transaction_hour: 0–23
th = df["transaction_hour"]
range_checks["transaction_hour"] = {
    "min": float(th.min()),
    "max": float(th.max()),
    "out_of_range_count": int(((th < 0) | (th > 23)).sum()),
    "expected_range": "[0, 23]",
}

# ip_risk_score: 0–1
irs = df["ip_risk_score"]
range_checks["ip_risk_score"] = {
    "min": float(irs.min()),
    "max": float(irs.max()),
    "out_of_range_count": int(((irs < 0) | (irs > 1)).sum()),
    "expected_range": "[0, 1]",
}

# fraud_label: 0 or 1
fl = df["fraud_label"]
range_checks["fraud_label"] = {
    "unique_values": sorted(fl.unique().tolist()),
    "out_of_range_count": int(~fl.isin([0, 1]).sum()),
    "expected_values": "[0, 1]",
}

# is_international: 0 or 1
ii = df["is_international"]
range_checks["is_international"] = {
    "unique_values": sorted(ii.unique().tolist()),
    "out_of_range_count": int(~ii.isin([0, 1]).sum()),
    "expected_values": "[0, 1]",
}

# transaction_amount positivity
ta = df["transaction_amount"]
range_checks["transaction_amount"] = {
    "min": float(ta.min()),
    "max": float(ta.max()),
    "non_positive_count": int((ta <= 0).sum()),
    "expected": "positive (> 0)",
}

print("Range checks complete.")

# ── Build profile dict ─────────────────────────────────────────────────────
profile = {
    "shape": {"rows": df.shape[0], "cols": df.shape[1]},
    "columns": all_cols,
    "dtypes": {c: str(dt) for c, dt in df.dtypes.items()},
    "column_classification": {
        "id_cols": id_cols,
        "target_cols": target_cols,
        "categorical_cols": cat_cols,
        "numerical_cols": num_cols,
    },
    "missing_values": {
        "counts": missing.to_dict(),
        "pct": missing_pct.to_dict(),
        "total_missing_cells": int(missing.sum()),
        "columns_with_missing": [c for c in all_cols if missing[c] > 0],
    },
    "duplicate_rows": dup_rows,
    "target_distribution": {
        "counts": {str(k): v for k, v in target_dist.items()},
        "percentages": {str(k): v for k, v in target_pct.items()},
    },
    "range_checks": range_checks,
    "numeric_summary": num_summary,
}

profile_path = OUTPUTS / "data_profile.json"
with open(profile_path, "w") as f:
    json.dump(profile, f, indent=2)
print(f"  Saved: {profile_path}")

# ── Missingness bar plot ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
miss_series = missing_pct[missing_pct > 0]
if miss_series.empty:
    ax.text(0.5, 0.5, "No missing values detected", ha="center", va="center", fontsize=14)
    ax.set_title("Missing Values (%) per Column")
else:
    miss_series.sort_values(ascending=False).plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Missing Values (%) per Column")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing %")
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
fig.savefig(PLOTS / "missingness_bar.png", dpi=150)
plt.close(fig)
print(f"  Saved: {PLOTS / 'missingness_bar.png'}")

# ── Cleaned dataset ────────────────────────────────────────────────────────
# Drop duplicate rows; no imputation needed if no missing values
df_clean = df.drop_duplicates().reset_index(drop=True)

# Ensure transaction_amount is positive (drop non-positive rows if any)
neg_mask = df_clean["transaction_amount"] <= 0
if neg_mask.sum() > 0:
    print(f"  Dropping {neg_mask.sum()} rows with non-positive transaction_amount")
    df_clean = df_clean[~neg_mask].reset_index(drop=True)

cleaned_path = OUTPUTS / "cleaned.csv"
df_clean.to_csv(cleaned_path, index=False)
print(f"  Saved cleaned dataset: {cleaned_path}  shape={df_clean.shape}")

# ── Summary printout ───────────────────────────────────────────────────────
print("\n=== Data Audit Summary ===")
print(f"  Original shape : {df.shape}")
print(f"  Cleaned shape  : {df_clean.shape}")
print(f"  Missing cells  : {int(missing.sum())}")
print(f"  Duplicate rows : {dup_rows}")
print(f"  Fraud rate     : {target_pct.get(1, target_pct.get('1', 0)):.2f}%")
print("Task A complete.")
