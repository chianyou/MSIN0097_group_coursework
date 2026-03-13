from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MPL_DIR = Path.cwd() / ".mplconfig"
DEFAULT_MPL_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPL_DIR))

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Digital_Payment_Fraud_Detection_Dataset.csv"
TOOL_NAME = "codex"
RUN_ID = "YYYY-MM-DD_run1"
RANDOM_SEED = 42
TARGET_COL = "fraud_label"
ID_COLS = ["transaction_id", "user_id"]
TASK_NAMES = {
    "A": "taskA_data_audit",
    "B": "taskB_eda",
    "C": "taskC_baseline_model",
    "D": "taskD_model_improvement",
    "E": "taskE_bug_leakage_debug",
    "F": "taskF_reproducibility_reporting",
}
TASK_SPECS = {
    "A": "Load the dataset, validate the schema, profile data quality, run required range checks, and export a cleaned dataset.",
    "B": "Perform EDA with required plots and write evidence-based insights backed by summary statistics.",
    "C": "Train exactly one LogisticRegression baseline using a stratified 70/15/15 split and evaluate on the test set only.",
    "D": "Evaluate at least three approved candidate models using train/validation selection, then report final test metrics for the best model.",
    "E": "Run a deliberately flawed pipeline, record invalid behavior, explain the defects, then rerun a corrected pipeline.",
    "F": "Assemble documentation, reproducibility assets, and a summary report covering the full benchmark run.",
}


def configure_environment() -> None:
    mpl_dir = ROOT / ".mplconfig"
    mpl_dir.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    matplotlib.use("Agg")
    plt.style.use("ggplot")


def artifact_root() -> Path:
    return ROOT / "artifacts" / TOOL_NAME / RUN_ID


def ensure_task_dirs(task_key: str) -> dict[str, Path]:
    task_root = artifact_root() / TASK_NAMES[task_key]
    outputs = task_root / "outputs"
    plots = task_root / "plots"
    outputs.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    for name, content in {
        "spec.md": f"# Task {task_key}\n\n{TASK_SPECS[task_key]}\n",
        "commands.txt": "",
        "logs.txt": "",
        "notes.md": f"# Task {task_key} Notes\n\n",
    }.items():
        path = task_root / name
        if not path.exists():
            path.write_text(content, encoding="utf-8")
    return {"root": task_root, "outputs": outputs, "plots": plots}


def append_text(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def reset_text(path: Path, text: str = "") -> None:
    path.write_text(text, encoding="utf-8")


def record_command(task_key: str, command: str) -> None:
    append_text(artifact_root() / TASK_NAMES[task_key] / "commands.txt", command.strip() + "\n")


def log_message(task_key: str, message: str) -> None:
    append_text(artifact_root() / TASK_NAMES[task_key] / "logs.txt", message.rstrip() + "\n")


def write_notes(task_key: str, text: str) -> None:
    reset_text(artifact_root() / TASK_NAMES[task_key] / "notes.md", text)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df[TARGET_COL],
        random_state=RANDOM_SEED,
    )
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[TARGET_COL],
        random_state=RANDOM_SEED,
    )
    return train_df.copy(), valid_df.copy(), test_df.copy()


def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    feature_df = df.drop(columns=[TARGET_COL], errors="ignore")
    categorical = [
        col
        for col in feature_df.columns
        if pd.api.types.is_object_dtype(feature_df[col])
        or pd.api.types.is_string_dtype(feature_df[col])
        or pd.api.types.is_categorical_dtype(feature_df[col])
    ]
    categorical = [col for col in categorical if col not in ID_COLS]
    numerical = [col for col in feature_df.columns if col not in categorical and col not in ID_COLS]
    return {
        "id_columns": [col for col in ID_COLS if col in df.columns],
        "categorical_columns": categorical,
        "numerical_columns": numerical,
        "target_column": [TARGET_COL] if TARGET_COL in df.columns else [],
    }


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = [
        col
        for col in X.columns
        if pd.api.types.is_object_dtype(X[col])
        or pd.api.types.is_string_dtype(X[col])
        or pd.api.types.is_categorical_dtype(X[col])
    ]
    numerical = [col for col in X.columns if col not in categorical]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )


def prepare_features(df: pd.DataFrame, *, feature_engineering: bool = False) -> pd.DataFrame:
    X = df.drop(columns=ID_COLS + [TARGET_COL], errors="ignore").copy()
    if feature_engineering:
        denom = X["avg_transaction_amount"].replace(0, np.nan)
        X["amount_to_average_ratio"] = (X["transaction_amount"] / denom).replace([np.inf, -np.inf], np.nan)
        X["high_risk_hour"] = X["transaction_hour"].isin([0, 1, 2, 3, 4, 22, 23]).astype(int)
        X["amount_risk_interaction"] = X["transaction_amount"] * X["ip_risk_score"]
    return X


@dataclass
class MetricBundle:
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float
    confusion_matrix: list[list[int]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "threshold": self.threshold,
            "confusion_matrix": self.confusion_matrix,
        }


def compute_metrics(y_true: pd.Series, scores: np.ndarray, threshold: float = 0.5) -> MetricBundle:
    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    return MetricBundle(
        pr_auc=float(average_precision_score(y_true, scores)),
        roc_auc=float(roc_auc_score(y_true, scores)),
        precision=float(precision_score(y_true, preds, zero_division=0)),
        recall=float(recall_score(y_true, preds, zero_division=0)),
        f1=float(f1_score(y_true, preds, zero_division=0)),
        threshold=float(threshold),
        confusion_matrix=cm.tolist(),
    )


def plot_confusion(y_true: pd.Series, scores: np.ndarray, out_path: Path, threshold: float) -> list[list[int]]:
    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix @ threshold={threshold:.2f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm.tolist()


def plot_pr_curve(y_true: pd.Series, scores: np.ndarray, out_path: Path, title: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label="PR curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true: pd.Series, scores: np.ndarray, out_path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label="ROC curve")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def dump_model(path: Path, model: Any) -> None:
    joblib.dump(model, path)
