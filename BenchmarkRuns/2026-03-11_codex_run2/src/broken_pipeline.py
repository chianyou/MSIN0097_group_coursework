from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .common import DATA_PATH, TARGET_COL, save_json


def run_broken_pipeline(output_path: str | Path) -> dict:
    df = pd.read_csv(DATA_PATH)

    # Deliberate defects:
    # 1. IDs retained as predictors.
    # 2. Target leaked directly into features.
    # 3. Non-stratified split with no fixed seed.
    # 4. Preprocessing fit on the full dataset before the split.
    # 5. Evaluation on the training set.
    X = df.copy()
    y = X[TARGET_COL]

    categorical = [
        col
        for col in X.columns
        if pd.api.types.is_object_dtype(X[col])
        or pd.api.types.is_string_dtype(X[col])
        or pd.api.types.is_categorical_dtype(X[col])
    ]
    numerical = [col for col in X.columns if col not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    X_all = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.30, stratify=None)

    model = Pipeline(
        steps=[
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)

    train_scores = model.predict_proba(X_train)[:, 1]
    metrics = {
        "evaluation_set": "training",
        "pr_auc": float(average_precision_score(y_train, train_scores)),
        "roc_auc": float(roc_auc_score(y_train, train_scores)),
        "n_train": int(len(y_train)),
        "n_test_unused": int(len(y_test)),
        "flaws": [
            "IDs used as predictors",
            "Target leakage via fraud_label inside features",
            "Preprocessing fit on full data before split",
            "Non-stratified split",
            "No fixed seed",
            "Evaluation performed on training set",
        ],
    }
    save_json(Path(output_path), metrics)
    return metrics
