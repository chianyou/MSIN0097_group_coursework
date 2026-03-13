from __future__ import annotations

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .common import (
    ID_COLS,
    TARGET_COL,
    build_preprocessor,
    compute_metrics,
    load_dataset,
    prepare_features,
    save_json,
    split_dataset,
)


def run_fixed_pipeline(output_path: str | Path) -> dict:
    df = load_dataset()
    train_df, valid_df, test_df = split_dataset(df)

    X_train = prepare_features(train_df, feature_engineering=False)
    X_test = prepare_features(test_df, feature_engineering=False)
    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)
    test_scores = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, test_scores).as_dict()
    metrics["evaluation_set"] = "test"
    metrics["split"] = {"train": len(train_df), "valid": len(valid_df), "test": len(test_df)}
    metrics["excluded_columns"] = ID_COLS + [TARGET_COL]
    save_json(Path(output_path), metrics)
    return metrics
