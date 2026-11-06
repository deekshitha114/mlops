"""Train, evaluate, and persist e-commerce return prediction models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    TARGET_LABELS,
    engineer_features,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "Section7_Q4_Ecommerce_Product_Return_CICD.csv"
DEFAULT_ARTIFACTS = ROOT / "artifacts"
RANDOM_STATE = 42


def _preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS + [
                "Delivery_Delay_Days",
                "Customer_Return_Frequency",
            ]),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


def _load_training_data(dataset_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    raw = pd.read_csv(dataset_path)
    if TARGET_COLUMN not in raw:
        raise ValueError(f"Dataset must contain target column {TARGET_COLUMN!r}")

    valid_target = raw[TARGET_COLUMN].isin(TARGET_LABELS)
    raw = raw.loc[valid_target].copy()
    if raw.empty:
        raise ValueError("Dataset contains no rows with a valid return status")

    features = engineer_features(raw)
    target = raw[TARGET_COLUMN].map({"Not Returned": 0, "Returned": 1})
    return features, target.astype("int64")


def train_models(
    dataset_path: Path = DEFAULT_DATASET,
    artifact_dir: Path = DEFAULT_ARTIFACTS,
) -> dict[str, Any]:
    """Compare logistic regression and random forest on a held-out test split."""
    X, y = _load_training_data(dataset_path)
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Training data must contain at least two examples per class")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    candidates = {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", _preprocessor(scale_numeric=True)),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", _preprocessor(scale_numeric=False)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    results: dict[str, Any] = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_returned": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall_returned": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1_returned": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        }

    best_name = max(
        results,
        key=lambda name: (
            results[name]["f1_returned"],
            results[name]["roc_auc"],
        ),
    )
    best_model = candidates[best_name]
    best_model.fit(X, y)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, artifact_dir / "return_model.joblib")
    report: dict[str, Any] = {
        "dataset": dataset_path.name,
        "rows_used": len(y),
        "target_counts": {
            "Not Returned": int((y == 0).sum()),
            "Returned": int((y == 1).sum()),
        },
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
        "feature_columns": FEATURE_COLUMNS,
        "model_metrics": results,
        "selected_model": best_name,
        "selection_metric": "f1_returned (ROC AUC tie-breaker)",
    }
    (artifact_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args()
    print(json.dumps(train_models(args.dataset, args.artifact_dir), indent=2))


if __name__ == "__main__":
    main()
