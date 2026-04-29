"""Reusable training pipeline for the end-to-end ML deployment project."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


logger = logging.getLogger(__name__)


@dataclass
class TrainingArtifacts:
    pipeline: Pipeline
    feature_columns: list[str]


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    data_path = Path(data_path)
    logger.info("Loading dataset from %s", data_path)
    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError("Input dataset is empty.")
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def build_training_pipeline(X: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=10,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_pipeline(
    data_path: str | Path,
    target_column: str,
    model_output_path: str | Path,
) -> Tuple[Pipeline, dict]:
    df = load_dataset(data_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() <= 20 else None,
    )

    pipeline = build_training_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    metrics = evaluate_pipeline(pipeline, X_test, y_test)
    save_pipeline(pipeline, model_output_path)
    return pipeline, metrics


def evaluate_pipeline(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = pipeline.predict(X_test)
    metrics = {"accuracy": float((predictions == y_test).mean())}
    if hasattr(pipeline, "predict_proba") and y_test.nunique() == 2:
        from sklearn.metrics import roc_auc_score

        probabilities = pipeline.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
    return metrics


def save_pipeline(pipeline: Pipeline, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    logger.info("Saved pipeline to %s", output_path)


def load_pipeline(model_path: str | Path) -> Pipeline:
    return joblib.load(model_path)
