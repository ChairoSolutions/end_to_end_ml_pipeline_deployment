"""Training entrypoint for the end-to-end ML pipeline project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline import train_pipeline
from src.utils import configure_logging


ARTIFACT_DIR = Path("artifacts")


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="Train the end-to-end ML pipeline.")
    parser.add_argument("--data", required=True, help="Path to the training CSV.")
    parser.add_argument("--target", required=True, help="Target column name.")
    args = parser.parse_args()

    model_path = ARTIFACT_DIR / "model_pipeline.joblib"
    _, metrics = train_pipeline(args.data, args.target, model_path)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
