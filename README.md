# End-to-End ML Pipeline + Deployment

This project is a production-style machine learning template that combines preprocessing, training, model persistence, and FastAPI deployment in one modular codebase.

## Project structure

```text
end_to_end_ml_pipeline_deployment/
|-- main.py
|-- train.py
|-- requirements.txt
|-- README.md
`-- src/
    |-- pipeline.py
    |-- schemas.py
    |-- service.py
    `-- utils.py
```

## Features

- sklearn `Pipeline` and `ColumnTransformer` for preprocessing and modeling
- Random Forest model for strong baseline performance
- joblib-based persistence for the full pipeline
- FastAPI endpoint at `/predict`
- input validation with Pydantic
- structured logging
- error handling for startup and prediction failures

## Installation

```bash
pip install -r requirements.txt
```

## Sample dataset

You can generate a sample CSV locally with:

```bash
python generate_sample_data.py
```

It writes `data/training_data.csv` using scikit-learn's Breast Cancer Wisconsin dataset:

- scikit-learn docs: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

This is a convenient built-in binary classification dataset, so the project can run immediately without an external download step.

## Train the model

```bash
python train.py --data data/training_data.csv --target target
```

Artifacts are stored in `artifacts/`:

- `model_pipeline.joblib`
- `metrics.json`

## Run locally

```bash
uvicorn main:app --reload
```

## API request format

```json
{
  "features": {
    "age": 42,
    "income": 85000,
    "product": "premium"
  }
}
```

## Deployment notes

- Train the model before starting the API in production.
- Keep request payloads aligned with the training schema.
- Extend `src/schemas.py` with stricter field-level validation if the feature contract is fixed.
