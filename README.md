# End-to-End ML Pipeline + Deployment

This project is a production-style machine learning template that combines preprocessing, training, model persistence, logging, input validation, and FastAPI deployment in one modular codebase.

## Business problem

Many machine learning projects stop at the modeling stage. In practice, that is rarely enough. Teams need a repeatable pipeline that can train on historical data, save a production-ready artifact, expose a stable API, and fail gracefully when inputs are bad or artifacts are missing.

This project addresses that need by providing:

- a reusable sklearn preprocessing and modeling pipeline
- a training entrypoint for building artifacts from CSV data
- a prediction API for real-time scoring
- structured logging and error handling
- a modular layout that can be adapted to many tabular classification use cases

Unlike the fraud and churn projects, this repository is intentionally domain-agnostic. It is meant to act as a starter template for deployable ML systems.

## Objective

The goal of the system is to provide a simple but realistic pattern for moving from raw tabular data to a deployable inference service. It supports two practical use cases:

- batch training on a labeled CSV dataset
- real-time predictions through a FastAPI endpoint

The value of this project is less about a single business domain and more about showing sound ML engineering structure.

## Who this is for

This repository is written for:

- engineers who want a reusable deployment template
- interviewers who want to see practical ML system design
- data scientists who want to productionize a tabular model quickly
- teams who need a clean starting point for an internal scoring service

## Project structure

```text
end_to_end_ml_pipeline_deployment/
|-- generate_sample_data.py
|-- main.py
|-- train.py
|-- requirements.txt
|-- README.md
`-- src/
    |-- __init__.py
    |-- pipeline.py
    |-- schemas.py
    |-- service.py
    `-- utils.py
```

## End-to-end workflow

The system works in the following sequence:

1. a labeled dataset is loaded from CSV
2. numeric and categorical features are preprocessed
3. a model is trained inside an sklearn `Pipeline`
4. the full pipeline is saved with joblib
5. the API loads the pipeline artifact
6. new JSON inputs are validated and scored through `/predict`

Conceptually:

```text
raw tabular data -> preprocessing pipeline -> model training
-> persisted artifact -> FastAPI service -> prediction response
```

## Features

- sklearn `Pipeline` and `ColumnTransformer` for preprocessing and modeling
- Random Forest model for a strong tabular baseline
- joblib-based persistence for the full pipeline
- FastAPI endpoint at `/predict`
- input validation with Pydantic
- structured logging
- error handling for startup and prediction failures

## Architecture and file responsibilities

### `generate_sample_data.py`

Creates a local sample dataset so the repository can be run immediately without waiting for an external data source.

### `train.py`

Acts as the training entrypoint:

- reads a CSV dataset
- accepts the target column as an argument
- calls the training pipeline
- saves both the pipeline artifact and metrics file

### `main.py`

Defines the FastAPI service:

- loads the trained model at startup when available
- exposes a health endpoint
- exposes `/predict`
- returns clean errors if the model is missing or input is invalid

### `src/pipeline.py`

Contains the reusable ML training logic:

- dataset loading
- preprocessing construction
- sklearn pipeline assembly
- training
- evaluation
- model persistence

This module is the core of the project’s ML workflow.

### `src/service.py`

Contains inference helpers:

- load the saved pipeline
- prepare a single request payload
- return prediction outputs in API-friendly form

### `src/schemas.py`

Defines request and response schemas with Pydantic. This improves API clarity and protects the service from malformed payloads.

### `src/utils.py`

Provides logging configuration so training and serving share a consistent logging pattern.

## Why this design is production-oriented

This repository is intentionally simple, but it includes the core ingredients of a deployable ML system:

- preprocessing and model logic packaged together in one sklearn pipeline
- artifact persistence for consistent inference behavior
- logging for observability
- structured validation for API requests
- error handling for missing models or bad inputs
- clear module boundaries between training, serving, and utility code

These are the habits that help a project move from experimental code toward something maintainable.

## Assumptions and limitations

This project keeps the scope focused:

- it uses a single tabular model family rather than broad model comparison
- it does not include CI/CD, Docker, cloud deployment, or orchestration
- the sample dataset is synthetic and generic
- monitoring, drift detection, and retraining automation are not included
- authentication and rate limiting are not yet added to the API

That is appropriate for a clean starter template and interview-ready codebase.

## Installation

```bash
pip install -r requirements.txt
```

## Sample dataset

You can generate a sample CSV locally with:

```bash
python generate_sample_data.py
```

It writes `data/training_data.csv`.

The sample is synthetic and generic so the project can run immediately without any external download.

## Train the model

```bash
python train.py --data data/training_data.csv --target target
```

Artifacts are stored in `artifacts/`:

- `model_pipeline.joblib`
- `metrics.json`

The metrics file records training-time evaluation so you can quickly inspect whether the trained artifact is behaving reasonably.

## What success looks like

From a system-design perspective, success means:

- the pipeline can be trained end to end from a CSV file
- the same preprocessing logic is reused during inference
- the model artifact can be saved and reloaded predictably
- the API validates request format and returns clean responses
- the codebase is easy to adapt to a different tabular use case

## Run locally

```bash
uvicorn main:app --reload
```

You can also test the service interactively through FastAPI's autogenerated documentation at:

```text
http://127.0.0.1:8000/docs
```

## API request format

```json
{
  "features": {
    "age": 42,
    "income": 85000,
    "tenure_months": 36,
    "product": "premium",
    "region": "east"
  }
}
```

Example response:

```json
{
  "prediction": 1,
  "probability": 0.81
}
```

## How to adapt this template

To reuse this project for a real use case, you would typically:

- replace the sample dataset with your own labeled CSV
- adjust the target column passed to `train.py`
- update the request schema if your input contract is fixed
- swap the estimator in `src/pipeline.py` if another model is more suitable
- tighten validation, logging, and deployment settings for your environment

## Future improvements

Natural next steps for this template would be:

- model registry integration
- Docker packaging
- CI test coverage
- cloud deployment configuration
- monitoring and drift alerts
- authentication and request tracing

## Conclusion

This project demonstrates how to package a tabular ML workflow into a reusable deployment template. It is suitable as:

- a portfolio project
- an interview walkthrough
- a starter foundation for internal ML APIs

It shows not just how to train a model, but how to organize the code so that training and inference work together in a maintainable way.
