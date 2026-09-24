# Olist Late Delivery MLOps

An end-to-end MLOps project for predicting whether an Olist e-commerce order will be delivered late.

The project converts the exploratory and modeling work from the previous notebooks into a reproducible inference service using FastAPI, MLflow, DVC, Great Expectations, Docker, automated tests, CI, logging, and monitoring.

## Project Structure

```text
olist_MLops/
├── app/                  # FastAPI application and API schemas
├── config/               # Project configuration
├── data/                 # Versioned dataset
├── models/               # Model and preprocessing artifacts
├── notebooks/            # Original analysis/modeling notebooks
├── requirements/         # Runtime and development dependencies
├── src/                  # Reusable inference and validation modules
├── tests/                # Automated tests
├── .github/workflows/    # CI workflow
├── .pre-commit-config.yaml
├── compose.yaml
├── Dockerfile
├── models.dvc
└── README.md
```

## Machine Learning Model

The service predicts late delivery as a binary classification problem.

The production model is loaded through MLflow Model Registry and is not retrained when the API starts or when a prediction request is received.

Model configuration:

- MLflow model name: `olist-late-delivery-model`
- MLflow registry version: `1`
- API model version: `1.0.0`

Preprocessing artifacts and the feature list are loaded from saved artifacts.

## Configuration

Project settings are stored in:

```text
config/config.yaml
```

Configuration is separated from the application code. Environment variables are used where appropriate, for example:

```text
MLFLOW_TRACKING_URI=http://mlflow:5000
```

## Installation

The project targets Python 3.11.

Runtime dependencies are pinned in:

```text
requirements/runtime.txt
```

Development dependencies are stored separately in:

```text
requirements/dev.txt
```

## Running with Docker Compose

Docker Compose starts the API and MLflow services.

```bash
docker compose up -d --build
```

Check the running containers:

```bash
docker compose ps
```

The API is available at:

```text
http://localhost:8000
```

MLflow is available at:

```text
http://localhost:5000
```

FastAPI Swagger documentation is available at:

```text
http://localhost:8000/docs
```

Stop the services with:

```bash
docker compose down
```

## API Endpoints

### Health Check

```http
GET /health
```

Returns the service health status.

### Model Information

```http
GET /model-info
```

Returns information about the model and model version.

### Single Prediction

```http
POST /predict
```

Returns:

- prediction
- probability
- model version

### Batch Prediction

```http
POST /predict/batch
```

Accepts multiple orders and returns predictions for all submitted orders.

### Monitoring Metrics

```http
GET /metrics
```

Exposes Prometheus-compatible service and prediction metrics.

## Logging and Error Handling

The application uses Python logging instead of `print()` for production inference.

Prediction requests log information including:

- request input
- prediction output
- probability
- model version
- request latency

Invalid requests are handled by FastAPI/Pydantic validation and prediction failures are logged with exception information.

## Monitoring

The API exposes Prometheus-compatible metrics through `/metrics`.

Custom metrics include:

```text
prediction_requests_total
prediction_errors_total
prediction_results_total
prediction_latency_seconds
```

These metrics allow monitoring of:

- prediction request count
- prediction errors
- latency
- prediction distribution

Error rate can be calculated from:

```text
prediction_errors_total / prediction_requests_total
```

The prediction distribution can be monitored over time to detect changes in model output behavior and potential drift.

Prediction inputs, outputs, probabilities, model version, and latency are also written to application logs. These logs can later be combined with ground-truth delivery outcomes for model performance evaluation.

### Alerting Decision

Suggested operational alerts:

- error rate above 5% during a 5-minute period
- p95 prediction latency above 1 second
- significant sustained change in the prediction distribution compared with the established baseline

These thresholds should be adjusted after observing normal production traffic.

## Data Validation

Great Expectations is used to validate data quality before model use.

Validation covers relevant checks such as:

- expected schema
- required columns
- missing values
- valid ranges
- allowed categorical values

Validation failures stop the affected processing step rather than silently accepting invalid data.

## Data and Artifact Versioning

DVC is used to version the dataset and model artifacts.

Examples:

```text
data/olist_merged_orders.csv.dvc
models.dvc
```

This separates large data/model artifacts from normal Git source-code versioning.

## MLflow

MLflow is used for experiment/model tracking and model registry.

The inference service loads the registered production model from MLflow instead of training a new model.

Docker Compose provides the MLflow tracking service used by the API.

## Tests

The project contains automated tests for data processing, preprocessing, inference, validation, CLI behavior, and API behavior.

Run the full test suite with:

```bash
pytest -q
```

A failing test returns a non-zero exit code and therefore stops the CI pipeline.

## Code Quality

Ruff is used for linting and formatting.

Run linting:

```bash
ruff check app src tests
```

Check formatting:

```bash
ruff format --check app src tests
```

Pre-commit hooks are configured in:

```text
.pre-commit-config.yaml
```

Install the hooks with:

```bash
pre-commit install
```

## CI

GitHub Actions runs automatically on pushes and pull requests.

The CI pipeline:

1. checks out the repository
2. sets up Python 3.11
3. installs runtime and development dependencies
4. runs Ruff linting
5. checks formatting
6. runs the automated tests
7. builds the Docker image if the previous steps succeed

A failure in linting, formatting, or tests stops the pipeline.

## Reproducibility

The project separates source code, configuration, dependencies, data, model artifacts, and infrastructure definitions.

The main application can be started with:

```bash
docker compose up -d --build
```

Inference uses saved preprocessing artifacts and the registered MLflow model; model retraining is not part of the API runtime.