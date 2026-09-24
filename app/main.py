import pandas as pd
import time
from src.logger import logger
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from src.monitoring import (
    REQUEST_COUNT,
    ERROR_COUNT,
    PREDICTION_COUNT,
    PREDICTION_LATENCY,
)

from app.schemas import (
    OrderInput,
    PredictionResponse,
    BatchOrderInput,
    BatchPredictionResponse,
)
from src.config import load_config
from src.inference import run_inference


app = FastAPI(
    title="Olist Late Delivery API",
    version="1.0.0",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(
        "Invalid request | path=%s | errors=%s",
        request.url.path,
        exc.errors(),
    )

    raise HTTPException(
        status_code=422,
        detail=exc.errors(),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/model-info")
def model_info():
    config = load_config()

    return {
        "model_name": config["project"]["name"],
        "model_version": config["project"]["model_version"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_order(order: OrderInput):
    start_time = time.perf_counter()
    REQUEST_COUNT.inc()

    logger.info(
        "Prediction request received | input=%s",
        order.model_dump(),
    )

    df = pd.DataFrame([order.model_dump()])

    try:
        result = run_inference(df)
    except Exception:
        ERROR_COUNT.inc()
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        )

    latency = time.perf_counter() - start_time
    PREDICTION_LATENCY.observe(latency)

    prediction = result["prediction"][0]
    probability = result["probability"][0]

    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

    logger.info(
        "Prediction completed | prediction=%s | probability=%.4f | "
        "model_version=%s | latency=%.4fs",
        prediction,
        probability,
        result["model_version"],
        latency,
    )

    return {
        "prediction": prediction,
        "probability": probability,
        "model_version": result["model_version"],
    }


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchOrderInput):
    start_time = time.perf_counter()
    REQUEST_COUNT.inc()

    logger.info(
        "Batch prediction request received | number_of_orders=%s | input=%s",
        len(batch.orders),
        batch.model_dump(),
    )

    df = pd.DataFrame([order.model_dump() for order in batch.orders])

    try:
        result = run_inference(df)
    except Exception:
        ERROR_COUNT.inc()
        logger.exception("Batch prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed",
        )

    latency = time.perf_counter() - start_time
    PREDICTION_LATENCY.observe(latency)

    predictions = []

    for prediction, probability in zip(
        result["prediction"],
        result["probability"],
    ):
        PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

        predictions.append(
            {
                "prediction": prediction,
                "probability": probability,
                "model_version": result["model_version"],
            }
        )

    logger.info(
        "Batch prediction completed | predictions=%s | probabilities=%s | "
        "model_version=%s | latency=%.4fs",
        result["prediction"],
        result["probability"],
        result["model_version"],
        latency,
    )

    return {"predictions": predictions}
