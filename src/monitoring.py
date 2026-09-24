from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
)

ERROR_COUNT = Counter(
    "prediction_errors_total",
    "Total number of prediction errors",
)

PREDICTION_COUNT = Counter(
    "prediction_results_total",
    "Distribution of prediction results",
    ["prediction"],
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency in seconds",
)
