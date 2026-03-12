"""
Prometheus custom metrics for Heart Disease Risk Prediction API.
Defined as module-level singletons to avoid duplicate registration errors.
"""

from prometheus_client import Counter, Histogram

# Count of predictions by risk level label (Low / Moderate / High)
PREDICTION_RISK_LEVEL_COUNTER = Counter(
    "heart_disease_prediction_risk_level_total",
    "Count of predictions by risk level",
    ["risk_level"],
)

# Time spent in ML model inference (preprocessing + predict_proba + confidence interval)
MODEL_INFERENCE_HISTOGRAM = Histogram(
    "heart_disease_model_inference_seconds",
    "Time spent in ML model inference",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

# Time spent waiting for LLM response (cache hits are excluded)
LLM_RESPONSE_HISTOGRAM = Histogram(
    "heart_disease_llm_response_seconds",
    "Time spent waiting for LLM response",
    buckets=[1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0],
)
