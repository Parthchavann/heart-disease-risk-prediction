"""
FastAPI main application for Heart Disease Risk Prediction API.
"""

import os
import sys
from contextlib import asynccontextmanager

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator

from api.routes import health, prediction, report, auth
from api.middleware.error_handling import (
    ErrorHandlingMiddleware,
    validation_exception_handler,
    not_found_exception_handler,
    generic_exception_handler
)
from api.middleware.validation import RequestLoggingMiddleware, RateLimitingMiddleware
from config.settings import settings
from config.logging_config import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    from api.database import init_db
    init_db()
    logger.info("Database initialised")
    logger.info("Starting Heart Disease Risk Prediction API...")
    logger.info(f"API version: {settings.VERSION}")
    logger.info(f"Debug mode: {settings.DEBUG}")

    try:
        # Test service initialization
        from src.prediction_service import HeartDiseasePredictionService
        test_service = HeartDiseasePredictionService()
        health_status = test_service.health_check()
        logger.info(f"Service health check: {health_status['service_status']}")

        if health_status['service_status'] == 'failed':
            logger.error("Service initialization failed")
            raise Exception("Service health check failed")

        logger.info("API startup completed successfully")

    except Exception as e:
        logger.error(f"API startup failed: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Heart Disease Risk Prediction API...")


# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Prometheus instrumentation — auto-tracks request count + latency per endpoint
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Add middleware
_allowed_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitingMiddleware, max_requests_per_minute=100)
app.add_middleware(ErrorHandlingMiddleware)

# Add exception handlers
app.add_exception_handler(ValueError, validation_exception_handler)
app.add_exception_handler(FileNotFoundError, not_found_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Include routers
app.include_router(health.router)
app.include_router(prediction.router)
app.include_router(report.router)
app.include_router(auth.router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "description": settings.DESCRIPTION,
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "disclaimer": settings.MEDICAL_DISCLAIMER
    }


@app.get("/info")
async def get_api_info():
    """Get detailed API information."""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "description": settings.DESCRIPTION,
        "endpoints": {
            "health_check": "/health",
            "model_info": "/health/model",
            "predict": "/predict",
            "predict_simple": "/predict/simple",
            "predict_batch": "/predict/batch",
            "example_request": "/predict/example",
            "feature_info": "/predict/features"
        },
        "features": [
            "Heart disease risk prediction",
            "SHAP-based explanations",
            "LLM-generated patient-friendly explanations",
            "Batch processing",
            "Comprehensive health monitoring"
        ],
        "model_info": {
            "type": "Machine Learning Ensemble",
            "algorithms": ["Random Forest", "Logistic Regression", "XGBoost"],
            "dataset": "UCI Heart Disease (Cleveland)",
            "features": 13
        },
        "disclaimer": settings.MEDICAL_DISCLAIMER
    }


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )