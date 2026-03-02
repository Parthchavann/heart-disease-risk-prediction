"""
Health check and system status endpoints.
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any

from api.models import HealthCheckResponse, ModelInfoResponse, ErrorResponse
from src.prediction_service import HeartDiseasePredictionService
from config.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


def get_prediction_service() -> HeartDiseasePredictionService:
    """Dependency to get prediction service instance."""
    # In production, this would be a singleton or cached instance
    return HeartDiseasePredictionService()


@router.get("/", response_model=HealthCheckResponse)
async def health_check(service: HeartDiseasePredictionService = Depends(get_prediction_service)):
    """
    Perform health check of the service and all components.

    Returns:
        HealthCheckResponse: Status of the service and its components
    """
    logger.info("Health check requested")

    try:
        health_status = service.health_check()
        logger.info(f"Health check completed: {health_status['service_status']}")
        return HealthCheckResponse(**health_status)

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        error_response = {
            "timestamp": "",
            "service_status": "failed",
            "components": {"error": str(e)}
        }
        return HealthCheckResponse(**error_response)


@router.get("/model", response_model=ModelInfoResponse)
async def get_model_info(service: HeartDiseasePredictionService = Depends(get_prediction_service)):
    """
    Get information about the loaded ML model.

    Returns:
        ModelInfoResponse: Model metadata and performance information
    """
    logger.info("Model info requested")

    try:
        model_info = service.get_model_info()

        if "error" in model_info:
            logger.error(f"Model info error: {model_info['error']}")
            raise Exception(model_info['error'])

        logger.info(f"Model info retrieved: {model_info['model_name']}")
        return ModelInfoResponse(**model_info)

    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise


@router.get("/status")
async def get_service_status():
    """
    Get simple service status.

    Returns:
        dict: Basic service status information
    """
    return {
        "status": "online",
        "service": "Heart Disease Risk Prediction API",
        "version": "1.0.0"
    }


@router.get("/ready")
async def readiness_probe(service: HeartDiseasePredictionService = Depends(get_prediction_service)):
    """
    Kubernetes-style readiness probe.

    Returns:
        dict: Readiness status
    """
    try:
        health_status = service.health_check()

        if health_status['service_status'] in ['healthy', 'degraded']:
            return {"status": "ready"}
        else:
            return {"status": "not_ready", "reason": "service_unhealthy"}

    except Exception as e:
        logger.error(f"Readiness probe failed: {str(e)}")
        return {"status": "not_ready", "reason": str(e)}


@router.get("/live")
async def liveness_probe():
    """
    Kubernetes-style liveness probe.

    Returns:
        dict: Liveness status
    """
    return {"status": "alive"}