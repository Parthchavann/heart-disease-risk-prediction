"""
Prediction endpoints for heart disease risk assessment.
"""

import asyncio
import time
import uuid
from typing import List
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from datetime import datetime

from api.models import (
    PredictionRequest, PredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, PatientDataRequest, PredictionOptions
)
from src.prediction_service import HeartDiseasePredictionService
from api.middleware.validation import validate_patient_data_ranges, sanitize_input_data
from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)

router = APIRouter(prefix="/predict", tags=["prediction"])


def get_prediction_service() -> HeartDiseasePredictionService:
    """Dependency to get prediction service instance."""
    try:
        return HeartDiseasePredictionService()
    except Exception as e:
        logger.error(f"Failed to initialize prediction service: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Prediction service unavailable. Please try again later."
        )


@router.post("/", response_model=PredictionResponse)
async def predict_heart_disease_risk(
    request: PredictionRequest,
    service: HeartDiseasePredictionService = Depends(get_prediction_service)
):
    """
    Predict heart disease risk for a single patient.

    Args:
        request: Patient data and prediction options

    Returns:
        PredictionResponse: Complete prediction result with explanations
    """
    logger.info("Prediction request received")

    try:
        # Extract patient data as dictionary
        patient_data = request.patient_data.dict()

        # Validate and sanitize input
        validate_patient_data_ranges(patient_data)
        patient_data = sanitize_input_data(patient_data)

        # Make prediction with options — run in thread so the LLM call (30-60 s for
        # local Ollama/Gemma2) doesn't block the FastAPI event loop.
        result = await asyncio.to_thread(
            service.predict,
            patient_data,
            request.options.include_explanation,
            request.options.include_llm_explanation,
        )

        logger.info(f"Prediction completed: {result['prediction_id']}")

        # Convert to response model
        return PredictionResponse(**result)

    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again later."
        )


@router.post("/simple", response_model=PredictionResponse)
async def predict_simple(
    patient_data: PatientDataRequest,
    include_explanation: bool = True,
    include_llm_explanation: bool = True,
    service: HeartDiseasePredictionService = Depends(get_prediction_service)
):
    """
    Simple prediction endpoint with patient data directly in request body.

    Args:
        patient_data: Patient health indicators
        include_explanation: Whether to include SHAP explanation
        include_llm_explanation: Whether to include LLM explanation

    Returns:
        PredictionResponse: Complete prediction result
    """
    # Create request object
    request = PredictionRequest(
        patient_data=patient_data,
        options=PredictionOptions(
            include_explanation=include_explanation,
            include_llm_explanation=include_llm_explanation
        )
    )

    # Use the main prediction endpoint
    return await predict_heart_disease_risk(request, service)


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    service: HeartDiseasePredictionService = Depends(get_prediction_service)
):
    """
    Batch prediction for multiple patients.

    Args:
        request: Batch of patient data
        background_tasks: FastAPI background tasks

    Returns:
        BatchPredictionResponse: Results for all patients
    """
    batch_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(f"Batch prediction request {batch_id} with {len(request.patients)} patients")

    # Limit batch size
    max_batch_size = 100
    if len(request.patients) > max_batch_size:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size {len(request.patients)} exceeds maximum {max_batch_size}"
        )

    predictions = []
    successful_count = 0
    failed_count = 0

    for i, patient_data in enumerate(request.patients):
        try:
            # Convert patient data to dict
            patient_dict = patient_data.dict()

            # Validate and sanitize
            validate_patient_data_ranges(patient_dict)
            patient_dict = sanitize_input_data(patient_dict)

            # Make prediction
            result = service.predict(
                patient_dict,
                include_explanation=request.options.include_explanation,
                include_llm_explanation=request.options.include_llm_explanation
            )

            predictions.append(PredictionResponse(**result))
            successful_count += 1

        except Exception as e:
            logger.error(f"Failed to process patient {i} in batch {batch_id}: {str(e)}")

            # Create error response for this patient
            error_result = {
                'prediction_id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e),
                'medical_disclaimer': settings.MEDICAL_DISCLAIMER
            }

            predictions.append(PredictionResponse(**error_result))
            failed_count += 1

    processing_time = time.time() - start_time

    batch_response = BatchPredictionResponse(
        batch_id=batch_id,
        timestamp=datetime.now().isoformat(),
        total_predictions=len(request.patients),
        successful_predictions=successful_count,
        failed_predictions=failed_count,
        predictions=predictions,
        processing_time_seconds=processing_time
    )

    logger.info(f"Batch prediction {batch_id} completed: "
               f"{successful_count} successful, {failed_count} failed, "
               f"{processing_time:.2f}s")

    return batch_response


@router.get("/example")
async def get_example_request():
    """
    Get an example prediction request for testing.

    Returns:
        dict: Example request structure
    """
    return {
        "patient_data": {
            "age": 45,
            "sex": 1,
            "cp": 2,
            "trestbps": 130,
            "chol": 250,
            "fbs": 0,
            "restecg": 0,
            "thalach": 175,
            "exang": 0,
            "oldpeak": 1.2,
            "slope": 1,
            "ca": 1,
            "thal": 2
        },
        "options": {
            "include_explanation": True,
            "include_llm_explanation": True
        }
    }


@router.get("/features")
async def get_feature_descriptions():
    """
    Get descriptions of all input features.

    Returns:
        dict: Feature descriptions and valid ranges
    """
    from utils.constants import FEATURE_DESCRIPTIONS, FEATURE_RANGES

    feature_info = {}
    for feature in FEATURE_DESCRIPTIONS:
        feature_info[feature] = {
            "description": FEATURE_DESCRIPTIONS[feature],
            "range": FEATURE_RANGES.get(feature, {"min": None, "max": None})
        }

    return {
        "features": feature_info,
        "note": "All values should be provided as numbers. See API documentation for detailed descriptions."
    }