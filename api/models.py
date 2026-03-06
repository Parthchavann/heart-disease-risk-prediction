"""
Pydantic models for API request/response schemas.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
from utils.constants import RiskLevel


class PatientDataRequest(BaseModel):
    """Request model for patient data input."""

    age: int = Field(..., ge=18, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0=female, 1=male)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=80, le=200, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (0=false, 1=true)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (0=no, 1=yes)")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0-3)")

    class Config:
        schema_extra = {
            "example": {
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
            }
        }


class PredictionOptions(BaseModel):
    """Options for prediction request."""

    include_explanation: bool = Field(True, description="Include SHAP-based explanation")
    include_llm_explanation: bool = Field(True, description="Include LLM-generated explanation")


class PredictionRequest(BaseModel):
    """Complete prediction request model."""

    patient_data: PatientDataRequest
    options: Optional[PredictionOptions] = PredictionOptions()


class RiskFactor(BaseModel):
    """Risk factor information."""

    feature: str = Field(..., description="Feature name")
    contribution: float = Field(..., description="SHAP contribution value")
    feature_value: float = Field(..., description="Actual feature value")
    explanation: str = Field(..., description="Human-readable explanation")


class ProtectiveFactor(BaseModel):
    """Protective factor information."""

    feature: str = Field(..., description="Feature name")
    contribution: float = Field(..., description="SHAP contribution value")
    feature_value: float = Field(..., description="Actual feature value")
    explanation: str = Field(..., description="Human-readable explanation")


class ModelInfo(BaseModel):
    """Model metadata information."""

    model_name: str = Field(..., description="Name of the ML model")
    model_version: str = Field(..., description="Version of the model")
    training_timestamp: str = Field(..., description="When the model was trained")


class FeatureContribution(BaseModel):
    """Individual feature contribution from SHAP."""

    shap_value: float = Field(..., description="SHAP contribution value")
    feature_value: float = Field(..., description="Actual feature value")
    impact: str = Field(..., description="Impact direction: increases_risk or decreases_risk")


class DetailedExplanation(BaseModel):
    """Detailed SHAP explanation."""

    prediction_probability: float = Field(..., description="Predicted risk probability")
    expected_value: float = Field(..., description="Model's expected baseline value")
    feature_contributions: Dict[str, FeatureContribution] = Field(
        ..., description="SHAP values for each feature"
    )
    top_risk_factors: List[RiskFactor] = Field(..., description="Top contributing risk factors")
    top_protective_factors: List[ProtectiveFactor] = Field(..., description="Top protective factors")


class LLMExplanation(BaseModel):
    """LLM-generated patient-friendly explanation."""

    risk_explanation: str = Field(..., description="Patient-friendly risk explanation")
    lifestyle_recommendations: List[str] = Field(..., description="Personalized lifestyle recommendations")
    doctor_consultation_questions: List[str] = Field(..., description="Questions to ask healthcare provider")
    generated_timestamp: str = Field(..., description="When explanation was generated")
    risk_level: str = Field(..., description="Risk level category")


class PredictionResponse(BaseModel):
    """Complete prediction response model."""

    prediction_id: str = Field(..., description="Unique prediction identifier")
    timestamp: str = Field(..., description="Prediction timestamp")
    success: bool = Field(..., description="Whether prediction succeeded")

    # Core prediction results
    prediction: Optional[int] = Field(None, description="Binary prediction (0=no disease, 1=disease)")
    risk_probability: Optional[float] = Field(None, description="Risk probability (0-1)")
    risk_level: Optional[str] = Field(None, description="Risk level: Low, Moderate, High")
    confidence_interval: Optional[Tuple[float, float]] = Field(None, description="Confidence interval")

    # Model information
    model_info: Optional[ModelInfo] = Field(None, description="Information about the model used")

    # Explanations
    explanation: Optional[DetailedExplanation] = Field(None, description="SHAP-based explanation")
    llm_explanation: Optional[LLMExplanation] = Field(None, description="LLM-generated explanation")

    # Error handling
    error: Optional[str] = Field(None, description="Error message if prediction failed")
    validation_errors: Optional[List[str]] = Field(None, description="Input validation errors")

    # Compliance
    medical_disclaimer: str = Field(..., description="Medical disclaimer")

    class Config:
        schema_extra = {
            "example": {
                "prediction_id": "123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2024-01-15T10:30:00Z",
                "success": True,
                "prediction": 0,
                "risk_probability": 0.35,
                "risk_level": "Moderate",
                "confidence_interval": [0.25, 0.45],
                "model_info": {
                    "model_name": "random_forest",
                    "model_version": "1.0.0",
                    "training_timestamp": "2024-01-10T15:20:00Z"
                },
                "medical_disclaimer": "This tool is for educational purposes only..."
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class HealthCheckResponse(BaseModel):
    """Health check response model."""

    timestamp: str = Field(..., description="Health check timestamp")
    service_status: str = Field(..., description="Overall service status")
    components: Dict[str, str] = Field(..., description="Status of individual components")

    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2024-01-15T10:30:00Z",
                "service_status": "healthy",
                "components": {
                    "model": "healthy",
                    "data_processor": "healthy",
                    "explainer": "healthy",
                    "llm_generator": "healthy"
                }
            }
        }


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_name: str = Field(..., description="Name of the ML model")
    model_version: str = Field(..., description="Version of the model")
    training_timestamp: str = Field(..., description="When the model was trained")
    metrics: Optional[Dict[str, float]] = Field(None, description="Model performance metrics")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")
    training_config: Optional[Dict[str, Any]] = Field(None, description="Training configuration")

    class Config:
        schema_extra = {
            "example": {
                "model_name": "random_forest",
                "model_version": "1.0.0",
                "training_timestamp": "2024-01-10T15:20:00Z",
                "metrics": {
                    "accuracy": 0.85,
                    "precision": 0.83,
                    "recall": 0.87,
                    "f1_score": 0.85,
                    "roc_auc": 0.92
                },
                "feature_importance": {
                    "ca": 0.15,
                    "thal": 0.12,
                    "cp": 0.11
                }
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""

    patients: List[PatientDataRequest] = Field(..., min_length=1, description="List of patient data")
    options: Optional[PredictionOptions] = PredictionOptions()

    class Config:
        schema_extra = {
            "example": {
                "patients": [
                    {
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
                    }
                ],
                "options": {
                    "include_explanation": True,
                    "include_llm_explanation": False
                }
            }
        }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""

    batch_id: str = Field(..., description="Unique batch identifier")
    timestamp: str = Field(..., description="Batch processing timestamp")
    total_predictions: int = Field(..., description="Total number of predictions")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    predictions: List[PredictionResponse] = Field(..., description="Individual prediction results")
    processing_time_seconds: float = Field(..., description="Total processing time")

    class Config:
        schema_extra = {
            "example": {
                "batch_id": "batch_123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2024-01-15T10:30:00Z",
                "total_predictions": 2,
                "successful_predictions": 2,
                "failed_predictions": 0,
                "predictions": [],
                "processing_time_seconds": 1.25
            }
        }