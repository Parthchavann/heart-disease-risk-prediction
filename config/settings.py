"""
Configuration settings for the Heart Disease Risk Prediction system.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Project metadata
    PROJECT_NAME: str = "Heart Disease Risk Prediction Assistant"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Explainable ML system for heart disease risk prediction"

    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False

    # Model settings
    MODEL_VERSION: str = "1.0.0"
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.15
    VAL_SIZE: float = 0.15
    CV_FOLDS: int = 5

    # Data paths
    DATA_DIR: str = "data"
    RAW_DATA_DIR: str = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR: str = os.path.join(DATA_DIR, "processed")
    MODELS_DIR: str = os.path.join(DATA_DIR, "models")

    # UCI dataset URL
    UCI_HEART_DISEASE_URL: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

    # LLM settings
    LLM_PROVIDER: str = "ollama"  # "ollama" | "gemini" | "openai"
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "gemma2"
    LLM_MAX_TOKENS: int = 500
    LLM_TEMPERATURE: float = 0.3

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"

    # CORS — comma-separated list of allowed origins, e.g. "https://your-app.lovable.app,https://your-app.railway.app"
    ALLOWED_ORIGINS: str = "*"

    # Security
    SECRET_KEY: str = "your-secret-key-here"  # Change in production

    # Medical disclaimers
    MEDICAL_DISCLAIMER: str = (
        "This tool is for educational and preventive awareness purposes only. "
        "It is NOT a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult with a qualified healthcare provider for medical decisions."
    )

    # Risk level thresholds
    LOW_RISK_THRESHOLD: float = 0.3
    HIGH_RISK_THRESHOLD: float = 0.7

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()