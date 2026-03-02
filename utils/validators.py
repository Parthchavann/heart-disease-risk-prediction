"""
Data validation utilities for the Heart Disease Risk Prediction system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from pydantic import BaseModel, Field, validator
import logging

from utils.constants import FEATURE_RANGES, FEATURE_NAMES
from config.logging_config import get_logger

logger = get_logger(__name__)


class PatientData(BaseModel):
    """Pydantic model for validating patient input data."""

    age: int = Field(..., ge=18, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0=female, 1=male)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type")
    trestbps: int = Field(..., ge=80, le=200, description="Resting blood pressure")
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results")
    thalach: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia")

    @validator('age')
    def validate_age(cls, v):
        if not 18 <= v <= 120:
            raise ValueError('Age must be between 18 and 120')
        return v

    @validator('trestbps')
    def validate_blood_pressure(cls, v):
        if v < 80:
            logger.warning(f"Very low blood pressure: {v}")
        elif v > 180:
            logger.warning(f"Very high blood pressure: {v}")
        return v

    @validator('chol')
    def validate_cholesterol(cls, v):
        if v < 120:
            logger.warning(f"Very low cholesterol: {v}")
        elif v > 400:
            logger.warning(f"Very high cholesterol: {v}")
        return v

    @validator('thalach')
    def validate_heart_rate(cls, v, values):
        if 'age' in values:
            max_theoretical = 220 - values['age']
            if v > max_theoretical:
                logger.warning(f"Heart rate {v} exceeds theoretical max {max_theoretical}")
        return v

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


class DataValidator:
    """Validates datasets and individual samples."""

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate a complete DataFrame."""

        errors = []

        # Check if required columns exist
        missing_features = set(FEATURE_NAMES) - set(df.columns)
        if missing_features:
            errors.append(f"Missing features: {missing_features}")

        # Check data types
        for feature in FEATURE_NAMES:
            if feature in df.columns:
                if feature == 'oldpeak':
                    if not pd.api.types.is_numeric_dtype(df[feature]):
                        errors.append(f"Feature '{feature}' must be numeric")
                else:
                    if not pd.api.types.is_integer_dtype(df[feature]) and not pd.api.types.is_numeric_dtype(df[feature]):
                        errors.append(f"Feature '{feature}' must be numeric")

        # Check value ranges
        for feature, ranges in FEATURE_RANGES.items():
            if feature in df.columns:
                min_val, max_val = ranges['min'], ranges['max']
                out_of_range = ((df[feature] < min_val) | (df[feature] > max_val)).sum()
                if out_of_range > 0:
                    errors.append(f"Feature '{feature}': {out_of_range} values out of range [{min_val}, {max_val}]")

        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            errors.append(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")

        return len(errors) == 0, errors

    @staticmethod
    def validate_sample(sample: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a single sample."""

        try:
            PatientData(**sample)
            return True, []
        except Exception as e:
            return False, [str(e)]

    @staticmethod
    def get_feature_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for features."""

        summary = {}

        for feature in FEATURE_NAMES:
            if feature in df.columns:
                feature_data = df[feature]
                summary[feature] = {
                    'count': len(feature_data),
                    'missing': feature_data.isnull().sum(),
                    'mean': feature_data.mean() if pd.api.types.is_numeric_dtype(feature_data) else None,
                    'std': feature_data.std() if pd.api.types.is_numeric_dtype(feature_data) else None,
                    'min': feature_data.min(),
                    'max': feature_data.max(),
                    'unique_values': feature_data.nunique()
                }

                # Add percentiles for numerical features
                if pd.api.types.is_numeric_dtype(feature_data):
                    percentiles = feature_data.quantile([0.25, 0.5, 0.75])
                    summary[feature].update({
                        'q25': percentiles[0.25],
                        'median': percentiles[0.5],
                        'q75': percentiles[0.75]
                    })

        return summary

    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality check."""

        quality_report = {
            'total_samples': len(df),
            'total_features': len(df.columns),
            'missing_data': {},
            'data_types': {},
            'outliers': {},
            'duplicates': df.duplicated().sum(),
            'class_distribution': {}
        }

        # Missing data analysis
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            quality_report['missing_data'][col] = {
                'count': missing_count,
                'percentage': (missing_count / len(df)) * 100
            }

        # Data types
        for col in df.columns:
            quality_report['data_types'][col] = str(df[col].dtype)

        # Outlier detection for numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            quality_report['outliers'][col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100
            }

        # Class distribution (if target exists)
        if 'target' in df.columns:
            quality_report['class_distribution'] = df['target'].value_counts().to_dict()

        return quality_report

    @staticmethod
    def sanitize_input(data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize and clean input data."""

        sanitized = {}

        for feature, value in data.items():
            if feature in FEATURE_NAMES:
                # Convert to appropriate type
                if feature == 'oldpeak':
                    try:
                        sanitized[feature] = float(value)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert {feature} value {value} to float")
                        sanitized[feature] = 0.0
                else:
                    try:
                        sanitized[feature] = int(value)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert {feature} value {value} to int")
                        sanitized[feature] = 0

                # Clip to valid range
                if feature in FEATURE_RANGES:
                    min_val = FEATURE_RANGES[feature]['min']
                    max_val = FEATURE_RANGES[feature]['max']
                    sanitized[feature] = max(min_val, min(max_val, sanitized[feature]))

        return sanitized


def validate_training_data(train_path: str, val_path: str, test_path: str) -> bool:
    """Validate training data files."""

    try:
        # Load datasets
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        logger.info("Validating training data...")

        # Validate each dataset
        for name, df in [("Training", train_df), ("Validation", val_df), ("Test", test_df)]:
            is_valid, errors = DataValidator.validate_dataframe(df)

            if not is_valid:
                logger.error(f"{name} dataset validation failed: {errors}")
                return False

            logger.info(f"{name} dataset: {df.shape[0]} samples, validation passed")

        # Check class balance
        for name, df in [("Training", train_df), ("Validation", val_df), ("Test", test_df)]:
            if 'target' in df.columns:
                class_dist = df['target'].value_counts(normalize=True)
                logger.info(f"{name} class distribution: {class_dist.to_dict()}")

        logger.info("All datasets validated successfully")
        return True

    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        return False