"""
Basic tests for the Heart Disease Risk Prediction system.
"""

import pytest
import os
import sys
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.settings import settings
from utils.constants import FEATURE_NAMES, FEATURE_RANGES
from utils.validators import DataValidator, PatientData
from src.data_processing import DataProcessor


class TestConfiguration:
    """Test configuration and constants."""

    def test_settings_loaded(self):
        """Test that settings are loaded correctly."""
        assert settings.PROJECT_NAME is not None
        assert settings.VERSION is not None
        assert settings.RANDOM_STATE == 42

    def test_feature_constants(self):
        """Test that feature constants are defined."""
        assert len(FEATURE_NAMES) == 13
        assert len(FEATURE_RANGES) >= 13

    def test_feature_ranges_valid(self):
        """Test that feature ranges are valid."""
        for feature, ranges in FEATURE_RANGES.items():
            assert 'min' in ranges
            assert 'max' in ranges
            assert ranges['min'] <= ranges['max']


class TestDataValidation:
    """Test data validation functionality."""

    def test_valid_patient_data(self):
        """Test validation with valid patient data."""
        valid_data = {
            'age': 45,
            'sex': 1,
            'cp': 2,
            'trestbps': 130,
            'chol': 250,
            'fbs': 0,
            'restecg': 0,
            'thalach': 175,
            'exang': 0,
            'oldpeak': 1.2,
            'slope': 1,
            'ca': 1,
            'thal': 2
        }

        is_valid, errors = DataValidator.validate_sample(valid_data)
        assert is_valid
        assert len(errors) == 0

    def test_invalid_patient_data(self):
        """Test validation with invalid patient data."""
        invalid_data = {
            'age': 200,  # Invalid age
            'sex': 1,
            'cp': 2,
            'trestbps': 130,
            'chol': 250,
            'fbs': 0,
            'restecg': 0,
            'thalach': 175,
            'exang': 0,
            'oldpeak': 1.2,
            'slope': 1,
            'ca': 1,
            'thal': 2
        }

        is_valid, errors = DataValidator.validate_sample(invalid_data)
        assert not is_valid
        assert len(errors) > 0

    def test_pydantic_validation(self):
        """Test Pydantic model validation."""
        # Valid data should pass
        valid_data = {
            'age': 45,
            'sex': 1,
            'cp': 2,
            'trestbps': 130,
            'chol': 250,
            'fbs': 0,
            'restecg': 0,
            'thalach': 175,
            'exang': 0,
            'oldpeak': 1.2,
            'slope': 1,
            'ca': 1,
            'thal': 2
        }

        patient = PatientData(**valid_data)
        assert patient.age == 45
        assert patient.sex == 1

        # Invalid data should raise exception
        invalid_data = valid_data.copy()
        invalid_data['age'] = 200

        with pytest.raises(Exception):
            PatientData(**invalid_data)


class TestDataProcessing:
    """Test data processing functionality."""

    def test_data_processor_initialization(self):
        """Test that data processor initializes correctly."""
        processor = DataProcessor()
        assert processor.scaler is not None
        assert processor.label_encoders is not None

    def test_feature_range_validation(self):
        """Test feature range validation."""
        processor = DataProcessor()

        # Create test data within ranges
        valid_data = pd.DataFrame({
            'age': [45, 50, 35],
            'trestbps': [120, 130, 140],
            'chol': [200, 250, 180]
        })

        validated_data = processor._validate_feature_ranges(valid_data)
        assert len(validated_data) == len(valid_data)

        # Create test data outside ranges
        invalid_data = pd.DataFrame({
            'age': [200, 50, 35],  # First age is invalid
            'trestbps': [120, 130, 140],
            'chol': [200, 250, 180]
        })

        validated_data = processor._validate_feature_ranges(invalid_data)
        # Age should be clipped to valid range
        assert validated_data.iloc[0]['age'] <= FEATURE_RANGES['age']['max']


class TestUtilities:
    """Test utility functions."""

    def test_helpers_import(self):
        """Test that helper functions can be imported."""
        from utils.helpers import (
            ensure_directory_exists,
            format_timestamp,
            calculate_age_category,
            format_probability
        )

        assert callable(ensure_directory_exists)
        assert callable(format_timestamp)
        assert callable(calculate_age_category)
        assert callable(format_probability)

    def test_age_categorization(self):
        """Test age categorization function."""
        from utils.helpers import calculate_age_category

        assert calculate_age_category(25) == "young"
        assert calculate_age_category(35) == "middle_aged"
        assert calculate_age_category(55) == "older"
        assert calculate_age_category(70) == "elderly"

    def test_probability_formatting(self):
        """Test probability formatting."""
        from utils.helpers import format_probability

        assert format_probability(0.25) == "25.0%"
        assert format_probability(0.333, 2) == "33.33%"


class TestAPIModels:
    """Test API model schemas."""

    def test_patient_data_model(self):
        """Test PatientData model from API."""
        from api.models import PatientDataRequest

        valid_data = {
            'age': 45,
            'sex': 1,
            'cp': 2,
            'trestbps': 130,
            'chol': 250,
            'fbs': 0,
            'restecg': 0,
            'thalach': 175,
            'exang': 0,
            'oldpeak': 1.2,
            'slope': 1,
            'ca': 1,
            'thal': 2
        }

        patient = PatientDataRequest(**valid_data)
        assert patient.age == 45
        assert patient.oldpeak == 1.2

    def test_prediction_response_model(self):
        """Test PredictionResponse model."""
        from api.models import PredictionResponse

        response_data = {
            'prediction_id': 'test-123',
            'timestamp': '2024-01-01T00:00:00Z',
            'success': True,
            'medical_disclaimer': 'Test disclaimer'
        }

        response = PredictionResponse(**response_data)
        assert response.prediction_id == 'test-123'
        assert response.success is True


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])