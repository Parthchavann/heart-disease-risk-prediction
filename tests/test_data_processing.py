"""
Tests for the DataProcessor class.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_processing import DataProcessor
from utils.constants import FEATURE_NAMES


@pytest.fixture(scope="module")
def processor():
    """Create a DataProcessor instance for tests."""
    return DataProcessor()


@pytest.fixture(scope="module")
def raw_df(processor):
    """Load raw data once for all tests."""
    return processor.load_raw_data()


@pytest.fixture(scope="module")
def clean_df(processor, raw_df):
    """Clean data once for all tests."""
    p = DataProcessor()
    return p.clean_data(raw_df)


@pytest.fixture(scope="module")
def engineered_df(processor, raw_df):
    """Engineer features once for all tests."""
    p = DataProcessor()
    clean = p.clean_data(raw_df)
    return p.engineer_features(clean)


def test_load_raw_data(processor):
    """Test that raw data loads successfully with expected shape."""
    df = processor.load_raw_data()
    assert isinstance(df, pd.DataFrame)
    # Dataset has 13 features + 1 target
    assert df.shape[1] == 14
    assert df.shape[0] > 0
    assert 'target' in df.columns
    for feature in FEATURE_NAMES:
        assert feature in df.columns


def test_clean_data(raw_df):
    """Test data cleaning: missing values, outlier removal, binary target."""
    p = DataProcessor()
    df_clean = p.clean_data(raw_df)

    assert isinstance(df_clean, pd.DataFrame)
    # Target should be binary
    assert set(df_clean['target'].unique()).issubset({0, 1})
    # No missing values should remain
    assert df_clean.isnull().sum().sum() == 0
    # Should have at least some rows
    assert df_clean.shape[0] > 50


def test_engineer_features(raw_df):
    """Test that feature engineering creates expected new features."""
    p = DataProcessor()
    df_clean = p.clean_data(raw_df)
    df_engineered = p.engineer_features(df_clean)

    # Check new features exist
    assert 'age_group' in df_engineered.columns
    assert 'bp_chol_ratio' in df_engineered.columns
    assert 'hr_reserve' in df_engineered.columns
    assert 'multiple_risk_factors' in df_engineered.columns

    # Check ratios are positive
    assert (df_engineered['bp_chol_ratio'] > 0).all()
    assert (df_engineered['hr_reserve'] > 0).all()
    assert (df_engineered['multiple_risk_factors'] >= 0).all()


def test_preprocess_features(raw_df):
    """Test that feature preprocessing with fit_transformers=True scales correctly."""
    p = DataProcessor()
    df_clean = p.clean_data(raw_df)
    df_engineered = p.engineer_features(df_clean)
    df_processed = p.preprocess_features(df_engineered, fit_transformers=True)

    assert isinstance(df_processed, pd.DataFrame)
    assert p.is_fitted is True
    # Target should remain in processed df
    assert 'target' in df_processed.columns
    # No NaN values in result
    assert df_processed.isnull().sum().sum() == 0


def test_split_data(raw_df):
    """Test data splitting returns 3 DataFrames with correct proportions."""
    p = DataProcessor()
    df_clean = p.clean_data(raw_df)
    df_engineered = p.engineer_features(df_clean)
    df_processed = p.preprocess_features(df_engineered, fit_transformers=True)
    train_df, val_df, test_df = p.split_data(df_processed)

    total = len(df_processed)

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    # Each split should be non-empty
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    # Together they should cover all rows
    assert len(train_df) + len(val_df) + len(test_df) == total

    # Train should be the largest split (~70%)
    assert len(train_df) > len(val_df)
    assert len(train_df) > len(test_df)

    # All splits must have target column
    assert 'target' in train_df.columns
    assert 'target' in val_df.columns
    assert 'target' in test_df.columns


def test_process_single_sample():
    """Test that a single sample can be processed and returns correct shape."""
    p = DataProcessor()
    # Load transformers from saved processed data
    try:
        p.load_transformers()
    except Exception:
        # If transformers not saved yet, fit on raw data first
        raw_df = p.load_raw_data()
        clean = p.clean_data(raw_df)
        engineered = p.engineer_features(clean)
        p.preprocess_features(engineered, fit_transformers=True)

    sample = {
        'age': 55, 'sex': 1, 'cp': 2, 'trestbps': 130, 'chol': 240,
        'fbs': 0, 'restecg': 0, 'thalach': 170, 'exang': 0,
        'oldpeak': 1.5, 'slope': 1, 'ca': 0, 'thal': 2
    }

    result = p.process_single_sample(sample)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape[0] == 1
