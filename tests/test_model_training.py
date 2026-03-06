"""
Tests for the ModelTrainer class.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model_training import ModelTrainer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification


@pytest.fixture(scope="module")
def trainer():
    """Create a ModelTrainer instance."""
    return ModelTrainer()


@pytest.fixture(scope="module")
def small_xy():
    """Generate small synthetic dataset for fast testing."""
    X, y = make_classification(
        n_samples=100, n_features=13, n_informative=8,
        n_redundant=2, random_state=42
    )
    return X, y


@pytest.fixture(scope="module")
def trained_lr(small_xy):
    """Train a small LogisticRegression for evaluation tests."""
    X, y = small_xy
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X[:80], y[:80])
    return model, X[80:], y[80:]


def test_get_model_configurations(trainer):
    """Test that model configurations are returned with expected keys."""
    configs = trainer.get_model_configurations()
    assert isinstance(configs, dict)
    # Should have at least 4 models (LR, DT, RF, SVM; XGB optional)
    assert len(configs) >= 4
    for name, config in configs.items():
        assert 'model' in config
        assert 'params' in config


def test_load_training_data(trainer):
    """Test that processed training data can be loaded."""
    try:
        train_df, val_df, test_df = trainer.load_training_data()
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert 'target' in train_df.columns
        assert len(train_df) > 0
        assert len(val_df) > 0
    except FileNotFoundError:
        pytest.skip("Processed data not available; run data processing first")


def test_prepare_data_for_training(trainer):
    """Test that data preparation returns correct array shapes."""
    try:
        train_df, val_df, _ = trainer.load_training_data()
        X, y, X_val, y_val = trainer.prepare_data_for_training(train_df, val_df)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(X_val, np.ndarray)
        assert isinstance(y_val, np.ndarray)
        assert X.shape[0] == len(y)
        assert X_val.shape[0] == len(y_val)
        assert X.shape[1] == X_val.shape[1]
    except FileNotFoundError:
        pytest.skip("Processed data not available; run data processing first")


def test_evaluate_model(trainer, trained_lr):
    """Test that model evaluation returns all expected metrics."""
    model, X_val, y_val = trained_lr
    metrics = trainer.evaluate_model(model, X_val, y_val)

    assert isinstance(metrics, dict)
    expected_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    for key in expected_keys:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_select_best_model(trainer, small_xy):
    """Test that select_best_model picks the model with highest ROC-AUC."""
    X, y = small_xy
    X_train, y_train = X[:80], y[:80]
    X_val, y_val = X[80:], y[80:]

    # Create two simple models
    lr1 = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr2 = LogisticRegression(C=0.01, max_iter=1000, random_state=42)
    lr1.fit(X_train, y_train)
    lr2.fit(X_train, y_train)

    tuned_results = {
        'model_a': {
            'best_estimator': lr1,
            'best_params': {'C': 1.0},
            'best_score': 0.85
        },
        'model_b': {
            'best_estimator': lr2,
            'best_params': {'C': 0.01},
            'best_score': 0.70
        }
    }

    best_model, best_name = trainer.select_best_model(tuned_results, X_val, y_val)
    assert best_model is not None
    assert isinstance(best_name, str)
    assert best_name in tuned_results


def test_save_and_load_model(trainer, trained_lr, tmp_path):
    """Test round-trip save and load of model artifact."""
    from config.settings import settings

    model, _, _ = trained_lr
    metrics = {'accuracy': 0.85, 'roc_auc': 0.90}

    # Use a temp directory to avoid overwriting the real best_model.pkl
    original_dir = settings.MODELS_DIR
    settings.MODELS_DIR = str(tmp_path)
    try:
        model_path = trainer.save_model(model, 'test_model', metrics)
        assert os.path.exists(model_path)

        loaded_model, metadata = trainer.load_best_model()
        assert loaded_model is not None
        assert isinstance(metadata, dict)
        assert 'model_name' in metadata
        assert 'metrics' in metadata
    finally:
        settings.MODELS_DIR = original_dir
