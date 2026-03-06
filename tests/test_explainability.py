"""
Tests for the ModelExplainer class.
"""

import os
import sys
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

pytestmark = pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")


@pytest.fixture(scope="module")
def lr_model_and_data():
    """Create a fitted LogisticRegression with synthetic data."""
    X, y = make_classification(
        n_samples=120, n_features=13, n_informative=8,
        n_redundant=2, random_state=42
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X[:100], y[:100])
    return model, X, y


@pytest.fixture(scope="module")
def explainer_fitted(lr_model_and_data):
    """Return an initialized ModelExplainer."""
    from src.explainability import ModelExplainer
    model, X, _ = lr_model_and_data
    exp = ModelExplainer()
    exp.initialize_explainer(model, X[:100])
    return exp, X


def test_initialize_explainer(lr_model_and_data):
    """Test that explainer is set after initialization."""
    from src.explainability import ModelExplainer
    model, X, _ = lr_model_and_data
    exp = ModelExplainer()
    assert exp.explainer is None
    exp.initialize_explainer(model, X[:50])
    assert exp.explainer is not None


def test_explain_predictions(explainer_fitted, lr_model_and_data):
    """Test that explain_predictions returns array with correct shape."""
    exp, X = explainer_fitted
    _, _, y = lr_model_and_data
    shap_values = exp.explain_predictions(X[:20])

    assert isinstance(shap_values, np.ndarray)
    # Should have rows equal to number of samples, columns equal to features
    assert shap_values.shape[0] == 20
    assert shap_values.shape[1] == X.shape[1]


def test_get_feature_importance(explainer_fitted, lr_model_and_data):
    """Test that feature importance returns a sorted dict with feature names."""
    exp, X = explainer_fitted
    shap_values = exp.explain_predictions(X[:20])
    importance = exp.get_feature_importance(shap_values)

    assert isinstance(importance, dict)
    assert len(importance) > 0

    # Values should be non-negative (mean absolute SHAP)
    for name, val in importance.items():
        assert val >= 0.0

    # Should be sorted in descending order
    values = list(importance.values())
    assert values == sorted(values, reverse=True)


def test_explain_single_prediction(explainer_fitted, lr_model_and_data):
    """Test that single prediction explanation returns required keys."""
    exp, X = explainer_fitted
    model, _, _ = lr_model_and_data
    X_sample = X[0:1]
    pred_proba = float(model.predict_proba(X_sample)[0, 1])

    result = exp.explain_single_prediction(X_sample, pred_proba)

    assert isinstance(result, dict)
    assert 'top_risk_factors' in result
    assert 'top_protective_factors' in result
    assert 'prediction_probability' in result
    assert 'feature_contributions' in result
    assert isinstance(result['top_risk_factors'], list)
    assert isinstance(result['top_protective_factors'], list)


def test_generate_text_explanation(explainer_fitted, lr_model_and_data):
    """Test that text explanation returns non-empty string."""
    exp, X = explainer_fitted
    model, _, _ = lr_model_and_data
    X_sample = X[0:1]
    pred_proba = float(model.predict_proba(X_sample)[0, 1])

    explanation = exp.explain_single_prediction(X_sample, pred_proba)
    text = exp.generate_text_explanation(explanation)

    assert isinstance(text, str)
    assert len(text) > 0
    assert 'Risk' in text or 'risk' in text
