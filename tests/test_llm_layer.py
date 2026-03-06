"""
Tests for the LLMExplanationGenerator class.
"""

import os
import sys
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.llm_layer import LLMExplanationGenerator


SAMPLE_RISK_FACTORS = [
    {'feature': 'age', 'contribution': 0.15, 'feature_value': 60},
    {'feature': 'chol', 'contribution': 0.12, 'feature_value': 280},
    {'feature': 'trestbps', 'contribution': 0.08, 'feature_value': 145},
]

SAMPLE_PROTECTIVE_FACTORS = [
    {'feature': 'thalach', 'contribution': -0.06, 'feature_value': 175},
]


@pytest.fixture(scope="module")
def llm_no_key():
    """Create LLM generator without API key (fallback mode)."""
    return LLMExplanationGenerator(api_key=None)


def test_init_without_api_key(llm_no_key):
    """Test that generator initializes without exception when no API key provided."""
    from config.settings import settings
    # Ollama provider does not require an API key, so client may be set if server is running.
    # For key-based providers (openai/gemini) without a key, client should be None.
    if settings.LLM_PROVIDER.lower() == "ollama":
        assert llm_no_key is not None
    else:
        assert llm_no_key.client is None


def test_generate_risk_explanation_fallback(llm_no_key):
    """Test that risk explanation returns non-empty string in fallback mode."""
    result = llm_no_key.generate_risk_explanation(
        risk_probability=0.65,
        risk_factors=SAMPLE_RISK_FACTORS,
        protective_factors=SAMPLE_PROTECTIVE_FACTORS,
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_lifestyle_recommendations_fallback(llm_no_key):
    """Test that lifestyle recommendations return a list of strings in fallback mode."""
    result = llm_no_key.generate_lifestyle_recommendations(
        risk_probability=0.65,
        risk_factors=SAMPLE_RISK_FACTORS,
    )
    assert isinstance(result, list)
    assert len(result) > 0
    for rec in result:
        assert isinstance(rec, str)
        assert len(rec) > 0


def test_generate_doctor_questions_fallback(llm_no_key):
    """Test that doctor questions return a list of strings in fallback mode."""
    result = llm_no_key.generate_doctor_questions(
        risk_probability=0.65,
        risk_factors=SAMPLE_RISK_FACTORS,
    )
    assert isinstance(result, list)
    assert len(result) > 0
    for q in result:
        assert isinstance(q, str)
        assert len(q) > 0


def test_generate_comprehensive_explanation_structure(llm_no_key):
    """Test that comprehensive explanation returns dict with all required keys."""
    prediction_result = {'risk_probability': 0.65, 'risk_level': 'Moderate'}
    shap_explanation = {
        'top_risk_factors': SAMPLE_RISK_FACTORS,
        'top_protective_factors': SAMPLE_PROTECTIVE_FACTORS,
    }

    result = llm_no_key.generate_comprehensive_explanation(
        prediction_result=prediction_result,
        shap_explanation=shap_explanation,
    )

    assert isinstance(result, dict)
    required_keys = [
        'risk_explanation',
        'lifestyle_recommendations',
        'doctor_consultation_questions',
        'generated_timestamp',
        'risk_level',
        'medical_disclaimer'
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"

    assert isinstance(result['lifestyle_recommendations'], list)
    assert isinstance(result['doctor_consultation_questions'], list)
    assert len(result['risk_explanation']) > 0
    assert len(result['medical_disclaimer']) > 0
