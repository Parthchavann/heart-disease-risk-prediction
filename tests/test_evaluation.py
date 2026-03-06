"""
Tests for the ModelEvaluator class.
"""

import os
import sys
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.evaluation import ModelEvaluator


@pytest.fixture(scope="module")
def evaluator():
    """Create ModelEvaluator instance."""
    return ModelEvaluator()


@pytest.fixture(scope="module")
def binary_predictions():
    """Create synthetic binary predictions for evaluation tests."""
    np.random.seed(42)
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1,
                       0, 1, 1, 0, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 1,
                       0, 1, 1, 0, 0, 1, 0, 0, 0, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.4, 0.15, 0.85, 0.3, 0.6, 0.75,
                             0.05, 0.95, 0.7, 0.25, 0.2, 0.88, 0.12, 0.45, 0.18, 0.82])
    return y_true, y_pred, y_pred_proba


def test_calculate_basic_metrics(evaluator, binary_predictions):
    """Test that basic metrics are calculated correctly and within [0, 1]."""
    y_true, y_pred, y_pred_proba = binary_predictions
    metrics = evaluator.calculate_basic_metrics(y_true, y_pred, y_pred_proba)

    assert isinstance(metrics, dict)
    expected_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc',
                     'specificity', 'sensitivity', 'npv', 'ppv']
    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"
        assert 0.0 <= metrics[key] <= 1.0, f"Metric {key}={metrics[key]} out of [0,1] range"


def test_calculate_clinical_metrics(evaluator, binary_predictions):
    """Test that clinical metrics include confusion matrix and sensitivity/specificity."""
    y_true, y_pred, y_pred_proba = binary_predictions
    metrics = evaluator.calculate_clinical_metrics(y_true, y_pred, y_pred_proba)

    assert isinstance(metrics, dict)
    assert 'confusion_matrix' in metrics
    assert 'true_positives' in metrics
    assert 'true_negatives' in metrics
    assert 'false_positives' in metrics
    assert 'false_negatives' in metrics
    assert 'sensitivity' in metrics
    assert 'specificity' in metrics
    assert 'ppv' in metrics
    assert 'npv' in metrics

    # Confusion matrix should be 2x2
    cm = metrics['confusion_matrix']
    assert len(cm) == 2
    assert len(cm[0]) == 2

    # All rates should be in [0, 1]
    for key in ['sensitivity', 'specificity', 'ppv', 'npv',
                'false_positive_rate', 'false_negative_rate']:
        assert 0.0 <= metrics[key] <= 1.0


def test_evaluate_at_different_thresholds(evaluator, binary_predictions):
    """Test threshold analysis returns optimal threshold in (0, 1)."""
    y_true, _, y_pred_proba = binary_predictions
    result = evaluator.evaluate_at_different_thresholds(y_true, y_pred_proba)

    assert isinstance(result, dict)
    assert 'threshold_analysis' in result
    assert 'optimal_threshold' in result
    assert 'optimal_metrics' in result

    optimal_threshold = result['optimal_threshold']
    assert 0.0 < optimal_threshold < 1.0

    # Threshold analysis should have entries for multiple thresholds
    assert len(result['threshold_analysis']) > 0


def test_generate_evaluation_report(evaluator, binary_predictions):
    """Test that evaluation report returns non-empty string."""
    y_true, y_pred, y_pred_proba = binary_predictions

    basic_metrics = evaluator.calculate_basic_metrics(y_true, y_pred, y_pred_proba)
    clinical_metrics = evaluator.calculate_clinical_metrics(y_true, y_pred, y_pred_proba)
    threshold_analysis = evaluator.evaluate_at_different_thresholds(y_true, y_pred_proba)

    results = {
        'model_name': 'TestModel',
        'evaluation_timestamp': '2026-01-01T00:00:00',
        'test_samples': len(y_true),
        'basic_metrics': basic_metrics,
        'clinical_metrics': clinical_metrics,
        'threshold_analysis': threshold_analysis
    }

    report = evaluator.generate_evaluation_report(results)

    assert isinstance(report, str)
    assert len(report) > 0
    assert 'TestModel' in report
    assert 'ROC-AUC' in report or 'roc_auc' in report.lower()
