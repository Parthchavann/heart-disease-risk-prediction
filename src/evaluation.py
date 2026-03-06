"""
Model evaluation module for Heart Disease Risk Prediction.
Provides comprehensive evaluation metrics and visualizations.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List, Optional
import joblib
from datetime import datetime

# Sklearn metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

from config.settings import settings
from config.logging_config import get_logger
from utils.constants import FEATURE_NAMES

logger = get_logger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ModelEvaluator:
    """Comprehensive model evaluation and performance analysis."""

    def __init__(self):
        self.evaluation_results = {}
        self.plots_dir = os.path.join("logs", "evaluation_plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    def load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load test data for evaluation."""

        test_path = os.path.join(settings.PROCESSED_DATA_DIR, "test.csv")

        if not os.path.exists(test_path):
            raise FileNotFoundError("Test data not found. Please run data processing first.")

        test_df = pd.read_csv(test_path)
        self.feature_names = [c for c in test_df.columns if c != 'target']
        X_test = test_df.drop('target', axis=1).values
        y_test = test_df['target'].values

        logger.info(f"Loaded test data: {X_test.shape[0]} samples")

        return X_test, y_test

    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics."""

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'specificity': self._calculate_specificity(y_true, y_pred),
            'sensitivity': recall_score(y_true, y_pred),  # Same as recall
            'npv': self._calculate_npv(y_true, y_pred),
            'ppv': precision_score(y_true, y_pred)  # Same as precision
        }

        return metrics

    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    def _calculate_npv(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate negative predictive value."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fn) if (tn + fn) > 0 else 0.0

    def calculate_clinical_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate clinical performance metrics."""

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Calculate rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity/Recall
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision/PPV
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # NPV

        # False rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False positive rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False negative rate

        # Clinical metrics
        clinical_metrics = {
            'confusion_matrix': cm.tolist(),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'sensitivity': tpr,  # True positive rate
            'specificity': tnr,  # True negative rate
            'ppv': ppv,  # Positive predictive value
            'npv': npv,  # Negative predictive value
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'prevalence': np.mean(y_true),
            'balanced_accuracy': (tpr + tnr) / 2
        }

        return clinical_metrics

    def evaluate_at_different_thresholds(self, y_true: np.ndarray,
                                       y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance at different probability thresholds."""

        thresholds = np.arange(0.1, 1.0, 0.1)
        threshold_results = []

        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)

            metrics = self.calculate_basic_metrics(y_true, y_pred_thresh, y_pred_proba)
            metrics['threshold'] = threshold

            threshold_results.append(metrics)

        # Find optimal threshold (maximize F1-score)
        f1_scores = [result['f1_score'] for result in threshold_results]
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = threshold_results[optimal_idx]['threshold']

        return {
            'threshold_analysis': threshold_results,
            'optimal_threshold': optimal_threshold,
            'optimal_metrics': threshold_results[optimal_idx]
        }

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            model_name: str = "Model") -> str:
        """Plot confusion matrix."""

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'])

        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        plot_path = os.path.join(self.plots_dir, f'confusion_matrix_{model_name.lower()}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Confusion matrix plot saved: {plot_path}")
        return plot_path

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      model_name: str = "Model") -> str:
        """Plot ROC curve."""

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'{model_name} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")

        plot_path = os.path.join(self.plots_dir, f'roc_curve_{model_name.lower()}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"ROC curve plot saved: {plot_path}")
        return plot_path

    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   model_name: str = "Model") -> str:
        """Plot Precision-Recall curve."""

        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkblue', lw=2,
                label=f'{model_name} (AUC = {pr_auc:.3f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")

        plot_path = os.path.join(self.plots_dir, f'pr_curve_{model_name.lower()}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Precision-Recall curve plot saved: {plot_path}")
        return plot_path

    def plot_feature_importance(self, model: Any, feature_names: List[str] = None,
                              model_name: str = "Model", top_n: int = 10) -> str:
        """Plot feature importance."""

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            logger.warning("Model does not have feature importance information")
            return ""

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(10, 6))
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.ylabel('Importance')

        plot_path = os.path.join(self.plots_dir, f'feature_importance_{model_name.lower()}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Feature importance plot saved: {plot_path}")
        return plot_path

    def plot_threshold_analysis(self, threshold_results: List[Dict[str, float]],
                              model_name: str = "Model") -> str:
        """Plot threshold analysis results."""

        thresholds = [r['threshold'] for r in threshold_results]
        precisions = [r['precision'] for r in threshold_results]
        recalls = [r['recall'] for r in threshold_results]
        f1_scores = [r['f1_score'] for r in threshold_results]

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, 'o-', label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, 's-', label='Recall', linewidth=2)
        plt.plot(thresholds, f1_scores, '^-', label='F1-Score', linewidth=2)

        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(f'Metrics vs Threshold - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plot_path = os.path.join(self.plots_dir, f'threshold_analysis_{model_name.lower()}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Threshold analysis plot saved: {plot_path}")
        return plot_path

    def comprehensive_evaluation(self, model: Any, model_name: str = "Model") -> Dict[str, Any]:
        """Perform comprehensive model evaluation."""

        logger.info(f"Starting comprehensive evaluation for {model_name}...")

        # Load test data
        X_test, y_test = self.load_test_data()

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        basic_metrics = self.calculate_basic_metrics(y_test, y_pred, y_pred_proba)
        clinical_metrics = self.calculate_clinical_metrics(y_test, y_pred, y_pred_proba)
        threshold_analysis = self.evaluate_at_different_thresholds(y_test, y_pred_proba)

        # Generate classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)

        # Create visualizations
        plots = {
            'confusion_matrix': self.plot_confusion_matrix(y_test, y_pred, model_name),
            'roc_curve': self.plot_roc_curve(y_test, y_pred_proba, model_name),
            'pr_curve': self.plot_precision_recall_curve(y_test, y_pred_proba, model_name),
            'feature_importance': self.plot_feature_importance(model, getattr(self, 'feature_names', FEATURE_NAMES), model_name),
            'threshold_analysis': self.plot_threshold_analysis(
                threshold_analysis['threshold_analysis'], model_name
            )
        }

        # Combine all results
        evaluation_results = {
            'model_name': model_name,
            'evaluation_timestamp': datetime.now().isoformat(),
            'test_samples': len(y_test),
            'basic_metrics': basic_metrics,
            'clinical_metrics': clinical_metrics,
            'threshold_analysis': threshold_analysis,
            'classification_report': class_report,
            'plots': plots
        }

        self.evaluation_results[model_name] = evaluation_results

        logger.info(f"Comprehensive evaluation completed for {model_name}")
        return evaluation_results

    def cross_validation_evaluation(self, model: Any, X: np.ndarray, y: np.ndarray,
                                   model_name: str = "Model") -> Dict[str, Any]:
        """Perform cross-validation evaluation."""

        logger.info(f"Performing cross-validation for {model_name}...")

        cv = StratifiedKFold(n_splits=settings.CV_FOLDS, shuffle=True, random_state=settings.RANDOM_STATE)

        # Multiple scoring metrics
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_results = {}

        for metric in scoring_metrics:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
            cv_results[metric] = {
                'scores': scores.tolist(),
                'mean': scores.mean(),
                'std': scores.std(),
                'confidence_interval': [
                    scores.mean() - 1.96 * scores.std(),
                    scores.mean() + 1.96 * scores.std()
                ]
            }

        logger.info(f"Cross-validation completed for {model_name}")
        return cv_results

    def save_evaluation_results(self, results: Dict[str, Any], model_name: str) -> str:
        """Save evaluation results to file."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{model_name}_{timestamp}.json"
        filepath = os.path.join("logs", filename)

        # Create logs directory
        os.makedirs("logs", exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Evaluation results saved: {filepath}")
        return filepath

    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable evaluation report."""

        model_name = results['model_name']
        basic_metrics = results['basic_metrics']
        clinical_metrics = results['clinical_metrics']

        report = f"""
HEART DISEASE RISK PREDICTION MODEL EVALUATION REPORT
=====================================================

Model: {model_name}
Evaluation Date: {results['evaluation_timestamp']}
Test Samples: {results['test_samples']}

PERFORMANCE METRICS
-------------------
Accuracy: {basic_metrics['accuracy']:.4f}
Precision (PPV): {basic_metrics['precision']:.4f}
Recall (Sensitivity): {basic_metrics['recall']:.4f}
Specificity: {basic_metrics['specificity']:.4f}
F1-Score: {basic_metrics['f1_score']:.4f}
ROC-AUC: {basic_metrics['roc_auc']:.4f}
NPV: {basic_metrics['npv']:.4f}

CLINICAL PERFORMANCE
--------------------
True Positives: {clinical_metrics['true_positives']}
True Negatives: {clinical_metrics['true_negatives']}
False Positives: {clinical_metrics['false_positives']}
False Negatives: {clinical_metrics['false_negatives']}

Balanced Accuracy: {clinical_metrics['balanced_accuracy']:.4f}
False Positive Rate: {clinical_metrics['false_positive_rate']:.4f}
False Negative Rate: {clinical_metrics['false_negative_rate']:.4f}

OPTIMAL THRESHOLD ANALYSIS
--------------------------
Optimal Threshold: {results['threshold_analysis']['optimal_threshold']:.2f}
F1-Score at Optimal: {results['threshold_analysis']['optimal_metrics']['f1_score']:.4f}
Precision at Optimal: {results['threshold_analysis']['optimal_metrics']['precision']:.4f}
Recall at Optimal: {results['threshold_analysis']['optimal_metrics']['recall']:.4f}

CLINICAL INTERPRETATION
-----------------------
- The model correctly identifies {basic_metrics['recall']:.1%} of patients with heart disease (Sensitivity)
- The model correctly identifies {basic_metrics['specificity']:.1%} of patients without heart disease (Specificity)
- When the model predicts heart disease, it's correct {basic_metrics['precision']:.1%} of the time (PPV)
- When the model predicts no heart disease, it's correct {basic_metrics['npv']:.1%} of the time (NPV)

RECOMMENDATIONS
---------------
- Model performance: {'Excellent' if basic_metrics['roc_auc'] > 0.9 else 'Good' if basic_metrics['roc_auc'] > 0.8 else 'Fair' if basic_metrics['roc_auc'] > 0.7 else 'Needs Improvement'}
- Recommended for: Educational and awareness purposes only
- Clinical validation: Required before medical use
- Consider retraining: {'No' if basic_metrics['roc_auc'] > 0.8 else 'Yes'}
        """

        return report


def main():
    """Main evaluation script."""

    from config.logging_config import setup_logging
    from src.model_training import ModelTrainer

    setup_logging()

    try:
        # Load best model
        trainer = ModelTrainer()
        model, metadata = trainer.load_best_model()

        # Initialize evaluator
        evaluator = ModelEvaluator()

        # Comprehensive evaluation
        results = evaluator.comprehensive_evaluation(model, metadata['model_name'])

        # Save results
        results_file = evaluator.save_evaluation_results(results, metadata['model_name'])

        # Generate and save report
        report = evaluator.generate_evaluation_report(results)
        report_file = os.path.join("logs", f"evaluation_report_{metadata['model_name']}.txt")

        with open(report_file, 'w') as f:
            f.write(report)

        print(report)
        print(f"\nDetailed results saved to: {results_file}")
        print(f"Report saved to: {report_file}")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()