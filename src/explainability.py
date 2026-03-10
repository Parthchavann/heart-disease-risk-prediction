"""
Explainability module for Heart Disease Risk Prediction.
Provides SHAP-based explanations for model predictions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional, Union
import joblib
import json

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from config.settings import settings
from config.logging_config import get_logger
from utils.constants import FEATURE_NAMES, RISK_FACTOR_EXPLANATIONS

logger = get_logger(__name__)


class ModelExplainer:
    """Provides model explanations using SHAP and other interpretability methods."""

    def __init__(self):
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for explainability. Please install with: pip install shap")

        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        self.feature_names = FEATURE_NAMES
        self.plots_dir = os.path.join("logs", "explainability_plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    def initialize_explainer(self, model: Any, X_background: np.ndarray) -> None:
        """Initialize SHAP explainer based on model type."""

        logger.info("Initializing SHAP explainer...")

        # Determine model type and create appropriate explainer
        if isinstance(model, (RandomForestClassifier, DecisionTreeClassifier)):
            self.explainer = shap.TreeExplainer(model)
            logger.info("Using TreeExplainer for tree-based model")

        elif isinstance(model, xgb.XGBClassifier):
            self.explainer = shap.TreeExplainer(model)
            logger.info("Using TreeExplainer for XGBoost model")

        elif isinstance(model, LogisticRegression):
            self.explainer = shap.LinearExplainer(model, X_background)
            logger.info("Using LinearExplainer for logistic regression")

        else:
            # Fallback to KernelExplainer (slower but works with any model)
            self.explainer = shap.KernelExplainer(
                model.predict_proba,
                X_background[:100]  # Use subset for efficiency
            )
            logger.info("Using KernelExplainer (fallback)")

        # Calculate expected value
        if hasattr(self.explainer, 'expected_value'):
            if isinstance(self.explainer.expected_value, (list, np.ndarray)):
                ev = self.explainer.expected_value
                self.expected_value = ev[1] if len(ev) > 1 else ev[0]  # For binary classification
            else:
                self.expected_value = self.explainer.expected_value
        else:
            self.expected_value = 0.5  # Default for binary classification

        logger.info("SHAP explainer initialized successfully")

    def explain_predictions(self, X: np.ndarray, max_samples: int = 100) -> np.ndarray:
        """Generate SHAP values for predictions."""

        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer first.")

        logger.info(f"Generating SHAP values for {min(len(X), max_samples)} samples...")

        # Limit samples for efficiency
        X_explain = X[:max_samples] if len(X) > max_samples else X

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_explain)

        # Handle binary classification case
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]  # Use positive class

        self.shap_values = shap_values
        logger.info("SHAP values generated successfully")

        return shap_values

    def get_feature_importance(self, shap_values: np.ndarray = None) -> Dict[str, float]:
        """Get global feature importance from SHAP values."""

        if shap_values is None:
            shap_values = self.shap_values

        if shap_values is None:
            raise ValueError("No SHAP values available. Call explain_predictions first.")

        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        # Create feature importance dictionary
        importance_dict = {}
        for i, feature in enumerate(self.feature_names):
            if i < len(mean_abs_shap):
                importance_dict[feature] = float(mean_abs_shap[i])

        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(),
                                      key=lambda x: x[1], reverse=True))

        return sorted_importance

    def explain_single_prediction(self, X_sample: np.ndarray,
                                 prediction_proba: float) -> Dict[str, Any]:
        """Explain a single prediction."""

        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer first.")

        # Ensure X_sample is 2D
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)

        # Get SHAP values for this sample
        shap_values_single = self.explainer.shap_values(X_sample)

        # Handle list output from binary classifiers (list of 2 arrays)
        if isinstance(shap_values_single, list) and len(shap_values_single) == 2:
            shap_values_single = shap_values_single[1]

        shap_values_single = np.array(shap_values_single)

        # Handle 3D output (n_samples, n_features, n_classes) from KernelExplainer
        if shap_values_single.ndim == 3:
            shap_values_single = shap_values_single[:, :, 1]  # positive class

        # Get SHAP values for the single sample
        if shap_values_single.ndim > 1:
            sample_shap = shap_values_single[0]
        else:
            sample_shap = shap_values_single

        # Create feature contributions
        feature_contributions = {}
        risk_factors = []
        protective_factors = []

        for i, feature in enumerate(self.feature_names):
            if i < len(sample_shap):
                contribution = float(sample_shap[i])
                feature_value = float(X_sample[0][i])

                feature_contributions[feature] = {
                    'shap_value': contribution,
                    'feature_value': feature_value,
                    'impact': 'increases_risk' if contribution > 0 else 'decreases_risk'
                }

                # Categorize as risk or protective factor
                if abs(contribution) > 0.01:  # Threshold for significance
                    factor_info = {
                        'feature': feature,
                        'contribution': contribution,
                        'feature_value': feature_value,
                        'explanation': RISK_FACTOR_EXPLANATIONS.get(feature, f"Feature {feature} affects risk")
                    }

                    if contribution > 0:
                        risk_factors.append(factor_info)
                    else:
                        protective_factors.append(factor_info)

        # Sort by absolute contribution
        risk_factors.sort(key=lambda x: abs(x['contribution']), reverse=True)
        protective_factors.sort(key=lambda x: abs(x['contribution']), reverse=True)

        explanation = {
            'prediction_probability': prediction_proba,
            'expected_value': self.expected_value,
            'feature_contributions': feature_contributions,
            'top_risk_factors': risk_factors[:5],
            'top_protective_factors': protective_factors[:3],
            'shap_values': sample_shap.tolist()
        }

        return explanation

    def plot_summary(self, X: np.ndarray, shap_values: np.ndarray = None,
                    model_name: str = "Model", max_display: int = 10) -> str:
        """Create SHAP summary plot."""

        if shap_values is None:
            shap_values = self.shap_values

        if shap_values is None:
            raise ValueError("No SHAP values available")

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X[:len(shap_values)],
            feature_names=self.feature_names[:shap_values.shape[1]],
            max_display=max_display,
            show=False
        )

        plot_path = os.path.join(self.plots_dir, f'shap_summary_{model_name.lower()}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"SHAP summary plot saved: {plot_path}")
        return plot_path

    def plot_waterfall(self, explanation: Dict[str, Any], sample_index: int = 0,
                      model_name: str = "Model") -> str:
        """Create SHAP waterfall plot for a single prediction."""

        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer first.")

        plt.figure(figsize=(10, 8))

        # Use shap_values from explanation dict if available, else fall back to self.shap_values
        if 'shap_values' in explanation:
            shap_values_for_plot = np.array(explanation['shap_values'])
        elif self.shap_values is not None:
            shap_values_for_plot = self.shap_values[sample_index]
        else:
            raise ValueError("No SHAP values available. Pass explanation with 'shap_values' key.")
        expected_value = self.expected_value

        # Manual waterfall plot since shap.waterfall_plot might not be available in all versions
        feature_names_subset = self.feature_names[:len(shap_values_for_plot)]
        contributions = shap_values_for_plot

        # Sort by absolute contribution
        indices = np.argsort(np.abs(contributions))[::-1][:10]  # Top 10

        y_pos = np.arange(len(indices))
        contributions_sorted = contributions[indices]
        features_sorted = [feature_names_subset[i] for i in indices]

        colors = ['red' if c > 0 else 'blue' for c in contributions_sorted]

        plt.barh(y_pos, contributions_sorted, color=colors, alpha=0.7)
        plt.yticks(y_pos, features_sorted)
        plt.xlabel('SHAP Value (impact on model output)')
        plt.title(f'Feature Contributions - {model_name} (Sample {sample_index})')
        plt.grid(True, alpha=0.3)

        # Add expected value line
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.8)

        plot_path = os.path.join(self.plots_dir, f'shap_waterfall_{model_name.lower()}_{sample_index}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"SHAP waterfall plot saved: {plot_path}")
        return plot_path

    def plot_feature_importance_bar(self, feature_importance: Dict[str, float],
                                   model_name: str = "Model", top_n: int = 10) -> str:
        """Create bar plot of feature importance."""

        # Get top N features
        top_features = dict(list(feature_importance.items())[:top_n])

        plt.figure(figsize=(10, 6))
        features = list(top_features.keys())
        importances = list(top_features.values())

        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Mean |SHAP Value|')
        plt.title(f'Feature Importance - {model_name}')
        plt.grid(True, alpha=0.3)

        plot_path = os.path.join(self.plots_dir, f'shap_importance_{model_name.lower()}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"SHAP feature importance plot saved: {plot_path}")
        return plot_path

    def generate_text_explanation(self, explanation: Dict[str, Any],
                                 patient_context: Dict[str, Any] = None) -> str:
        """Generate human-readable explanation text."""

        risk_prob = explanation['prediction_probability']
        risk_level = "High" if risk_prob > 0.7 else "Moderate" if risk_prob > 0.3 else "Low"

        # Start with risk assessment
        text_explanation = f"**Risk Assessment: {risk_level} Risk ({risk_prob:.1%})**\n\n"

        # Explain top risk factors
        if explanation['top_risk_factors']:
            text_explanation += "**Key Risk Factors:**\n"
            for i, factor in enumerate(explanation['top_risk_factors'][:3], 1):
                feature = factor['feature']
                value = factor['feature_value']
                impact = factor['contribution']
                explanation_text = factor['explanation']

                text_explanation += f"{i}. {explanation_text}\n"
                text_explanation += f"   Your value: {value:.1f}, Impact: {impact:.3f}\n\n"

        # Explain protective factors
        if explanation['top_protective_factors']:
            text_explanation += "**Protective Factors:**\n"
            for i, factor in enumerate(explanation['top_protective_factors'][:2], 1):
                feature = factor['feature']
                value = factor['feature_value']
                impact = abs(factor['contribution'])
                explanation_text = factor['explanation']

                text_explanation += f"{i}. {explanation_text}\n"
                text_explanation += f"   Your value: {value:.1f}, Protective effect: {impact:.3f}\n\n"

        # Add medical disclaimer
        text_explanation += "\n**Important Disclaimer:**\n"
        text_explanation += settings.MEDICAL_DISCLAIMER

        return text_explanation

    def create_explanation_report(self, model: Any, X_sample: np.ndarray,
                                 prediction_proba: float, patient_data: Dict[str, Any] = None,
                                 model_name: str = "Model") -> Dict[str, Any]:
        """Create comprehensive explanation report for a prediction."""

        # Get detailed explanation
        explanation = self.explain_single_prediction(X_sample, prediction_proba)

        # Generate plots
        plots = {}
        try:
            plots['waterfall'] = self.plot_waterfall(explanation, 0, model_name)
        except Exception as e:
            logger.warning(f"Could not create waterfall plot: {e}")

        # Generate text explanation
        text_explanation = self.generate_text_explanation(explanation, patient_data)

        # Create comprehensive report
        report = {
            'model_name': model_name,
            'prediction_probability': prediction_proba,
            'risk_level': "High" if prediction_proba > 0.7 else "Moderate" if prediction_proba > 0.3 else "Low",
            'detailed_explanation': explanation,
            'text_explanation': text_explanation,
            'plots': plots,
            'confidence_interval': self._calculate_confidence_interval(prediction_proba),
            'medical_disclaimer': settings.MEDICAL_DISCLAIMER
        }

        return report

    def _calculate_confidence_interval(self, prediction_proba: float,
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for prediction (simplified)."""

        # This is a simplified confidence interval calculation
        # In practice, you'd want to use bootstrap or other methods
        margin = 0.1  # Simplified margin of error

        lower = max(0.0, prediction_proba - margin)
        upper = min(1.0, prediction_proba + margin)

        return (lower, upper)

    def save_explanation(self, report: Dict[str, Any], sample_id: str = None) -> str:
        """Save explanation report to file."""

        if sample_id is None:
            sample_id = "sample"

        filename = f"explanation_{sample_id}.json"
        filepath = os.path.join("logs", filename)

        # Create logs directory
        os.makedirs("logs", exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Explanation saved: {filepath}")
        return filepath


def main():
    """Main explainability demonstration."""

    from config.logging_config import setup_logging
    from src.model_training import ModelTrainer
    from src.data_processing import DataProcessor

    setup_logging()

    try:
        # Load model and data
        trainer = ModelTrainer()
        model, metadata = trainer.load_best_model()

        # Load test data
        processor = DataProcessor()
        processor.load_transformers()

        test_path = os.path.join(settings.PROCESSED_DATA_DIR, "test.csv")
        test_df = pd.read_csv(test_path)
        X_test = test_df.drop('target', axis=1).values
        y_test = test_df['target'].values

        # Initialize explainer
        explainer = ModelExplainer()
        explainer.feature_names = [c for c in test_df.columns if c != 'target']
        explainer.initialize_explainer(model, X_test[:100])  # Use subset as background

        # Generate explanations for a few samples
        shap_values = explainer.explain_predictions(X_test[:50])

        # Create summary plots
        summary_plot = explainer.plot_summary(X_test[:50], shap_values, metadata['model_name'])
        feature_importance = explainer.get_feature_importance(shap_values)
        importance_plot = explainer.plot_feature_importance_bar(feature_importance, metadata['model_name'])

        # Explain a specific prediction
        sample_idx = 0
        X_sample = X_test[sample_idx:sample_idx+1]
        prediction_proba = model.predict_proba(X_sample)[0, 1]

        explanation_report = explainer.create_explanation_report(
            model, X_sample, prediction_proba, model_name=metadata['model_name']
        )

        # Save explanation
        explainer.save_explanation(explanation_report, f"test_sample_{sample_idx}")

        print("\n" + "="*60)
        print("EXPLAINABILITY ANALYSIS COMPLETED")
        print("="*60)
        print(f"Model: {metadata['model_name']}")
        print(f"Sample analyzed: Test sample {sample_idx}")
        print(f"Predicted risk probability: {prediction_proba:.3f}")
        print(f"Actual outcome: {'Disease' if y_test[sample_idx] == 1 else 'No Disease'}")
        print(f"\nPlots saved in: {explainer.plots_dir}")
        print(f"Explanation saved in: logs/explanation_test_sample_{sample_idx}.json")
        print("="*60)

    except Exception as e:
        logger.error(f"Explainability analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()