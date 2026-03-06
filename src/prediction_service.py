"""
Core prediction service that orchestrates all components.
Coordinates data processing, ML prediction, explainability, and LLM explanations.
"""

import os
import uuid
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime
import json

from config.settings import settings
from config.logging_config import get_logger
from src.model_training import ModelTrainer
from src.data_processing import DataProcessor
from src.explainability import ModelExplainer
from src.llm_layer import LLMExplanationGenerator
from utils.validators import DataValidator
from utils.constants import RiskLevel, FEATURE_NAMES

logger = get_logger(__name__)


class HeartDiseasePredictionService:
    """Main service class that orchestrates all prediction components."""

    def __init__(self, openai_api_key: Optional[str] = None):
        self.model = None
        self.model_metadata = None
        self.data_processor = None
        self.explainer = None
        self.llm_generator = None
        self.background_data = None

        # Initialize all components
        self._initialize_components(openai_api_key)

    def _initialize_components(self, openai_api_key: Optional[str] = None) -> None:
        """Initialize all service components."""

        logger.info("Initializing Heart Disease Prediction Service...")

        try:
            # Load trained model
            trainer = ModelTrainer()
            self.model, self.model_metadata = trainer.load_best_model()
            logger.info(f"Loaded model: {self.model_metadata['model_name']}")

            # Initialize data processor
            self.data_processor = DataProcessor()
            self.data_processor.load_transformers()
            logger.info("Data processor initialized")

            # Load background data for SHAP explainer
            self._load_background_data()

            # Initialize explainer
            self.explainer = ModelExplainer()
            self.explainer.initialize_explainer(self.model, self.background_data)
            logger.info("Explainer initialized")

            # Initialize LLM generator
            self.llm_generator = LLMExplanationGenerator(openai_api_key)
            logger.info("LLM generator initialized")

            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise

    def _load_background_data(self) -> None:
        """Load background data for SHAP explainer."""

        try:
            # Load training data as background
            train_path = os.path.join(settings.PROCESSED_DATA_DIR, "train.csv")
            if os.path.exists(train_path):
                import pandas as pd
                train_df = pd.read_csv(train_path)
                self.background_data = train_df.drop('target', axis=1).values[:100]  # Use subset
                logger.info("Background data loaded for explainer")
            else:
                logger.warning("Background data not found. Using dummy data.")
                self.background_data = np.random.randn(100, len(FEATURE_NAMES))

        except Exception as e:
            logger.error(f"Failed to load background data: {str(e)}")
            # Create dummy background data
            self.background_data = np.random.randn(100, len(FEATURE_NAMES))

    def _determine_risk_level(self, risk_probability: float) -> str:
        """Determine risk level category."""

        if risk_probability >= settings.HIGH_RISK_THRESHOLD:
            return RiskLevel.HIGH
        elif risk_probability >= settings.LOW_RISK_THRESHOLD:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW

    def _calculate_confidence_interval(self, risk_probability: float) -> Tuple[float, float]:
        """Calculate confidence interval for the prediction."""

        # Simplified confidence interval calculation
        # In production, you might want to use bootstrap or ensemble methods
        margin = 0.1  # 10% margin of error

        lower = max(0.0, risk_probability - margin)
        upper = min(1.0, risk_probability + margin)

        return (lower, upper)

    def validate_input(self, patient_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate input patient data."""

        # Use the data validator
        is_valid, errors = DataValidator.validate_sample(patient_data)

        if not is_valid:
            logger.warning(f"Input validation failed: {errors}")

        return is_valid, errors

    def preprocess_input(self, patient_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess patient data for prediction."""

        # Sanitize input first
        sanitized_data = DataValidator.sanitize_input(patient_data)

        # Process through data processor
        processed_data = self.data_processor.process_single_sample(sanitized_data)

        return processed_data

    def make_prediction(self, X: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """Make prediction using the trained model."""

        # Get prediction and probability
        prediction = self.model.predict(X)[0]
        prediction_proba = self.model.predict_proba(X)[0]
        risk_probability = prediction_proba[1]  # Probability of positive class

        logger.info(f"Prediction made: {prediction}, Risk probability: {risk_probability:.3f}")

        return prediction, risk_probability, prediction_proba

    def generate_explanation(self, X: np.ndarray, risk_probability: float,
                           patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SHAP-based explanation for the prediction."""

        try:
            # Generate SHAP explanation
            shap_explanation = self.explainer.explain_single_prediction(X, risk_probability)

            # Create explanation report
            explanation_report = self.explainer.create_explanation_report(
                self.model, X, risk_probability, patient_data, self.model_metadata['model_name']
            )

            return explanation_report

        except Exception as e:
            logger.error(f"Failed to generate explanation: {str(e)}")
            # Return minimal explanation
            return {
                'prediction_probability': risk_probability,
                'risk_level': self._determine_risk_level(risk_probability),
                'detailed_explanation': {'feature_contributions': {}},
                'text_explanation': "Explanation generation failed. Please consult healthcare provider.",
                'medical_disclaimer': settings.MEDICAL_DISCLAIMER
            }

    def generate_llm_explanation(self, prediction_result: Dict[str, Any],
                                shap_explanation: Dict[str, Any],
                                patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate LLM-based patient-friendly explanation."""

        try:
            # Extract relevant information for LLM
            prediction_for_llm = {
                'risk_probability': prediction_result['risk_probability'],
                'risk_level': prediction_result['risk_level']
            }

            # Generate comprehensive LLM explanation
            llm_explanation = self.llm_generator.generate_comprehensive_explanation(
                prediction_for_llm, shap_explanation.get('detailed_explanation', {}), patient_data
            )

            return llm_explanation

        except Exception as e:
            logger.error(f"Failed to generate LLM explanation: {str(e)}")
            # Return fallback explanation
            return {
                'risk_explanation': "Your cardiovascular risk assessment has been completed. Please discuss these results with your healthcare provider.",
                'lifestyle_recommendations': [
                    "Maintain a heart-healthy diet",
                    "Exercise regularly as recommended by your doctor",
                    "Monitor your blood pressure",
                    "Avoid smoking and limit alcohol"
                ],
                'doctor_consultation_questions': [
                    "What does my risk assessment mean?",
                    "What lifestyle changes should I prioritize?",
                    "How often should I be monitored?"
                ],
                'medical_disclaimer': settings.MEDICAL_DISCLAIMER
            }

    def predict(self, patient_data: Dict[str, Any],
               include_explanation: bool = True,
               include_llm_explanation: bool = True) -> Dict[str, Any]:
        """Complete prediction pipeline with explanations."""

        # Generate unique prediction ID
        prediction_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        logger.info(f"Starting prediction {prediction_id}")

        try:
            # Step 1: Validate input
            is_valid, validation_errors = self.validate_input(patient_data)
            if not is_valid:
                return {
                    'prediction_id': prediction_id,
                    'timestamp': timestamp,
                    'success': False,
                    'error': 'Input validation failed',
                    'validation_errors': validation_errors,
                    'medical_disclaimer': settings.MEDICAL_DISCLAIMER
                }

            # Step 2: Preprocess input
            X = self.preprocess_input(patient_data)

            # Step 3: Make prediction
            prediction, risk_probability, prediction_proba = self.make_prediction(X)

            # Step 4: Determine risk level and confidence interval
            risk_level = self._determine_risk_level(risk_probability)
            confidence_interval = self._calculate_confidence_interval(risk_probability)

            # Step 5: Build base result
            result = {
                'prediction_id': prediction_id,
                'timestamp': timestamp,
                'success': True,
                'patient_data': patient_data,
                'prediction': int(prediction),
                'risk_probability': float(risk_probability),
                'risk_level': risk_level,
                'confidence_interval': confidence_interval,
                'model_info': {
                    'model_name': self.model_metadata['model_name'],
                    'model_version': self.model_metadata['model_version'],
                    'training_timestamp': self.model_metadata['training_timestamp']
                },
                'medical_disclaimer': settings.MEDICAL_DISCLAIMER
            }

            # Step 6: Generate explanations if requested
            if include_explanation:
                try:
                    shap_explanation = self.generate_explanation(X, risk_probability, patient_data)
                    result['explanation'] = shap_explanation
                except Exception as e:
                    logger.error(f"SHAP explanation failed: {str(e)}")
                    result['explanation_error'] = str(e)

            if include_llm_explanation and include_explanation and 'explanation' in result:
                try:
                    llm_explanation = self.generate_llm_explanation(
                        result, result['explanation'], patient_data
                    )
                    result['llm_explanation'] = llm_explanation
                except Exception as e:
                    logger.error(f"LLM explanation failed: {str(e)}")
                    result['llm_explanation_error'] = str(e)

            logger.info(f"Prediction {prediction_id} completed successfully")
            return result

        except Exception as e:
            logger.error(f"Prediction {prediction_id} failed: {str(e)}")
            return {
                'prediction_id': prediction_id,
                'timestamp': timestamp,
                'success': False,
                'error': str(e),
                'medical_disclaimer': settings.MEDICAL_DISCLAIMER
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""

        if self.model_metadata:
            return {
                'model_name': self.model_metadata['model_name'],
                'model_version': self.model_metadata['model_version'],
                'training_timestamp': self.model_metadata['training_timestamp'],
                'metrics': self.model_metadata.get('metrics', {}),
                'feature_importance': self.model_metadata.get('feature_importance', {}),
                'training_config': self.model_metadata.get('training_config', {})
            }
        else:
            return {'error': 'No model loaded'}

    def health_check(self) -> Dict[str, Any]:
        """Perform health check of all components."""

        health_status = {
            'timestamp': datetime.now().isoformat(),
            'service_status': 'healthy',
            'components': {}
        }

        # Check model
        try:
            if self.model is not None:
                health_status['components']['model'] = 'healthy'
            else:
                health_status['components']['model'] = 'failed'
                health_status['service_status'] = 'degraded'
        except Exception:
            health_status['components']['model'] = 'failed'
            health_status['service_status'] = 'degraded'

        # Check data processor
        try:
            if self.data_processor is not None and self.data_processor.is_fitted:
                health_status['components']['data_processor'] = 'healthy'
            else:
                health_status['components']['data_processor'] = 'failed'
                health_status['service_status'] = 'degraded'
        except Exception:
            health_status['components']['data_processor'] = 'failed'
            health_status['service_status'] = 'degraded'

        # Check explainer
        try:
            if self.explainer is not None and self.explainer.explainer is not None:
                health_status['components']['explainer'] = 'healthy'
            else:
                health_status['components']['explainer'] = 'degraded'
        except Exception:
            health_status['components']['explainer'] = 'failed'

        # Check LLM generator
        try:
            if self.llm_generator is not None:
                health_status['components']['llm_generator'] = 'healthy'
            else:
                health_status['components']['llm_generator'] = 'failed'
        except Exception:
            health_status['components']['llm_generator'] = 'failed'

        return health_status


def main():
    """Demonstration of the complete prediction service."""

    from config.logging_config import setup_logging
    setup_logging()

    try:
        # Initialize service
        service = HeartDiseasePredictionService()

        # Example patient data
        patient_data = {
            'age': 45,
            'sex': 1,  # Male
            'cp': 2,  # Atypical angina
            'trestbps': 130,  # Resting BP
            'chol': 250,  # Cholesterol
            'fbs': 0,  # Fasting blood sugar
            'restecg': 0,  # Resting ECG
            'thalach': 175,  # Max heart rate
            'exang': 0,  # Exercise angina
            'oldpeak': 1.2,  # ST depression
            'slope': 1,  # ST slope
            'ca': 1,  # Major vessels
            'thal': 2  # Thalassemia
        }

        # Make prediction
        result = service.predict(
            patient_data,
            include_explanation=True,
            include_llm_explanation=True
        )

        # Display results
        print("\n" + "="*80)
        print("HEART DISEASE PREDICTION SERVICE DEMO")
        print("="*80)

        if result['success']:
            print(f"Prediction ID: {result['prediction_id']}")
            print(f"Risk Probability: {result['risk_probability']:.3f}")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Confidence Interval: {result['confidence_interval']}")

            if 'llm_explanation' in result:
                print(f"\nPatient-Friendly Explanation:")
                print(result['llm_explanation']['risk_explanation'])

                print(f"\nRecommendations:")
                for i, rec in enumerate(result['llm_explanation']['lifestyle_recommendations'], 1):
                    print(f"{i}. {rec}")

        else:
            print(f"Prediction failed: {result.get('error', 'Unknown error')}")

        print("\n" + "="*80)

        # Save result
        result_file = os.path.join("logs", f"prediction_demo_{result['prediction_id']}.json")
        os.makedirs("logs", exist_ok=True)
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"Full result saved to: {result_file}")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()