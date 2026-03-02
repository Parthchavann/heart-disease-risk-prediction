"""
LLM integration layer for Heart Disease Risk Prediction.
Converts technical ML outputs into patient-friendly explanations and recommendations.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

from config.settings import settings
from config.logging_config import get_logger
from utils.constants import RiskLevel, RISK_FACTOR_EXPLANATIONS

logger = get_logger(__name__)


class LLMExplanationGenerator:
    """Generates patient-friendly explanations using LLM."""

    def __init__(self, api_key: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI not available. LLM features will be limited.")
            self.client = None
            return

        # Initialize OpenAI client
        api_key = api_key or settings.OPENAI_API_KEY
        if api_key:
            openai.api_key = api_key
            self.client = openai
            logger.info("OpenAI client initialized")
        else:
            logger.warning("No OpenAI API key provided. LLM features disabled.")
            self.client = None

    def _make_llm_request(self, prompt: str, max_tokens: int = None,
                         temperature: float = None) -> str:
        """Make a request to the LLM."""

        if not self.client:
            return self._generate_fallback_explanation(prompt)

        try:
            max_tokens = max_tokens or settings.LLM_MAX_TOKENS
            temperature = temperature or settings.LLM_TEMPERATURE

            response = self.client.ChatCompletion.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM request failed: {str(e)}")
            return self._generate_fallback_explanation(prompt)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""

        return """You are a medical AI assistant specializing in cardiovascular health education.
Your role is to explain heart disease risk predictions in clear, patient-friendly language.

Guidelines:
- Use simple, non-technical language
- Be encouraging and supportive
- Always emphasize that this is educational, not diagnostic
- Focus on actionable lifestyle recommendations
- Be empathetic and understanding
- Avoid causing anxiety while being honest about risks
- Always recommend consulting healthcare providers
- Keep explanations concise but thorough

Remember: You are providing educational information only, not medical diagnosis or treatment advice."""

    def _generate_fallback_explanation(self, prompt: str) -> str:
        """Generate a fallback explanation when LLM is not available."""

        # Simple template-based explanation
        if "risk probability" in prompt.lower():
            return ("Based on your health indicators, our analysis suggests a certain level of "
                   "cardiovascular risk. Please consult with your healthcare provider to discuss "
                   "these findings and develop an appropriate health plan.")

        return ("Your health assessment has been completed. Please review the detailed results "
               "and consult with a healthcare professional for personalized medical advice.")

    def generate_risk_explanation(self, risk_probability: float,
                                 risk_factors: List[Dict[str, Any]],
                                 protective_factors: List[Dict[str, Any]],
                                 patient_context: Dict[str, Any] = None) -> str:
        """Generate patient-friendly risk explanation."""

        risk_level = self._determine_risk_level(risk_probability)

        # Prepare risk factors text
        risk_factors_text = ""
        if risk_factors:
            risk_factors_text = "Key factors contributing to your risk include:\n"
            for factor in risk_factors[:3]:  # Top 3
                factor_name = factor.get('feature', 'Unknown factor')
                explanation = RISK_FACTOR_EXPLANATIONS.get(factor_name,
                                                         f"The {factor_name} measurement")
                risk_factors_text += f"- {explanation}\n"

        # Prepare protective factors text
        protective_factors_text = ""
        if protective_factors:
            protective_factors_text = "Factors working in your favor include:\n"
            for factor in protective_factors[:2]:  # Top 2
                factor_name = factor.get('feature', 'Unknown factor')
                explanation = RISK_FACTOR_EXPLANATIONS.get(factor_name,
                                                         f"The {factor_name} measurement")
                protective_factors_text += f"- {explanation} appears to be protective\n"

        prompt = f"""
Please explain the following heart disease risk assessment results in patient-friendly language:

Risk Level: {risk_level}
Risk Probability: {risk_probability:.1%}

{risk_factors_text}

{protective_factors_text}

Please provide:
1. A clear, reassuring explanation of what this risk level means
2. Context about how this assessment works
3. Emphasis on the educational nature of this tool

Keep the explanation warm, supportive, and educational.
"""

        return self._make_llm_request(prompt)

    def generate_lifestyle_recommendations(self, risk_probability: float,
                                         risk_factors: List[Dict[str, Any]],
                                         patient_context: Dict[str, Any] = None) -> List[str]:
        """Generate personalized lifestyle recommendations."""

        risk_level = self._determine_risk_level(risk_probability)

        # Identify key modifiable risk factors
        modifiable_factors = []
        for factor in risk_factors:
            feature = factor.get('feature', '')
            if feature in ['chol', 'trestbps', 'fbs', 'thalach', 'oldpeak']:
                modifiable_factors.append(feature)

        prompt = f"""
Based on a {risk_level} cardiovascular risk assessment, please provide 5-7 specific,
actionable lifestyle recommendations. The key risk factors identified include: {', '.join(modifiable_factors[:3])}.

Please provide recommendations that are:
- Specific and actionable
- Evidence-based
- Suitable for general health improvement
- Encouraging and positive
- Appropriate for someone with {risk_level.lower()} cardiovascular risk

Focus on diet, exercise, stress management, and general wellness.
Format as a simple list.
"""

        response = self._make_llm_request(prompt)

        # Parse response into list
        recommendations = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or
                        line.startswith('1.') or line.startswith('2.') or
                        any(line.startswith(f'{i}.') for i in range(1, 10))):
                # Clean up formatting
                clean_line = line.lstrip('-•').strip()
                if clean_line and len(clean_line) > 10:  # Filter out very short items
                    recommendations.append(clean_line)

        # Fallback recommendations if parsing failed
        if not recommendations:
            recommendations = self._get_fallback_recommendations(risk_level)

        return recommendations[:7]  # Limit to 7 recommendations

    def generate_doctor_questions(self, risk_probability: float,
                                 risk_factors: List[Dict[str, Any]],
                                 patient_context: Dict[str, Any] = None) -> List[str]:
        """Generate relevant questions for doctor consultation."""

        risk_level = self._determine_risk_level(risk_probability)

        # Key risk factors for question generation
        key_factors = [factor.get('feature', '') for factor in risk_factors[:3]]

        prompt = f"""
A person has received a {risk_level} cardiovascular risk assessment. The main contributing
factors are: {', '.join(key_factors)}.

Please suggest 5-6 specific, relevant questions this person should ask their doctor during
their next appointment. The questions should:
- Be directly related to cardiovascular health
- Help them understand their personal risk
- Lead to actionable medical advice
- Be appropriate for their risk level
- Cover both assessment validation and prevention strategies

Format as a simple list of questions.
"""

        response = self._make_llm_request(prompt)

        # Parse response into list
        questions = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and ('?' in line or line.startswith('-') or line.startswith('•') or
                        any(line.startswith(f'{i}.') for i in range(1, 10))):
                clean_line = line.lstrip('-•').strip()
                if clean_line and len(clean_line) > 15:  # Filter out very short items
                    if not clean_line.endswith('?'):
                        clean_line += '?'
                    questions.append(clean_line)

        # Fallback questions if parsing failed
        if not questions:
            questions = self._get_fallback_questions(risk_level)

        return questions[:6]  # Limit to 6 questions

    def _determine_risk_level(self, risk_probability: float) -> str:
        """Determine risk level category."""

        if risk_probability >= settings.HIGH_RISK_THRESHOLD:
            return RiskLevel.HIGH
        elif risk_probability >= settings.LOW_RISK_THRESHOLD:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW

    def _get_fallback_recommendations(self, risk_level: str) -> List[str]:
        """Provide fallback recommendations when LLM is not available."""

        base_recommendations = [
            "Maintain a heart-healthy diet rich in fruits, vegetables, and whole grains",
            "Engage in regular physical activity as approved by your healthcare provider",
            "Monitor your blood pressure regularly",
            "Maintain a healthy weight",
            "Avoid smoking and limit alcohol consumption",
            "Manage stress through relaxation techniques",
            "Get adequate sleep (7-9 hours per night)"
        ]

        if risk_level == RiskLevel.HIGH:
            base_recommendations.extend([
                "Schedule regular check-ups with your healthcare provider",
                "Consider discussing medication options with your doctor",
                "Monitor cholesterol levels regularly"
            ])

        return base_recommendations

    def _get_fallback_questions(self, risk_level: str) -> List[str]:
        """Provide fallback questions when LLM is not available."""

        base_questions = [
            "What does my cardiovascular risk assessment mean for my health?",
            "What additional tests might be helpful to assess my heart health?",
            "What lifestyle changes would be most beneficial for me?",
            "How often should I monitor my cardiovascular health?",
            "Are there any warning signs I should watch for?"
        ]

        if risk_level == RiskLevel.HIGH:
            base_questions.extend([
                "Should I consider medication for cardiovascular protection?",
                "What specialists might I need to see?"
            ])

        return base_questions

    def generate_comprehensive_explanation(self, prediction_result: Dict[str, Any],
                                         shap_explanation: Dict[str, Any],
                                         patient_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive patient-friendly explanation."""

        risk_probability = prediction_result.get('risk_probability', 0.0)
        risk_factors = shap_explanation.get('top_risk_factors', [])
        protective_factors = shap_explanation.get('top_protective_factors', [])

        # Generate all explanation components
        risk_explanation = self.generate_risk_explanation(
            risk_probability, risk_factors, protective_factors, patient_context
        )

        recommendations = self.generate_lifestyle_recommendations(
            risk_probability, risk_factors, patient_context
        )

        doctor_questions = self.generate_doctor_questions(
            risk_probability, risk_factors, patient_context
        )

        # Compile comprehensive explanation
        comprehensive_explanation = {
            'risk_explanation': risk_explanation,
            'lifestyle_recommendations': recommendations,
            'doctor_consultation_questions': doctor_questions,
            'generated_timestamp': datetime.now().isoformat(),
            'risk_level': self._determine_risk_level(risk_probability),
            'medical_disclaimer': settings.MEDICAL_DISCLAIMER
        }

        return comprehensive_explanation

    def save_explanation(self, explanation: Dict[str, Any], filename: str = None) -> str:
        """Save LLM explanation to file."""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"llm_explanation_{timestamp}.json"

        filepath = os.path.join("logs", filename)
        os.makedirs("logs", exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(explanation, f, indent=2)

        logger.info(f"LLM explanation saved: {filepath}")
        return filepath


def main():
    """Demonstration of LLM explanation generation."""

    from config.logging_config import setup_logging
    setup_logging()

    # Initialize LLM generator
    llm_generator = LLMExplanationGenerator()

    # Example prediction result
    prediction_result = {
        'risk_probability': 0.65,
        'risk_level': 'Moderate'
    }

    # Example SHAP explanation
    shap_explanation = {
        'top_risk_factors': [
            {'feature': 'age', 'contribution': 0.15, 'feature_value': 55},
            {'feature': 'chol', 'contribution': 0.12, 'feature_value': 280},
            {'feature': 'trestbps', 'contribution': 0.08, 'feature_value': 145}
        ],
        'top_protective_factors': [
            {'feature': 'thalach', 'contribution': -0.05, 'feature_value': 170}
        ]
    }

    # Example patient context
    patient_context = {
        'age': 55,
        'sex': 1  # Male
    }

    # Generate comprehensive explanation
    explanation = llm_generator.generate_comprehensive_explanation(
        prediction_result, shap_explanation, patient_context
    )

    # Save and display results
    saved_file = llm_generator.save_explanation(explanation)

    print("\n" + "="*60)
    print("LLM EXPLANATION GENERATION DEMO")
    print("="*60)
    print(f"Risk Level: {explanation['risk_level']}")
    print(f"\nRisk Explanation:")
    print(explanation['risk_explanation'])
    print(f"\nLifestyle Recommendations ({len(explanation['lifestyle_recommendations'])}):")
    for i, rec in enumerate(explanation['lifestyle_recommendations'], 1):
        print(f"{i}. {rec}")
    print(f"\nDoctor Questions ({len(explanation['doctor_consultation_questions'])}):")
    for i, question in enumerate(explanation['doctor_consultation_questions'], 1):
        print(f"{i}. {question}")
    print(f"\nExplanation saved to: {saved_file}")
    print("="*60)


if __name__ == "__main__":
    main()