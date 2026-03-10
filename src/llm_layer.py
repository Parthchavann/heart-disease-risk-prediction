"""
LLM integration layer for Heart Disease Risk Prediction.
Converts technical ML outputs into patient-friendly explanations and recommendations.
"""

import hashlib
import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

try:
    from google import genai as google_genai
    GEMINI_AVAILABLE = True
    GEMINI_LEGACY = False
except ImportError:
    try:
        import google.generativeai as google_genai  # type: ignore
        GEMINI_AVAILABLE = True
        GEMINI_LEGACY = True
    except ImportError:
        GEMINI_AVAILABLE = False
        GEMINI_LEGACY = False
        google_genai = None

import requests

from config.settings import settings
from config.logging_config import get_logger
from utils.constants import RiskLevel, RISK_FACTOR_EXPLANATIONS

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Simple in-memory TTL cache for LLM responses.
# Keyed on a hash of (risk_level, sorted top-3 feature names).
# Identical risk profiles reuse cached output for LLM_CACHE_TTL_SECONDS.
# ---------------------------------------------------------------------------
_LLM_CACHE: Dict[str, Dict[str, Any]] = {}
_LLM_CACHE_TTL_SECONDS: int = 3600  # 1 hour


def _cache_key(risk_level: str, risk_factors: List[Dict[str, Any]]) -> str:
    top3 = sorted(f.get('feature', '') for f in risk_factors[:3])
    raw = f"{risk_level}|{'|'.join(top3)}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    entry = _LLM_CACHE.get(key)
    if entry and (time.time() - entry['ts']) < _LLM_CACHE_TTL_SECONDS:
        return entry['value']
    if entry:
        del _LLM_CACHE[key]
    return None


def _cache_set(key: str, value: Dict[str, Any]) -> None:
    _LLM_CACHE[key] = {'value': value, 'ts': time.time()}


class LLMExplanationGenerator:
    """Generates patient-friendly explanations using LLM.

    Supports multiple providers via LLM_PROVIDER setting:
      - ollama  : local Gemma2 (no API key required)
      - gemini  : Google Gemini API (requires GEMINI_API_KEY)
      - openai  : OpenAI API (requires OPENAI_API_KEY)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.provider = settings.LLM_PROVIDER.lower()
        self.client = None

        if self.provider == "ollama":
            self._init_ollama()
        elif self.provider == "gemini":
            self._init_gemini(api_key)
        elif self.provider == "openai":
            self._init_openai(api_key)
        else:
            logger.warning(f"Unknown LLM_PROVIDER '{self.provider}'. Using fallback.")

    def _init_ollama(self):
        """Initialize Ollama local provider."""
        try:
            resp = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if resp.status_code == 200:
                self.client = "ollama"
                logger.info(f"Ollama client initialized (model: {settings.LLM_MODEL})")
            else:
                logger.warning("Ollama server not reachable. Using fallback.")
        except Exception as e:
            logger.warning(f"Ollama not available: {e}. Using fallback.")

    def _init_gemini(self, api_key: Optional[str] = None):
        """Initialize Google Gemini provider (new google.genai SDK)."""
        if not GEMINI_AVAILABLE:
            logger.warning("google-genai not installed. Run: pip install google-genai")
            return
        api_key = api_key or settings.GEMINI_API_KEY
        if not api_key:
            logger.warning("No GEMINI_API_KEY provided. Using fallback.")
            return
        if GEMINI_LEGACY:
            # Old google.generativeai SDK (deprecated)
            google_genai.configure(api_key=api_key)
            self.client = google_genai.GenerativeModel(settings.LLM_MODEL)
        else:
            # New google.genai SDK
            self.client = google_genai.Client(api_key=api_key)
        logger.info(f"Gemini client initialized (model: {settings.LLM_MODEL}, "
                    f"sdk={'legacy' if GEMINI_LEGACY else 'new'})")

    def _init_openai(self, api_key: Optional[str] = None):
        """Initialize OpenAI provider."""
        if not OPENAI_AVAILABLE:
            logger.warning("openai package not installed. Using fallback.")
            return
        api_key = api_key or settings.OPENAI_API_KEY
        if api_key:
            self.client = OpenAI(api_key=api_key)
            logger.info(f"OpenAI client initialized (model: {settings.LLM_MODEL})")
        else:
            logger.warning("No OPENAI_API_KEY provided. Using fallback.")

    def _make_llm_request(self, prompt: str, max_tokens: int = None,
                         temperature: float = None) -> str:
        """Make a request to the configured LLM provider."""

        if not self.client:
            return self._generate_fallback_explanation(prompt)

        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        temperature = temperature or settings.LLM_TEMPERATURE

        try:
            if self.provider == "ollama":
                return self._request_ollama(prompt, max_tokens, temperature)
            elif self.provider == "gemini":
                return self._request_gemini(prompt)
            elif self.provider == "openai":
                return self._request_openai(prompt, max_tokens, temperature)
        except Exception as e:
            logger.error(f"LLM request failed: {str(e)}")

        return self._generate_fallback_explanation(prompt)

    def _request_ollama(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Make request to local Ollama server."""
        full_prompt = f"{self._get_system_prompt()}\n\n{prompt}"
        response = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={"model": settings.LLM_MODEL, "prompt": full_prompt, "stream": False,
                  "options": {"temperature": temperature, "num_predict": max_tokens}},
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    def _request_gemini(self, prompt: str) -> str:
        """Make request to Google Gemini API."""
        full_prompt = f"{self._get_system_prompt()}\n\n{prompt}"
        if GEMINI_LEGACY:
            response = self.client.generate_content(full_prompt)
        else:
            response = self.client.models.generate_content(
                model=settings.LLM_MODEL, contents=full_prompt
            )
        return response.text.strip()

    def _request_openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Make request to OpenAI API."""
        response = self.client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

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

    def _generate_disclaimer(self, risk_probability: float,
                             risk_factors: List[Dict[str, Any]]) -> str:
        """Generate a personalised, context-aware medical disclaimer using the LLM."""

        risk_level = self._determine_risk_level(risk_probability)
        top_factors = ', '.join([f.get('feature', '') for f in risk_factors[:3] if f.get('feature')])

        prompt = f"""
Write a brief, personalised medical disclaimer (2-3 sentences) for a patient who just received
a {risk_level} cardiovascular risk assessment ({risk_probability*100:.0f}% risk probability).
Their top contributing factors are: {top_factors}.

The disclaimer should:
- Acknowledge their specific risk level and top factors
- Clarify this is a predictive tool, not a clinical diagnosis
- Encourage them to consult a qualified cardiologist or GP
- Be empathetic and reassuring, not alarming

Write only the disclaimer text, no headings or labels.
"""
        try:
            return self._make_llm_request(prompt, max_tokens=150)
        except Exception:
            return settings.MEDICAL_DISCLAIMER

    def generate_comprehensive_explanation(self, prediction_result: Dict[str, Any],
                                         shap_explanation: Dict[str, Any],
                                         patient_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive patient-friendly explanation.

        Results are cached for _LLM_CACHE_TTL_SECONDS (1 h) by risk level +
        top-3 risk factor names so identical profiles skip the LLM call.
        """

        risk_probability = prediction_result.get('risk_probability', 0.0)
        risk_factors = shap_explanation.get('top_risk_factors', [])
        protective_factors = shap_explanation.get('top_protective_factors', [])

        # Check cache
        risk_level_key = self._determine_risk_level(risk_probability)
        ck = _cache_key(risk_level_key, risk_factors)
        cached = _cache_get(ck)
        if cached:
            logger.info("LLM explanation served from cache")
            return cached

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

        # Generate personalised disclaimer via LLM
        disclaimer = self._generate_disclaimer(risk_probability, risk_factors)

        # Compile comprehensive explanation
        comprehensive_explanation = {
            'risk_explanation': risk_explanation,
            'lifestyle_recommendations': recommendations,
            'doctor_consultation_questions': doctor_questions,
            'generated_timestamp': datetime.now().isoformat(),
            'risk_level': self._determine_risk_level(risk_probability),
            'medical_disclaimer': disclaimer
        }

        # Store in cache for reuse
        _cache_set(ck, comprehensive_explanation)

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