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
from utils.metrics import LLM_RESPONSE_HISTOGRAM

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Simple in-memory TTL cache for LLM responses.
# Keyed on a hash of (risk_level, sorted top-3 feature names).
# Identical risk profiles reuse cached output for LLM_CACHE_TTL_SECONDS.
# ---------------------------------------------------------------------------
_LLM_CACHE: Dict[str, Dict[str, Any]] = {}
_LLM_CACHE_TTL_SECONDS: int = 3600  # 1 hour


def _cache_key(risk_level: str, risk_factors: List[Dict[str, Any]]) -> str:
    # Include feature name + rounded value so different measurements get different responses
    top3 = sorted(
        f"{f.get('feature', '')}:{round(float(f.get('feature_value', 0)), 1)}"
        for f in risk_factors[:3]
    )
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
            _t0 = time.perf_counter()
            if self.provider == "ollama":
                result = self._request_ollama(prompt, max_tokens, temperature)
            elif self.provider == "gemini":
                result = self._request_gemini(prompt)
            elif self.provider == "openai":
                result = self._request_openai(prompt, max_tokens, temperature)
            else:
                result = self._generate_fallback_explanation(prompt)
            LLM_RESPONSE_HISTOGRAM.observe(time.perf_counter() - _t0)
            return result
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

        # Prepare risk factors text with actual values
        risk_factors_text = ""
        if risk_factors:
            risk_factors_text = "Key factors contributing to your risk:\n"
            for factor in risk_factors[:3]:
                factor_name = factor.get('feature', 'Unknown factor')
                val = factor.get('feature_value', '')
                explanation = RISK_FACTOR_EXPLANATIONS.get(factor_name,
                                                         f"The {factor_name} measurement")
                val_str = f" (your value: {round(float(val), 2)})" if isinstance(val, (int, float)) else ""
                risk_factors_text += f"- {explanation}{val_str}\n"

        # Prepare protective factors text
        protective_factors_text = ""
        if protective_factors:
            protective_factors_text = "Factors working in your favor:\n"
            for factor in protective_factors[:2]:
                factor_name = factor.get('feature', 'Unknown factor')
                val = factor.get('feature_value', '')
                explanation = RISK_FACTOR_EXPLANATIONS.get(factor_name,
                                                         f"The {factor_name} measurement")
                val_str = f" (your value: {round(float(val), 2)})" if isinstance(val, (int, float)) else ""
                protective_factors_text += f"- {explanation}{val_str} appears protective\n"

        age = (patient_context or {}).get('age', '')
        sex = 'male' if (patient_context or {}).get('sex') == 1 else 'female' if (patient_context or {}).get('sex') == 0 else ''
        patient_line = f"Patient: {age}-year-old {sex}".strip() if age or sex else ""

        prompt = f"""
Please explain the following heart disease risk assessment in warm, patient-friendly language:

{patient_line}
Risk Level: {risk_level}
Risk Probability: {risk_probability:.1%}

{risk_factors_text}
{protective_factors_text}

In 2-3 sentences: explain what this risk level means for this specific patient given their
actual measurements, and reassure them this is educational, not a diagnosis.
"""

        return self._make_llm_request(prompt)

    def generate_lifestyle_recommendations(self, risk_probability: float,
                                         risk_factors: List[Dict[str, Any]],
                                         patient_context: Dict[str, Any] = None) -> List[str]:
        """Generate personalized lifestyle recommendations."""

        risk_level = self._determine_risk_level(risk_probability)
        patient_context = patient_context or {}

        # Build a human-readable profile from patient values
        FEATURE_LABELS = {
            'age': ('Age', 'years'), 'trestbps': ('Resting blood pressure', 'mm Hg'),
            'chol': ('Serum cholesterol', 'mg/dl'), 'thalach': ('Max heart rate', 'bpm'),
            'oldpeak': ('ST depression', ''), 'fbs': ('Fasting blood sugar >120', ''),
            'ca': ('Major vessels blocked', ''), 'thal': ('Thalassemia type', ''),
            'cp': ('Chest pain type', ''), 'exang': ('Exercise angina', ''),
        }

        # Patient measurements from context
        patient_profile_lines = []
        for key, (label, unit) in FEATURE_LABELS.items():
            val = patient_context.get(key)
            if val is not None:
                patient_profile_lines.append(f"  - {label}: {val} {unit}".strip())

        patient_profile = "\n".join(patient_profile_lines) if patient_profile_lines else "  Not provided"

        # Top risk factors with actual values
        risk_factor_lines = []
        for f in risk_factors[:5]:
            name = f.get('feature', '').replace('_', ' ')
            val = f.get('feature_value', '')
            contrib = f.get('contribution', 0)
            tip = f.get('explanation', '')
            risk_factor_lines.append(
                f"  - {name} = {round(float(val), 2) if isinstance(val, (int, float)) else val}"
                f" (SHAP contribution: +{contrib:.3f})"
                + (f" — {tip}" if tip else "")
            )

        risk_factors_text = "\n".join(risk_factor_lines) if risk_factor_lines else "  None identified"

        prompt = f"""
A patient has received a {risk_level} cardiovascular risk assessment ({risk_probability:.1%} probability).

Patient profile:
{patient_profile}

Top contributing risk factors (with actual measured values):
{risk_factors_text}

Please provide 5-7 specific, personalised lifestyle recommendations that directly address
THIS patient's actual measurements and risk factors above. For example:
- If their cholesterol is high, give specific dietary advice targeting cholesterol reduction
- If their blood pressure is elevated, give targeted BP-lowering tips
- If they have exercise angina, tailor exercise advice accordingly
- If ST depression is elevated, address cardiac rehabilitation

Requirements:
- Tie each recommendation to at least one of their actual measurements
- Be specific (e.g. "aim for cholesterol below 200 mg/dl" not just "eat healthy")
- Be encouraging and positive
- Appropriate urgency for {risk_level.lower()} risk

Format: numbered list, one recommendation per line, no sub-bullets.
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
        patient_context = patient_context or {}

        # Top risk factors with values
        factor_lines = []
        for f in risk_factors[:4]:
            name = f.get('feature', '').replace('_', ' ')
            val = f.get('feature_value', '')
            tip = f.get('explanation', '')
            factor_lines.append(
                f"  - {name} = {round(float(val), 2) if isinstance(val, (int, float)) else val}"
                + (f" ({tip})" if tip else "")
            )
        factors_text = "\n".join(factor_lines) if factor_lines else "  None identified"

        age = patient_context.get('age', 'unknown')
        sex = 'male' if patient_context.get('sex') == 1 else 'female' if patient_context.get('sex') == 0 else 'unknown'

        prompt = f"""
A {age}-year-old {sex} patient has received a {risk_level} cardiovascular risk assessment
({risk_probability:.1%} probability).

Their top contributing risk factors are:
{factors_text}

Suggest 5-6 specific questions they should ask their doctor that are directly relevant to
THEIR measurements above. For example, if their thalassemia type is abnormal, ask about it.
If their blood pressure is high, ask about targets. If ST depression is present, ask about
exercise stress testing.

Format: numbered list of questions, one per line.
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

        # Only cache if LLM actually responded (not fallback static text)
        fallback_markers = ["consult with your healthcare provider to discuss", "Please review the detailed results"]
        is_fallback = any(m in risk_explanation for m in fallback_markers)
        if not is_fallback:
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