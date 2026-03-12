"""
Constants and enums for the Heart Disease Risk Prediction system.
"""

from enum import Enum
from typing import Dict, List


class RiskLevel(str, Enum):
    """Risk level categories."""
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"


class ChestPainType(int, Enum):
    """Chest pain type encoding."""
    TYPICAL_ANGINA = 0
    ATYPICAL_ANGINA = 1
    NON_ANGINAL_PAIN = 2
    ASYMPTOMATIC = 3


class RestingECG(int, Enum):
    """Resting ECG results encoding."""
    NORMAL = 0
    ST_T_ABNORMALITY = 1
    LV_HYPERTROPHY = 2


class STSlope(int, Enum):
    """ST slope encoding."""
    UPSLOPING = 0
    FLAT = 1
    DOWNSLOPING = 2


class Thalassemia(int, Enum):
    """Thalassemia encoding."""
    NORMAL = 0
    FIXED_DEFECT = 1
    REVERSIBLE_DEFECT = 2
    NOT_DESCRIBED = 3


# Feature names and descriptions
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "age": "Age in years",
    "sex": "Sex (0: female, 1: male)",
    "cp": "Chest pain type (0-3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (0: false, 1: true)",
    "restecg": "Resting electrocardiographic results (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise induced angina (0: no, 1: yes)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of the peak exercise ST segment (0-2)",
    "ca": "Number of major vessels colored by fluoroscopy (0-4)",
    "thal": "Thalassemia (0-3)"
}

# Feature ranges for validation
FEATURE_RANGES: Dict[str, Dict[str, float]] = {
    "age": {"min": 18, "max": 120},
    "sex": {"min": 0, "max": 1},
    "cp": {"min": 0, "max": 3},
    "trestbps": {"min": 80, "max": 200},
    "chol": {"min": 100, "max": 600},
    "fbs": {"min": 0, "max": 1},
    "restecg": {"min": 0, "max": 2},
    "thalach": {"min": 60, "max": 220},
    "exang": {"min": 0, "max": 1},
    "oldpeak": {"min": 0, "max": 10},
    "slope": {"min": 0, "max": 2},
    "ca": {"min": 0, "max": 4},
    "thal": {"min": 0, "max": 3}
}

# Feature names in order
FEATURE_NAMES: List[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# Categorical features
CATEGORICAL_FEATURES: List[str] = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]

# Numerical features
NUMERICAL_FEATURES: List[str] = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

# Model evaluation metrics
EVALUATION_METRICS: List[str] = [
    "accuracy", "precision", "recall", "f1_score", "roc_auc"
]

# Risk factor explanations for LLM
RISK_FACTOR_EXPLANATIONS: Dict[str, str] = {
    "age": "Advanced age is associated with increased cardiovascular risk",
    "sex": "Male sex is associated with higher risk of heart disease",
    "cp": "Certain types of chest pain indicate higher cardiac risk",
    "trestbps": "High resting blood pressure increases cardiovascular stress",
    "chol": "Elevated cholesterol levels contribute to arterial plaque buildup",
    "fbs": "High fasting blood sugar indicates diabetes risk",
    "restecg": "Abnormal ECG patterns may indicate heart problems",
    "thalach": "Lower maximum heart rate may indicate reduced cardiac fitness",
    "exang": "Exercise-induced angina suggests coronary artery disease",
    "oldpeak": "ST depression indicates possible ischemia during exercise",
    "slope": "Abnormal ST slope patterns suggest cardiac stress",
    "ca": "Blocked major vessels directly increase heart disease risk",
    "thal": "Thalassemia defects affect heart muscle function",
    # Engineered features — human-readable descriptions
    "age_group": "Age group category affects baseline cardiovascular risk",
    "bp_chol_ratio": "Combined blood pressure and cholesterol burden on arteries",
    "hr_reserve": "Heart rate reserve reflects cardiovascular fitness capacity",
    "multiple_risk_factors": "Combination of multiple simultaneous risk factors",
    "age_chol": "Age-cholesterol interaction — risk compounds with age",
    "bp_hr_ratio": "Blood pressure to heart rate ratio reflects cardiac workload",
    "oldpeak_exang": "ST depression combined with exercise angina indicates ischemia",
    "ca_thal_risk": "Combined vessel blockage and thalassemia defect severity",
    "cp_exang_combo": "Chest pain type combined with exercise-induced angina"
}

# Human-readable display names for features (for UI)
FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "age": "Age", "sex": "Sex", "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure", "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar", "restecg": "Resting ECG",
    "thalach": "Max Heart Rate", "exang": "Exercise Angina",
    "oldpeak": "ST Depression", "slope": "ST Slope",
    "ca": "Major Vessels Blocked", "thal": "Thalassemia",
    "age_group": "Age Group", "bp_chol_ratio": "BP-Cholesterol Load",
    "hr_reserve": "Heart Rate Reserve", "multiple_risk_factors": "Multiple Risk Factors",
    "age_chol": "Age × Cholesterol", "bp_hr_ratio": "BP-Heart Rate Ratio",
    "oldpeak_exang": "ST Depression + Angina", "ca_thal_risk": "Vessel + Thalassemia Risk",
    "cp_exang_combo": "Chest Pain + Exercise Angina",
}

# Feature units for display
FEATURE_UNITS: Dict[str, str] = {
    "age": "years", "trestbps": "mm Hg", "chol": "mg/dl",
    "thalach": "bpm", "oldpeak": "",
}