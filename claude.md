# Explainable Heart Disease Risk Prediction & Preventive Health Assistant

## 🎯 Problem Statement

Build a production-ready ML + LLM system that collects patient health indicators, predicts heart disease risk using ML, explains contributing factors, and generates preventive guidance with structured JSON responses. This is NOT a diagnostic tool - it's a preventive awareness assistant.

## 🏗️ System Architecture

### High-Level Architecture
```
[Patient Input] → [Data Validation] → [ML Prediction] → [Explainability] → [LLM Layer] → [Structured Response]
```

### Core Components
1. **Data Processing Layer**: Input validation, feature engineering, preprocessing
2. **ML Prediction Engine**: Trained models with uncertainty quantification
3. **Explainability Module**: SHAP/LIME-based feature importance
4. **LLM Integration Layer**: Patient-friendly explanations and recommendations
5. **API Layer**: FastAPI endpoints with structured responses
6. **Logging & Monitoring**: Request tracking, model performance monitoring

## 📁 Project Structure

```
heart_disease_assistant/
├── claude.md                          # This file - project specification
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore file
├── README.md                          # Project documentation
├── config/
│   ├── __init__.py
│   ├── settings.py                    # Configuration management
│   └── logging_config.py              # Logging configuration
├── data/
│   ├── raw/                          # Original UCI dataset
│   ├── processed/                    # Cleaned and preprocessed data
│   └── models/                       # Trained model artifacts
├── src/
│   ├── __init__.py
│   ├── data_processing.py            # Data cleaning and preprocessing
│   ├── model_training.py             # ML model training pipeline
│   ├── evaluation.py                 # Model evaluation metrics
│   ├── explainability.py             # SHAP/LIME explanations
│   ├── llm_layer.py                  # LLM integration for explanations
│   └── prediction_service.py         # Core prediction orchestration
├── api/
│   ├── __init__.py
│   ├── main.py                       # FastAPI application
│   ├── models.py                     # Pydantic models/schemas
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py                 # Health check endpoints
│   │   └── prediction.py             # Prediction endpoints
│   └── middleware/
│       ├── __init__.py
│       ├── validation.py             # Input validation
│       └── error_handling.py         # Error handling middleware
├── utils/
│   ├── __init__.py
│   ├── constants.py                  # Project constants
│   ├── helpers.py                    # Utility functions
│   └── validators.py                 # Data validation utilities
├── tests/
│   ├── __init__.py
│   ├── test_data_processing.py
│   ├── test_model_training.py
│   ├── test_evaluation.py
│   ├── test_explainability.py
│   ├── test_llm_layer.py
│   └── test_api.py
├── logs/                             # Application logs
├── notebooks/                        # Jupyter notebooks for EDA
│   └── exploratory_analysis.ipynb
└── scripts/
    ├── download_data.py              # UCI dataset download
    ├── train_model.py                # Model training script
    └── run_evaluation.py             # Evaluation script
```

## 📊 Data Pipeline

### Dataset: UCI Heart Disease (Cleveland)
- **Source**: UCI ML Repository
- **Features**: 14 attributes (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
- **Target**: Binary classification (0: no heart disease, 1: heart disease)
- **Size**: ~303 samples

### Data Processing Steps
1. **Data Collection**: Download UCI dataset programmatically
2. **Data Cleaning**:
   - Handle missing values (imputation/removal)
   - Remove outliers using IQR method
   - Validate data types and ranges
3. **Feature Engineering**:
   - Categorical encoding (one-hot/label encoding)
   - Feature scaling (StandardScaler)
   - Feature selection based on correlation analysis
4. **Data Splitting**: 70/15/15 train/validation/test split
5. **Cross-Validation**: 5-fold stratified CV

## 🤖 ML Training Pipeline

### Model Selection Strategy
1. **Baseline Models**:
   - Logistic Regression
   - Decision Tree
   - Random Forest
2. **Advanced Models**:
   - XGBoost
   - Support Vector Machine
3. **Model Comparison**: ROC-AUC, Precision, Recall, F1-Score

### Training Process
1. **Hyperparameter Tuning**: GridSearchCV for each model
2. **Feature Importance**: Track feature contributions
3. **Model Persistence**: Save best model using joblib
4. **Version Control**: Model versioning with metadata

### Training Configuration
- Cross-validation: 5-fold stratified
- Scoring metric: ROC-AUC (primary), Precision/Recall (secondary)
- Random state: 42 for reproducibility

## 📈 Evaluation Plan

### Metrics
- **Primary**: ROC-AUC, Confusion Matrix
- **Secondary**: Precision, Recall, F1-Score, Accuracy
- **Clinical**: Sensitivity, Specificity, PPV, NPV

### Evaluation Framework
1. **Hold-out Test Set**: Final model performance
2. **Cross-Validation**: Training stability assessment
3. **Feature Importance**: Model interpretability
4. **Threshold Analysis**: Optimal decision threshold
5. **Error Analysis**: False positive/negative analysis

## 🔍 Explainability Plan

### SHAP Integration
- **Global Explanations**: Feature importance across all predictions
- **Local Explanations**: Per-prediction feature contributions
- **Visualization**: Waterfall plots, force plots, summary plots

### Implementation
1. **SHAP Explainer**: TreeExplainer for tree-based models, LinearExplainer for linear models
2. **Feature Attribution**: Numerical values for each feature contribution
3. **Risk Factors**: Identify top contributing risk factors
4. **Protective Factors**: Identify factors reducing risk

## 🧠 LLM Integration Plan

### LLM Responsibilities
1. **Medical Translation**: Convert technical terms to patient-friendly language
2. **Risk Explanation**: Explain why certain factors increase/decrease risk
3. **Lifestyle Recommendations**: Generate personalized preventive advice
4. **Doctor Consultation**: Suggest relevant questions for healthcare providers

### LLM Integration Architecture
```python
# Pseudo-code structure
def generate_explanation(prediction_result, shap_values, patient_data):
    risk_factors = extract_top_risk_factors(shap_values)
    protective_factors = extract_protective_factors(shap_values)

    explanation = llm_client.generate_explanation(
        risk_level=prediction_result.risk_probability,
        risk_factors=risk_factors,
        protective_factors=protective_factors,
        patient_context=patient_data
    )

    recommendations = llm_client.generate_recommendations(
        patient_profile=patient_data,
        risk_factors=risk_factors
    )

    return explanation, recommendations
```

### LLM Provider
- **Primary**: OpenAI GPT-4 (configurable)
- **Fallback**: Local model or alternative API
- **Rate Limiting**: Implement request throttling

## 🛠️ API Structure

### FastAPI Implementation

#### Core Endpoints
```python
POST /predict
- Input: Patient health data
- Output: Risk prediction + explanation + recommendations

GET /health
- System health check

GET /model/info
- Model version and metadata

POST /explain
- Input: Prediction result
- Output: Detailed SHAP explanations
```

#### Request/Response Schema
```python
# Input Schema
class PatientData(BaseModel):
    age: int = Field(..., ge=18, le=120)
    sex: int = Field(..., ge=0, le=1)  # 0=female, 1=male
    cp: int = Field(..., ge=0, le=3)   # chest_pain_type
    trestbps: int = Field(..., ge=80, le=200)  # resting_blood_pressure
    chol: int = Field(..., ge=100, le=600)     # serum_cholesterol
    fbs: int = Field(..., ge=0, le=1)  # fasting_blood_sugar
    restecg: int = Field(..., ge=0, le=2)      # resting_ecg
    thalach: int = Field(..., ge=60, le=220)   # max_heart_rate
    exang: int = Field(..., ge=0, le=1)        # exercise_angina
    oldpeak: float = Field(..., ge=0, le=10)   # st_depression
    slope: int = Field(..., ge=0, le=2)        # st_slope
    ca: int = Field(..., ge=0, le=4)           # major_vessels
    thal: int = Field(..., ge=0, le=3)         # thalassemia

# Output Schema
class PredictionResponse(BaseModel):
    patient_id: str
    risk_probability: float
    risk_level: str  # "Low", "Moderate", "High"
    confidence_interval: Tuple[float, float]
    risk_factors: List[RiskFactor]
    protective_factors: List[ProtectiveFactor]
    explanation: str
    recommendations: List[str]
    doctor_questions: List[str]
    model_version: str
    timestamp: datetime
    disclaimer: str
```

## 🚀 Deployment Strategy

### Development Environment
- Local development with hot-reload
- Docker containerization
- Environment-specific configs

### Production Considerations
1. **Model Serving**: FastAPI + Uvicorn
2. **Containerization**: Docker with multi-stage builds
3. **Monitoring**: Request logging, model drift detection
4. **Scaling**: Load balancing capabilities
5. **Security**: Input validation, rate limiting, HTTPS

### Infrastructure (Future)
- **Cloud Provider**: AWS/GCP/Azure
- **Container Orchestration**: Kubernetes/Docker Swarm
- **Model Registry**: MLflow/DVC
- **Monitoring**: Prometheus + Grafana

## ⚖️ Ethical Considerations

### Medical Disclaimers
```
"This tool is for educational and preventive awareness purposes only.
It is NOT a substitute for professional medical advice, diagnosis, or treatment.
Always consult with a qualified healthcare provider for medical decisions."
```

### Bias & Fairness
1. **Dataset Limitations**: Cleveland Clinic data may not represent all populations
2. **Demographic Bias**: Monitor performance across age, gender, ethnicity
3. **Clinical Validation**: Requires clinical validation before medical use
4. **Transparency**: Clear explanation of model limitations

### Privacy & Security
1. **Data Handling**: No persistent storage of patient data
2. **Anonymization**: Remove identifying information from logs
3. **Encryption**: Secure data transmission
4. **Compliance**: HIPAA consideration for future medical use

## 📅 3-Week Development Roadmap

### Week 1: Foundation & Data Pipeline
**Days 1-2**: Project Setup
- Initialize repository structure
- Set up development environment
- Create configuration management
- Download and explore UCI dataset

**Days 3-5**: Data Processing
- Implement data cleaning pipeline
- Feature engineering and selection
- Data validation and quality checks
- Train/validation/test split

**Days 6-7**: Baseline ML Models
- Implement logistic regression baseline
- Basic evaluation framework
- Model persistence and loading

### Week 2: ML Pipeline & Explainability
**Days 8-10**: Advanced ML Models
- Implement Random Forest, XGBoost
- Hyperparameter tuning
- Model comparison and selection
- Cross-validation framework

**Days 11-12**: Explainability Integration
- SHAP implementation
- Feature importance analysis
- Visualization capabilities
- Local and global explanations

**Days 13-14**: Evaluation & Testing
- Comprehensive evaluation metrics
- Error analysis and model diagnostics
- Unit tests for ML components

### Week 3: LLM Integration & API
**Days 15-17**: LLM Layer
- LLM integration for explanations
- Patient-friendly language translation
- Recommendation generation
- Template-based response formatting

**Days 18-19**: API Development
- FastAPI application structure
- Request/response schemas
- Input validation and error handling
- Health check endpoints

**Days 20-21**: Production Features
- Logging and monitoring
- Security measures
- Documentation
- Final testing and deployment preparation

## 📝 Commit Strategy

### Commit Message Format
```
<type>: <description>

<optional body>
```

### Commit Types
- `feat`: New features
- `fix`: Bug fixes
- `refactor`: Code restructuring
- `test`: Adding tests
- `docs`: Documentation updates
- `chore`: Maintenance tasks

### Example Commits
1. `feat: initialize repository structure`
2. `feat: add UCI dataset download script`
3. `feat: implement data preprocessing pipeline`
4. `feat: add logistic regression baseline model`
5. `feat: implement Random Forest with hyperparameter tuning`
6. `feat: integrate SHAP explainability module`
7. `feat: add LLM explanation generation`
8. `feat: create FastAPI prediction endpoint`
9. `feat: implement input validation and error handling`
10. `docs: add API documentation and usage examples`

## 🔧 Code Quality Standards

### Python Standards
- **Style**: PEP 8 compliance
- **Type Hints**: All functions and methods
- **Docstrings**: Google/NumPy format
- **Testing**: Minimum 80% coverage
- **Linting**: flake8, black, isort

### Project Standards
- **Modularity**: Single responsibility principle
- **Configuration**: Environment-based configs
- **Error Handling**: Comprehensive exception handling
- **Logging**: Structured logging with appropriate levels
- **Security**: Input validation and sanitization

### Dependencies
```
Core ML/Data:
- pandas>=1.5.0
- numpy>=1.24.0
- scikit-learn>=1.2.0
- xgboost>=1.7.0
- shap>=0.41.0

API/Web:
- fastapi>=0.100.0
- uvicorn>=0.23.0
- pydantic>=2.0.0

LLM Integration:
- openai>=1.0.0
- tiktoken>=0.4.0

Utilities:
- python-dotenv>=1.0.0
- joblib>=1.3.0
- python-multipart>=0.0.6

Development:
- pytest>=7.4.0
- black>=23.7.0
- flake8>=6.0.0
- mypy>=1.5.0
```

## 🎯 Success Criteria

### Technical Metrics
- Model ROC-AUC > 0.80 on test set
- API response time < 500ms
- 100% test coverage for critical paths
- Zero security vulnerabilities

### Functional Requirements
- Accurate risk predictions with confidence intervals
- Clear, actionable explanations
- Patient-friendly language
- Robust error handling
- Complete API documentation

### Production Readiness
- Comprehensive logging
- Input validation
- Error handling
- Health monitoring
- Security measures
- Deployment documentation

---

**Note**: This specification serves as the single source of truth for the project. All implementation decisions must align with this document. Any major architectural changes require explicit approval and document updates.