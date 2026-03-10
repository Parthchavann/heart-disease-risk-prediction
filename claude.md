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

---

## ✅ Completed Work (chronological)

### Phase 1 — Foundation
- [x] Full project structure initialised (`src/`, `api/`, `config/`, `utils/`, `tests/`, `scripts/`)
- [x] UCI Heart Disease dataset downloaded from 4 locations — **920 total samples**
- [x] Data cleaning: missing value imputation, IQR outlier removal, type validation
- [x] Feature engineering: 13 original → **17 features** (added `age_group`, `bp_chol_ratio`, `hr_reserve`, `multiple_risk_factors`)
- [x] Train / val / test split: 644 / 138 / 138 (stratified)
- [x] Transformers saved: `data/processed/transformers/` (scaler, label encoders, imputers)

### Phase 2 — ML Pipeline
- [x] Baseline models: Logistic Regression, Decision Tree
- [x] Advanced models: Random Forest, SVM, LightGBM (GridSearchCV, 5-fold stratified CV)
- [x] XGBoost tuned via **Optuna** (150 trials, Bayesian search) — avoids Windows crash from n_jobs=-1
- [x] Stacking ensemble **removed** — unnecessary complexity for 920-sample dataset
- [x] Youden-J optimal threshold computed on val set and stored in metadata
- [x] **Best model: SVM** (GridSearch winner across all comparisons)
  - Val:  AUC=0.845, Acc=81.2%, Recall=82.9%
  - Test: AUC=0.904, Acc=84.8%, Recall=92.1%, F1=87.0%
- [x] Model saved: `data/models/best_model.pkl` + `best_model_metadata.json`

### Phase 3 — Explainability
- [x] SHAP explainability module (`src/explainability.py`) — TreeExplainer / LinearExplainer
- [x] Waterfall, force, and summary plots saved in `logs/explainability_plots/`
- [x] Per-prediction risk factors and protective factors extracted

### Phase 4 — LLM Layer
- [x] Multi-provider LLM layer (`src/llm_layer.py`):
  - `ollama` — Gemma2 local (active, no API key needed)
  - `gemini` — production-ready (add `GEMINI_API_KEY`)
  - `openai` — fallback (key stored, quota exhausted)
- [x] Patient-friendly explanations + personalised disclaimer per prediction

### Phase 5 — API & Frontend
- [x] FastAPI server (`api/main.py`) with endpoints: `POST /predict`, `GET /health`, `GET /model/info`
- [x] End-to-end prediction pipeline working (`success: true`)
- [x] Streamlit dashboard (`app.py`) — risk gauge, SHAP bar chart, LLM explanation
- [x] Docker containerisation (`Dockerfile` + `docker-compose.yml`)

### Phase 6 — Testing & Quality
- [x] 46/46 tests passing (full test suite across all modules)
- [x] Full pipeline script (`scripts/run_full_pipeline.py`) — 8/8 steps pass

### Key Bugs Fixed (for reference)
- `ModuleNotFoundError: config` — fixed via `PYTHONPATH=.`
- `UnicodeEncodeError` on Windows — replaced emoji with ASCII in pipeline script
- Feature mismatch 13 vs 17 in SHAP/evaluation — fixed
- LabelEncoder unseen labels — graceful fallback
- OpenAI v0→v1 API migration (`chat.completions.create`)
- `test_save_and_load_model` corrupting `best_model.pkl` — fixed with `tmp_path`
- Waterfall plot `ValueError` — uses explanation dict shap_values directly
- Batch endpoint accepting empty list — fixed with `min_length=1`
- XGBoost GridSearchCV Windows crash (n_jobs=-1 + Python 3.13) — XGBoost moved to Optuna only

---

## 🚀 How to Resume This Project

### Step 1 — Start Ollama (required for LLM)
```cmd
curl http://localhost:11434/api/tags
```
If not running: `ollama serve`

### Step 2 — Start the API
```cmd
cd "C:\Users\Parth Chavan\heart-disease-risk-prediction"
set PYTHONPATH=. && python api/main.py
```
Swagger UI: `http://localhost:8000/docs`

### Step 3 — Sample predict request
```json
{
  "patient_data": {
    "age": 55, "sex": 1, "cp": 3, "trestbps": 140, "chol": 250,
    "fbs": 0, "restecg": 1, "thalach": 150, "exang": 1,
    "oldpeak": 2.5, "slope": 1, "ca": 1, "thal": 2
  },
  "options": { "include_explanation": true, "include_llm_explanation": true }
}
```
Note: Response takes 30-60 s (Gemma2 local).

### Streamlit Dashboard
```cmd
set PYTHONPATH=. && streamlit run app.py
```
Open: `http://localhost:8501`

### Docker
```cmd
docker-compose up --build
```
API: `http://localhost:8000/docs` | Streamlit: `http://localhost:8501`

### Re-run Full Pipeline (if data/model need refresh)
```cmd
set PYTHONPATH=. && python scripts/run_full_pipeline.py
```
IMPORTANT: After running the pipeline, always retrain the model immediately —
the pipeline re-splits data, which makes any previously saved model misaligned.

---

## 📋 Remaining Work (do in this order)

### STEP 1 — Verify tests still pass after model_training.py cleanup
```cmd
set PYTHONPATH=. && python -m pytest tests/ -v
```
Expected: 46/46 pass. Fix any failures before proceeding.

### STEP 2 — Fix optimal_threshold None bug in prediction_service.py
The `optimal_threshold` stored in `best_model_metadata.json` can be `None`
(if `optimize_threshold` returns None). `prediction_service.py` line ~148 does:
```python
threshold = self.model_metadata.get('optimal_threshold', 0.5)
```
`dict.get()` returns `None` (not 0.5) when the key exists with value `None`.
Fix: `threshold = self.model_metadata.get('optimal_threshold') or 0.5`

### STEP 3 — Switch LLM to Gemini (production, faster than Gemma2)
1. Get free API key: https://aistudio.google.com
2. Update `.env`:
   ```
   LLM_PROVIDER=gemini
   LLM_MODEL=gemini-1.5-pro
   GEMINI_API_KEY=your-key-here
   ```
3. Install: `pip install google-generativeai`
4. Test: restart API and call `/predict` — response should be <5 s instead of 30-60 s

### STEP 4 — Async Ollama calls (remove API blocking)
Currently `prediction_service.py` calls the LLM synchronously, blocking FastAPI for 30-60 s.
Fix: make `generate_llm_explanation` async and use `httpx.AsyncClient` for Ollama calls.
This lets the API serve other requests while Gemma2 generates.

### STEP 5 — Response caching (reduce repeated LLM calls)
Add an in-memory cache (e.g. `functools.lru_cache` or `cachetools.TTLCache`) keyed on
a hash of (risk_level, top-3 risk factors). Identical risk profiles reuse cached LLM output.

### STEP 6 — Cloud deployment
Use existing `Dockerfile` + `docker-compose.yml`.
Recommended path: push image to Docker Hub → deploy on Railway / Render (free tier) or AWS ECS.
Set env vars (`GEMINI_API_KEY`, `LLM_PROVIDER=gemini`) in the cloud dashboard.
No code changes needed — Docker setup is production-ready.

### STEP 7 (optional) — Monitoring
Add Prometheus metrics endpoint + Grafana dashboard for:
- Request count / latency
- Model prediction distribution (% High / Moderate / Low risk)
- LLM response time