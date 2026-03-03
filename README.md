# Heart Disease Risk Prediction & Preventive Health Assistant

A production-ready ML + LLM system that predicts heart disease risk using machine learning, provides SHAP-based explanations, and generates patient-friendly recommendations through LLM integration.

## 🎯 Overview

This system is designed for **educational and preventive awareness purposes only**. It is NOT a diagnostic tool and should never replace professional medical advice.

### Features

- 🧠 **Machine Learning Prediction**: Multiple algorithms (Random Forest, Logistic Regression, XGBoost) with hyperparameter tuning
- 📊 **Explainable AI**: SHAP-based feature importance and contribution analysis
- 🤖 **LLM Integration**: Patient-friendly explanations and personalized recommendations
- 🚀 **Production API**: FastAPI backend with comprehensive validation and error handling
- 📈 **Comprehensive Evaluation**: Detailed performance metrics and visualizations
- 🔒 **Security**: Input validation, rate limiting, and audit logging

## 📁 Project Structure

```
heart-disease-risk-prediction/
├── claude.md                    # Project specification and architecture
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── config/                      # Configuration management
├── data/                        # Dataset storage
│   ├── raw/                     # Original UCI dataset
│   ├── processed/               # Cleaned data
│   └── models/                  # Trained models
├── src/                         # Core ML components
│   ├── data_processing.py       # Data cleaning and preprocessing
│   ├── model_training.py        # ML model training
│   ├── evaluation.py            # Model evaluation and metrics
│   ├── explainability.py        # SHAP explanations
│   ├── llm_layer.py            # LLM integration
│   └── prediction_service.py    # Main orchestration service
├── api/                         # FastAPI application
│   ├── main.py                  # FastAPI app entry point
│   ├── models.py                # Pydantic schemas
│   ├── routes/                  # API endpoints
│   └── middleware/              # Middleware components
├── utils/                       # Utility functions
├── tests/                       # Unit tests
├── scripts/                     # Automation scripts
└── logs/                        # Application logs
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Parthchavann/heart-disease-risk-prediction.git
cd heart-disease-risk-prediction

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Download data, process, and train models
python scripts/train_model.py
```

This will:
- Download the UCI Heart Disease dataset
- Clean and preprocess the data
- Train multiple ML models with hyperparameter tuning
- Select the best performing model
- Save the trained model and metadata

### 3. Run the API

```bash
# Start the FastAPI server
python api/main.py
```

The API will be available at:
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Prediction Endpoint**: http://localhost:8000/predict

### 4. Make Predictions

#### Using the API

```python
import requests

# Example patient data
patient_data = {
    "age": 45,
    "sex": 1,  # 1=male, 0=female
    "cp": 2,   # chest pain type
    "trestbps": 130,  # resting blood pressure
    "chol": 250,      # cholesterol
    "fbs": 0,         # fasting blood sugar
    "restecg": 0,     # resting ECG
    "thalach": 175,   # max heart rate
    "exang": 0,       # exercise angina
    "oldpeak": 1.2,   # ST depression
    "slope": 1,       # ST slope
    "ca": 1,          # major vessels
    "thal": 2         # thalassemia
}

# Make prediction
response = requests.post(
    "http://localhost:8000/predict/simple",
    json=patient_data
)

result = response.json()
print(f"Risk Level: {result['risk_level']}")
print(f"Risk Probability: {result['risk_probability']:.2%}")
```

#### Using the Service Directly

```python
from src.prediction_service import HeartDiseasePredictionService

# Initialize service
service = HeartDiseasePredictionService()

# Make prediction
result = service.predict(patient_data)
print(f"Risk Level: {result['risk_level']}")
```

## 📊 Model Performance

The system trains multiple models and selects the best performer:

- **Random Forest**: Tree-based ensemble method
- **Logistic Regression**: Linear baseline model
- **XGBoost**: Gradient boosting (if available)
- **Decision Tree**: Simple interpretable model
- **SVM**: Support vector machine

Typical performance metrics:
- **Accuracy**: 80-85%
- **ROC-AUC**: 85-92%
- **Precision**: 80-85%
- **Recall**: 80-85%

## 🔍 Explainability Features

### SHAP Explanations
- **Global**: Overall feature importance across all predictions
- **Local**: Per-prediction feature contributions
- **Visualizations**: Waterfall plots, summary plots, force plots

### LLM-Generated Explanations
- **Patient-friendly language**: Technical terms converted to understandable explanations
- **Personalized recommendations**: Lifestyle advice based on risk factors
- **Doctor consultation questions**: Relevant questions for healthcare providers

## 🛡️ Input Features

| Feature | Description | Range |
|---------|-------------|-------|
| age | Age in years | 18-120 |
| sex | Sex (0=female, 1=male) | 0-1 |
| cp | Chest pain type | 0-3 |
| trestbps | Resting blood pressure (mm Hg) | 80-200 |
| chol | Serum cholesterol (mg/dl) | 100-600 |
| fbs | Fasting blood sugar > 120 mg/dl | 0-1 |
| restecg | Resting ECG results | 0-2 |
| thalach | Maximum heart rate achieved | 60-220 |
| exang | Exercise induced angina | 0-1 |
| oldpeak | ST depression induced by exercise | 0-10 |
| slope | Slope of peak exercise ST segment | 0-2 |
| ca | Number of major vessels (0-4) | 0-4 |
| thal | Thalassemia | 0-3 |

## 🌐 API Endpoints

### Health & Information
- `GET /` - API information
- `GET /health` - Health check
- `GET /health/model` - Model metadata
- `GET /predict/features` - Feature descriptions

### Prediction
- `POST /predict` - Full prediction with explanations
- `POST /predict/simple` - Simple prediction endpoint
- `POST /predict/batch` - Batch predictions
- `GET /predict/example` - Example request format

### Example API Response

```json
{
  "prediction_id": "123e4567-e89b-12d3-a456-426614174000",
  "timestamp": "2024-01-15T10:30:00Z",
  "success": true,
  "risk_probability": 0.35,
  "risk_level": "Moderate",
  "confidence_interval": [0.25, 0.45],
  "explanation": {
    "top_risk_factors": [
      {
        "feature": "age",
        "contribution": 0.15,
        "explanation": "Advanced age increases cardiovascular risk"
      }
    ],
    "top_protective_factors": [
      {
        "feature": "thalach",
        "contribution": -0.05,
        "explanation": "Good maximum heart rate indicates fitness"
      }
    ]
  },
  "llm_explanation": {
    "risk_explanation": "Your assessment indicates moderate cardiovascular risk...",
    "lifestyle_recommendations": [
      "Maintain a heart-healthy diet rich in fruits and vegetables",
      "Engage in regular aerobic exercise as approved by your doctor"
    ],
    "doctor_consultation_questions": [
      "What does my moderate risk level mean for my health?",
      "What preventive measures would be most beneficial for me?"
    ]
  }
}
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# LLM Configuration
OPENAI_API_KEY=your_openai_key_here
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.3

# Logging
LOG_LEVEL=INFO

# Security
SECRET_KEY=your-secret-key-here
```

### Settings
Modify `config/settings.py` for:
- Risk thresholds (low/moderate/high)
- Model parameters
- API rate limits
- File paths

## 📋 Requirements

### Core Dependencies
```
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.2.0
xgboost>=1.7.0
shap>=0.41.0
fastapi>=0.100.0
uvicorn>=0.23.0
openai>=1.0.0
```

See `requirements.txt` for complete list.

## 🧪 Testing

```bash
# Run evaluation on trained model
python scripts/run_evaluation.py

# Test individual components
python src/data_processing.py
python src/model_training.py
python src/explainability.py
python src/llm_layer.py
```

## 📈 Model Evaluation

The system provides comprehensive evaluation:

### Metrics
- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Probabilistic**: ROC-AUC, Precision-Recall AUC
- **Clinical**: Sensitivity, Specificity, PPV, NPV

### Visualizations
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Feature importance plots
- SHAP summary plots

### Reports
Detailed evaluation reports saved in `logs/`:
- Performance metrics
- Model comparison
- Clinical interpretation
- Recommendations

## 🚨 Important Disclaimers

### Medical Disclaimer
**This tool is for educational and preventive awareness purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical decisions.**

### Limitations
- Based on the UCI Heart Disease dataset (Cleveland Clinic)
- May not represent all populations
- Requires clinical validation before medical use
- Should not be used for actual medical diagnosis
- Performance may vary with different populations

### Data Privacy
- No patient data is stored permanently
- All predictions are processed in memory only
- Audit logs contain only anonymized metadata
- Comply with applicable privacy regulations

## 🛠️ Development

### Project Architecture
The system follows a modular architecture:

1. **Data Layer**: Processing and validation
2. **ML Layer**: Training and prediction
3. **Explanation Layer**: SHAP and LLM integration
4. **API Layer**: FastAPI with middleware
5. **Utilities**: Helpers and constants

### Contributing
1. Fork the repository
2. Create a feature branch
3. Follow the existing code style
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

### Code Quality
- Type hints for all functions
- Comprehensive logging
- Error handling
- Input validation
- Security best practices

## 📚 References

- **Dataset**: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- **SHAP**: [Explainable AI Library](https://github.com/slundberg/shap)
- **FastAPI**: [Modern Python Web Framework](https://fastapi.tiangolo.com/)
- **Scikit-learn**: [Machine Learning Library](https://scikit-learn.org/)

## 📞 Support

For questions, issues, or contributions:
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check the API docs at `/docs` endpoint
- **Architecture**: See `claude.md` for detailed specifications

---

**Remember**: This is an educational tool. Always consult healthcare professionals for medical advice.