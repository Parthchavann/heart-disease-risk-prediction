"""
Utility helper functions for the Heart Disease Risk Prediction system.
"""

import os
import json
import pickle
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


def ensure_directory_exists(directory_path: str) -> None:
    """Ensure that a directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        logger.info(f"Created directory: {directory_path}")


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary data to JSON file."""
    ensure_directory_exists(os.path.dirname(filepath))

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"JSON data saved to: {filepath}")


def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    logger.info(f"JSON data loaded from: {filepath}")
    return data


def save_pickle(obj: Any, filepath: str) -> None:
    """Save object to pickle file."""
    ensure_directory_exists(os.path.dirname(filepath))

    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

    logger.info(f"Object saved to pickle: {filepath}")


def load_pickle(filepath: str) -> Any:
    """Load object from pickle file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pickle file not found: {filepath}")

    with open(filepath, 'rb') as f:
        obj = pickle.load(f)

    logger.info(f"Object loaded from pickle: {filepath}")
    return obj


def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """Format timestamp for consistent display."""
    if timestamp is None:
        timestamp = datetime.now()

    return timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")


def calculate_age_category(age: int) -> str:
    """Categorize age into groups."""
    if age < 30:
        return "young"
    elif age < 45:
        return "middle_aged"
    elif age < 65:
        return "older"
    else:
        return "elderly"


def format_probability(prob: float, decimal_places: int = 1) -> str:
    """Format probability as percentage string."""
    return f"{prob * 100:.{decimal_places}f}%"


def format_confidence_interval(ci: tuple, decimal_places: int = 1) -> str:
    """Format confidence interval as string."""
    lower, upper = ci
    return f"[{lower * 100:.{decimal_places}f}% - {upper * 100:.{decimal_places}f}%]"


def get_risk_category_color(risk_level: str) -> str:
    """Get color code for risk category."""
    color_map = {
        "Low": "#28a745",    # Green
        "Moderate": "#ffc107",  # Yellow
        "High": "#dc3545"    # Red
    }
    return color_map.get(risk_level, "#6c757d")  # Gray as default


def validate_file_exists(filepath: str, description: str = "File") -> None:
    """Validate that a file exists and is readable."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{description} not found: {filepath}")

    if not os.path.isfile(filepath):
        raise ValueError(f"{description} is not a file: {filepath}")

    if not os.access(filepath, os.R_OK):
        raise PermissionError(f"{description} is not readable: {filepath}")


def cleanup_old_files(directory: str, max_age_days: int = 30,
                     file_pattern: str = "*.log") -> int:
    """Clean up old files in a directory."""
    if not os.path.exists(directory):
        return 0

    cutoff_date = datetime.now() - timedelta(days=max_age_days)
    deleted_count = 0

    for filename in os.listdir(directory):
        if file_pattern == "*" or filename.endswith(file_pattern.replace("*", "")):
            filepath = os.path.join(directory, filename)

            if os.path.isfile(filepath):
                file_date = datetime.fromtimestamp(os.path.getmtime(filepath))

                if file_date < cutoff_date:
                    try:
                        os.remove(filepath)
                        deleted_count += 1
                        logger.info(f"Deleted old file: {filepath}")
                    except OSError as e:
                        logger.error(f"Failed to delete file {filepath}: {e}")

    if deleted_count > 0:
        logger.info(f"Cleaned up {deleted_count} old files from {directory}")

    return deleted_count


def safe_division(numerator: float, denominator: float,
                 default: float = 0.0) -> float:
    """Perform safe division with default value for zero division."""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ValueError):
        return default


def normalize_feature_name(feature_name: str) -> str:
    """Normalize feature name for display."""
    # Convert snake_case to Title Case
    normalized = feature_name.replace('_', ' ').title()

    # Handle special cases
    name_mapping = {
        'Cp': 'Chest Pain Type',
        'Trestbps': 'Resting Blood Pressure',
        'Chol': 'Cholesterol',
        'Fbs': 'Fasting Blood Sugar',
        'Restecg': 'Resting ECG',
        'Thalach': 'Max Heart Rate',
        'Exang': 'Exercise Angina',
        'Oldpeak': 'ST Depression',
        'Ca': 'Major Vessels',
        'Thal': 'Thalassemia'
    }

    return name_mapping.get(normalized, normalized)


def round_to_significant_figures(value: float, sig_figs: int = 3) -> float:
    """Round value to specified number of significant figures."""
    if value == 0:
        return 0

    return round(value, sig_figs - 1 - int(np.floor(np.log10(abs(value)))))


def create_summary_statistics(values: List[float]) -> Dict[str, float]:
    """Create summary statistics for a list of values."""
    if not values:
        return {}

    values_array = np.array(values)

    return {
        'count': len(values),
        'mean': float(np.mean(values_array)),
        'median': float(np.median(values_array)),
        'std': float(np.std(values_array)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'q25': float(np.percentile(values_array, 25)),
        'q75': float(np.percentile(values_array, 75))
    }


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def log_prediction_request(patient_data: Dict[str, Any],
                         prediction_id: str,
                         client_info: Optional[Dict[str, str]] = None) -> None:
    """Log prediction request for monitoring and audit."""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'prediction_id': prediction_id,
        'patient_data_hash': hash(str(sorted(patient_data.items()))),
        'features_count': len(patient_data),
        'client_info': client_info or {}
    }

    # Log to file for audit trail
    log_file = os.path.join(settings.LOG_DIR, "prediction_audit.log")
    ensure_directory_exists(settings.LOG_DIR)

    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


def validate_model_performance(metrics: Dict[str, float],
                             min_requirements: Dict[str, float] = None) -> bool:
    """Validate that model performance meets minimum requirements."""
    if min_requirements is None:
        min_requirements = {
            'accuracy': 0.70,
            'precision': 0.65,
            'recall': 0.65,
            'f1_score': 0.65,
            'roc_auc': 0.75
        }

    for metric, min_value in min_requirements.items():
        if metric in metrics:
            if metrics[metric] < min_value:
                logger.warning(f"Model {metric} ({metrics[metric]:.3f}) "
                             f"below minimum requirement ({min_value})")
                return False

    return True


def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging."""
    import platform
    import psutil

    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
        'disk_usage_gb': round(psutil.disk_usage('/').used / (1024**3), 2),
        'timestamp': datetime.now().isoformat()
    }


def mask_sensitive_data(data: Dict[str, Any],
                       sensitive_keys: List[str] = None) -> Dict[str, Any]:
    """Mask sensitive information in data dictionary."""
    if sensitive_keys is None:
        sensitive_keys = ['api_key', 'password', 'secret', 'token']

    masked_data = data.copy()

    for key, value in masked_data.items():
        if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
            masked_data[key] = '***MASKED***'
        elif isinstance(value, dict):
            masked_data[key] = mask_sensitive_data(value, sensitive_keys)

    return masked_data