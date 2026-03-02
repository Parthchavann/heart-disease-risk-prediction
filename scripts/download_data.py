"""
Script to download and prepare the UCI Heart Disease dataset.
"""

import os
import logging
import requests
import pandas as pd
from typing import None
from config.settings import settings
from config.logging_config import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)


def download_uci_heart_disease_data() -> None:
    """Download the UCI Heart Disease dataset."""

    # Create directories if they don't exist
    os.makedirs(settings.RAW_DATA_DIR, exist_ok=True)

    # Define file path
    file_path = os.path.join(settings.RAW_DATA_DIR, "heart_disease.data")

    # Check if file already exists
    if os.path.exists(file_path):
        logger.info(f"Dataset already exists at {file_path}")
        return

    try:
        logger.info(f"Downloading UCI Heart Disease dataset from {settings.UCI_HEART_DISEASE_URL}")

        # Download the dataset
        response = requests.get(settings.UCI_HEART_DISEASE_URL, timeout=30)
        response.raise_for_status()

        # Save the raw data
        with open(file_path, 'w') as f:
            f.write(response.text)

        logger.info(f"Dataset downloaded successfully to {file_path}")

        # Create a CSV version with proper column names
        create_csv_with_headers(file_path)

    except Exception as e:
        logger.error(f"Failed to download dataset: {str(e)}")
        raise


def create_csv_with_headers(data_file_path: str) -> None:
    """Create a CSV file with proper column headers."""

    # Define column names based on UCI documentation
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]

    try:
        # Read the raw data
        df = pd.read_csv(data_file_path, header=None, names=column_names)

        # Replace missing values marked as '?' with NaN
        df = df.replace('?', pd.NA)

        # Save as CSV with headers
        csv_path = os.path.join(settings.RAW_DATA_DIR, "heart_disease.csv")
        df.to_csv(csv_path, index=False)

        logger.info(f"CSV file created with headers: {csv_path}")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Missing values per column:\n{df.isnull().sum()}")

    except Exception as e:
        logger.error(f"Failed to create CSV with headers: {str(e)}")
        raise


if __name__ == "__main__":
    download_uci_heart_disease_data()