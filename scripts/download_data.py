"""
Script to download and prepare the UCI Heart Disease dataset.
Downloads all 4 location datasets (Cleveland, Hungarian, Switzerland, VA Long Beach)
and combines them for a larger, more representative training set (~920 samples).
"""

import os
import logging
import requests
import pandas as pd
from typing import Optional
from config.settings import settings
from config.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

COLUMN_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

UCI_DATASETS = {
    'cleveland':  'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
    'hungarian':  'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data',
    'switzerland':'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data',
    'va':         'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data',
}


def download_single_dataset(name: str, url: str) -> Optional[pd.DataFrame]:
    """Download one UCI dataset and return as DataFrame."""
    file_path = os.path.join(settings.RAW_DATA_DIR, f"heart_disease_{name}.data")

    if not os.path.exists(file_path):
        logger.info(f"Downloading {name} dataset from {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(file_path, 'w') as f:
                f.write(response.text)
            logger.info(f"Saved {name} dataset to {file_path}")
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
            return None
    else:
        logger.info(f"{name} dataset already exists, skipping download")

    try:
        df = pd.read_csv(file_path, header=None, names=COLUMN_NAMES)
        df = df.replace('?', pd.NA)
        df['source'] = name
        logger.info(f"  {name}: {len(df)} rows loaded")
        return df
    except Exception as e:
        logger.error(f"Failed to parse {name}: {e}")
        return None


def download_uci_heart_disease_data() -> None:
    """Download all 4 UCI Heart Disease datasets and combine into one CSV."""
    os.makedirs(settings.RAW_DATA_DIR, exist_ok=True)

    csv_path = os.path.join(settings.RAW_DATA_DIR, "heart_disease.csv")

    frames = []
    for name, url in UCI_DATASETS.items():
        df = download_single_dataset(name, url)
        if df is not None:
            frames.append(df)

    if not frames:
        raise RuntimeError("Failed to download any dataset.")

    combined = pd.concat(frames, ignore_index=True)

    # Drop the helper 'source' column before saving
    combined = combined.drop(columns=['source'])

    combined.to_csv(csv_path, index=False)

    total = len(combined)
    logger.info(f"Combined dataset saved: {csv_path}  ({total} rows from {len(frames)} locations)")
    logger.info(f"Missing values per column:\n{combined.isnull().sum()}")
    logger.info(f"Target distribution:\n{combined['target'].value_counts()}")


if __name__ == "__main__":
    download_uci_heart_disease_data()
