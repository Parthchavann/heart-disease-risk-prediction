"""
Script to download and prepare the UCI Heart Disease dataset.
Downloads all 4 UCI location datasets + Statlog Heart dataset and generates
CTGAN synthetic samples for a larger, more representative training set.

Data sources:
  - UCI Cleveland, Hungarian, Switzerland, VA Long Beach (~920 real samples)
  - UCI Statlog Heart (~270 real samples, same features)
  - CTGAN synthetic samples generated from all real data combined
"""

import os
import logging
import requests
import numpy as np
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

# Statlog Heart dataset — space-separated, same 13 features, target 1=absent 2=present
STATLOG_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat'

# Number of CTGAN synthetic samples to generate
SYNTHETIC_SAMPLES = 1500


def normalize_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise feature encodings to 0-indexed Cleveland standard.

    Raw UCI files use 1-indexed values for several fields:
      cp    : 1-4  →  0-3  (subtract 1 if any value > 3)
      slope : 1-3  →  0-2  (subtract 1 if any value > 2)
      thal  : Cleveland uses 3/6/7; others may use 1/2/3 or 0/1/2
               Map 3→0 (normal), 6→1 (fixed), 7→2 (reversible)
               If already in 0-3 range leave untouched.
    """
    df = df.copy()

    # cp: 1-4 → 0-3
    if pd.to_numeric(df['cp'], errors='coerce').max() > 3:
        df['cp'] = pd.to_numeric(df['cp'], errors='coerce') - 1

    # slope: 1-3 → 0-2
    if pd.to_numeric(df['slope'], errors='coerce').max() > 2:
        df['slope'] = pd.to_numeric(df['slope'], errors='coerce') - 1

    # thal: 3/6/7 → 0/1/2 (Cleveland original encoding)
    thal_num = pd.to_numeric(df['thal'], errors='coerce')
    if thal_num.isin([6, 7]).any():
        thal_map = {3: 0, 6: 1, 7: 2}
        df['thal'] = thal_num.map(thal_map).fillna(thal_num)

    return df


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
        df = normalize_encodings(df)
        df['source'] = name
        logger.info(f"  {name}: {len(df)} rows loaded")
        return df
    except Exception as e:
        logger.error(f"Failed to parse {name}: {e}")
        return None


def download_statlog_dataset() -> Optional[pd.DataFrame]:
    """Download UCI Statlog Heart dataset (270 samples, same features as Cleveland)."""
    file_path = os.path.join(settings.RAW_DATA_DIR, "heart_disease_statlog.data")

    if not os.path.exists(file_path):
        logger.info(f"Downloading Statlog Heart dataset from {STATLOG_URL}")
        try:
            response = requests.get(STATLOG_URL, timeout=30)
            response.raise_for_status()
            with open(file_path, 'w') as f:
                f.write(response.text)
            logger.info(f"Saved Statlog dataset to {file_path}")
        except Exception as e:
            logger.error(f"Failed to download Statlog: {e}")
            return None

    try:
        df = pd.read_csv(file_path, sep=r'\s+', header=None, names=COLUMN_NAMES)
        # Statlog target: 1=absent → 0, 2=present → 1
        df['target'] = (df['target'] == 2).astype(int)
        df = normalize_encodings(df)
        df['source'] = 'statlog'
        logger.info(f"  statlog: {len(df)} rows loaded")
        return df
    except Exception as e:
        logger.error(f"Failed to parse Statlog: {e}")
        return None


def generate_synthetic_data(real_df: pd.DataFrame, n_samples: int = SYNTHETIC_SAMPLES) -> pd.DataFrame:
    """Generate synthetic heart disease samples using CTGAN.

    CTGAN (Conditional Tabular GAN) learns the joint distribution of the real
    data and generates realistic synthetic rows that preserve feature correlations
    and class balance.
    """
    try:
        from ctgan import CTGAN

        logger.info(f"Training CTGAN on {len(real_df)} real samples...")

        # Work on a clean copy with only the 14 model columns
        train_df = real_df[COLUMN_NAMES].copy()

        # Convert target to int for CTGAN
        train_df['target'] = train_df['target'].astype(int)

        # CTGAN does not support NaN in continuous columns — impute before fitting
        from sklearn.impute import SimpleImputer
        num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        train_df[num_cols] = SimpleImputer(strategy='median').fit_transform(train_df[num_cols])
        train_df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(train_df[cat_cols])
        train_df = train_df.astype({c: int for c in cat_cols})

        # Discrete columns — CTGAN treats these as categorical
        discrete_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']

        model = CTGAN(epochs=300, verbose=False)
        model.fit(train_df, discrete_columns=discrete_cols)

        synthetic = model.sample(n_samples)

        # Clip continuous features to valid ranges
        synthetic['age']      = synthetic['age'].clip(18, 100).round().astype(int)
        synthetic['trestbps'] = synthetic['trestbps'].clip(80, 200).round().astype(int)
        synthetic['chol']     = synthetic['chol'].clip(100, 600).round().astype(int)
        synthetic['thalach']  = synthetic['thalach'].clip(60, 220).round().astype(int)
        synthetic['oldpeak']  = synthetic['oldpeak'].clip(0, 10).round(1)
        synthetic['target']   = synthetic['target'].astype(int).clip(0, 1)

        logger.info(f"Generated {len(synthetic)} synthetic samples")
        logger.info(f"Synthetic target distribution:\n{synthetic['target'].value_counts()}")

        synthetic['source'] = 'synthetic_ctgan'
        return synthetic

    except Exception as e:
        logger.error(f"CTGAN generation failed: {e}")
        return pd.DataFrame()


def download_uci_heart_disease_data() -> None:
    """Download all datasets, generate synthetic data, and combine into one CSV."""
    os.makedirs(settings.RAW_DATA_DIR, exist_ok=True)

    csv_path = os.path.join(settings.RAW_DATA_DIR, "heart_disease.csv")

    # --- Real datasets ---
    frames = []
    for name, url in UCI_DATASETS.items():
        df = download_single_dataset(name, url)
        if df is not None:
            frames.append(df)

    statlog_df = download_statlog_dataset()
    if statlog_df is not None:
        frames.append(statlog_df)

    if not frames:
        raise RuntimeError("Failed to download any dataset.")

    real_combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Total real samples: {len(real_combined)}")

    # --- Synthetic data ---
    real_for_ctgan = real_combined.drop(columns=['source'])
    synthetic_df = generate_synthetic_data(real_for_ctgan, n_samples=SYNTHETIC_SAMPLES)

    # --- Combine real + synthetic ---
    all_frames = [real_combined]
    if not synthetic_df.empty:
        all_frames.append(synthetic_df)

    combined = pd.concat(all_frames, ignore_index=True)
    # Keep 'source' column so data_processing.py can restrict synthetic rows to train only
    combined.to_csv(csv_path, index=False)

    total = len(combined)
    real_count = len(real_combined)
    synth_count = len(synthetic_df) if not synthetic_df.empty else 0

    logger.info(f"Combined dataset saved: {csv_path}")
    logger.info(f"  Real samples  : {real_count}")
    logger.info(f"  Synthetic     : {synth_count}")
    logger.info(f"  Total         : {total}")
    logger.info(f"Missing values per column:\n{combined.isnull().sum()}")
    logger.info(f"Target distribution:\n{combined['target'].value_counts()}")


if __name__ == "__main__":
    download_uci_heart_disease_data()
