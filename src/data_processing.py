"""
Data processing module for the Heart Disease Risk Prediction system.
Handles data loading, cleaning, preprocessing, and feature engineering.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import logging

from config.settings import settings
from config.logging_config import get_logger
from utils.constants import FEATURE_NAMES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES

logger = get_logger(__name__)


class DataProcessor:
    """Handles all data preprocessing operations."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.imputer_numerical = SimpleImputer(strategy='median')
        self.imputer_categorical = SimpleImputer(strategy='most_frequent')
        self.is_fitted = False

    def load_raw_data(self) -> pd.DataFrame:
        """Load the raw UCI Heart Disease dataset."""

        data_path = os.path.join(settings.RAW_DATA_DIR, "heart_disease.csv")

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Dataset not found at {data_path}. "
                "Please run scripts/download_data.py first."
            )

        try:
            df = pd.read_csv(data_path)
            logger.info(f"Loaded raw data with shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the raw dataset."""

        logger.info("Starting data cleaning...")

        # Create a copy to avoid modifying the original
        df_clean = df.copy()

        # Log initial data info
        logger.info(f"Initial shape: {df_clean.shape}")
        logger.info(f"Missing values:\n{df_clean.isnull().sum()}")

        # Convert target to binary (0: no disease, 1: disease)
        # Original dataset has values 0-4, where 0 means no disease
        df_clean['target'] = (df_clean['target'] > 0).astype(int)

        # Handle missing values for numerical features
        if df_clean[NUMERICAL_FEATURES].isnull().any().any():
            logger.info("Imputing missing values for numerical features")
            df_clean[NUMERICAL_FEATURES] = self.imputer_numerical.fit_transform(
                df_clean[NUMERICAL_FEATURES]
            )

        # Handle missing values for categorical features
        categorical_with_missing = [col for col in CATEGORICAL_FEATURES
                                  if col in df_clean.columns and df_clean[col].isnull().any()]

        if categorical_with_missing:
            logger.info(f"Imputing missing values for categorical features: {categorical_with_missing}")
            df_clean[categorical_with_missing] = self.imputer_categorical.fit_transform(
                df_clean[categorical_with_missing]
            )

        # Remove outliers using IQR method for numerical features
        df_clean = self._remove_outliers(df_clean)

        # Validate feature ranges
        df_clean = self._validate_feature_ranges(df_clean)

        logger.info(f"Cleaned data shape: {df_clean.shape}")
        logger.info("Data cleaning completed")

        return df_clean

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method."""

        df_no_outliers = df.copy()
        initial_rows = len(df_no_outliers)

        for feature in NUMERICAL_FEATURES:
            if feature in df_no_outliers.columns:
                Q1 = df_no_outliers[feature].quantile(0.25)
                Q3 = df_no_outliers[feature].quantile(0.75)
                IQR = Q3 - Q1

                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Remove outliers
                mask = (df_no_outliers[feature] >= lower_bound) & (df_no_outliers[feature] <= upper_bound)
                df_no_outliers = df_no_outliers[mask]

        removed_rows = initial_rows - len(df_no_outliers)
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} outliers ({removed_rows/initial_rows:.1%} of data)")

        return df_no_outliers

    def _validate_feature_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that features are within expected ranges."""

        from utils.constants import FEATURE_RANGES

        df_valid = df.copy()

        for feature, ranges in FEATURE_RANGES.items():
            if feature in df_valid.columns:
                min_val, max_val = ranges['min'], ranges['max']

                # Check for values outside valid ranges
                invalid_mask = (df_valid[feature] < min_val) | (df_valid[feature] > max_val)
                invalid_count = invalid_mask.sum()

                if invalid_count > 0:
                    logger.warning(f"Found {invalid_count} invalid values for {feature}")
                    # Clip values to valid range
                    df_valid[feature] = df_valid[feature].clip(min_val, max_val)

        return df_valid

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features from the existing ones."""

        logger.info("Starting feature engineering...")

        df_engineered = df.copy()

        # Age groups
        df_engineered['age_group'] = pd.cut(
            df_engineered['age'],
            bins=[0, 40, 55, 70, 100],
            labels=['young', 'middle', 'senior', 'elderly']
        )

        # BMI proxy (not perfect but useful)
        # Using a simplified calculation since we don't have height/weight
        df_engineered['bp_chol_ratio'] = df_engineered['trestbps'] / df_engineered['chol']

        # Heart rate reserve (max HR - resting, approximated)
        estimated_max_hr = 220 - df_engineered['age']
        df_engineered['hr_reserve'] = df_engineered['thalach'] / estimated_max_hr

        # Risk indicators combination
        df_engineered['multiple_risk_factors'] = (
            (df_engineered['fbs'] == 1).astype(int) +
            (df_engineered['exang'] == 1).astype(int) +
            (df_engineered['chol'] > 240).astype(int) +
            (df_engineered['trestbps'] > 140).astype(int)
        )

        logger.info(f"Feature engineering completed. New shape: {df_engineered.shape}")

        return df_engineered

    def preprocess_features(self, df: pd.DataFrame, fit_transformers: bool = True) -> pd.DataFrame:
        """Preprocess features for machine learning."""

        logger.info("Starting feature preprocessing...")

        df_processed = df.copy()

        # Separate features and target
        if 'target' in df_processed.columns:
            target = df_processed['target']
            features = df_processed.drop('target', axis=1)
        else:
            features = df_processed.copy()

        # Handle categorical features with label encoding
        for feature in CATEGORICAL_FEATURES:
            if feature in features.columns:
                if fit_transformers:
                    if feature not in self.label_encoders:
                        self.label_encoders[feature] = LabelEncoder()
                    features[feature] = self.label_encoders[feature].fit_transform(features[feature])
                else:
                    if feature in self.label_encoders:
                        features[feature] = self.label_encoders[feature].transform(features[feature])

        # Handle engineered categorical features
        if 'age_group' in features.columns:
            if fit_transformers:
                if 'age_group' not in self.label_encoders:
                    self.label_encoders['age_group'] = LabelEncoder()
                features['age_group'] = self.label_encoders['age_group'].fit_transform(features['age_group'])
            else:
                if 'age_group' in self.label_encoders:
                    features['age_group'] = self.label_encoders['age_group'].transform(features['age_group'])

        # Scale numerical features
        numerical_cols = features.select_dtypes(include=[np.number]).columns

        if fit_transformers:
            features[numerical_cols] = self.scaler.fit_transform(features[numerical_cols])
            self.is_fitted = True
        else:
            if self.is_fitted:
                features[numerical_cols] = self.scaler.transform(features[numerical_cols])
            else:
                logger.warning("Transformers not fitted yet. Fitting on current data.")
                features[numerical_cols] = self.scaler.fit_transform(features[numerical_cols])
                self.is_fitted = True

        # Reconstruct DataFrame with target if it existed
        if 'target' in df_processed.columns:
            df_processed = pd.concat([features, target], axis=1)
        else:
            df_processed = features

        logger.info("Feature preprocessing completed")

        return df_processed

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets."""

        logger.info("Splitting data...")

        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']

        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=settings.TEST_SIZE,
            random_state=settings.RANDOM_STATE,
            stratify=y
        )

        # Second split: train and validation
        val_size_adjusted = settings.VAL_SIZE / (1 - settings.TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=settings.RANDOM_STATE,
            stratify=y_temp
        )

        # Create DataFrames
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        logger.info(f"Data split completed:")
        logger.info(f"  Train: {train_df.shape[0]} samples")
        logger.info(f"  Validation: {val_df.shape[0]} samples")
        logger.info(f"  Test: {test_df.shape[0]} samples")

        return train_df, val_df, test_df

    def save_processed_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                           test_df: pd.DataFrame) -> None:
        """Save processed datasets and transformers."""

        # Create processed data directory
        os.makedirs(settings.PROCESSED_DATA_DIR, exist_ok=True)

        # Save datasets
        train_df.to_csv(os.path.join(settings.PROCESSED_DATA_DIR, "train.csv"), index=False)
        val_df.to_csv(os.path.join(settings.PROCESSED_DATA_DIR, "val.csv"), index=False)
        test_df.to_csv(os.path.join(settings.PROCESSED_DATA_DIR, "test.csv"), index=False)

        # Save transformers
        transformers_dir = os.path.join(settings.PROCESSED_DATA_DIR, "transformers")
        os.makedirs(transformers_dir, exist_ok=True)

        joblib.dump(self.scaler, os.path.join(transformers_dir, "scaler.pkl"))
        joblib.dump(self.label_encoders, os.path.join(transformers_dir, "label_encoders.pkl"))
        joblib.dump(self.imputer_numerical, os.path.join(transformers_dir, "imputer_numerical.pkl"))
        joblib.dump(self.imputer_categorical, os.path.join(transformers_dir, "imputer_categorical.pkl"))

        logger.info("Processed data and transformers saved successfully")

    def load_transformers(self) -> None:
        """Load saved transformers."""

        transformers_dir = os.path.join(settings.PROCESSED_DATA_DIR, "transformers")

        try:
            self.scaler = joblib.load(os.path.join(transformers_dir, "scaler.pkl"))
            self.label_encoders = joblib.load(os.path.join(transformers_dir, "label_encoders.pkl"))
            self.imputer_numerical = joblib.load(os.path.join(transformers_dir, "imputer_numerical.pkl"))
            self.imputer_categorical = joblib.load(os.path.join(transformers_dir, "imputer_categorical.pkl"))
            self.is_fitted = True
            logger.info("Transformers loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load transformers: {str(e)}")
            raise

    def process_single_sample(self, sample_data: Dict[str, Any]) -> np.ndarray:
        """Process a single sample for prediction."""

        if not self.is_fitted:
            self.load_transformers()

        # Create DataFrame from sample
        df = pd.DataFrame([sample_data])

        # Apply the same preprocessing steps (without fitting)
        df_processed = self.preprocess_features(df, fit_transformers=False)

        return df_processed.values


def main():
    """Main data processing pipeline."""

    logger.info("Starting data processing pipeline...")

    # Initialize processor
    processor = DataProcessor()

    # Load and process data
    df_raw = processor.load_raw_data()
    df_clean = processor.clean_data(df_raw)
    df_engineered = processor.engineer_features(df_clean)
    df_processed = processor.preprocess_features(df_engineered, fit_transformers=True)

    # Split data
    train_df, val_df, test_df = processor.split_data(df_processed)

    # Save processed data
    processor.save_processed_data(train_df, val_df, test_df)

    logger.info("Data processing pipeline completed successfully")


if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging()
    main()