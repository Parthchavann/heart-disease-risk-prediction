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
        """Clip numerical features to medically valid hard bounds.

        Uses domain-specific ranges instead of statistical IQR removal so that
        patients from non-Cleveland populations (different distributions) are
        retained. Values outside the bounds are clipped, not dropped.
        """
        from utils.constants import FEATURE_RANGES

        df_clipped = df.copy()

        for feature in NUMERICAL_FEATURES:
            if feature in df_clipped.columns and feature in FEATURE_RANGES:
                lo = FEATURE_RANGES[feature]['min']
                hi = FEATURE_RANGES[feature]['max']
                clipped = df_clipped[feature].clip(lo, hi)
                changed = (clipped != df_clipped[feature]).sum()
                if changed:
                    logger.info(f"Clipped {changed} out-of-range values in '{feature}' to [{lo}, {hi}]")
                df_clipped[feature] = clipped

        logger.info(f"Outlier clipping complete — retained all {len(df_clipped)} rows")
        return df_clipped

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

        # --- Interaction features ---
        # Age × cholesterol (older + higher chol = compounding risk)
        df_engineered['age_chol'] = (
            df_engineered['age'] * df_engineered['chol'] / 10000.0
        )

        # Blood-pressure-to-heart-rate ratio (haemodynamic stress proxy)
        df_engineered['bp_hr_ratio'] = (
            df_engineered['trestbps'] / (df_engineered['thalach'] + 1e-6)
        )

        # ST depression amplified by exercise-induced angina
        df_engineered['oldpeak_exang'] = (
            df_engineered['oldpeak'] * df_engineered['exang']
        )

        # Number of major vessels × thalassemia type (combined structural risk)
        df_engineered['ca_thal_risk'] = (
            df_engineered['ca'] * df_engineered['thal']
        )

        # Chest pain type × exercise angina (symptom severity combo)
        df_engineered['cp_exang_combo'] = (
            df_engineered['cp'] * (df_engineered['exang'] + 1)
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
                        le = self.label_encoders[feature]
                        # Handle unseen labels by mapping to nearest known class
                        known = set(le.classes_)
                        features[feature] = features[feature].apply(
                            lambda v: le.transform([v])[0] if v in known
                            else le.transform([min(le.classes_, key=lambda c: abs(c - v))])[0]
                        )

        # Handle engineered categorical features
        if 'age_group' in features.columns:
            if fit_transformers:
                if 'age_group' not in self.label_encoders:
                    self.label_encoders['age_group'] = LabelEncoder()
                features['age_group'] = self.label_encoders['age_group'].fit_transform(features['age_group'])
            else:
                if 'age_group' in self.label_encoders:
                    le = self.label_encoders['age_group']
                    known = set(le.classes_)
                    features['age_group'] = features['age_group'].apply(
                        lambda v: le.transform([v])[0] if v in known else le.transform([le.classes_[0]])[0]
                    )

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
        """Split data into train, validation, and test sets.

        If a 'source' column is present (set by download_data.py), synthetic rows
        (source == 'synthetic_ctgan') are kept exclusively in the training set so that
        val/test metrics are computed only on real clinical data.
        """

        logger.info("Splitting data...")

        # Separate synthetic from real if source column is available
        has_source = 'source' in df.columns
        if has_source:
            synthetic_mask = df['source'] == 'synthetic_ctgan'
            synthetic_df = df[synthetic_mask].drop(columns=['source'])
            real_df = df[~synthetic_mask].drop(columns=['source'])
            logger.info(f"  Real samples: {len(real_df)}, Synthetic (train-only): {len(synthetic_df)}")
        else:
            real_df = df
            synthetic_df = pd.DataFrame(columns=df.columns)

        # Split real data into train+val and test
        X_real = real_df.drop('target', axis=1)
        y_real = real_df['target']

        X_temp, X_test, y_temp, y_test = train_test_split(
            X_real, y_real,
            test_size=settings.TEST_SIZE,
            random_state=settings.RANDOM_STATE,
            stratify=y_real
        )

        # Second split: train and validation (from real data only)
        val_size_adjusted = settings.VAL_SIZE / (1 - settings.TEST_SIZE)
        X_train_real, X_val, y_train_real, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=settings.RANDOM_STATE,
            stratify=y_temp
        )

        # Combine real train portion with all synthetic rows
        train_real_df = pd.concat([X_train_real, y_train_real], axis=1)
        if not synthetic_df.empty:
            train_df = pd.concat([train_real_df, synthetic_df], ignore_index=True)
        else:
            train_df = train_real_df

        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        logger.info(f"Data split completed:")
        logger.info(f"  Train: {train_df.shape[0]} samples (real + synthetic)")
        logger.info(f"  Validation: {val_df.shape[0]} samples (real only)")
        logger.info(f"  Test: {test_df.shape[0]} samples (real only)")

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

        # Apply feature engineering (same as training pipeline)
        df = self.engineer_features(df)

        # Convert categorical dtype to string to avoid LabelEncoder issues
        if 'age_group' in df.columns:
            df['age_group'] = df['age_group'].astype(str)

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