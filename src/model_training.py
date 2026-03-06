"""
Machine learning model training module for Heart Disease Risk Prediction.
Implements multiple algorithms with hyperparameter tuning and model comparison.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import joblib
from datetime import datetime

# Sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from config.settings import settings
from config.logging_config import get_logger

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

logger = get_logger(__name__)


class ModelTrainer:
    """Handles model training, hyperparameter tuning, and model selection."""

    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.training_results = {}
        self.cv_results = {}

    def get_model_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Get model configurations with hyperparameter grids."""

        configs = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=settings.RANDOM_STATE, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=settings.RANDOM_STATE),
                'params': {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=settings.RANDOM_STATE),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'svm': {
                'model': SVC(random_state=settings.RANDOM_STATE, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
                }
            }
        }

        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            configs['xgboost'] = {
                'model': xgb.XGBClassifier(
                    random_state=settings.RANDOM_STATE,
                    eval_metric='logloss'
                ),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 4, 5, 6],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            }

        return configs

    def load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load preprocessed training data."""

        train_path = os.path.join(settings.PROCESSED_DATA_DIR, "train.csv")
        val_path = os.path.join(settings.PROCESSED_DATA_DIR, "val.csv")
        test_path = os.path.join(settings.PROCESSED_DATA_DIR, "test.csv")

        if not all(os.path.exists(path) for path in [train_path, val_path, test_path]):
            raise FileNotFoundError(
                "Processed data not found. Please run data processing pipeline first."
            )

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        logger.info(f"Loaded training data: {train_df.shape}, validation: {val_df.shape}, test: {test_df.shape}")

        return train_df, val_df, test_df

    def prepare_data_for_training(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training by separating features and targets."""

        # Combine training and validation for hyperparameter tuning
        combined_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

        X = combined_df.drop('target', axis=1).values
        y = combined_df['target'].values

        # Also prepare validation set separately for final evaluation
        X_val = val_df.drop('target', axis=1).values
        y_val = val_df['target'].values

        logger.info(f"Prepared data for training: X shape {X.shape}, y distribution {np.bincount(y)}")

        return X, y, X_val, y_val

    def train_baseline_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train baseline models without hyperparameter tuning."""

        logger.info("Training baseline models...")

        baseline_results = {}
        configs = self.get_model_configurations()

        for model_name, config in configs.items():
            logger.info(f"Training baseline {model_name}...")

            model = config['model']

            # Perform cross-validation
            cv_scores = cross_val_score(
                model, X, y,
                cv=StratifiedKFold(n_splits=settings.CV_FOLDS, shuffle=True, random_state=settings.RANDOM_STATE),
                scoring='roc_auc'
            )

            # Fit model on full training data
            model.fit(X, y)

            baseline_results[model_name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            }

            logger.info(f"{model_name} baseline - ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return baseline_results

    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform hyperparameter tuning for all models."""

        logger.info("Starting hyperparameter tuning...")

        tuned_results = {}
        configs = self.get_model_configurations()

        cv = StratifiedKFold(n_splits=settings.CV_FOLDS, shuffle=True, random_state=settings.RANDOM_STATE)

        for model_name, config in configs.items():
            logger.info(f"Tuning {model_name}...")

            # Create parameter grid
            param_grid = config['params']
            base_model = config['model']

            # Perform grid search
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(X, y)

            tuned_results[model_name] = {
                'best_estimator': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': {
                    'mean_scores': grid_search.cv_results_['mean_test_score'].tolist(),
                    'std_scores': grid_search.cv_results_['std_test_score'].tolist()
                }
            }

            logger.info(f"{model_name} tuned - Best ROC-AUC: {grid_search.best_score_:.4f}")
            logger.info(f"{model_name} best params: {grid_search.best_params_}")

        return tuned_results

    def evaluate_model(self, model: Any, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Evaluate a model on validation data."""

        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba)
        }

        return metrics

    def select_best_model(self, tuned_results: Dict[str, Any], X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Any, str]:
        """Select the best model based on validation performance."""

        logger.info("Selecting best model...")

        best_score = -1
        best_model = None
        best_model_name = None

        model_comparison = {}

        for model_name, results in tuned_results.items():
            model = results['best_estimator']

            # Evaluate on validation set
            val_metrics = self.evaluate_model(model, X_val, y_val)

            model_comparison[model_name] = {
                'cv_score': results['best_score'],
                'val_metrics': val_metrics,
                'best_params': results['best_params']
            }

            # Use ROC-AUC as the primary metric for model selection
            current_score = val_metrics['roc_auc']

            if current_score > best_score:
                best_score = current_score
                best_model = model
                best_model_name = model_name

            logger.info(f"{model_name} validation metrics: {val_metrics}")

        logger.info(f"Best model: {best_model_name} with ROC-AUC: {best_score:.4f}")

        self.model_comparison_results = model_comparison

        return best_model, best_model_name

    def get_feature_importance(self, model: Any, feature_names: List[str] = None) -> Dict[str, float]:
        """Extract feature importance from the model."""

        importance_dict = {}

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            logger.warning("Model does not have feature importance information")
            return importance_dict

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        # Sort features by importance
        indices = np.argsort(importances)[::-1]

        for i in indices:
            importance_dict[feature_names[i]] = float(importances[i])

        return importance_dict

    def save_model(self, model: Any, model_name: str, metrics: Dict[str, Any],
                   feature_names: List[str] = None) -> str:
        """Save the trained model and its metadata."""

        # Create models directory
        os.makedirs(settings.MODELS_DIR, exist_ok=True)

        # Create model filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.pkl"
        model_path = os.path.join(settings.MODELS_DIR, model_filename)

        # Save model
        joblib.dump(model, model_path)

        # Save metadata
        metadata = {
            'model_name': model_name,
            'model_version': settings.MODEL_VERSION,
            'training_timestamp': timestamp,
            'model_file': model_filename,
            'metrics': metrics,
            'feature_importance': self.get_feature_importance(model, feature_names),
            'training_config': {
                'random_state': settings.RANDOM_STATE,
                'cv_folds': settings.CV_FOLDS
            }
        }

        metadata_filename = f"{model_name}_{timestamp}_metadata.json"
        metadata_path = os.path.join(settings.MODELS_DIR, metadata_filename)

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Also save as latest model
        latest_model_path = os.path.join(settings.MODELS_DIR, "best_model.pkl")
        latest_metadata_path = os.path.join(settings.MODELS_DIR, "best_model_metadata.json")

        joblib.dump(model, latest_model_path)
        with open(latest_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved: {model_path}")
        logger.info(f"Metadata saved: {metadata_path}")

        return model_path

    def optuna_tune_xgboost(self, X: np.ndarray, y: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            n_trials: int = 100) -> Dict[str, Any]:
        """Tune XGBoost using Optuna Bayesian optimisation."""

        if not OPTUNA_AVAILABLE or not XGBOOST_AVAILABLE:
            logger.warning("Optuna or XGBoost not available — skipping Optuna tuning.")
            return {}

        logger.info(f"Starting Optuna tuning for XGBoost ({n_trials} trials)...")

        cv = StratifiedKFold(n_splits=settings.CV_FOLDS, shuffle=True,
                             random_state=settings.RANDOM_STATE)

        def objective(trial):
            params = {
                'n_estimators':      trial.suggest_int('n_estimators', 50, 500),
                'max_depth':         trial.suggest_int('max_depth', 2, 8),
                'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_child_weight':  trial.suggest_int('min_child_weight', 1, 10),
                'gamma':             trial.suggest_float('gamma', 0, 5),
                'random_state':      settings.RANDOM_STATE,
                'eval_metric':       'logloss',
                'use_label_encoder': False,
            }
            model = xgb.XGBClassifier(**params)
            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=settings.RANDOM_STATE))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_params.update({'random_state': settings.RANDOM_STATE,
                            'eval_metric': 'logloss', 'use_label_encoder': False})

        logger.info(f"Optuna best CV ROC-AUC: {study.best_value:.4f}")
        logger.info(f"Optuna best params: {best_params}")

        # Train final model on full data with best params
        best_model = xgb.XGBClassifier(**best_params)
        best_model.fit(X, y)

        val_metrics = self.evaluate_model(best_model, X_val, y_val)
        logger.info(f"Optuna XGBoost validation metrics: {val_metrics}")

        return {
            'best_estimator': best_model,
            'best_params': best_params,
            'best_score': study.best_value,
            'val_metrics': val_metrics,
        }

    def train_complete_pipeline(self) -> Dict[str, Any]:
        """Execute the complete training pipeline."""

        logger.info("Starting complete training pipeline...")

        # Load data
        train_df, val_df, test_df = self.load_training_data()
        X, y, X_val, y_val = self.prepare_data_for_training(train_df, val_df)

        # Train baseline models
        baseline_results = self.train_baseline_models(X, y)

        # Hyperparameter tuning (GridSearchCV for all models)
        tuned_results = self.hyperparameter_tuning(X, y)

        # Select best model from GridSearchCV results
        best_model, best_model_name = self.select_best_model(tuned_results, X_val, y_val)

        # Optuna tuning for XGBoost (Bayesian search — finds better params)
        optuna_result = self.optuna_tune_xgboost(X, y, X_val, y_val, n_trials=100)
        if optuna_result:
            optuna_val_auc = optuna_result['val_metrics']['roc_auc']
            grid_val_auc = self.evaluate_model(best_model, X_val, y_val)['roc_auc']
            if optuna_val_auc > grid_val_auc:
                logger.info(f"Optuna XGBoost ({optuna_val_auc:.4f}) beats GridSearch best "
                            f"({grid_val_auc:.4f}) — using Optuna model.")
                best_model = optuna_result['best_estimator']
                best_model_name = 'xgboost_optuna'
            else:
                logger.info(f"GridSearch best ({grid_val_auc:.4f}) >= Optuna "
                            f"({optuna_val_auc:.4f}) — keeping GridSearch model.")

        # Get final validation metrics
        final_metrics = self.evaluate_model(best_model, X_val, y_val)

        # Capture feature names from training data for proper importance labelling
        feature_names = [c for c in train_df.columns if c != 'target']

        # Save best model (pass feature names so importance dict uses real names)
        model_path = self.save_model(best_model, best_model_name, final_metrics,
                                     feature_names=feature_names)

        # Store results
        self.best_model = best_model
        self.best_model_name = best_model_name
        self.training_results = {
            'baseline_results': baseline_results,
            'tuned_results': tuned_results,
            'model_comparison': self.model_comparison_results,
            'best_model_name': best_model_name,
            'final_metrics': final_metrics,
            'model_path': model_path
        }

        logger.info("Training pipeline completed successfully")
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Final validation metrics: {final_metrics}")

        return self.training_results

    def load_best_model(self) -> Tuple[Any, Dict[str, Any]]:
        """Load the best saved model."""

        model_path = os.path.join(settings.MODELS_DIR, "best_model.pkl")
        metadata_path = os.path.join(settings.MODELS_DIR, "best_model_metadata.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError("No trained model found. Please train a model first.")

        # Load model
        model = joblib.load(model_path)

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        logger.info(f"Loaded model: {metadata['model_name']}")
        logger.info(f"Model metrics: {metadata['metrics']}")

        return model, metadata


def main():
    """Main training script."""

    from config.logging_config import setup_logging
    setup_logging()

    trainer = ModelTrainer()
    results = trainer.train_complete_pipeline()

    print("\n" + "="*50)
    print("TRAINING RESULTS SUMMARY")
    print("="*50)
    print(f"Best Model: {results['best_model_name']}")
    print(f"Final Metrics: {results['final_metrics']}")
    print(f"Model Path: {results['model_path']}")


if __name__ == "__main__":
    main()