"""
Script to train the heart disease prediction model.
Downloads data, processes it, and trains the model.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.logging_config import setup_logging, get_logger
from scripts.download_data import download_uci_heart_disease_data
from src.data_processing import main as process_data
from src.model_training import ModelTrainer
from utils.validators import validate_training_data
from config.settings import settings

# Initialize logging
setup_logging()
logger = get_logger(__name__)


def main():
    """Complete training pipeline from data download to model training."""

    logger.info("Starting complete training pipeline...")

    try:
        # Step 1: Download data
        logger.info("Step 1: Downloading UCI Heart Disease dataset...")
        download_uci_heart_disease_data()

        # Step 2: Process data
        logger.info("Step 2: Processing data...")
        process_data()

        # Step 3: Validate processed data
        logger.info("Step 3: Validating processed data...")
        train_path = os.path.join(settings.PROCESSED_DATA_DIR, "train.csv")
        val_path = os.path.join(settings.PROCESSED_DATA_DIR, "val.csv")
        test_path = os.path.join(settings.PROCESSED_DATA_DIR, "test.csv")

        if not validate_training_data(train_path, val_path, test_path):
            raise ValueError("Data validation failed")

        # Step 4: Train model
        logger.info("Step 4: Training models...")
        trainer = ModelTrainer()
        results = trainer.train_complete_pipeline()

        # Step 5: Display results
        logger.info("Step 5: Training completed successfully!")
        print("\n" + "="*60)
        print("HEART DISEASE PREDICTION MODEL TRAINING COMPLETE")
        print("="*60)
        print(f"Best Model: {results['best_model_name']}")
        print(f"Validation ROC-AUC: {results['final_metrics']['roc_auc']:.4f}")
        print(f"Validation Accuracy: {results['final_metrics']['accuracy']:.4f}")
        print(f"Model saved to: {results['model_path']}")
        print("="*60)

    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()