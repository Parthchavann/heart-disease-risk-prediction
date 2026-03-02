"""
Script to run comprehensive model evaluation.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.logging_config import setup_logging, get_logger
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator

# Initialize logging
setup_logging()
logger = get_logger(__name__)


def main():
    """Run comprehensive model evaluation."""

    logger.info("Starting model evaluation...")

    try:
        # Load the best trained model
        trainer = ModelTrainer()
        model, metadata = trainer.load_best_model()

        logger.info(f"Loaded model: {metadata['model_name']}")
        logger.info(f"Training metrics: {metadata['metrics']}")

        # Initialize evaluator
        evaluator = ModelEvaluator()

        # Perform comprehensive evaluation
        results = evaluator.comprehensive_evaluation(model, metadata['model_name'])

        # Save results
        results_file = evaluator.save_evaluation_results(results, metadata['model_name'])

        # Generate evaluation report
        report = evaluator.generate_evaluation_report(results)

        # Save report
        report_file = os.path.join("logs", f"evaluation_report_{metadata['model_name']}.txt")
        with open(report_file, 'w') as f:
            f.write(report)

        # Display summary
        print("\n" + "="*80)
        print("MODEL EVALUATION COMPLETED")
        print("="*80)
        print(f"Model: {metadata['model_name']}")
        print(f"Test Accuracy: {results['basic_metrics']['accuracy']:.4f}")
        print(f"Test ROC-AUC: {results['basic_metrics']['roc_auc']:.4f}")
        print(f"Test F1-Score: {results['basic_metrics']['f1_score']:.4f}")
        print(f"Sensitivity: {results['basic_metrics']['recall']:.4f}")
        print(f"Specificity: {results['basic_metrics']['specificity']:.4f}")
        print(f"\nDetailed results: {results_file}")
        print(f"Evaluation report: {report_file}")
        print(f"Plots saved in: logs/evaluation_plots/")
        print("="*80)

        # Show quick interpretation
        roc_auc = results['basic_metrics']['roc_auc']
        if roc_auc >= 0.9:
            performance = "Excellent"
        elif roc_auc >= 0.8:
            performance = "Good"
        elif roc_auc >= 0.7:
            performance = "Fair"
        else:
            performance = "Needs Improvement"

        print(f"\nModel Performance: {performance}")
        print("Note: This model is for educational purposes only.")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()