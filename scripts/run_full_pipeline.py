"""
Script to run the complete end-to-end pipeline.
Downloads data, processes it, trains models, evaluates performance, and starts the API.
"""

import os
import sys
import time
import subprocess
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.logging_config import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)


def run_command(command, description):
    """Run a system command and log the results."""
    logger.info(f"Starting: {description}")
    logger.info(f"Command: {command}")

    try:
        start_time = time.time()
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        duration = time.time() - start_time

        logger.info(f"Completed: {description} in {duration:.2f} seconds")

        if result.stdout:
            logger.info(f"Output: {result.stdout}")

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}")
        logger.error(f"Error: {e}")
        if e.stdout:
            logger.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return False


def main():
    """Run the complete pipeline."""

    print("\n" + "="*80)
    print("HEART DISEASE RISK PREDICTION - COMPLETE PIPELINE")
    print("="*80)

    logger.info("Starting complete pipeline execution...")

    steps = [
        {
            "command": "python scripts/download_data.py",
            "description": "Download UCI Heart Disease dataset"
        },
        {
            "command": "python src/data_processing.py",
            "description": "Process and clean data"
        },
        {
            "command": "python scripts/train_model.py",
            "description": "Train machine learning models"
        },
        {
            "command": "python scripts/run_evaluation.py",
            "description": "Evaluate model performance"
        },
        {
            "command": "python src/explainability.py",
            "description": "Generate SHAP explanations"
        },
        {
            "command": "python src/llm_layer.py",
            "description": "Test LLM integration"
        },
        {
            "command": "python src/prediction_service.py",
            "description": "Test prediction service"
        }
    ]

    successful_steps = 0
    total_steps = len(steps)

    for i, step in enumerate(steps, 1):
        print(f"\n[{i}/{total_steps}] {step['description']}")
        print("-" * 60)

        success = run_command(step["command"], step["description"])

        if success:
            successful_steps += 1
            print(f"✅ {step['description']} - COMPLETED")
        else:
            print(f"❌ {step['description']} - FAILED")

    # Run basic tests
    print(f"\n[{total_steps+1}/{total_steps+1}] Running basic tests")
    print("-" * 60)

    test_success = run_command("python -m pytest tests/test_basic.py -v", "Run basic tests")

    if test_success:
        successful_steps += 1
        print("✅ Basic tests - COMPLETED")
    else:
        print("❌ Basic tests - FAILED")

    # Final summary
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    print(f"Total steps: {total_steps + 1}")
    print(f"Successful: {successful_steps}")
    print(f"Failed: {(total_steps + 1) - successful_steps}")

    if successful_steps == total_steps + 1:
        print("\n🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
        print("\nYou can now:")
        print("1. Start the API: python api/main.py")
        print("2. Access docs: http://localhost:8000/docs")
        print("3. Make predictions: http://localhost:8000/predict")

        # Optionally start the API
        start_api = input("\nStart the API now? (y/n): ").lower().strip()
        if start_api == 'y':
            print("\nStarting API server...")
            print("Access the API at: http://localhost:8000")
            print("API Documentation: http://localhost:8000/docs")
            print("Press Ctrl+C to stop the server")

            try:
                subprocess.run("python api/main.py", shell=True, check=True)
            except KeyboardInterrupt:
                print("\nAPI server stopped.")

    else:
        print("\n⚠️ SOME STEPS FAILED!")
        print("Please check the logs for details and resolve any issues.")
        logger.error(f"Pipeline completed with {(total_steps + 1) - successful_steps} failures")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()