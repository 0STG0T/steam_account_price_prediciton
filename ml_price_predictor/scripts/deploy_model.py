#!/usr/bin/env python3
"""Automated model deployment script with validation and monitoring."""

import argparse
import logging
import shutil
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deploy_model(model_path, validation_data, threshold_metrics=None):
    """Deploy model with validation checks."""
    try:
        # Validate model performance
        from scripts.evaluate_model import evaluate_model
        metrics = evaluate_model(model_path, validation_data, "validation_results")

        # Check if metrics meet thresholds
        if threshold_metrics:
            for metric, threshold in threshold_metrics.items():
                if metrics.get(metric, float('inf')) > threshold:
                    raise ValueError(f"{metric} exceeds threshold: {metrics[metric]} > {threshold}")

        # Setup model monitoring
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        production_path = Path("models/production")
        production_path.mkdir(parents=True, exist_ok=True)

        # Archive current production model if exists
        current_model = production_path / "model.onnx"
        if current_model.exists():
            archive_path = production_path / "archive" / f"model_{timestamp}.onnx"
            archive_path.parent.mkdir(exist_ok=True)
            shutil.move(str(current_model), str(archive_path))

        # Deploy new model
        shutil.copy(model_path, str(current_model))
        logger.info(f"Model successfully deployed to production: {current_model}")

        return True

    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy model to production")
    parser.add_argument("--model-path", required=True, help="Path to model file")
    parser.add_argument("--validation-data", required=True, help="Path to validation data")
    parser.add_argument("--rmse-threshold", type=float, default=100.0, help="RMSE threshold")
    parser.add_argument("--mae-threshold", type=float, default=50.0, help="MAE threshold")

    args = parser.parse_args()

    threshold_metrics = {
        'rmse': args.rmse_threshold,
        'mae': args.mae_threshold
    }

    success = deploy_model(args.model_path, args.validation_data, threshold_metrics)
    exit(0 if success else 1)
