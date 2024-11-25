#!/usr/bin/env python3
import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Optional

class SteamPricePredictor:
    def __init__(self):
        self.data_dir = "/app/data"
        self.model_dir = "/app/models/onnx"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def upload_dataset(self, file_path: str) -> bool:
        """Upload and validate dataset file."""
        try:
            # Validate JSON format
            with open(file_path, 'r') as f:
                json.load(f)  # Verify valid JSON

            # Copy file to data directory
            destination = os.path.join(self.data_dir, "dataset.json")
            shutil.copy2(file_path, destination)
            print(f"Dataset uploaded successfully to {destination}")
            return True
        except json.JSONDecodeError:
            print("Error: Invalid JSON file format")
            return False
        except Exception as e:
            print(f"Error uploading dataset: {str(e)}")
            return False

    def train(self):
        """Train the model."""
        try:
            from train import main as train_main
            train_main()
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return False
        return True

    def validate(self):
        """Validate the model."""
        try:
            from validate import main as validate_main
            validate_main()
        except Exception as e:
            print(f"Error during validation: {str(e)}")
            return False
        return True

    def predict(self):
        """Make predictions."""
        try:
            from predict import main as predict_main
            predict_main()
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return False
        return True

    def finetune(self):
        """Finetune the model."""
        try:
            from train import ModelTrainer
            trainer = ModelTrainer()

            # Load existing model and continue training with different parameters
            trainer.train(
                learning_rate=0.01,
                iterations=10000,
                early_stopping_rounds=100,
                use_best_model=True
            )
        except Exception as e:
            print(f"Error during finetuning: {str(e)}")
            return False
        return True

def main():
    parser = argparse.ArgumentParser(
        description="Steam Account Price Prediction CLI",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        'action',
        choices=['train', 'validate', 'predict', 'finetune'],
        help='Action to perform:\n'
             'train    - Train a new model\n'
             'validate - Validate model performance\n'
             'predict  - Make predictions\n'
             'finetune - Finetune existing model'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to custom dataset.json file'
    )

    args = parser.parse_args()

    predictor = SteamPricePredictor()

    # Handle dataset upload if provided
    if args.dataset:
        if not predictor.upload_dataset(args.dataset):
            sys.exit(1)

    # Execute requested action
    action_map = {
        'train': predictor.train,
        'validate': predictor.validate,
        'predict': predictor.predict,
        'finetune': predictor.finetune
    }

    if not action_map[args.action]():
        sys.exit(1)

if __name__ == "__main__":
    main()
