#!/usr/bin/env python3
import os
import sys
import json
import shutil
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional
from single_cat_predictor import SingleCategoryModel

class SteamPricePredictor:
    def __init__(self):
        self.data_dir = "/app/data"
        self.model_dir = "/app/models/onnx"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.model = None

    def load_dataset(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load dataset from file or use default."""
        try:
            if file_path:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                print(f"Loaded custom dataset from {file_path}")
            else:
                default_path = os.path.join(self.data_dir, "dataset.json")
                with open(default_path, 'r') as f:
                    data = json.load(f)
                print(f"Loaded default dataset from {default_path}")

            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            sys.exit(1)

    def train(self, dataset_path: Optional[str] = None):
        """Train the model."""
        try:
            data = self.load_dataset(dataset_path)
            self.model = SingleCategoryModel(category_number=1)
            self.model.train(data)

            output_path = os.path.join(self.model_dir, "model.onnx")
            self.model.export(output_path)
            print(f"Model trained and exported to {output_path}")
            return True
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return False

    def validate(self, dataset_path: Optional[str] = None):
        """Validate the model."""
        try:
            if not self.model:
                model_path = os.path.join(self.model_dir, "model.onnx")
                self.model = SingleCategoryModel(category_number=1)
                self.model.load_model(model_path)

            data = self.load_dataset(dataset_path)
            metrics = self.model.validate(data)

            metrics_path = os.path.join(self.data_dir, "validation_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Validation metrics saved to {metrics_path}")
            return True
        except Exception as e:
            print(f"Error during validation: {str(e)}")
            return False

    def predict(self, dataset_path: Optional[str] = None):
        """Make predictions."""
        try:
            if not self.model:
                model_path = os.path.join(self.model_dir, "model.onnx")
                self.model = SingleCategoryModel(category_number=1)
                self.model.load_model(model_path)

            data = self.load_dataset(dataset_path)
            processed_data = self.model.preprocess_data(data)
            predictions = self.model.meta_model.predict(processed_data)

            pred_path = os.path.join(self.data_dir, "predictions.json")
            with open(pred_path, 'w') as f:
                json.dump({"predictions": predictions.tolist()}, f, indent=2)
            print(f"Predictions saved to {pred_path}")
            return True
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return False

    def finetune(self, dataset_path: Optional[str] = None):
        """Finetune the model."""
        try:
            if not self.model:
                model_path = os.path.join(self.model_dir, "model.onnx")
                self.model = SingleCategoryModel(category_number=1)
                self.model.load_model(model_path)

            data = self.load_dataset(dataset_path)
            self.model.finetune(data)

            output_path = os.path.join(self.model_dir, "model_finetuned.onnx")
            self.model.export(output_path)
            print(f"Model finetuned and exported to {output_path}")
            return True
        except Exception as e:
            print(f"Error during finetuning: {str(e)}")
            return False

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
    action_map = {
        'train': predictor.train,
        'validate': predictor.validate,
        'predict': predictor.predict,
        'finetune': predictor.finetune
    }

    action_map[args.action](args.dataset)

if __name__ == "__main__":
    main()
