#!/usr/bin/env python3

import argparse
import os
from datetime import datetime

from src.models.single_cat_model import SingleCategoryModel
from src.utils.logging_utils import create_log_file, log_training_metrics
from catboost import CatBoostRegressor

def parse_args():
    parser = argparse.ArgumentParser(description='Export trained model to ONNX format')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save ONNX model')
    parser.add_argument('--category-id', type=int, required=True, help='Category ID for model metadata')
    return parser.parse_args()

def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    try:
        print(f"Loading model from {args.model_path}")

        # Initialize model
        model = SingleCategoryModel(category_number=args.category_id)
        model.meta_model = CatBoostRegressor()
        model.meta_model.load_model(args.model_path)  # Load native CatBoost model

        # Export to ONNX
        print(f"Exporting model to ONNX format...")
        model.export(args.output_path)  # Use the correct export method

        print(f"Model successfully exported to: {args.output_path}")

    except Exception as e:
        print(f"Error during ONNX export: {str(e)}")
        raise

if __name__ == '__main__':
    main()
