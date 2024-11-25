#!/usr/bin/env python3

import argparse
import os
from datetime import datetime
import pandas as pd
from tqdm import tqdm

from src.models.single_cat_model import SingleCategoryModel
from src.utils.log_manager import LogManager
from src.utils.logging_utils import log_validation_results

def parse_args():
    parser = argparse.ArgumentParser(description='Validate a trained price prediction model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--data-path', type=str, required=True, help='Path to validation data JSON file')
    parser.add_argument('--category-id', type=int, required=True, help='Category ID for validation')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save validation results')
    parser.add_argument('--plot-name', type=str, help='Custom name for validation plot')
    return parser.parse_args()

def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize log manager and create log file
    log_manager = LogManager()
    log_file = log_manager.get_log_path('validation')

    try:
        # Load data
        print(f"Loading validation data from {args.data_path}")
        df = pd.read_json(args.data_path)

        # Filter data for specific category
        valid_df = df[df['category_id'] == args.category_id].copy()
        if len(valid_df) == 0:
            raise ValueError(f"No validation data found for category_id {args.category_id}")

        print(f"Validating model for category {args.category_id}")
        print(f"Total validation samples: {len(valid_df)}")

        # Initialize and load model
        model = SingleCategoryModel(category_number=args.category_id)
        model.load_model(args.model_path)

        # Set up plot path using log manager
        plot_name = args.plot_name or f'cat_{args.category_id}_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plot_path = log_manager.get_log_path('reports', plot_name)

        # Validate model
        metrics = model.validate(df=valid_df, save_plot_path=plot_path)

        # Log validation results
        log_validation_results(log_file, args.category_id, metrics, plot_path)

        print("Validation Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
        print(f"Validation plot saved to: {plot_path}")

    except Exception as e:
        print(f"Error during validation: {str(e)}")
        raise

if __name__ == '__main__':
    main()
