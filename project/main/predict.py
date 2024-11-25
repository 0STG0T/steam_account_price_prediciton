#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime

from single_cat_model import SingleCategoryModel
import onnxruntime as ort
from utils.logging_utils import create_log_file, log_predictions

def parse_args():
    parser = argparse.ArgumentParser(description='Make predictions using a trained ONNX model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--data-path', type=str, required=True, help='Path to input data JSON file')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save predictions CSV')
    parser.add_argument('--category-id', type=int, required=True, help='Category ID for preprocessing')
    return parser.parse_args()

def load_onnx_model(model_path):
    try:
        session = ort.InferenceSession(model_path)
        print(f"ONNX model loaded successfully from {model_path}")
        return session
    except Exception as e:
        print(f"Failed to load ONNX model from {model_path}: {e}")
        raise

def predict_with_onnx(session, X):
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    predictions = session.run([output_name], {input_name: X.astype(np.float32)})[0]
    return predictions

def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Create log file
    log_file = create_log_file('prediction')

    try:
        # Load data
        print(f"Loading data from {args.data_path}")
        df = pd.read_json(args.data_path)

        # Filter data for specific category
        category_df = df[df['category_id'] == args.category_id].copy()
        if len(category_df) == 0:
            raise ValueError(f"No data found for category_id {args.category_id}")

        print(f"Making predictions for category {args.category_id}")
        print(f"Total samples: {len(category_df)}")

        # Initialize model for preprocessing
        model = SingleCategoryModel(category_number=args.category_id)

        # Preprocess data
        processed_data = model.preprocess_data(category_df)

        # Load ONNX model and make predictions
        onnx_session = load_onnx_model(args.model_path)
        predictions = predict_with_onnx(onnx_session, processed_data)

        # Save predictions
        results_df = pd.DataFrame({
            'id': category_df.index,
            'predicted_price': predictions.flatten()
        })
        results_df.to_csv(args.output_path, index=False)

        # Log prediction results
        log_predictions(log_file, args.category_id, predictions, args.data_path)

        print(f"Predictions saved to {args.output_path}")
        print(f"Prediction summary:")
        print(f"Mean prediction: {predictions.mean():.2f}")
        print(f"Std prediction: {predictions.std():.2f}")
        print(f"Min prediction: {predictions.min():.2f}")
        print(f"Max prediction: {predictions.max():.2f}")

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

if __name__ == '__main__':
    main()
