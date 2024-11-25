#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import onnxruntime as rt
from datetime import datetime

class ModelValidator:
    def __init__(self, data_dir="/app/data", model_dir="/app/models/onnx"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.validation_results_dir = os.path.join(model_dir, "validation_results")
        os.makedirs(self.validation_results_dir, exist_ok=True)

    def load_and_preprocess_data(self, filename="dataset.json"):
        """Load and preprocess the dataset."""
        with open(os.path.join(self.data_dir, filename), 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)

        # Preprocess
        df = df.drop(columns=[
            'steam_cards_count',
            'steam_cards_games',
            'category_id',
            'is_sticky'
        ])

        # One-hot encode categories
        if 'category_name' in df.columns:
            dummies = pd.get_dummies(df['category_name'], prefix='category_name')
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=['category_name'], inplace=True)

        return df

    def cross_validate(self, df, category, n_splits=5):
        """Perform k-fold cross-validation for a specific category."""
        model_path = os.path.join(self.model_dir, f"category_{category}_model.onnx")
        session = rt.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        X = df.drop('sold_price', axis=1)
        y = df['sold_price']

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            # Make predictions
            predictions = session.run(
                [output_name],
                {input_name: X_val.values.astype(np.float32)}
            )[0]

            # Calculate metrics
            metrics = {
                'fold': fold,
                'mae': mean_absolute_error(y_val, predictions),
                'mse': mean_squared_error(y_val, predictions),
                'rmse': np.sqrt(mean_squared_error(y_val, predictions)),
                'r2': r2_score(y_val, predictions)
            }
            fold_metrics.append(metrics)

        return fold_metrics

    def validate_price_ranges(self, df, category):
        """Validate model performance across different price ranges."""
        model_path = os.path.join(self.model_dir, f"category_{category}_model.onnx")
        session = rt.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Create price range bins
        df['price_range'] = pd.qcut(df['sold_price'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        range_metrics = {}

        for price_range in df['price_range'].unique():
            range_df = df[df['price_range'] == price_range]
            X = range_df.drop(['sold_price', 'price_range'], axis=1)
            y = range_df['sold_price']

            predictions = session.run(
                [output_name],
                {input_name: X.values.astype(np.float32)}
            )[0]

            range_metrics[str(price_range)] = {
                'mae': mean_absolute_error(y, predictions),
                'mse': mean_squared_error(y, predictions),
                'rmse': np.sqrt(mean_squared_error(y, predictions)),
                'r2': r2_score(y, predictions),
                'sample_size': len(y)
            }

        return range_metrics

def main():
    validator = ModelValidator()

    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        df = validator.load_and_preprocess_data()

        # Get unique categories from model files
        categories = [f.replace('category_', '').replace('_model.onnx', '')
                     for f in os.listdir(validator.model_dir)
                     if f.endswith('.onnx') and f.startswith('category_')]

        validation_results = {}
        for category in categories:
            print(f"\nValidating model for category: {category}")

            # Perform cross-validation
            cv_metrics = validator.cross_validate(df, category)

            # Validate across price ranges
            price_range_metrics = validator.validate_price_ranges(df, category)

            validation_results[category] = {
                'cross_validation': cv_metrics,
                'price_range_performance': price_range_metrics
            }

        # Save validation results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(
            validator.validation_results_dir,
            f'validation_results_{timestamp}.json'
        )

        with open(results_file, 'w') as f:
            json.dump(validation_results, f, indent=4)

        print(f"\nValidation completed successfully! Results saved to {results_file}")

    except Exception as e:
        print(f"Error during validation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
