#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
import onnxruntime as rt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

class ModelTester:
    def __init__(self, data_dir="/app/data", model_dir="/app/models/onnx"):
        self.data_dir = data_dir
        self.model_dir = model_dir

    def load_data(self, filename="dataset.json"):
        """Load and preprocess the dataset."""
        data_path = os.path.join(self.data_dir, filename)
        with open(data_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df

    def preprocess_data(self, df):
        """Preprocess the data for testing."""
        # Drop unnecessary columns
        df = df.drop(columns=[
            'steam_cards_count',
            'steam_cards_games',
            'category_id',
            'is_sticky'
        ])

        # Define categorical features
        cat_features = ['category_name']

        # One-hot encoding for categorical features
        for col in cat_features:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                df.drop(columns=[col], inplace=True)

        return df

    def load_onnx_model(self, model_path):
        """Load an ONNX model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        return rt.InferenceSession(model_path)

    def predict_with_onnx(self, session, X):
        """Make predictions using ONNX model."""
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        return session.run([output_name], {input_name: X.astype(np.float32)})[0]

    def test_model(self, df, category=None):
        """Test a model for a specific category."""
        if category:
            df = df[df['category_name'] == category]
            model_path = os.path.join(self.model_dir, f"category_{category}_model.onnx")
        else:
            model_path = os.path.join(self.model_dir, "full_model.onnx")

        # Split features and target
        X = df.drop('sold_price', axis=1)
        y = df['sold_price']

        # Load model
        session = self.load_onnx_model(model_path)

        # Make predictions
        predictions = self.predict_with_onnx(session, X.values)

        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y, predictions),
            'mse': mean_squared_error(y, predictions),
            'r2': r2_score(y, predictions),
            'sample_size': len(y)
        }

        return metrics

def main():
    tester = ModelTester()

    try:
        # Load and preprocess data
        print("Loading data...")
        df = tester.load_data()
        df = tester.preprocess_data(df)

        # Test models for each category
        categories = df['category_name'].unique()
        results = {}

        for category in categories:
            print(f"\nTesting model for category: {category}")
            try:
                metrics = tester.test_model(df, category)
                results[category] = metrics
                print(f"Category {category} metrics:")
                print(f"MAE: {metrics['mae']:.4f}")
                print(f"MSE: {metrics['mse']:.4f}")
                print(f"R2: {metrics['r2']:.4f}")
                print(f"Sample size: {metrics['sample_size']}")
            except Exception as e:
                print(f"Error testing category {category}: {str(e)}")
                results[category] = {"error": str(e)}

        # Save results
        results_path = os.path.join(tester.model_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        print("\nTesting completed successfully!")

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
