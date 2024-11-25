#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import onnxmltools

class ModelTrainer:
    def __init__(self, data_dir="/app/data", model_dir="/app/models/onnx"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def load_data(self, filename="dataset.json"):
        """Load and preprocess the dataset."""
        data_path = os.path.join(self.data_dir, filename)
        with open(data_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df

    def preprocess_data(self, df):
        """Preprocess the data for training."""
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

    def train_model(self, df, category=None):
        """Train a CatBoost model for a specific category."""
        if category:
            df = df[df['category_name'] == category]

        # Split features and target
        X = df.drop('sold_price', axis=1)
        y = df['sold_price']

        # Train-test split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Initialize and train model
        model = CatBoostRegressor(
            iterations=60000,
            l2_leaf_reg=2.7,
            learning_rate=0.01,
            use_best_model=True,
            early_stopping_rounds=300,
            verbose=100
        )

        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=100
        )

        # Convert to ONNX
        onnx_model = onnxmltools.convert_catboost(
            model,
            initial_types=[('input', onnxmltools.convert.common.data_types.FloatTensorType([None, X_train.shape[1]]))],
            target_opset=13
        )

        # Save model
        model_name = f"category_{category}_model.onnx" if category else "full_model.onnx"
        model_path = os.path.join(self.model_dir, model_name)
        onnxmltools.utils.save_model(onnx_model, model_path)

        # Calculate metrics
        val_predictions = model.predict(X_val)
        metrics = {
            'mae': mean_absolute_error(y_val, val_predictions),
            'mse': mean_squared_error(y_val, val_predictions),
            'r2': r2_score(y_val, val_predictions)
        }

        return model_path, metrics

def main():
    trainer = ModelTrainer()

    try:
        # Load and preprocess data
        print("Loading data...")
        df = trainer.load_data()
        df = trainer.preprocess_data(df)

        # Train models for each category
        categories = df['category_name'].unique()
        results = {}

        for category in categories:
            print(f"\nTraining model for category: {category}")
            model_path, metrics = trainer.train_model(df, category)
            results[category] = {
                'model_path': model_path,
                'metrics': metrics
            }
            print(f"Category {category} metrics:")
            print(f"MAE: {metrics['mae']:.4f}")
            print(f"MSE: {metrics['mse']:.4f}")
            print(f"R2: {metrics['r2']:.4f}")

        # Save results
        results_path = os.path.join(trainer.model_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        print("\nTraining completed successfully!")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
