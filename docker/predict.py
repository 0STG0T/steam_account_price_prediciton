#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
import onnxruntime as rt

class ModelPredictor:
    def __init__(self, model_dir="/app/models/onnx"):
        self.model_dir = model_dir
        self.sessions = {}
        self._load_models()

    def _load_models(self):
        """Load all available ONNX models."""
        for file in os.listdir(self.model_dir):
            if file.endswith('.onnx'):
                model_path = os.path.join(self.model_dir, file)
                category = file.replace('category_', '').replace('_model.onnx', '')
                self.sessions[category] = rt.InferenceSession(model_path)

    def preprocess_input(self, input_data):
        """Preprocess input data for prediction."""
        df = pd.DataFrame([input_data])

        # One-hot encode category if present
        if 'category_name' in df.columns:
            dummies = pd.get_dummies(df['category_name'], prefix='category_name')
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=['category_name'], inplace=True)

        # Drop unnecessary columns if present
        columns_to_drop = [
            'steam_cards_count',
            'steam_cards_games',
            'category_id',
            'is_sticky',
            'sold_price'  # Drop if present in input
        ]

        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        return df

    def predict(self, input_data):
        """Make price prediction for input data."""
        # Determine category and get appropriate model
        category = input_data.get('category_name')
        if category not in self.sessions:
            raise ValueError(f"No model available for category: {category}")

        session = self.sessions[category]

        # Preprocess input
        X = self.preprocess_input(input_data)

        # Make prediction
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        prediction = session.run(
            [output_name],
            {input_name: X.values.astype(np.float32)}
        )[0][0]

        return {
            'predicted_price': float(prediction),
            'category': category,
            'input_features': input_data
        }

def main():
    predictor = ModelPredictor()

    try:
        # Example input data (can be replaced with actual input)
        example_input = {
            'category_name': 'dota2',
            'price': 100.0,
            'view_count': 50,
            'steam_level': 20,
            'steam_friend_count': 100,
            'steam_hours_played_recently': 40
        }

        # Make prediction
        result = predictor.predict(example_input)

        # Print results
        print("\nPrediction Results:")
        print(f"Category: {result['category']}")
        print(f"Predicted Price: ${result['predicted_price']:.2f}")

        # Save prediction to file
        output_dir = "/app/predictions"
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, 'latest_prediction.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)

        print(f"\nPrediction saved to {output_file}")

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    main()
