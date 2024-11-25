#!/usr/bin/env python3
"""Generate comprehensive model evaluation reports."""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

from src.models.single_cat_model import SingleCategoryModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

def evaluate_model(model_path, data_path, output_dir, save_predictions=None):
    """Run comprehensive model evaluation."""
    print(f"Loading model from {model_path}")
    model = SingleCategoryModel.load(model_path)

    print(f"Loading validation data from {data_path}")
    df = pd.read_json(data_path, orient='records')

    # Make predictions
    print("Making predictions...")
    predictions = model.predict(df)

    # Try different possible target column names
    target_columns = ['target', 'sold_price', 'price']
    actuals = None
    for col in target_columns:
        if col in df.columns:
            actuals = df[col]
            print(f"Using '{col}' as target column")
            break

    if actuals is None:
        raise ValueError(f"No target column found. Expected one of: {target_columns}")

    print("Calculating metrics...")
    metrics = {
        'mae': mean_absolute_error(actuals, predictions),
        'mse': mean_squared_error(actuals, predictions),
        'rmse': root_mean_squared_error(actuals, predictions),
        'r2': r2_score(actuals, predictions)
    }

    # Save predictions if requested
    if save_predictions:
        pred_df = pd.DataFrame({
            'prediction': predictions,
            'actual': actuals
        })
        pred_df.to_csv(save_predictions, index=False)
        print(f"Predictions saved to {save_predictions}")

    # Generate plots
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics['timestamp'] = timestamp
    metrics['num_samples'] = len(actuals)

    # Save timestamped metrics
    with open(output_dir / f'metrics_{timestamp}.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save latest metrics
    with open(output_dir / 'metrics_latest.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Generate and save plots
    plot_evaluation_results(actuals, predictions, output_dir, timestamp)

    return metrics

def plot_evaluation_results(actuals, predictions, output_dir, timestamp):
    """Generate evaluation plots."""
    # Scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=actuals, y=predictions, alpha=0.5)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction vs Actual Values')
    plt.savefig(output_dir / f'scatter_{timestamp}.png')
    plt.close()

    # Residual plot
    residuals = predictions - actuals
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Count')
    plt.title('Residual Distribution')
    plt.savefig(output_dir / f'residuals_{timestamp}.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance")
    parser.add_argument("--model-path", required=True, help="Path to the model file")
    parser.add_argument("--data-path", required=True, help="Path to evaluation data")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory")
    parser.add_argument("--save-predictions", help="Path to save predictions CSV")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.data_path, args.output_dir, args.save_predictions)
