#!/usr/bin/env python3
"""Monitor model drift and performance degradation."""

import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def calculate_drift_metrics(baseline_data, current_data):
    """Calculate distribution drift between baseline and current data."""
    drift_metrics = {}

    # Get common numerical columns
    common_columns = set(baseline_data.columns) & set(current_data.columns)
    numerical_columns = [col for col in common_columns
                        if baseline_data[col].dtype in [np.float64, np.int64]]

    for column in numerical_columns:
        # KS test for numerical features
        statistic, p_value = stats.ks_2samp(
            baseline_data[column].dropna(),
            current_data[column].dropna()
        )
        drift_metrics[column] = {
            'statistic': float(statistic),  # Convert numpy types to native Python
            'p_value': float(p_value),      # Convert numpy types to native Python
            'is_drifting': bool(p_value < 0.01)  # Convert numpy bool to Python bool
        }

    return drift_metrics

def monitor_performance(predictions_file, actuals_file, baseline_metrics_file):
    """Monitor model performance and data drift."""
    # Load data
    predictions_df = pd.read_csv(predictions_file)
    actuals_df = pd.read_csv(actuals_file)
    with open(baseline_metrics_file) as f:
        baseline_metrics = json.load(f)

    # Determine target column name
    target_col = 'sold_price' if 'sold_price' in actuals_df.columns else 'target'
    if target_col not in actuals_df.columns:
        raise ValueError(f"No target column found in actuals file. Expected 'sold_price' or 'target'")

    # Load configuration
    with open('configs/monitoring_config.json') as f:
        config = json.load(f)

    # Calculate current metrics
    current_metrics = {
        'rmse': root_mean_squared_error(actuals_df[target_col], predictions_df['prediction']),
        'mae': mean_absolute_error(actuals_df[target_col], predictions_df['prediction'])
    }

    # Calculate metric degradation
    degradation = {
        metric: (current_metrics[metric] - baseline_metrics[metric]) / baseline_metrics[metric]
        for metric in current_metrics
    }

    # Check against thresholds
    alerts = []
    if degradation['rmse'] > config['performance_thresholds']['rmse_degradation']:
        alerts.append(f"RMSE degradation ({degradation['rmse']:.1%}) exceeds threshold")
    if degradation['mae'] > config['performance_thresholds']['mae_degradation']:
        alerts.append(f"MAE degradation ({degradation['mae']:.1%}) exceeds threshold")

    # Create comparison dataframes with just the values we want to compare
    baseline_compare = pd.DataFrame({'value': actuals_df[target_col]})
    current_compare = pd.DataFrame({'value': predictions_df['prediction']})

    # Calculate drift metrics on the predictions vs actuals
    drift_metrics = calculate_drift_metrics(baseline_compare, current_compare)

    return {
        'current_metrics': current_metrics,
        'baseline_metrics': baseline_metrics,
        'degradation': degradation,
        'drift_metrics': drift_metrics,
        'alerts': alerts,
        'timestamp': datetime.now().isoformat()
    }

def main():
    parser = argparse.ArgumentParser(description="Monitor model drift and performance")
    parser.add_argument("--predictions", required=True, help="Path to predictions CSV")
    parser.add_argument("--actuals", required=True, help="Path to actuals CSV")
    parser.add_argument("--baseline-metrics", required=True, help="Path to baseline metrics JSON")
    parser.add_argument("--output-dir", default="monitoring_results", help="Output directory")
    args = parser.parse_args()

    results = monitor_performance(args.predictions, args.actuals, args.baseline_metrics)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    with open(output_dir / f'monitoring_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\nMonitoring Results:")
    print("==================")
    print(f"Current RMSE: {results['current_metrics']['rmse']:.2f}")
    print(f"Baseline RMSE: {results['baseline_metrics']['rmse']:.2f}")
    print(f"RMSE Degradation: {results['degradation']['rmse']*100:.1f}%")

    print("\nDrift Detection:")
    drifting_features = [
        f for f, m in results['drift_metrics'].items()
        if m['is_drifting']
    ]
    if drifting_features:
        print("WARNING: Detected drift in features:", ", ".join(drifting_features))
    else:
        print("No significant drift detected")

if __name__ == "__main__":
    main()
