#!/usr/bin/env python3
"""Validate model performance against deployment criteria."""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from scipy import stats

def validate_model_performance(metrics_file, criteria_file=None):
    """Validate model metrics against deployment criteria."""
    # Default criteria
    criteria = {
        'rmse_threshold': 100.0,
        'mae_threshold': 50.0,
        'r2_threshold': 0.7,
        'drift_p_value_threshold': 0.01
    }

    # Load custom criteria if provided
    if criteria_file:
        with open(criteria_file) as f:
            criteria.update(json.load(f))

    # Ensure directory exists
    Path(metrics_file).parent.mkdir(parents=True, exist_ok=True)

    # Load metrics
    with open(metrics_file) as f:
        metrics = json.load(f)

    # Handle different key names for R² score
    r2_value = metrics.get('r2', metrics.get('r2_score', 0.0))

    # Get drift p-value if available
    drift_p_value = metrics.get('drift_metrics', {}).get('p_value', 1.0)

    # Validate metrics
    validation_results = {
        'rmse_passed': metrics['rmse'] < criteria['rmse_threshold'],
        'mae_passed': metrics['mae'] < criteria['mae_threshold'],
        'r2_passed': r2_value > criteria['r2_threshold'],
        'drift_passed': drift_p_value >= criteria['drift_p_value_threshold'],
        'metrics': {**metrics, 'r2': r2_value, 'drift_p_value': drift_p_value},
        'criteria': criteria
    }

    return validation_results

def main():
    parser = argparse.ArgumentParser(description="Validate model for deployment")
    parser.add_argument("--metrics-file", required=True, help="Path to metrics JSON file")
    parser.add_argument("--criteria-file", help="Path to custom criteria JSON file")
    parser.add_argument("--output-file", help="Path to save validation results")
    args = parser.parse_args()

    results = validate_model_performance(args.metrics_file, args.criteria_file)

    # Print results
    print("\nModel Validation Results:")
    print("========================")
    print(f"RMSE: {results['metrics']['rmse']:.2f} (threshold: {results['criteria']['rmse_threshold']}) - {'✓' if results['rmse_passed'] else '✗'}")
    print(f"MAE: {results['metrics']['mae']:.2f} (threshold: {results['criteria']['mae_threshold']}) - {'✓' if results['mae_passed'] else '✗'}")
    print(f"R² Score: {results['metrics']['r2']:.2f} (threshold: {results['criteria']['r2_threshold']}) - {'✓' if results['r2_passed'] else '✗'}")
    print(f"Drift p-value: {results['metrics']['drift_p_value']:.3f} (threshold: {results['criteria']['drift_p_value_threshold']}) - {'✓' if results['drift_passed'] else '✗'}")

    all_passed = all([results['rmse_passed'], results['mae_passed'], results['r2_passed'], results['drift_passed']])
    print(f"\nOverall Status: {'PASSED' if all_passed else 'FAILED'}")

    # Save results if output file specified
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

    # Exit with status code
    exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
