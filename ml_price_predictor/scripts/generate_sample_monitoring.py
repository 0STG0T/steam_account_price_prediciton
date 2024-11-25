#!/usr/bin/env python3
"""Generate sample monitoring data for testing."""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def generate_sample_metrics(base_rmse=50.0, base_mae=30.0, n_days=30, drift_factor=0.01):
    """Generate sample metrics with gradual drift."""
    metrics = []

    for day in range(n_days):
        # Add some random variation and drift
        drift = drift_factor * day
        noise_rmse = np.random.normal(0, 2.0)
        noise_mae = np.random.normal(0, 1.0)

        current_metrics = {
            'current_metrics': {
                'rmse': base_rmse * (1 + drift) + noise_rmse,
                'mae': base_mae * (1 + drift) + noise_mae
            },
            'baseline_metrics': {
                'rmse': base_rmse,
                'mae': base_mae
            },
            'drift_metrics': {
                'p_value': max(0.001, 0.05 - (drift * 0.1)),
                'ks_statistic': min(0.999, drift * 2)
            },
            'alerts': [],
            'timestamp': (datetime.now() - timedelta(days=n_days-day-1)).isoformat()
        }

        # Add alerts if metrics exceed thresholds
        if drift > 0.1:
            current_metrics['alerts'].append(f"Performance degradation detected: RMSE increased by {drift*100:.1f}%")

        metrics.append(current_metrics)

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Generate sample monitoring data")
    parser.add_argument("--output-dir", default="monitoring_results", help="Output directory")
    parser.add_argument("--n-days", type=int, default=30, help="Number of days to generate")
    parser.add_argument("--drift-factor", type=float, default=0.01, help="Daily drift factor")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = generate_sample_metrics(n_days=args.n_days, drift_factor=args.drift_factor)

    # Save individual monitoring files
    for i, metric in enumerate(metrics):
        date_str = datetime.fromisoformat(metric['timestamp']).strftime('%Y%m%d')
        output_file = output_dir / f"monitoring_{date_str}.json"
        with open(output_file, 'w') as f:
            json.dump(metric, f, indent=2)

        # Create/update latest.json symlink for the most recent file
        if i == len(metrics) - 1:
            latest_file = output_dir / "monitoring_latest.json"
            with open(latest_file, 'w') as f:
                json.dump(metric, f, indent=2)

    print(f"Generated {args.n_days} days of sample monitoring data in {output_dir}")

if __name__ == "__main__":
    main()
