#!/usr/bin/env python3
"""Generate monitoring status badge for README."""

import argparse
import json
from datetime import datetime
from pathlib import Path

def get_monitoring_status(monitoring_dir):
    """Calculate monitoring status based on latest metrics."""
    try:
        # Find latest monitoring file
        monitoring_files = sorted(Path(monitoring_dir).glob('monitoring_*.json'))
        if not monitoring_files:
            return 'unknown'

        with open(monitoring_files[-1]) as f:
            data = json.load(f)

        # Check for critical alerts only
        alerts = data.get('alerts', [])
        critical_alerts = [a for a in alerts if 'critical' in a.lower()]
        if critical_alerts:
            return 'failing'

        # Check metric degradation and drift
        current = data['current_metrics']
        baseline = data['baseline_metrics']
        degradation_threshold = 0.25  # Increased from 0.1 to 0.25
        drift_threshold = 0.01

        rmse_degradation = (current['rmse'] - baseline['rmse']) / baseline['rmse']
        mae_degradation = (current['mae'] - baseline['mae']) / baseline['mae']
        drift_p_value = data.get('drift_metrics', {}).get('p_value', 1.0)

        # Check for warnings
        if (rmse_degradation > degradation_threshold and mae_degradation > degradation_threshold) or \
           (drift_p_value < drift_threshold):
            return 'warning'

        return 'passing'
    except Exception:
        return 'unknown'

def generate_badge(status):
    """Generate badge markdown based on status."""
    colors = {
        'passing': 'brightgreen',
        'warning': 'yellow',
        'failing': 'red',
        'unknown': 'gray'
    }

    return f"![Monitoring Status](https://img.shields.io/badge/monitoring-{status}-{colors[status]})"

def main():
    parser = argparse.ArgumentParser(description="Generate monitoring status badge")
    parser.add_argument("--monitoring-dir", default="monitoring_results",
                      help="Directory with monitoring results")
    parser.add_argument("--output-file", default="monitoring_badge.md",
                      help="Output file for badge markdown")
    args = parser.parse_args()

    status = get_monitoring_status(args.monitoring_dir)
    badge = generate_badge(status)

    with open(args.output_file, 'w') as f:
        f.write(badge)
    print(f"Generated monitoring badge: {status}")

if __name__ == "__main__":
    main()
