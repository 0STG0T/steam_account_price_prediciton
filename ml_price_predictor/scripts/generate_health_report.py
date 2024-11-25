#!/usr/bin/env python3
"""Generate comprehensive monitoring health report."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

def collect_health_metrics(monitoring_dir: Path) -> Dict[str, Any]:
    """Collect all health metrics from monitoring files."""
    try:
        # Get latest monitoring data
        monitoring_files = sorted(monitoring_dir.glob('monitoring_*.json'))
        if not monitoring_files:
            return {'status': 'unknown', 'error': 'No monitoring data found'}

        with open(monitoring_files[-1]) as f:
            monitoring_data = json.load(f)

        return {
            'timestamp': datetime.now().isoformat(),
            'status': 'warning' if monitoring_data.get('alerts') else 'passing',
            'metrics': {
                'current': monitoring_data['current_metrics'],
                'baseline': monitoring_data['baseline_metrics'],
                'drift_metrics': monitoring_data.get('drift_metrics', {})
            },
            'alerts': monitoring_data.get('alerts', []),
            'drift_detected': monitoring_data.get('drift_metrics', {}).get('p_value', 1.0) < 0.01  # More conservative threshold
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description="Generate health report")
    parser.add_argument("--monitoring-dir", default="monitoring_results",
                      help="Directory containing monitoring files")
    parser.add_argument("--output-file", default="health_report.json",
                      help="Output file for health report")
    args = parser.parse_args()

    monitoring_dir = Path(args.monitoring_dir)
    if not monitoring_dir.exists():
        print(f"Error: Directory not found: {args.monitoring_dir}")
        return

    health_report = collect_health_metrics(monitoring_dir)

    with open(args.output_file, 'w') as f:
        json.dump(health_report, f, indent=2)
    print(f"Generated health report: {args.output_file}")

if __name__ == "__main__":
    main()
