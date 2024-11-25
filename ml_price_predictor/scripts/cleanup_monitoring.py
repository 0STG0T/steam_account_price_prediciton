#!/usr/bin/env python3
"""Clean up old monitoring files while preserving important metrics."""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta

def cleanup_monitoring_files(monitoring_dir, reports_dir, max_age_days=30):
    """Clean up old monitoring files while preserving key metrics."""
    monitoring_dir = Path(monitoring_dir)
    reports_dir = Path(reports_dir)

    # Calculate cutoff date
    cutoff_date = datetime.now() - timedelta(days=max_age_days)

    # Process monitoring files
    monitoring_files = list(monitoring_dir.glob('monitoring_*.json'))
    metrics_history = []

    for file in monitoring_files:
        # Skip files with 'latest' in the name
        if 'latest' in file.stem:
            continue

        try:
            file_date = datetime.strptime(file.stem.split('_')[1], '%Y%m%d')
            if file_date < cutoff_date:
                # Extract key metrics before deletion
                with open(file) as f:
                    data = json.load(f)
                    metrics_history.append({
                        'date': file_date.isoformat(),
                        'rmse': data['current_metrics']['rmse'],
                        'mae': data['current_metrics']['mae']
                    })
                file.unlink()
        except (ValueError, IndexError):
            print(f"Skipping file with invalid date format: {file.name}")
            continue

    # Save metrics history
    if metrics_history:
        history_file = monitoring_dir / 'metrics_history.json'
        if history_file.exists():
            with open(history_file) as f:
                existing_history = json.load(f)
            metrics_history.extend(existing_history)

        with open(history_file, 'w') as f:
            json.dump(metrics_history, f, indent=2)

    # Clean up old reports
    for file in reports_dir.glob('monitoring_report_*.html'):
        file_date = datetime.strptime(file.stem.split('_')[2], '%Y%m%d')
        if file_date < cutoff_date:
            file.unlink()

def main():
    parser = argparse.ArgumentParser(description="Clean up old monitoring files")
    parser.add_argument("--monitoring-dir", default="monitoring_results", help="Monitoring results directory")
    parser.add_argument("--reports-dir", default="reports", help="Reports directory")
    parser.add_argument("--max-age-days", type=int, default=30, help="Maximum age of files to keep")
    args = parser.parse_args()

    cleanup_monitoring_files(args.monitoring_dir, args.reports_dir, args.max_age_days)
    print(f"Cleaned up monitoring files older than {args.max_age_days} days")

if __name__ == "__main__":
    main()
