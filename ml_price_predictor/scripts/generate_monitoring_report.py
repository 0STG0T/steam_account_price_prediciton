#!/usr/bin/env python3
"""Generate comprehensive monitoring reports with visualizations."""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

def plot_metric_trends(monitoring_data, output_dir):
    """Plot performance metric trends over time."""
    df = pd.DataFrame(monitoring_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Plot RMSE and MAE trends
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['current_metrics'].apply(lambda x: x['rmse']), label='RMSE')
    plt.plot(df['timestamp'], df['current_metrics'].apply(lambda x: x['mae']), label='MAE')
    plt.axhline(y=df['baseline_metrics'].iloc[0]['rmse'], linestyle='--', color='r', label='Baseline RMSE')
    plt.xlabel('Time')
    plt.ylabel('Metric Value')
    plt.title('Model Performance Trends')
    plt.legend()
    plt.savefig(output_dir / 'metric_trends.png')
    plt.close()

def generate_report(monitoring_dir, output_dir):
    """Generate monitoring report with visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load monitoring data
    monitoring_files = sorted(Path(monitoring_dir).glob('monitoring_*.json'))
    monitoring_data = []
    for f in monitoring_files:
        with open(f) as file:
            monitoring_data.append(json.load(file))

    # Generate visualizations
    plot_metric_trends(monitoring_data, output_dir)

    # Generate HTML report
    report_html = f"""
    <html>
    <head>
        <title>Model Monitoring Report - {datetime.now().strftime('%Y-%m-%d')}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .metric {{ padding: 10px; margin: 5px; border: 1px solid #ddd; }}
            .alert {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>Model Monitoring Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Current Performance</h2>
        <div class="metric">
            <h3>Latest Metrics</h3>
            <p>RMSE: {monitoring_data[-1]['current_metrics']['rmse']:.2f}</p>
            <p>MAE: {monitoring_data[-1]['current_metrics']['mae']:.2f}</p>
        </div>

        <h2>Drift Analysis</h2>
        <img src="metric_trends.png" alt="Metric Trends">

        <h2>Alerts</h2>
        {'<p class="alert">' + '<br>'.join(monitoring_data[-1]['alerts']) + '</p>' if monitoring_data[-1]['alerts'] else '<p>No active alerts</p>'}
    </body>
    </html>
    """

    with open(output_dir / 'monitoring_report.html', 'w') as f:
        f.write(report_html)

def main():
    parser = argparse.ArgumentParser(description="Generate monitoring report")
    parser.add_argument("--monitoring-dir", default="monitoring_results", help="Directory with monitoring results")
    parser.add_argument("--output-dir", default="reports", help="Output directory for report")
    args = parser.parse_args()

    generate_report(args.monitoring_dir, args.output_dir)
    print(f"Report generated in {args.output_dir}")

if __name__ == "__main__":
    main()
