#!/usr/bin/env python3
"""Generate comprehensive monitoring dashboard."""

import argparse
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

def load_monitoring_data(monitoring_dir):
    """Load and combine all monitoring data."""
    monitoring_files = sorted(Path(monitoring_dir).glob('monitoring_*.json'))
    data = []
    for f in monitoring_files:
        with open(f) as file:
            data.append(json.load(file))
    return data

def create_dashboard(monitoring_data, output_dir):
    """Create interactive dashboard with Plotly."""
    df = pd.DataFrame([
        {
            'timestamp': d['timestamp'],
            'rmse': d['current_metrics']['rmse'],
            'mae': d['current_metrics']['mae'],
            'rmse_baseline': d['baseline_metrics']['rmse'],
            'mae_baseline': d['baseline_metrics']['mae'],
            'alerts': len(d['alerts'])
        }
        for d in monitoring_data
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Create subplot figure
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Model Performance Metrics', 'Performance vs Baseline', 'Alert Frequency'),
        vertical_spacing=0.1
    )

    # Performance metrics
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['rmse'], name='RMSE', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['mae'], name='MAE', line=dict(color='red')),
        row=1, col=1
    )

    # Performance vs baseline
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['rmse']/df['rmse_baseline'],
                  name='RMSE vs Baseline', line=dict(color='lightblue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['mae']/df['mae_baseline'],
                  name='MAE vs Baseline', line=dict(color='pink')),
        row=2, col=1
    )

    # Alert frequency
    fig.add_trace(
        go.Bar(x=df['timestamp'], y=df['alerts'], name='Alerts'),
        row=3, col=1
    )

    # Update layout
    fig.update_layout(
        height=900,
        title_text='Model Monitoring Dashboard',
        showlegend=True
    )

    # Save dashboard
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_dir / 'monitoring_dashboard.html')

    # Generate summary stats
    summary = {
        'latest_metrics': {
            'rmse': float(df['rmse'].iloc[-1]),
            'mae': float(df['mae'].iloc[-1])
        },
        'baseline_comparison': {
            'rmse_vs_baseline': float(df['rmse'].iloc[-1] / df['rmse_baseline'].iloc[-1]),
            'mae_vs_baseline': float(df['mae'].iloc[-1] / df['mae_baseline'].iloc[-1])
        },
        'alert_summary': {
            'total_alerts': int(df['alerts'].sum()),
            'recent_alerts': int(df['alerts'].tail(7).sum())
        },
        'generated_at': datetime.now().isoformat()
    }

    with open(output_dir / 'dashboard_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Generate monitoring dashboard")
    parser.add_argument("--monitoring-dir", default="monitoring_results",
                      help="Directory with monitoring results")
    parser.add_argument("--output-dir", default="reports",
                      help="Output directory for dashboard")
    args = parser.parse_args()

    monitoring_data = load_monitoring_data(args.monitoring_dir)
    create_dashboard(monitoring_data, args.output_dir)
    print(f"Dashboard generated in {args.output_dir}/monitoring_dashboard.html")

if __name__ == "__main__":
    main()
