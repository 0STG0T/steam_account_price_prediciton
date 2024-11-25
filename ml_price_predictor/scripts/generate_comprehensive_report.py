#!/usr/bin/env python3
"""Generate comprehensive monitoring report combining all monitoring aspects."""

import argparse
import json
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_monitoring_data(monitoring_dir: Path, schema_file: Path, health_file: Path):
    """Load all monitoring-related data."""
    data = {
        'timestamp': datetime.now().isoformat(),
        'monitoring_files': [],
        'schema_validation': None,
        'health_status': None
    }

    try:
        # Load monitoring files
        for file in sorted(monitoring_dir.glob('monitoring_*.json')):
            with open(file) as f:
                data['monitoring_files'].append(json.load(f))

        # Load schema validation results
        if schema_file.exists():
            with open(schema_file) as f:
                data['schema_validation'] = json.load(f)

        # Load health report with proper error handling
        if health_file.exists():
            try:
                with open(health_file) as f:
                    data['health_status'] = json.load(f)
                print(f"Loaded health report from {health_file}")
            except json.JSONDecodeError as e:
                print(f"Error decoding health report: {e}")
            except Exception as e:
                print(f"Error loading health report: {e}")
    except Exception as e:
        print(f"Error loading data: {e}")

    return data

def generate_html_report(data: dict, output_file: Path):
    """Generate HTML report with all monitoring information."""
    # Create plotly figure
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('RMSE Over Time', 'MAE Over Time', 'Drift P-Value',
                       'Alerts', 'Health Status', 'Drift Status'),
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "indicator"}, {"type": "indicator"}]]
    )

    # Extract time series data
    timestamps = []
    rmse_values = []
    mae_values = []
    drift_values = []
    alerts = []

    for entry in data['monitoring_files']:
        timestamps.append(entry['timestamp'])
        rmse_values.append(entry['current_metrics']['rmse'])
        mae_values.append(entry['current_metrics']['mae'])
        drift_values.append(entry.get('drift_metrics', {}).get('p_value', 1.0))
        alerts.append(len(entry.get('alerts', [])))

    # Add traces
    fig.add_trace(go.Scatter(x=timestamps, y=rmse_values, name='RMSE'), row=1, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=mae_values, name='MAE'), row=1, col=2)
    fig.add_trace(go.Scatter(x=timestamps, y=drift_values, name='Drift P-Value',
                            hovertemplate='p-value: %{y:.3f}'), row=1, col=3)
    fig.add_trace(go.Bar(x=timestamps, y=alerts, name='Alerts'), row=2, col=1)

    # Add health status indicator
    if data['health_status']:
        status = data['health_status'].get('status', 'unknown')
        drift_detected = data['health_status'].get('drift_detected', False)

        # Health Status
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=100 if status == 'passing' else 50 if status == 'warning' else 0,
                gauge={'axis': {'range': [0, 100]},
                       'steps': [
                           {'range': [0, 30], 'color': "red"},
                           {'range': [30, 70], 'color': "yellow"},
                           {'range': [70, 100], 'color': "green"}
                       ]},
                title={'text': f"Health Status: {status.upper()}"}
            ),
            row=2, col=2
        )

        # Drift Status
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=100 if not drift_detected else 0,
                gauge={'axis': {'range': [0, 100]},
                       'steps': [
                           {'range': [0, 30], 'color': "red"},
                           {'range': [30, 70], 'color': "yellow"},
                           {'range': [70, 100], 'color': "green"}
                       ]},
                title={'text': f"Drift Status: {'NO DRIFT' if not drift_detected else 'DRIFT DETECTED'}"}
            ),
            row=2, col=3
        )

    # Update layout
    fig.update_layout(height=800, showlegend=True, title_text="ML Price Predictor Monitoring Report")

    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Monitoring Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .section {{ margin: 20px 0; padding: 10px; border: 1px solid #ddd; }}
            .status-passing {{ color: green; }}
            .status-warning {{ color: orange; }}
            .status-failing {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>ML Price Predictor Monitoring Report</h1>
        <div class="section">
            <h2>Schema Validation</h2>
            <pre>{json.dumps(data['schema_validation'], indent=2) if data['schema_validation'] else 'No schema validation data'}</pre>
        </div>
        <div class="section">
            <h2>Health Status</h2>
            <pre>{json.dumps(data['health_status'], indent=2) if data['health_status'] else 'No health status data'}</pre>
        </div>
        <div class="section">
            <h2>Monitoring Metrics</h2>
            {fig.to_html(full_html=False)}
        </div>
    </body>
    </html>
    """

    output_file.write_text(html_content)
    print(f"Generated comprehensive report: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive monitoring report")
    parser.add_argument("--monitoring-dir", default="monitoring_results",
                      help="Directory containing monitoring files")
    parser.add_argument("--schema-file", default="reports/schema_validation.json",
                      help="Schema validation results file")
    parser.add_argument("--health-file", default="reports/health_report.json",
                      help="Health report file")
    parser.add_argument("--output-file", default="reports/comprehensive_report.html",
                      help="Output HTML report file")
    args = parser.parse_args()

    monitoring_dir = Path(args.monitoring_dir)
    schema_file = Path(args.schema_file)
    health_file = Path(args.health_file)
    output_file = Path(args.output_file)

    data = load_monitoring_data(monitoring_dir, schema_file, health_file)
    generate_html_report(data, output_file)

if __name__ == "__main__":
    main()
