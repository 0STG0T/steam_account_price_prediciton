#!/usr/bin/env python3
"""Monitor model performance and data drift in production."""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self, baseline_stats_path=None):
        """Initialize model monitor with optional baseline statistics."""
        self.baseline_stats = {}
        if baseline_stats_path:
            self.load_baseline_stats(baseline_stats_path)

    def compute_distribution_stats(self, data):
        """Compute distribution statistics for numerical columns."""
        stats_dict = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            stats_dict[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'median': data[col].median(),
                'q1': data[col].quantile(0.25),
                'q3': data[col].quantile(0.75)
            }
        return stats_dict

    def detect_drift(self, current_data, threshold=0.05):
        """Detect data drift using statistical tests."""
        drift_report = {}
        for col in self.baseline_stats.keys():
            if col in current_data:
                # Perform Kolmogorov-Smirnov test
                ks_stat, p_value = stats.ks_2samp(
                    self.baseline_stats[col]['data'],
                    current_data[col]
                )
                drift_report[col] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'drift_detected': p_value < threshold
                }
        return drift_report

    def log_predictions(self, predictions, actuals, timestamp=None):
        """Log prediction metrics over time."""
        if timestamp is None:
            timestamp = datetime.now()

        metrics = {
            'timestamp': timestamp,
            'mae': np.mean(np.abs(predictions - actuals)),
            'rmse': np.sqrt(np.mean((predictions - actuals) ** 2)),
            'correlation': np.corrcoef(predictions, actuals)[0, 1]
        }

        return metrics

    def save_monitoring_report(self, metrics, drift_report, output_dir):
        """Save monitoring results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path(output_dir) / f'monitoring_report_{timestamp}.json'

        report = {
            'metrics': metrics,
            'drift_analysis': drift_report,
            'timestamp': timestamp
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([report]).to_json(output_path, orient='records')
        logger.info(f"Monitoring report saved to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor model performance")
    parser.add_argument("--predictions-path", required=True, help="Path to predictions file")
    parser.add_argument("--actuals-path", required=True, help="Path to actuals file")
    parser.add_argument("--baseline-stats", help="Path to baseline statistics")
    parser.add_argument("--output-dir", default="monitoring_results", help="Output directory")

    args = parser.parse_args()

    monitor = ModelMonitor(args.baseline_stats)
    predictions = pd.read_csv(args.predictions_path)
    actuals = pd.read_csv(args.actuals_path)

    metrics = monitor.log_predictions(predictions['prediction'].values, actuals['actual'].values)
    drift_report = monitor.detect_drift(actuals) if args.baseline_stats else {}

    monitor.save_monitoring_report(metrics, drift_report, args.output_dir)
