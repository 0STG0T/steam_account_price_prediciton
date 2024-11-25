import os
from datetime import datetime
import json
import yaml
from pathlib import Path

class LogManager:
    """Centralized logging management for the ML Price Predictor project."""

    def __init__(self, base_log_dir="logs"):
        self.base_log_dir = Path(base_log_dir)
        self.subdirs = {
            'training': self.base_log_dir / 'training',
            'validation': self.base_log_dir / 'validation',
            'prediction': self.base_log_dir / 'prediction',
            'monitoring': self.base_log_dir / 'monitoring',
            'reports': self.base_log_dir / 'reports',
            'metrics': self.base_log_dir / 'metrics'
        }
        self._ensure_directories()

    def _ensure_directories(self):
        """Create all required log directories if they don't exist."""
        for dir_path in self.subdirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_log_path(self, log_type, filename=None):
        """Get the full path for a log file."""
        if log_type not in self.subdirs:
            raise ValueError(f"Invalid log type: {log_type}")

        if filename is None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"{log_type}_log_{timestamp}.csv"

        return str(self.subdirs[log_type] / filename)

    def save_metrics(self, metrics, model_name):
        """Save metrics to a JSON file."""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"metrics_{model_name}_{timestamp}.json"
        filepath = self.subdirs['metrics'] / filename

        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)

        return str(filepath)

    def save_report(self, report_content, report_type):
        """Save HTML or text reports."""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"{report_type}_report_{timestamp}.html"
        filepath = self.subdirs['reports'] / filename

        with open(filepath, 'w') as f:
            f.write(report_content)

        return str(filepath)

    def cleanup_old_logs(self, days_to_keep=30):
        """Clean up logs older than specified days."""
        # Implementation for log rotation and cleanup
        pass
