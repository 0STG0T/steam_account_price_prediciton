#!/usr/bin/env python3
"""Validate monitoring configuration parameters."""

import argparse
import json
import sys
from pathlib import Path

def validate_config(config):
    """Validate monitoring configuration parameters."""
    required_sections = ['performance_thresholds', 'drift_thresholds', 'alert_settings']
    required_params = {
        'performance_thresholds': {
            'rmse_degradation': (float, 0.0, 1.0),
            'mae_degradation': (float, 0.0, 1.0),
            'latency_threshold_ms': (float, 0.0, 1000.0)
        },
        'drift_thresholds': {
            'p_value_threshold': (float, 0.0, 1.0),
            'min_samples_required': (int, 100, 1000000)
        },
        'alert_settings': {
            'enable_email_alerts': (bool, None, None),
            'check_interval_minutes': (int, 1, 1440),
            'consecutive_failures_before_alert': (int, 1, 10)
        }
    }

    errors = []

    # Check required sections
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
            continue

        # Check required parameters
        for param, (param_type, min_val, max_val) in required_params[section].items():
            if param not in config[section]:
                errors.append(f"Missing parameter: {section}.{param}")
                continue

            value = config[section][param]
            if not isinstance(value, param_type):
                errors.append(f"Invalid type for {section}.{param}: expected {param_type.__name__}")
                continue

            if min_val is not None and max_val is not None:
                if value < min_val or value > max_val:
                    errors.append(f"Invalid value for {section}.{param}: must be between {min_val} and {max_val}")

    return errors

def main():
    parser = argparse.ArgumentParser(description="Validate monitoring configuration")
    parser.add_argument("--config-file", default="configs/monitoring_config.json", help="Path to config file")
    args = parser.parse_args()

    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config_file}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    errors = validate_config(config)

    if errors:
        print("Configuration validation failed:")
        for error in errors:
            print(f"- {error}")
        sys.exit(1)
    else:
        print("Configuration validation successful")

if __name__ == "__main__":
    main()
