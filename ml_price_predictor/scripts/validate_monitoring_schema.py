#!/usr/bin/env python3
"""Validate monitoring configuration against schema."""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

MONITORING_SCHEMA = {
    'required_fields': {
        'performance_thresholds': {
            'rmse_degradation': float,
            'mae_degradation': float,
            'latency_threshold_ms': float
        },
        'drift_thresholds': {
            'p_value_threshold': float,
            'min_samples_required': int
        },
        'alert_settings': {
            'enable_email_alerts': bool,
            'check_interval_minutes': int,
            'consecutive_failures_before_alert': int
        },
        'retention_policy': {
            'max_age_days': int,
            'min_reports_to_keep': int
        }
    },
    'optional_fields': {
        'email_recipients': list,
        'custom_metrics': dict
    }
}

def validate_schema(config: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Validate configuration against schema."""
    errors = []

    def check_type(value: Any, expected_type: Any, path: str) -> None:
        if not isinstance(value, expected_type):
            errors.append(f"Invalid type for {path}: expected {expected_type.__name__}, got {type(value).__name__}")

    def validate_section(config_section: Dict[str, Any], schema_section: Dict[str, Any], path: str = '') -> None:
        for field, expected_type in schema_section.items():
            field_path = f"{path}.{field}" if path else field
            if field not in config_section:
                errors.append(f"Missing required field: {field_path}")
                continue

            if isinstance(expected_type, dict):
                if not isinstance(config_section[field], dict):
                    errors.append(f"Invalid type for {field_path}: expected dict")
                    continue
                validate_section(config_section[field], expected_type, field_path)
            else:
                check_type(config_section[field], expected_type, field_path)

    # Validate required fields
    validate_section(config, MONITORING_SCHEMA['required_fields'])

    # Validate optional fields if present
    for field, expected_type in MONITORING_SCHEMA['optional_fields'].items():
        if field in config:
            check_type(config[field], expected_type, field)

    return errors

def main():
    parser = argparse.ArgumentParser(description="Validate monitoring configuration schema")
    parser.add_argument("--config-file", required=True,
                      help="Path to monitoring configuration file")
    parser.add_argument("--output-file", default="schema_validation.json",
                      help="Output file for validation results")
    args = parser.parse_args()

    try:
        with open(args.config_file) as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error reading configuration file: {e}")
        return

    errors = validate_schema(config, MONITORING_SCHEMA)

    validation_result = {
        'valid': len(errors) == 0,
        'errors': errors
    }

    with open(args.output_file, 'w') as f:
        json.dump(validation_result, f, indent=2)

    if errors:
        print("Schema validation failed:")
        for error in errors:
            print(f"- {error}")
    else:
        print("Schema validation passed!")

if __name__ == "__main__":
    main()
