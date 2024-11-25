#!/usr/bin/env python3
"""Validate monitoring data format and consistency."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

REQUIRED_FIELDS = {
    'current_metrics': {
        'rmse': float,
        'mae': float
    },
    'baseline_metrics': {
        'rmse': float,
        'mae': float
    },
    'timestamp': str,
    'alerts': list
}

def validate_data_structure(data: Dict[str, Any], filename: str) -> List[str]:
    """Validate monitoring data structure."""
    errors = []

    def check_field(data: Dict[str, Any], field: str, expected_type: Any, path: str = '') -> None:
        if field not in data:
            errors.append(f"{filename}: Missing required field: {path}{field}")
            return

        if isinstance(expected_type, dict):
            if not isinstance(data[field], dict):
                errors.append(f"{filename}: {path}{field} should be a dictionary")
                return
            for subfield, subtype in expected_type.items():
                check_field(data[field], subfield, subtype, f"{path}{field}.")
        else:
            if not isinstance(data[field], expected_type):
                errors.append(f"{filename}: {path}{field} should be of type {expected_type.__name__}")

    for field, expected_type in REQUIRED_FIELDS.items():
        check_field(data, field, expected_type)

    # Validate timestamp format
    try:
        datetime.fromisoformat(data['timestamp'])
    except (ValueError, KeyError):
        errors.append(f"{filename}: Invalid timestamp format")

    return errors

def main():
    parser = argparse.ArgumentParser(description="Validate monitoring data format")
    parser.add_argument("--monitoring-dir", default="monitoring_results",
                      help="Directory containing monitoring files")
    args = parser.parse_args()

    monitoring_dir = Path(args.monitoring_dir)
    if not monitoring_dir.exists():
        print(f"Error: Directory not found: {args.monitoring_dir}")
        sys.exit(1)

    all_errors = []
    for file in monitoring_dir.glob('monitoring_*.json'):
        try:
            with open(file) as f:
                data = json.load(f)
            errors = validate_data_structure(data, file.name)
            all_errors.extend(errors)
        except json.JSONDecodeError:
            all_errors.append(f"{file.name}: Invalid JSON format")

    if all_errors:
        print("Validation errors found:")
        for error in all_errors:
            print(f"- {error}")
        sys.exit(1)
    else:
        print("All monitoring data files are valid!")

if __name__ == "__main__":
    main()
