#!/usr/bin/env python3
"""Validate monitoring directory structure and file formats."""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

REQUIRED_DIRECTORIES = [
    'monitoring_results',
    'reports',
    'logs/monitoring'
]

REQUIRED_FILES = {
    'configs/monitoring_config.json': {
        'required_fields': ['performance_thresholds', 'drift_thresholds', 'alert_settings']
    },
    'reports/schema_validation.json': {
        'required_fields': ['valid', 'errors']
    },
    'reports/health_report.json': {
        'required_fields': ['status', 'metrics']
    }
}

def validate_directory_structure(base_dir: Path) -> List[str]:
    """Validate required directories exist."""
    errors = []
    for dir_path in REQUIRED_DIRECTORIES:
        full_path = base_dir / dir_path
        if not full_path.exists():
            errors.append(f"Missing required directory: {dir_path}")
        elif not full_path.is_dir():
            errors.append(f"Path exists but is not a directory: {dir_path}")
    return errors

def validate_file_structure(base_dir: Path) -> List[str]:
    """Validate required files exist with correct format."""
    errors = []
    for file_path, requirements in REQUIRED_FILES.items():
        full_path = base_dir / file_path
        if not full_path.exists():
            errors.append(f"Missing required file: {file_path}")
            continue

        try:
            with open(full_path) as f:
                content = json.load(f)
                for field in requirements['required_fields']:
                    if field not in content:
                        errors.append(f"Missing required field '{field}' in {file_path}")
        except json.JSONDecodeError:
            errors.append(f"Invalid JSON format in file: {file_path}")
        except Exception as e:
            errors.append(f"Error reading file {file_path}: {str(e)}")

    return errors

def validate_monitoring_files(monitoring_dir: Path) -> List[str]:
    """Validate monitoring file format and content."""
    errors = []
    monitoring_files = list(monitoring_dir.glob('monitoring_*.json'))

    if not monitoring_files:
        errors.append("No monitoring files found")
        return errors

    for file_path in monitoring_files:
        try:
            with open(file_path) as f:
                content = json.load(f)
                required_fields = ['timestamp', 'current_metrics', 'baseline_metrics']
                for field in required_fields:
                    if field not in content:
                        errors.append(f"Missing required field '{field}' in {file_path.name}")

                # Validate timestamp format
                try:
                    datetime.fromisoformat(content['timestamp'])
                except ValueError:
                    errors.append(f"Invalid timestamp format in {file_path.name}")

                # Validate metrics format
                for metric_type in ['current_metrics', 'baseline_metrics']:
                    if metric_type in content:
                        metrics = content[metric_type]
                        if not isinstance(metrics, dict):
                            errors.append(f"Invalid {metric_type} format in {file_path.name}")
                        elif not all(isinstance(v, (int, float)) for v in metrics.values()):
                            errors.append(f"Invalid metric values in {file_path.name}")
        except Exception as e:
            errors.append(f"Error processing {file_path.name}: {str(e)}")

    return errors

def main():
    parser = argparse.ArgumentParser(description="Validate monitoring directory structure")
    parser.add_argument("--base-dir", default=".",
                      help="Base directory of the project")
    parser.add_argument("--output-file", default="reports/structure_validation.json",
                      help="Output file for validation results")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Error: Base directory not found: {args.base_dir}")
        return

    # Collect all validation errors
    all_errors = []
    all_errors.extend(validate_directory_structure(base_dir))
    all_errors.extend(validate_file_structure(base_dir))
    all_errors.extend(validate_monitoring_files(base_dir / 'monitoring_results'))

    # Generate validation report
    validation_result = {
        'timestamp': datetime.now().isoformat(),
        'valid': len(all_errors) == 0,
        'errors': all_errors
    }

    output_file = base_dir / args.output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(validation_result, f, indent=2)

    if all_errors:
        print("Structure validation failed:")
        for error in all_errors:
            print(f"- {error}")
    else:
        print("Structure validation passed!")

if __name__ == "__main__":
    main()
