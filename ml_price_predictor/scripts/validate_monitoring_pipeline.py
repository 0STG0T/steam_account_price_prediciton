#!/usr/bin/env python3
"""Validate the entire monitoring pipeline end-to-end."""

import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime

def run_command(command):
    """Run a command and return its success status and output."""
    try:
        result = subprocess.run(command, shell=True, check=True,
                              capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def validate_pipeline(base_dir):
    """Run end-to-end validation of the monitoring pipeline."""
    results = {
        'timestamp': datetime.now().isoformat(),
        'steps': [],
        'success': True
    }

    pipeline_steps = [
        {
            'name': 'Directory Structure',
            'command': 'make monitor-structure',
            'required_files': ['monitoring_results', 'reports', 'logs/monitoring']
        },
        {
            'name': 'Sample Data Generation',
            'command': 'make monitor-sample-data',
            'required_files': ['monitoring_results/monitoring_latest.json']
        },
        {
            'name': 'Schema Validation',
            'command': 'make monitor-schema',
            'required_files': ['reports/schema_validation.json']
        },
        {
            'name': 'Health Report',
            'command': 'make monitor-health',
            'required_files': ['reports/health_report.json']
        },
        {
            'name': 'Comprehensive Report',
            'command': 'make monitor-comprehensive',
            'required_files': ['reports/comprehensive_report.html']
        },
        {
            'name': 'Dashboard Generation',
            'command': 'make monitor-dashboard',
            'required_files': ['reports/monitoring_dashboard.html']
        }
    ]

    for step in pipeline_steps:
        step_result = {
            'name': step['name'],
            'success': False,
            'details': []
        }

        # Run the command
        success, output = run_command(step['command'])
        step_result['success'] = success
        if not success:
            step_result['details'].append(f"Command failed: {output}")
            results['success'] = False

        # Verify required files
        for required_file in step['required_files']:
            file_path = Path(base_dir) / required_file
            if not file_path.exists():
                step_result['success'] = False
                step_result['details'].append(f"Missing required file: {required_file}")
                results['success'] = False

        results['steps'].append(step_result)

    return results

def main():
    parser = argparse.ArgumentParser(description="Validate monitoring pipeline")
    parser.add_argument("--base-dir", default=".",
                      help="Base directory of the project")
    parser.add_argument("--output-file",
                      default="reports/pipeline_validation.json",
                      help="Output file for validation results")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_file = base_dir / args.output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = validate_pipeline(base_dir)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if results['success']:
        print("Pipeline validation successful!")
    else:
        print("Pipeline validation failed. Check reports/pipeline_validation.json for details.")
        exit(1)

if __name__ == "__main__":
    main()
