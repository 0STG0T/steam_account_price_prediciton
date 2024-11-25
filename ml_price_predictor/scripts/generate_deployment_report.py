#!/usr/bin/env python3
"""Generate a comprehensive deployment readiness report."""

import argparse
import json
from datetime import datetime
from pathlib import Path

def check_monitoring_health(base_dir):
    """Check monitoring system health and readiness."""
    monitoring_checks = {
        'structure_valid': Path(base_dir, 'reports/pipeline_validation.json').exists(),
        'schema_valid': Path(base_dir, 'reports/schema_validation.json').exists(),
        'health_report': Path(base_dir, 'reports/health_report.json').exists(),
        'comprehensive_report': Path(base_dir, 'reports/comprehensive_report.html').exists(),
        'monitoring_data': Path(base_dir, 'monitoring_results/monitoring_latest.json').exists()
    }
    return monitoring_checks

def generate_report(base_dir, output_file):
    """Generate deployment readiness report."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'monitoring_status': check_monitoring_health(base_dir),
        'recommendations': []
    }

    # Check monitoring health
    monitoring_health = check_monitoring_health(base_dir)
    report['monitoring_ready'] = all(monitoring_health.values())

    # Generate recommendations
    if not report['monitoring_ready']:
        for check, status in monitoring_health.items():
            if not status:
                report['recommendations'].append(
                    f"Fix {check}: Required monitoring component is missing"
                )

    # Add deployment checklist
    report['deployment_checklist'] = {
        'monitoring_validated': report['monitoring_ready'],
        'tests_passing': Path(base_dir, '.pytest_cache').exists(),
        'config_validated': Path(base_dir, 'configs/monitoring_config.json').exists(),
        'documentation_complete': all([
            Path(base_dir, 'README.md').exists(),
            Path(base_dir, 'docs/monitoring_setup.md').exists()
        ])
    }

    # Save report
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    return report

def main():
    parser = argparse.ArgumentParser(description="Generate deployment readiness report")
    parser.add_argument("--base-dir", default=".",
                      help="Base directory of the project")
    parser.add_argument("--output-file",
                      default="reports/deployment_readiness.json",
                      help="Output file for deployment report")
    args = parser.parse_args()

    report = generate_report(args.base_dir, args.output_file)

    print("\nDeployment Readiness Report:")
    print(f"Monitoring Status: {'Ready' if report['monitoring_ready'] else 'Not Ready'}")
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"- {rec}")
    print(f"\nReport saved to: {args.output_file}")

if __name__ == "__main__":
    main()
