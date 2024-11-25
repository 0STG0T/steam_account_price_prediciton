#!/usr/bin/env python3
"""Test monitoring infrastructure functionality."""

import json
import pytest
from pathlib import Path
from datetime import datetime, timedelta

def test_monitoring_config_validation(tmp_path):
    """Test monitoring configuration validation."""
    from scripts.validate_monitoring_config import validate_config

    valid_config = {
        'performance_thresholds': {
            'rmse_degradation': 0.1,
            'mae_degradation': 0.1,
            'latency_threshold_ms': 100.0
        },
        'drift_thresholds': {
            'p_value_threshold': 0.05,
            'min_samples_required': 1000
        },
        'alert_settings': {
            'enable_email_alerts': False,
            'check_interval_minutes': 60,
            'consecutive_failures_before_alert': 3
        }
    }

    assert validate_config(valid_config) == []

def test_monitoring_report_generation(tmp_path):
    """Test monitoring report generation."""
    from scripts.generate_monitoring_report import generate_report

    # Create sample monitoring data
    monitoring_dir = tmp_path / "monitoring"
    monitoring_dir.mkdir()

    sample_data = {
        'current_metrics': {'rmse': 50.0, 'mae': 30.0},
        'baseline_metrics': {'rmse': 45.0, 'mae': 28.0},
        'alerts': [],
        'timestamp': datetime.now().isoformat()
    }

    with open(monitoring_dir / "monitoring_latest.json", "w") as f:
        json.dump(sample_data, f)

    # Generate report
    report_dir = tmp_path / "reports"
    generate_report(monitoring_dir, report_dir)

    assert (report_dir / "monitoring_report.html").exists()

def test_monitoring_cleanup(tmp_path):
    """Test monitoring file cleanup."""
    from scripts.cleanup_monitoring import cleanup_monitoring_files

    # Create old and new monitoring files
    monitoring_dir = tmp_path / "monitoring"
    monitoring_dir.mkdir()

    old_date = datetime.now() - timedelta(days=40)
    new_date = datetime.now() - timedelta(days=5)

    old_file = monitoring_dir / f"monitoring_{old_date.strftime('%Y%m%d')}.json"
    new_file = monitoring_dir / f"monitoring_{new_date.strftime('%Y%m%d')}.json"

    sample_data = {
        'current_metrics': {'rmse': 50.0, 'mae': 30.0},
        'timestamp': datetime.now().isoformat()
    }

    old_file.write_text(json.dumps(sample_data))
    new_file.write_text(json.dumps(sample_data))

    # Run cleanup
    cleanup_monitoring_files(monitoring_dir, tmp_path / "reports", max_age_days=30)

    assert not old_file.exists()
    assert new_file.exists()
    assert (monitoring_dir / "metrics_history.json").exists()

def test_monitoring_data_validation(tmp_path):
    """Test monitoring data format validation."""
    from scripts.validate_monitoring_data import validate_data_structure

    # Valid monitoring data
    valid_data = {
        'current_metrics': {'rmse': 50.0, 'mae': 30.0},
        'baseline_metrics': {'rmse': 45.0, 'mae': 28.0},
        'timestamp': datetime.now().isoformat(),
        'alerts': []
    }
    assert validate_data_structure(valid_data, "test.json") == []

    # Invalid data - missing fields
    invalid_data = {
        'current_metrics': {'rmse': 50.0},
        'timestamp': 'invalid_date'
    }
    errors = validate_data_structure(invalid_data, "test.json")
    assert len(errors) > 0
    assert any('missing' in err.lower() for err in errors)

def test_monitoring_dependencies():
    """Test monitoring dependency checking."""
    import importlib
    from scripts.check_monitoring_deps import REQUIRED_PACKAGES

    # Verify all required packages can be imported
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            pytest.fail(f"Required package {package} is not installed")

def test_monitoring_dashboard(tmp_path):
    """Test monitoring dashboard generation."""
    from scripts.generate_monitoring_dashboard import create_dashboard

    # Create sample monitoring data
    monitoring_data = []
    for i in range(7):
        data = {
            'timestamp': (datetime.now() - timedelta(days=i)).isoformat(),
            'current_metrics': {'rmse': 50.0 + i, 'mae': 30.0 + i},
            'baseline_metrics': {'rmse': 45.0, 'mae': 28.0},
            'alerts': [f"Alert {j}" for j in range(i)],
        }
        monitoring_data.append(data)

    # Generate dashboard
    output_dir = tmp_path / "reports"
    create_dashboard(monitoring_data, output_dir)

    # Verify outputs
    assert (output_dir / "monitoring_dashboard.html").exists()
    assert (output_dir / "dashboard_summary.json").exists()

    # Verify summary content
    with open(output_dir / "dashboard_summary.json") as f:
        summary = json.load(f)
        assert "latest_metrics" in summary
        assert "baseline_comparison" in summary
        assert "alert_summary" in summary

def test_monitoring_schema_validation():
    """Test monitoring schema validation functionality."""
    from scripts.validate_monitoring_schema import validate_schema, MONITORING_SCHEMA

    # Test valid configuration
    valid_config = {
        'performance_thresholds': {
            'rmse_degradation': 0.1,
            'mae_degradation': 0.1,
            'latency_threshold_ms': 100.0
        },
        'drift_thresholds': {
            'p_value_threshold': 0.05,
            'min_samples_required': 1000
        },
        'alert_settings': {
            'enable_email_alerts': False,
            'check_interval_minutes': 60,
            'consecutive_failures_before_alert': 3
        },
        'retention_policy': {
            'max_age_days': 30,
            'min_reports_to_keep': 10
        }
    }
    assert len(validate_schema(valid_config, MONITORING_SCHEMA)) == 0

    # Test invalid configuration
    invalid_config = {
        'performance_thresholds': {
            'rmse_degradation': '0.1',  # Wrong type
            'latency_threshold_ms': 100.0
        }
    }
    errors = validate_schema(invalid_config, MONITORING_SCHEMA)
    assert len(errors) > 0
    assert any('Invalid type' in err for err in errors)
    assert any('Missing required field' in err for err in errors)

def test_health_report_generation(tmp_path):
    """Test health report generation functionality."""
    from scripts.generate_health_report import collect_health_metrics

    # Create sample monitoring data
    monitoring_dir = tmp_path / "monitoring"
    monitoring_dir.mkdir()

    sample_data = {
        'current_metrics': {'rmse': 50.0, 'mae': 30.0},
        'baseline_metrics': {'rmse': 45.0, 'mae': 28.0},
        'alerts': ['Test alert'],
        'timestamp': datetime.now().isoformat()
    }

    with open(monitoring_dir / "monitoring_latest.json", "w") as f:
        json.dump(sample_data, f)

    # Generate and verify health report
    health_report = collect_health_metrics(monitoring_dir)

    assert 'status' in health_report
    assert 'metrics' in health_report
    assert 'changes' in health_report['metrics']
    assert 'drift_detected' in health_report
    assert health_report['status'] == 'warning'  # Due to >10% metric degradation

def test_monitoring_structure_validation(tmp_path):
    """Test monitoring structure validation functionality."""
    from scripts.validate_monitoring_structure import (
        validate_directory_structure,
        validate_file_structure,
        validate_monitoring_files
    )

    # Create test directory structure
    base_dir = tmp_path
    for dir_path in ['monitoring_results', 'reports', 'logs/monitoring', 'configs']:
        (base_dir / dir_path).mkdir(parents=True)

    # Create required files with valid content
    config = {
        'performance_thresholds': {'rmse': 0.1},
        'drift_thresholds': {'p_value': 0.05},
        'alert_settings': {'enabled': True}
    }
    (base_dir / 'configs/monitoring_config.json').write_text(json.dumps(config))

    schema_validation = {'valid': True, 'errors': []}
    (base_dir / 'reports/schema_validation.json').write_text(json.dumps(schema_validation))

    health_report = {'status': 'passing', 'metrics': {'rmse': 0.1}}
    (base_dir / 'reports/health_report.json').write_text(json.dumps(health_report))

    # Create sample monitoring file
    monitoring_data = {
        'timestamp': datetime.now().isoformat(),
        'current_metrics': {'rmse': 0.1},
        'baseline_metrics': {'rmse': 0.09}
    }
    (base_dir / 'monitoring_results/monitoring_latest.json').write_text(json.dumps(monitoring_data))

    # Test validations
    assert len(validate_directory_structure(base_dir)) == 0
    assert len(validate_file_structure(base_dir)) == 0
    assert len(validate_monitoring_files(base_dir / 'monitoring_results')) == 0

    # Test with missing directory
    (base_dir / 'monitoring_results/monitoring_latest.json').unlink()  # Remove file first
    (base_dir / 'monitoring_results').rmdir()
    assert len(validate_directory_structure(base_dir)) > 0

    # Test with invalid monitoring file
    (base_dir / 'monitoring_results').mkdir()
    (base_dir / 'monitoring_results/monitoring_invalid.json').write_text('invalid json')
    assert len(validate_monitoring_files(base_dir / 'monitoring_results')) > 0

def test_monitoring_pipeline_validation(tmp_path):
    """Test monitoring pipeline validation functionality."""
    from scripts.validate_monitoring_pipeline import validate_pipeline
    from pathlib import Path
    import json

    # Create test directory structure
    base_dir = tmp_path
    for dir_path in ['monitoring_results', 'reports', 'logs/monitoring']:
        (base_dir / dir_path).mkdir(parents=True)

    # Create sample files
    monitoring_data = {
        'timestamp': datetime.now().isoformat(),
        'current_metrics': {'rmse': 0.1},
        'baseline_metrics': {'rmse': 0.09}
    }
    (base_dir / 'monitoring_results/monitoring_latest.json').write_text(json.dumps(monitoring_data))
    (base_dir / 'reports/schema_validation.json').write_text(json.dumps({'valid': True}))
    (base_dir / 'reports/health_report.json').write_text(json.dumps({'status': 'passing'}))
    (base_dir / 'reports/comprehensive_report.html').write_text('<html></html>')
    (base_dir / 'reports/monitoring_dashboard.html').write_text('<html></html>')

    # Run pipeline validation
    results = validate_pipeline(base_dir)

    # Verify results structure
    assert 'timestamp' in results
    assert 'steps' in results
    assert 'success' in results
    assert len(results['steps']) > 0

    # Verify step structure
    for step in results['steps']:
        assert 'name' in step
        assert 'success' in step
        assert 'details' in step

def test_deployment_readiness_report(tmp_path):
    """Test deployment readiness report generation."""
    from scripts.generate_deployment_report import generate_report

    # Create test directory structure
    base_dir = tmp_path
    for dir_path in ['reports', 'monitoring_results', '.pytest_cache', 'configs', 'docs']:
        (base_dir / dir_path).mkdir(parents=True)

    # Create required files
    (base_dir / 'README.md').write_text('# Project Documentation')
    (base_dir / 'docs/monitoring_setup.md').write_text('# Monitoring Setup')
    (base_dir / 'configs/monitoring_config.json').write_text('{}')
    (base_dir / 'reports/pipeline_validation.json').write_text('{}')
    (base_dir / 'reports/schema_validation.json').write_text('{}')
    (base_dir / 'reports/health_report.json').write_text('{}')
    (base_dir / 'reports/comprehensive_report.html').write_text('<html></html>')
    (base_dir / 'monitoring_results/monitoring_latest.json').write_text('{}')

    # Generate and verify report
    report = generate_report(base_dir, base_dir / 'reports/deployment_readiness.json')

    assert 'timestamp' in report
    assert 'monitoring_status' in report
    assert 'recommendations' in report
    assert 'deployment_checklist' in report
    assert all(report['monitoring_status'].values()), "All monitoring checks should pass"
    assert report['deployment_checklist']['documentation_complete'], "Documentation should be complete"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
