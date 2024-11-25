# Monitoring Configuration Documentation

## Overview
This document describes all available configuration options for the ML Price Predictor monitoring system.

## Configuration Schema

### Performance Thresholds
- `rmse_degradation` (float): Maximum allowed RMSE increase relative to baseline (e.g., 0.1 = 10%)
- `mae_degradation` (float): Maximum allowed MAE increase relative to baseline
- `latency_threshold_ms` (float): Maximum allowed prediction latency in milliseconds

### Drift Thresholds
- `p_value_threshold` (float): P-value threshold for drift detection tests (default: 0.05)
- `min_samples_required` (int): Minimum number of samples required for drift detection

### Alert Settings
- `enable_email_alerts` (bool): Enable/disable email alerting
- `check_interval_minutes` (int): Interval between monitoring checks
- `consecutive_failures_before_alert` (int): Number of consecutive failures before triggering alert

### Retention Policy
- `max_age_days` (int): Maximum age of monitoring files before cleanup
- `min_reports_to_keep` (int): Minimum number of reports to retain regardless of age

### Optional Settings
- `email_recipients` (list): List of email addresses for alerts
- `custom_metrics` (dict): Additional custom metrics to track
  - `r2_score_threshold` (float): Minimum required R² score
  - `prediction_latency_p95_ms` (float): 95th percentile latency threshold

## Example Configuration
```json
{
    "performance_thresholds": {
        "rmse_degradation": 0.1,
        "mae_degradation": 0.1,
        "latency_threshold_ms": 100.0
    },
    "drift_thresholds": {
        "p_value_threshold": 0.05,
        "min_samples_required": 1000
    },
    "alert_settings": {
        "enable_email_alerts": false,
        "check_interval_minutes": 60,
        "consecutive_failures_before_alert": 3
    },
    "retention_policy": {
        "max_age_days": 30,
        "min_reports_to_keep": 10
    }
}
```

## Usage
1. Copy the example configuration
2. Modify thresholds according to your requirements
3. Validate configuration using:
   ```bash
   make monitor-schema
   ```
4. Run monitoring pipeline:
   ```bash
   make monitor
   ```
