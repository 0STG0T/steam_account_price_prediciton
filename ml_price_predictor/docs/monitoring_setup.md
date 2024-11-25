# ML Price Predictor Monitoring Setup Guide

## Overview
This guide explains how to set up and maintain the monitoring infrastructure for the ML Price Predictor system.

## Directory Structure
```
ml_price_predictor/
├── monitoring_results/    # Monitoring data files
├── reports/              # Generated reports
│   ├── comprehensive_report.html
│   ├── health_report.json
│   └── schema_validation.json
└── logs/monitoring/      # Monitoring logs
```

## Quick Start
```bash
# Initialize monitoring structure
make monitor-structure

# Generate initial reports
make monitor-comprehensive
```

## Configuration
1. Copy `configs/monitoring_config.example.json` to `configs/monitoring_config.json`
2. Adjust thresholds in monitoring_config.json:
   ```json
   {
     "performance_thresholds": {
       "rmse_degradation": 0.1,
       "mae_degradation": 0.1
     }
   }
   ```

## Validation Steps
1. Structure Validation
   ```bash
   make monitor-structure
   ```
2. Schema Validation
   ```bash
   make monitor-schema
   ```
3. Data Validation
   ```bash
   make monitor-validate-data
   ```

## Monitoring Reports
1. Health Report
   - Shows current model health
   - Tracks metric changes
   - Lists active alerts

2. Comprehensive Report
   - Interactive visualizations
   - Combined metrics view
   - Schema validation status
   - Historical performance

## Maintenance
1. Regular Cleanup
   ```bash
   make monitor-cleanup
   ```

2. Update Monitoring Badge
   ```bash
   make monitor-badge
   ```

## Troubleshooting
1. Check Dependencies
   ```bash
   make monitor-check-deps
   ```

2. Validate Configuration
   ```bash
   make monitor-validate-config
   ```

## Best Practices
1. Run complete monitoring pipeline daily
2. Review comprehensive report weekly
3. Update thresholds based on business requirements
4. Maintain historical data for trend analysis
5. Document custom metric additions

## Alert Handling
1. Warning Status:
   - Review comprehensive report
   - Check drift detection results
   - Analyze affected metrics

2. Failure Status:
   - Immediate investigation required
   - Check latest monitoring files
   - Review system logs
   - Validate data pipeline

## Extending Monitoring
1. Add Custom Metrics:
   - Update monitoring_config.json
   - Add metric collection logic
   - Update report generation

2. Custom Validations:
   - Add validation rules
   - Update schema validation
   - Test new validation logic
