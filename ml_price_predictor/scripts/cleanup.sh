#!/bin/bash

# Cleanup script for ml_price_predictor
# Removes all generated data, logs, plots, and cache files

echo "Cleaning up generated files and directories..."

# Create and ensure backup directory exists
mkdir -p .tmp_backup

# Backup important baseline files
if [ -f "models/trained/baseline_metrics.json" ]; then
    cp -f models/trained/baseline_metrics.json .tmp_backup/
fi
if [ -f "reports/schema_validation.json" ]; then
    cp -f reports/schema_validation.json .tmp_backup/
fi
if [ -f "reports/health_report.json" ]; then
    cp -f reports/health_report.json .tmp_backup/
fi

# Remove logs
rm -rf logs/predictions/*
rm -rf logs/training/*
rm -rf logs/validation/*

# Remove evaluation results
rm -rf evaluation_results/*
rm -rf validation_results/*

# Remove monitoring results and reports
rm -rf monitoring_results/*
rm -rf reports/*

# Remove model files
rm -rf models/trained/*
rm -rf models/onnx/*

# Remove plots
rm -rf plots/*

# Remove cache files
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete
find . -type f -name ".coverage" -delete
find . -type d -name "*.egg-info" -exec rm -rf {} +
find . -type d -name "*.egg" -exec rm -rf {} +
find . -type d -name ".pytest_cache" -exec rm -rf {} +
find . -type d -name ".eggs" -exec rm -rf {} +

# Remove temporary files
rm -rf tmp/*
rm -f .DS_Store
rm -rf .ipynb_checkpoints

# Create necessary directories
mkdir -p logs/{predictions,training,validation}
mkdir -p evaluation_results
mkdir -p validation_results
mkdir -p monitoring_results
mkdir -p reports
mkdir -p models/{trained,onnx}
mkdir -p plots
mkdir -p tmp

# Restore important baseline files with error checking
if [ -f ".tmp_backup/baseline_metrics.json" ]; then
    cp -f .tmp_backup/baseline_metrics.json models/trained/
fi
if [ -f ".tmp_backup/schema_validation.json" ]; then
    cp -f .tmp_backup/schema_validation.json reports/
fi
if [ -f ".tmp_backup/health_report.json" ]; then
    cp -f .tmp_backup/health_report.json reports/
fi

# Clean up backup directory
rm -rf .tmp_backup

echo "Cleanup complete!"
