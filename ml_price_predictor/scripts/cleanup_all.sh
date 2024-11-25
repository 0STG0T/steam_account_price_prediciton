#!/bin/bash

# Script to clean up all generated files and logs

echo "Starting comprehensive cleanup..."

# Clean Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete
find . -type f -name ".coverage" -delete
find . -type d -name "*.egg-info" -exec rm -rf {} +
find . -type d -name "*.egg" -exec rm -rf {} +
find . -type d -name ".pytest_cache" -exec rm -rf {} +
find . -type d -name ".eggs" -exec rm -rf {} +

# Clean generated data directories
rm -rf data/raw/*.json
rm -rf data/processed/*.csv

# Clean model artifacts (preserve baseline metrics and configs)
find models/trained -type f ! -name 'baseline_metrics.json' -delete
rm -rf models/onnx/*

# Clean logs and results
rm -rf logs/predictions/*
rm -rf logs/training/*
rm -rf logs/validation/*
rm -rf evaluation_results/*
rm -rf monitoring_results/*
rm -rf reports/*

# Clean temporary files
rm -rf .pytest_cache/
rm -rf .coverage
rm -rf htmlcov/
rm -rf dist/
rm -rf build/

# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models/trained
mkdir -p models/onnx
mkdir -p logs/predictions
mkdir -p logs/training
mkdir -p logs/validation
mkdir -p evaluation_results
mkdir -p monitoring_results
mkdir -p reports

echo "Cleanup complete!"
