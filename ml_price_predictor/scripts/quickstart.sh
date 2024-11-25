#!/bin/bash
set -e

echo "ML Price Predictor Quick Start"
echo "============================="

# Create required directories
echo "0. Creating directories..."
mkdir -p data/raw data/processed models/trained models/onnx evaluation_results validation_results monitoring_results reports logs/{training,validation,prediction}

# Setup
echo "1. Installing dependencies..."
make install

# Generate sample data
echo "2. Generating sample data..."
make generate-data

# Train and validate model
echo "3. Training model..."
python src/train.py \
    --category-id 1 \
    --data-path data/raw/sample_train.json \
    --output-dir models/trained/

echo "4. Exporting to ONNX..."
python src/export_onnx.py \
    --model-path models/trained/category_1_model.cbm \
    --output-path models/onnx/category_1_model.onnx

echo "5. Running evaluation..."
make evaluate

# Setup monitoring
echo "6. Setting up monitoring..."
make monitor-test

echo "Quick start complete! Check reports/ for comprehensive evaluation and monitoring results."
