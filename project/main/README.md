# Price Prediction Model CLI Tools

This directory contains command-line tools for training, validating, and making predictions with category-specific price prediction models.

## Installation

Install required dependencies:

```bash
pip install pandas numpy scikit-learn tqdm catboost onnxruntime statsmodels matplotlib seaborn
```

## Available Scripts

### 1. Training (`train.py`)

Train a category-specific price prediction model:

```bash
python train.py --category-id 1 \
                --data-path data/training_data.json \
                --output-dir models/ \
                --test-size 0.078 \
                --random-state 42
```

### 2. Validation (`validate.py`)

Validate a trained model and generate performance plots:

```bash
python validate.py --model-path models/category_1_model.onnx \
                  --data-path data/validation_data.json \
                  --category-id 1 \
                  --output-dir validation_results/ \
                  --plot-name validation_plot.png
```

### 3. ONNX Export (`export_onnx.py`)

Export a trained model to ONNX format:

```bash
python export_onnx.py --model-path models/category_1_model.pkl \
                     --output-path models/category_1_model.onnx \
                     --category-id 1
```

### 4. Prediction (`predict.py`)

Make predictions using a trained ONNX model:

```bash
python predict.py --model-path models/category_1_model.onnx \
                 --data-path data/test_data.json \
                 --output-path predictions/results.csv \
                 --category-id 1
```

## Logging

All operations are logged with timestamps in the `logs/` directory:
- Training metrics: `logs/training_YYYY-MM-DD_HH-MM-SS.csv`
- Validation results: `logs/validation_YYYY-MM-DD_HH-MM-SS.csv`
- Prediction summaries: `logs/prediction_YYYY-MM-DD_HH-MM-SS.csv`

## Directory Structure

```
.
├── train.py           # Training script
├── validate.py        # Validation script
├── export_onnx.py     # ONNX export script
├── predict.py         # Prediction script
├── models/           # Directory for saved models
├── logs/             # Directory for CSV logs
└── validation_results/ # Directory for validation plots
```

## Notes

- All scripts use consistent random seeds (default: 42) for reproducibility
- Validation results include MAE, MSE, RMSE, R² Score, and Pearson Correlation
- Predictions are saved in CSV format with timestamps
- All operations are logged for tracking and analysis
