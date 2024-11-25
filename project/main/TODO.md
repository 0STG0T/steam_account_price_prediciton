# Scripts Implementation Plan

## 1. train.py
Arguments:
- `--category-id`: Category ID to train model for
- `--data-path`: Path to training data JSON file
- `--output-dir`: Directory to save trained model
- `--test-size`: Test split size (default: 0.078)
- `--random-state`: Random seed (default: 42)

## 2. validate.py
Arguments:
- `--model-path`: Path to ONNX model
- `--data-path`: Path to validation data JSON file
- `--category-id`: Category ID for validation
- `--output-dir`: Directory to save validation results and plots
- `--plot-name`: Custom name for validation plot (optional)

## 3. export_onnx.py
Arguments:
- `--model-path`: Path to trained model
- `--output-path`: Path to save ONNX model
- `--category-id`: Category ID for model metadata

## 4. predict.py
Arguments:
- `--model-path`: Path to ONNX model
- `--data-path`: Path to input data JSON file
- `--output-path`: Path to save predictions CSV
- `--category-id`: Category ID for preprocessing

## Common Features Across Scripts:
1. CSV logging with timestamps for all operations
2. Error handling and validation
3. Progress bars using tqdm
4. Consistent data preprocessing
5. Detailed console output

## CSV Logging Format:
```
logs/
├── training_logs_YYYY-MM-DD_HH-MM-SS.csv
├── validation_logs_YYYY-MM-DD_HH-MM-SS.csv
└── prediction_logs_YYYY-MM-DD_HH-MM-SS.csv
```

## Implementation Order:
1. Create logging utility functions
2. Implement train.py
3. Implement export_onnx.py
4. Implement validate.py
5. Implement predict.py
