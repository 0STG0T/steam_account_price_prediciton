# Category-Specific Price Prediction Models

A machine learning project for training and deploying category-specific price prediction models with ONNX support.

## Overview

This project provides a framework for training, validating, and deploying machine learning models for price prediction across different product categories. The models can be exported to ONNX format for efficient deployment and inference.

## Key Features

- Category-specific model training
- ONNX model export and inference
- Model validation with performance metrics
- Model finetuning capabilities
- Easy-to-use inference API

## Project Structure

```
project/
├── main/
│   ├── models/
│   │   └── onnx/           # Exported ONNX models
│   ├── validation_plots/   # Model validation visualizations
│   ├── main_example.ipynb  # Main training pipeline example
│   ├── onnx_usage_example.ipynb  # ONNX inference example
│   └── single_cat_model.py # Core model implementation
```

## Usage

### Training Models

See `main_example.ipynb` for a complete training pipeline example. The notebook demonstrates:

1. Data preparation and category splitting
2. Model training for each category
3. Model export to ONNX format
4. Model validation with metrics and visualizations
5. Model finetuning capabilities

```python
from single_cat_model import SingleCategoryModel

# Initialize and train model
model = SingleCategoryModel(category_number=1)
model.train(category_data)

# Export model
model.export(output_path_onnx='./models/onnx/category_1_model.onnx')
```

### ONNX Inference

See `onnx_usage_example.ipynb` for inference using exported ONNX models:

```python
import onnxruntime as ort
from single_cat_model import SingleCategoryModel

# Load ONNX model
onnx_session = ort.InferenceSession("./models/onnx/category_1_model.onnx")

# Prepare data
model = SingleCategoryModel(category_number=1)
processed_data = model.preprocess_data(df)

# Run inference
predictions = predict_with_onnx(onnx_session, processed_data)
```

## Model Validation

The project includes built-in validation capabilities:

```python
metrics = model.validate(
    valid_df=test_data,
    save_plot_path='./validation_plots/model_performance.png'
)
```

## Model Finetuning

Models can be finetuned on new data:

```python
model.finetune(new_data)
metrics = model.validate(test_data, './validation_plots/finetuned_model.png')
```

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- onnxruntime
- matplotlib
- tqdm

## Getting Started

1. Clone the repository
2. Install dependencies
3. Use the example notebooks to understand the workflow
4. Adapt the code for your specific use case

For detailed examples, refer to the Jupyter notebooks in the project.
