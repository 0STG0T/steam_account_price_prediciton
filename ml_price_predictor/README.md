# ML Price Predictor

A production-ready machine learning system for predicting gaming account prices.

![Monitoring Status](https://img.shields.io/badge/monitoring-passing-brightgreen)

## Project Structure
```
ml_price_predictor/
├── src/                    # Source code
│   ├── models/            # ML model implementations
│   ├── data_processing/   # Data processing utilities
│   ├── utils/             # Common utilities
│   └── inference/         # Training and inference scripts
├── tests/                 # Test suite
├── configs/               # Configuration files
├── docs/                  # Documentation
├── notebooks/            # Jupyter notebooks
├── data/                 # Data directory
│   ├── raw/              # Raw input data
│   └── processed/        # Processed data
├── models/               # Saved models
│   ├── trained/         # CatBoost models
│   └── onnx/            # ONNX exported models
└── logs/                 # Logging directory
    ├── training/        # Training logs
    ├── validation/      # Validation logs
    └── prediction/      # Prediction logs
```

## Quick Start

### Quick Demo
```bash
# Clone the repository
git clone PATH_TO_YOUR_REPOSITORY

# Install dependencies
pip install -e ".[dev]"  # Required for plotting features (includes kaleido)
# Note: For inference-only, use: pip install -e .

# Run complete demo pipeline
./scripts/quickstart.sh
```
This script demonstrates the entire workflow from data generation to deployment, including monitoring setup.

### Installation
```bash
# Install package in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

1. Training a model:
```bash
python src/train.py \
    --category-id CATEGORY_ID \
    --data-path PATH_TO_YOUR_TRAINING_DATA \
    --output-dir PATH_TO_YOUR_MODEL_OUTPUT_DIR
```

2. Export to ONNX:
```bash
python src/export_onnx.py \
    --model-path PATH_TO_YOUR_CBM_MODEL \
    --output-path PATH_TO_YOUR_ONNX_MODEL
```

3. Making predictions:
```bash
python scripts/evaluate_model.py \
    --model-path PATH_TO_YOUR_ONNX_MODEL \
    --data-path PATH_TO_YOUR_TEST_DATA \
    --output-dir PATH_TO_YOUR_OUTPUT_DIR \
    --save-predictions PATH_TO_YOUR_PREDICTIONS_CSV
```

4. Cleanup generated files:
```bash
# Remove all generated files while preserving configurations
./scripts/cleanup_all.sh
```

## Development

### Testing
The test suite covers:
- Model initialization and configuration
- Data preprocessing and validation
- Training pipeline with early stopping
- Prediction accuracy and reliability
- Validation metrics and visualization
- Edge cases and error handling
- End-to-end pipeline validation
- Monitoring infrastructure:
  - Configuration validation
  - Report generation
  - Data cleanup and retention
  - Alert generation

Run tests with:
```bash
# Run all tests with coverage
make test

# Run monitoring-specific tests
make test-monitoring  # Includes end-to-end pipeline validation

# Run end-to-end pipeline test
make test-pipeline

# Run specific test file
pytest tests/models/test_single_cat_model.py -v
```

### Sample Data Generation
The test suite includes automatic sample data generation with:
- Realistic feature distributions
- Category-specific attributes
- Steam-related features
- Price and view count correlations
- Timestamp-based features

To generate sample data manually:
```bash
python scripts/generate_sample_data.py \
    --output-dir PATH_TO_YOUR_OUTPUT_DIR \
    --n-samples NUMBER_OF_SAMPLES
```

### Production Deployment and Monitoring

For detailed monitoring setup instructions, see [Monitoring Setup Guide](docs/monitoring_setup.md).

#### Deployment Requirements
- Model validation must pass all criteria defined in `configs/validation_criteria.json`:
  - RMSE < 100.0
  - MAE < 50.0
  - R² Score > 0.7
  - Minimum training samples: 1,000
  - Minimum validation samples: 200
  - Maximum missing rate: 10%
  - Feature importance threshold: 0.01
  - Prediction latency < 100ms
  - Drift detection threshold: p < 0.01 (conservative threshold for production stability)

#### Data Cleanup
```bash
# Clean all generated files and logs while preserving configurations
./scripts/cleanup_all.sh  # Preserves baseline_metrics.json and validation criteria
```

### Monitoring Capabilities
- Status Tracking:
  - Real-time monitoring status badge (passing/warning/failing)
  - Comprehensive health reporting with drift detection
  - Visual performance indicators and trends
- Comprehensive Reporting:
  - Interactive HTML dashboard
  - Combined monitoring metrics
  - Schema validation status
  - Health report integration
  - Performance visualizations
  - Alert history tracking
- Structure Validation:
  - Directory structure verification
  - Required file presence checks
  - JSON schema validation
  - Monitoring file format validation
  - Timestamp format verification
  - Metric type validation
- Data Quality:
  - Automated data format validation
  - Schema consistency checks
  - Timestamp validation
  - Required field verification
- Configuration:
  - Detailed configuration in `configs/monitoring_config.json`
  - Baseline metrics in `models/trained/baseline_metrics.json`
  - Validation criteria in `configs/validation_criteria.json`
  - Automated schema validation
  - Configurable drift thresholds

### Monitoring Commands
```bash
# Check monitoring dependencies
make monitor-check-deps

# Initialize monitoring structure
make monitor-structure

# Generate comprehensive report
make monitor-comprehensive

# Run complete monitoring pipeline
make monitor  # Includes all validations and reports

# Run monitoring with sample data
make monitor-test  # Useful for testing the monitoring setup
```

For configuration details, see `configs/monitoring_config.md`.

### Production Features
- Automated deployment pipeline with validation checks
- Model versioning and archiving
- Performance monitoring and drift detection
- Comprehensive evaluation reports including:
  - Model accuracy metrics (RMSE, MAE, R²)
  - Latency measurements (mean, p95, p99)
  - Feature importance analysis
  - Data quality checks
- Threshold-based deployment validation

### Performance Monitoring
```bash
# Run complete model evaluation
make evaluate  # Includes accuracy metrics and latency measurements

# Monitor production performance
make monitor  # Tracks drift and performance degradation
```

#### Deployment Commands
```bash
# Validate model against criteria
make validate  # Checks against validation_criteria.json

# Deploy model if validation passes
make deploy   # Includes monitoring setup and health check
```

### Model Architecture
- Uses CatBoost for robust price prediction with early stopping
- Supports category-specific model training with automatic feature preprocessing
- Handles both 'target' and 'sold_price' columns for flexible data input
- ONNX export for efficient production deployment
- Comprehensive metric tracking:
  - Root Mean Squared Error (RMSE): Primary metric for model evaluation
  - Mean Absolute Error (MAE): For interpretable error measurement
  - Mean Squared Error (MSE): For penalizing larger errors
  - R² Score: For explained variance measurement
  - Residual analysis with visualization
  - Feature importance tracking

### Deployment Process
The deployment pipeline includes:
- Automated validation against criteria in `validation_criteria.json`
- Performance monitoring with drift detection (p-value < 0.01)
- Health status tracking with warning indicators
- Comprehensive HTML reports with interactive visualizations
- Automated cleanup of old monitoring data (30-day retention)

Key validation checks:
- Model performance metrics (RMSE, MAE, R²)
- Latency requirements (p95 < 100ms)
- Data quality and completeness
- Drift detection and stability
- Configuration validation

For detailed monitoring setup, see [Monitoring Setup Guide](docs/monitoring_setup.md).

For detailed monitoring setup, see [Monitoring Setup Guide](docs/monitoring_setup.md).

### Feature Engineering
The model implements sophisticated preprocessing:
- Automatic handling of timestamp features
- One-hot encoding for categorical variables
- Missing value imputation
- Feature aggregation (sum, mean, std)
- Price-related feature engineering
- Steam-specific feature processing

### Configuration and Logging

#### Configuration Files
- Model configuration: `configs/model_config.yaml`
- Monitoring configuration: `configs/monitoring_config.json`
- Validation criteria: `configs/validation_criteria.json`
- Baseline metrics: `models/trained/baseline_metrics.json`

#### Directory Structure
```bash
logs/                  # Runtime logs
├── training/         # Training metrics and callbacks
├── validation/       # Validation results
└── prediction/       # Prediction outputs

monitoring_results/    # Monitoring data
├── drift/           # Drift detection results
└── metrics/         # Performance metrics

reports/              # Generated reports
├── comprehensive_report.html   # Interactive dashboard
├── health_report.json         # Current health status
└── schema_validation.json     # Configuration validation
```
