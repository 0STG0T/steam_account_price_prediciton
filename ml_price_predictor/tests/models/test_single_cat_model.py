import pytest
import numpy as np
from src.models.single_cat_model import SingleCategoryModel

def test_model_initialization():
    """Test model initialization with default parameters"""
    model = SingleCategoryModel(category_number=1)
    assert model.category_number == 1
    assert model.meta_model is None

def test_model_preprocessing(sample_data):
    """Test preprocessing doesn't affect target values"""
    model = SingleCategoryModel(category_number=1)
    original_targets = sample_data['target'].values
    processed_df = model.preprocess_data(sample_data)
    processed_targets = processed_df['target'].values

    # Check target values are different
    assert len(np.unique(processed_targets)) > 1, "All target values are equal after preprocessing"
    assert np.allclose(original_targets, processed_targets, rtol=1e-10), "Target values changed during preprocessing"

def test_model_training(sample_data):
    """Test model training with sample data"""
    model = SingleCategoryModel(category_number=1)
    model.train(sample_data)
    assert model.meta_model is not None
    assert hasattr(model.meta_model, 'predict')

def test_model_prediction(sample_data):
    """Test model prediction functionality"""
    model = SingleCategoryModel(category_number=1)
    model.train(sample_data)
    predictions = model.predict(sample_data)
    assert len(predictions) == len(sample_data)
    assert isinstance(predictions[0], (float, np.float32, np.float64))

def test_model_validation(sample_data):
    """Test model validation metrics"""
    model = SingleCategoryModel(category_number=1)
    model.train(sample_data)
    metrics = model.validate(sample_data)
    assert 'rmse' in metrics
    assert isinstance(metrics['rmse'], float)
