"""End-to-end test for the complete ML pipeline."""

import pytest
import os
import numpy as np
from datetime import datetime
import pandas as pd
from pathlib import Path
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.single_cat_model import SingleCategoryModel
from scripts.generate_sample_data import generate_sample_data
from scripts.evaluate_model import evaluate_model
from scripts.monitor_model import ModelMonitor

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    train_path = "data/raw/sample_train.json"
    val_path = "data/raw/sample_validation.json"

    # Generate sample data
    n_train = 100
    n_val = 50

    def generate_data(n):
        return {
            'target': np.random.uniform(10, 1000, n),
            'category': [1] * n,
            'views': np.random.randint(0, 1000, n),
            'created_at': [datetime.now().timestamp() for _ in range(n)],
            'steam_balance': ['USD 100.00'] * n,
            'price': np.random.uniform(10, 500, n),
            'view_count': np.random.randint(0, 1000, n),
            'steam_full_games': [{'total': 10, 'games': [{'playtime': 100}]} for _ in range(n)],
            'item_origin': ['market'] * n,
            'extended_guarantee': ['yes'] * n,
            'nsb': ['no'] * n,
            'email_type': ['gmail'] * n,
            'item_domain': ['steam'] * n,
            'resale_item_origin': ['market'] * n,
            'steam_country': ['US'] * n,
            'steam_community_ban': ['no'] * n,
            'steam_is_limited': ['no'] * n,
            'steam_cs2_wingman_rank_id': [1] * n,
            'steam_cs2_rank_id': [1] * n,
            'steam_cs2_ban_type': ['none'] * n
        }

    pd.DataFrame(generate_data(n_train)).to_json(train_path, orient='records')
    pd.DataFrame(generate_data(n_val)).to_json(val_path, orient='records')

    return train_path, val_path

def test_complete_pipeline(sample_data):
    """Test the complete pipeline from training to monitoring."""
    train_path, val_path = sample_data

    # 1. Train model
    model = SingleCategoryModel(category_number=1)
    train_df = pd.read_json(train_path, orient='records')
    model.train(train_df)

    # 2. Save and export model
    model_path = "models/trained/test_model.onnx"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.export(model_path)
    assert os.path.exists(model_path)

    # 3. Evaluate model
    metrics = evaluate_model(model_path, val_path, "evaluation_results")
    assert 'rmse' in metrics
    assert 'mae' in metrics

    # 4. Test monitoring
    monitor = ModelMonitor()
    val_df = pd.read_json(val_path, orient='records')
    predictions = model.predict(val_df)

    monitoring_metrics = monitor.log_predictions(
        predictions=predictions,
        actuals=val_df['target'].values
    )

    assert 'mae' in monitoring_metrics
    assert 'rmse' in monitoring_metrics
    assert 'correlation' in monitoring_metrics

    # Cleanup
    if os.path.exists(model_path):
        os.remove(model_path)
