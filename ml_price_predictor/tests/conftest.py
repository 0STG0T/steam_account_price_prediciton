import pytest
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    np.random.seed(42)
    n_samples = 100
    current_time = datetime.now().timestamp()

    # Generate more realistic price data
    base_prices = np.random.uniform(10, 1000, n_samples)
    view_counts = np.random.randint(10, 1000, n_samples)
    price_multiplier = 1 + np.random.normal(0, 0.3, n_samples)  # Add some noise
    prices = base_prices * price_multiplier

    return pd.DataFrame({
        'category_id': [1] * n_samples,
        'price': prices,
        'view_count': view_counts,
        'published_date': [current_time - np.random.randint(0, 365*24*3600) for _ in range(n_samples)],
        'steam_balance': [f'{np.random.uniform(0, 1000):.2f} USD' for _ in range(n_samples)],
        'steam_full_games': [{'total': np.random.randint(1, 100),
                             'games': [{'playtime': np.random.randint(0, 1000)} for _ in range(5)]}
                           for _ in range(n_samples)],
        'item_origin': np.random.choice(['market', 'trade', 'drop'], n_samples),
        'steam_country': np.random.choice(['US', 'EU', 'RU'], n_samples),
        'target': prices * (1 + np.random.normal(0, 0.2, n_samples))  # Target with realistic variation
    })

@pytest.fixture
def model_config():
    """Sample model configuration for testing"""
    return {
        'iterations': 100,
        'l2_leaf_reg': 2.7,
        'random_state': 42,
        'thread_count': 1
    }
