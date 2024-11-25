import os
import csv
from datetime import datetime
import pandas as pd

def get_timestamp():
    """Return current timestamp in YYYY-MM-DD_HH-MM-SS format"""
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def setup_logging_directory():
    """Create logs directory if it doesn't exist"""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def create_log_file(operation_type):
    """Create a new log file with timestamp"""
    log_dir = setup_logging_directory()
    timestamp = get_timestamp()
    filename = f'{operation_type}_logs_{timestamp}.csv'
    return os.path.join(log_dir, filename)

def log_training_metrics(log_file, category_id, epoch, metrics):
    """Log training metrics to CSV"""
    timestamp = get_timestamp()
    metrics_dict = {
        'timestamp': timestamp,
        'category_id': category_id,
        'epoch': epoch,
        **metrics
    }

    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_dict)

def log_validation_results(log_file, category_id, metrics, plot_path):
    """Log validation results to CSV"""
    timestamp = get_timestamp()
    metrics_dict = {
        'timestamp': timestamp,
        'category_id': category_id,
        'plot_path': plot_path,
        **metrics
    }

    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_dict)

def log_predictions(log_file, category_id, predictions, input_data_path):
    """Log prediction summary statistics to CSV"""
    timestamp = get_timestamp()
    stats = {
        'timestamp': timestamp,
        'category_id': category_id,
        'input_data_path': input_data_path,
        'mean_prediction': float(predictions.mean()),
        'std_prediction': float(predictions.std()),
        'min_prediction': float(predictions.min()),
        'max_prediction': float(predictions.max()),
        'num_predictions': len(predictions)
    }

    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=stats.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(stats)
