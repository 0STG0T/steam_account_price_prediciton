from src.utils.log_manager import LogManager
import json

def test_logging():
    log_manager = LogManager()

    # Test each log type
    for log_type in ['training', 'validation', 'prediction', 'monitoring', 'reports', 'metrics']:
        test_path = log_manager.get_log_path(log_type, f'test_{log_type}.log')
        with open(test_path, 'w') as f:
            f.write(f"Test log for {log_type}")
        print(f"Created test log: {test_path}")

    # Test metrics saving
    metrics = {"test_metric": 0.95}
    metrics_path = log_manager.save_metrics(metrics, "test_model")
    print(f"Saved test metrics to: {metrics_path}")

if __name__ == "__main__":
    test_logging()
