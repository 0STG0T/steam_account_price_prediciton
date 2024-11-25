.PHONY: install test test-pipeline lint format clean validate generate-data evaluate monitor monitor-drift monitor-report monitor-cleanup deploy setup-permissions demo

# Setup
setup-permissions:
	chmod +x scripts/setup_permissions.sh
	./scripts/setup_permissions.sh

install: setup-permissions
	pip install -e ".[dev]"

# Development
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-monitoring:
	pytest tests/test_monitoring.py -v
	python scripts/validate_monitoring_pipeline.py --base-dir .

test-pipeline:
	pytest tests/test_pipeline.py -v --log-cli-level=INFO

lint:
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

# Data and Model Operations
generate-data:
	python scripts/generate_sample_data.py --n-samples 1000 --category-id 1 --output-dir data/raw/

demo: clean generate-data
	@echo "=== Running complete demo pipeline ==="
	@echo "1. Generating sample data..."
	make generate-data
	@echo "\n2. Training model..."
	python src/train.py --data-path data/raw/sample_train.json --category-id 1 --output-dir models/trained/
	@echo "\n3. Exporting to ONNX..."
	python src/export_onnx.py --model-path models/trained/category_1_model.cbm --output-path models/onnx/category_1_model.onnx
	@echo "\n4. Running evaluation..."
	make evaluate
	@echo "\n5. Running monitoring checks..."
	make monitor-test

validate:
	python scripts/validate_deployment.py --metrics-file evaluation_results/metrics_latest.json --criteria-file configs/validation_criteria.json --output-file validation_results/validation_latest.json

evaluate:
	python scripts/evaluate_model.py --model-path models/onnx/category_1_model.onnx --data-path data/raw/sample_validation.json --output-dir evaluation_results --save-predictions logs/predictions/latest.csv && \
	python scripts/measure_latency.py --model-path models/onnx/category_1_model.onnx --output-file evaluation_results/latency_latest.json

# Production Monitoring
monitor-check-deps:
	python scripts/check_monitoring_deps.py

monitor-structure:
	python scripts/validate_monitoring_structure.py --base-dir .

monitor-validate-data:
	python scripts/validate_monitoring_data.py --monitoring-dir monitoring_results

monitor-validate-config:
	python scripts/validate_monitoring_config.py --config-file configs/monitoring_config.json

monitor-sample-data:
	python scripts/generate_sample_monitoring.py --output-dir monitoring_results --n-days 30 --drift-factor 0.01

monitor-badge:
	python scripts/generate_monitoring_badge.py --monitoring-dir monitoring_results --output-file monitoring_badge.md && \
	python scripts/update_readme_badge.py --readme README.md --badge monitoring_badge.md

monitor-schema:
	python scripts/validate_monitoring_schema.py --config-file configs/monitoring_config.json --output-file reports/schema_validation.json

monitor-dashboard:
	python scripts/generate_monitoring_dashboard.py --monitoring-dir monitoring_results --output-dir reports

monitor-drift:
	python scripts/monitor_drift.py --predictions logs/predictions/latest.csv --actuals data/raw/actuals.csv --baseline-metrics models/trained/baseline_metrics.json --output-dir monitoring_results

monitor-report:
	python scripts/generate_monitoring_report.py --monitoring-dir monitoring_results --output-dir reports

monitor-health:
	python scripts/generate_health_report.py --monitoring-dir monitoring_results --output-file reports/health_report.json

monitor-comprehensive:
	python scripts/generate_comprehensive_report.py \
		--monitoring-dir monitoring_results \
		--schema-file reports/schema_validation.json \
		--health-file reports/health_report.json \
		--output-file reports/comprehensive_report.html

monitor-cleanup:
	python scripts/cleanup_monitoring.py --monitoring-dir monitoring_results --reports-dir reports --max-age-days 30

monitor: monitor-check-deps monitor-structure monitor-schema monitor-validate-config monitor-validate-data monitor-drift monitor-report monitor-comprehensive monitor-dashboard monitor-badge monitor-health monitor-cleanup

monitor-test: monitor-sample-data monitor

deploy: monitor
	python scripts/generate_deployment_report.py --base-dir . --output-file reports/deployment_readiness.json
	python scripts/deploy_model.py --model-path models/trained/latest.onnx --validation-data data/validation.json --rmse-threshold 100.0 --mae-threshold 50.0

clean:
	# Clean Python cache files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".eggs" -exec rm -rf {} +
	# Clean generated files and directories
	./scripts/cleanup.sh

# Comprehensive cleanup
cleanup-all:
	./scripts/cleanup_all.sh

