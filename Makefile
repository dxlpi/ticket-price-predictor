.PHONY: install install-dev lint format typecheck test test-cov clean fetch-sample ingest-events collect-snapshots pipeline

# Install production dependencies
install:
	pip install -e .

# Install with development dependencies
install-dev:
	pip install -e ".[dev]"

# Run linting
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

# Auto-fix lint issues and format code
format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

# Run type checking
typecheck:
	mypy src/

# Run tests
test:
	pytest

# Run tests with coverage report
test-cov:
	pytest --cov=src/ticket_price_predictor --cov-report=html
	@echo "Coverage report generated in htmlcov/"

# Run all checks (lint, typecheck, test)
check: lint typecheck test

# Fetch sample events (requires TICKETMASTER_API_KEY)
fetch-sample:
	python scripts/fetch_sample_events.py

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

# Ingest events from Ticketmaster (requires TICKETMASTER_API_KEY)
ingest-events:
	python scripts/ingest_events.py --days-ahead 90

# Collect price snapshots for tracked events
collect-snapshots:
	python scripts/collect_snapshots.py --all-tracked

# Run full data pipeline (ingest + collect)
pipeline:
	python scripts/run_pipeline.py --days-ahead 90
