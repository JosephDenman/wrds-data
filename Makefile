.PHONY: setup test clean format lint

setup:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=wrds_data

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache *.egg-info

format:
	black src/ tests/
	isort src/ tests/

lint:
	flake8 src/ tests/
	mypy src/
