# Makefile for parametric-umap

.PHONY: help install test lint format build clean pre-commit

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install all dependencies
	uv sync --all-extras

test:  ## Run tests
	uv run pytest

lint:  ## Run linting checks
	uv run ruff check .

format:  ## Format code
	uv run ruff format .

build:  ## Build the package
	uv build

clean:  ## Clean build artifacts
	rm -rf dist/ build/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f .coverage coverage.xml
	rm -rf htmlcov/

pre-commit:  ## Install pre-commit hooks
	uv run pre-commit install
