# Makefile for parametric-umap development and release tasks

.PHONY: help install test lint format release release-dry version changelog clean build

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install development dependencies
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

version:  ## Show next version that would be released
	uv run python scripts/semantic_release.py version

changelog:  ## Generate changelog
	uv run python scripts/semantic_release.py changelog

release-dry:  ## Dry run of release process (show what would happen)
	uv run python scripts/semantic_release.py --dry-run

release:  ## Perform semantic release (version bump, build, publish, tag)
	uv run python scripts/semantic_release.py

# Alternative using semantic-release directly
release-semantic:  ## Release using semantic-release CLI directly
	uv run semantic-release version

release-semantic-dry:  ## Dry run using semantic-release CLI
	uv run semantic-release version --print