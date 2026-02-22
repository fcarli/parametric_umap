# Contributing to parametric-umap

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fcarli/parametric_umap.git
   cd parametric_umap
   ```

2. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies:**
   ```bash
   # macOS / Windows
   uv sync --extra dev --extra test --extra examples

   # Linux — CPU only
   uv sync --extra dev --extra test --extra examples --extra cpu
   ```
   > **Note:** Do not use `--all-extras` — the CUDA extras conflict with each other.

4. **Install pre-commit hooks:**
   ```bash
   uv run pre-commit install
   ```

## Code Style

- We use [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting
- Line length limit is **120 characters**
- All Ruff rules are enabled — check `pyproject.toml` for per-file ignores
- Type hints are expected for all public APIs
- Use NumPy-style docstrings for public methods

Run checks locally:
```bash
make lint      # Run ruff check
make format    # Run ruff format
```

## Testing

Tests are run with pytest:
```bash
make test
```

When adding new functionality, please include tests in the `tests/` directory.

## Commit Conventions

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation changes
- `test:` — adding or updating tests
- `refactor:` — code refactoring (no behavior change)
- `perf:` — performance improvement
- `ci:` — CI/CD changes
- `chore:` — maintenance tasks

Examples:
```
feat: add batch normalization option to encoder
fix: handle empty input in transform method
docs: update installation instructions
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`
2. Make your changes, ensuring all checks pass:
   ```bash
   make lint
   make test
   ```
3. Write clear commit messages following the conventions above
4. Open a pull request against `main` with a description of your changes
5. Ensure CI passes on your PR

## Reporting Issues

If you find a bug or have a feature request, please [open an issue](https://github.com/fcarli/parametric_umap/issues) with:
- A clear description of the problem or feature
- Steps to reproduce (for bugs)
- Your environment details (Python version, OS, GPU availability)
