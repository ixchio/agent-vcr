# Contributing to Agent VCR

Thank you for your interest in contributing to Agent VCR! We welcome bug reports, feature requests, documentation improvements, and code contributions.

## Development Setup

1. Fork and clone the repository.
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[dev,langgraph,tui]"
   ```
4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Running Tests

We use `pytest` for testing. To run the full test suite with coverage:

```bash
pytest tests/ -v --cov=agent_vcr --cov-report=term-missing
```

## Linting and Formatting

We use `ruff` for linting and formatting, and `mypy` for static type checking. These will run automatically if you have installed the pre-commit hooks. You can also run them manually:

```bash
ruff check .
ruff format .
mypy src/agent_vcr/
```

## Pull Request Process

1. Create a new branch for your feature or bug fix.
2. Ensure all tests pass.
3. Include tests for your new feature or bug fix.
4. Update the documentation (README.md, docstrings) as applicable.
5. Submit a pull request and describe your changes.
