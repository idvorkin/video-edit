set export

default:
    @just --list

# Install the package in development mode
install:
    uv pip install --editable .

# Install the package globally
global-install: install
    uv tool install --force --editable .

# Run tests
test:
    pytest -n auto

# Run tests with coverage
test-coverage:
    pytest -n auto --cov=. --cov-report=term-missing

# Format code using black
format:
    black .

# Run type checking
typecheck:
    mypy . 