default: check

install:
    pip install -e ".[dev]"

lint:
    ruff check .
    ruff format --check .

types:
    mypy

test:
    pytest

coverage:
    coverage run -m pytest
    coverage report --fail-under=85

replay:
    immune replay

docs:
    mkdocs build --strict

check: lint types coverage replay

release-check:
    python -m build
    twine check dist/*
