PYTHON ?= python3

.PHONY: install lint type test test-unit test-e2e bench build check-dist verify clean clean-runtime

install:
	$(PYTHON) -m pip install -e ".[dev,tui]"

lint:
	ruff check .

type:
	mypy src/agent_vcr/ src/openhands_sentinel/

test: test-unit

test-unit:
	pytest tests/unit/ -q

test-e2e:
	pytest tests/e2e/ -q --no-cov

bench:
	$(PYTHON) -m pytest tests/benchmarks/ -q --benchmark-only --no-cov --benchmark-name=short

build:
	$(PYTHON) -m build --sdist --wheel

check-dist: build
	$(PYTHON) -m twine check dist/*

verify: lint type test-unit test-e2e bench check-dist

clean:
	rm -rf .coverage .coverage.* coverage.xml htmlcov
	rm -rf .pytest_cache .mypy_cache .ruff_cache .benchmarks
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-runtime:
	rm -rf .vcr
