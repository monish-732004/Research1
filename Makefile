.PHONY: setup preprocess analyze simulate figures test lint clean

PYTHON := python
SRC    := src

# ── Environment ──────────────────────────────────────────────────────────────
setup:
	pip install -e ".[dev]"
	@echo "Setup complete. Activate your venv first if you haven't already."

# ── Pipeline ─────────────────────────────────────────────────────────────────
preprocess:
	$(PYTHON) -m $(SRC).preprocess.clock_align
	$(PYTHON) -m $(SRC).preprocess.state_machine
	$(PYTHON) -m $(SRC).preprocess.truncation
	$(PYTHON) -m $(SRC).preprocess.aggregate

analyze:
	$(PYTHON) -m $(SRC).analysis.fragmentation
	$(PYTHON) -m $(SRC).analysis.carbon_join
	$(PYTHON) -m $(SRC).analysis.correlation

simulate:
	$(PYTHON) -m $(SRC).simulation.runner --mode baseline
	$(PYTHON) -m $(SRC).simulation.runner --mode cads

figures:
	$(PYTHON) -m $(SRC).viz.figures

# ── Quality ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/

# ── Housekeeping ──────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ *.egg-info/

clean-outputs:
	rm -f outputs/figures/* outputs/tables/* outputs/logs/*

# Full reproducibility run (requires data in data/raw/)
all: preprocess analyze simulate figures
