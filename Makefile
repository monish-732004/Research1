.PHONY: setup validate preprocess preprocess-force analyze simulate figures test lint clean

PYTHON := python
SRC    := src

# ── Environment ──────────────────────────────────────────────────────────────
setup:
	pip install -e ".[dev]"
	@echo "Setup complete. Activate your venv first if you haven't already."

# ── Pipeline ─────────────────────────────────────────────────────────────────
validate:
	$(PYTHON) -m $(SRC).ingest.validate --trace v2026
	$(PYTHON) -m $(SRC).ingest.validate --trace v2023

preprocess:
	$(PYTHON) -m $(SRC).preprocess.pipeline --trace v2026

preprocess-force:
	$(PYTHON) -m $(SRC).preprocess.pipeline --trace v2026 --force

analyze:
	$(PYTHON) -m $(SRC).analysis.fragmentation
	$(PYTHON) -m $(SRC).analysis.wasted_hours
	$(PYTHON) -m $(SRC).analysis.carbon_join
	$(PYTHON) -m $(SRC).analysis.correlation

simulate:
	$(PYTHON) -m $(SRC).simulation.runner --mode baseline
	$(PYTHON) -m $(SRC).simulation.runner --mode cads

# ── Modeling (Phase 4.5) ──────────────────────────────────────────────────────
# Full run (~90 min).  Use FAST=1 for a 10% subsample (~5 min).
FAST_FLAG := $(if $(FAST),--fast,)

model: _model_features _model_split _model_train _model_evaluate _model_choose

_model_features:
	$(PYTHON) -m $(SRC).modeling.features $(FAST_FLAG)

_model_split:
	$(PYTHON) -m $(SRC).modeling.split

_model_train:
	$(PYTHON) -m $(SRC).modeling.train $(FAST_FLAG)

_model_evaluate:
	$(PYTHON) -m $(SRC).modeling.evaluate $(FAST_FLAG)

_model_choose:
	$(PYTHON) -m $(SRC).modeling.choose

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
