.PHONY: setup setup-auto install lint format format-check typecheck test check prototypes vocabulary ui spike

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python

setup:
	bash scripts/setup_env.sh

setup-auto:
	bash scripts/setup_auto.sh

install:
	uv pip install --python "$(PYTHON)" -e ".[baseline,tesseract,easyocr,experiments,arabic,ui]" pytest pytest-cov ruff mypy pre-commit

lint:
	uv run --no-sync ruff check src scripts tests

format:
	uv run --no-sync ruff format src scripts tests

format-check:
	uv run --no-sync ruff format --check src scripts tests

typecheck:
	uv run --no-sync mypy src

test:
	uv run --no-sync pytest

check: lint format-check typecheck test

prototypes:
	uv run --no-sync embedding-ocr build-prototypes --config configs/default.yaml

vocabulary:
	uv run --no-sync embedding-ocr build-vocabulary --config configs/default.yaml

ui:
	VENV="$(VENV)" bash scripts/run_ui.sh $(UI_ARGS)

spike:
	uv run --no-sync python scripts/phase0_az_spike.py --config configs/spike_az.yaml
