.PHONY: setup setup-auto install lint format format-check typecheck test check prototypes vocabulary ui spike

setup:
	bash scripts/setup_env.sh

setup-auto:
	bash scripts/setup_auto.sh

install:
	uv sync --group dev

lint:
	uv run ruff check src scripts tests

format:
	uv run ruff format src scripts tests

format-check:
	uv run ruff format --check src scripts tests

typecheck:
	uv run mypy src

test:
	uv run pytest

check: lint format-check typecheck test

prototypes:
	uv run embedding-ocr build-prototypes --config configs/default.yaml

vocabulary:
	uv run embedding-ocr build-vocabulary --config configs/default.yaml

ui:
	VENV="$(VENV)" bash scripts/run_ui.sh $(UI_ARGS)

spike:
	uv run python scripts/phase0_az_spike.py --config configs/spike_az.yaml
