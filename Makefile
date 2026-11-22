PY ?= .venv/bin/python
STAGE ?= stage1

.PHONY: install install-scrape lint format test predict animate backtest live site lock

install:
	python3 -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e '.[dev]'

install-scrape: install
	$(PY) -m pip install -e '.[dev,scrape]'
	$(PY) -m playwright install chromium

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

format:
	$(PY) -m ruff check --fix src tests
	$(PY) -m ruff format src tests

test:
	$(PY) -m pytest

predict:
	$(PY) -m cs2pickem.cli simulate --stage $(STAGE)
	$(PY) -m cs2pickem.cli pickem --stage $(STAGE)

animate:
	$(PY) -m cs2pickem.cli animate --stage $(STAGE)

backtest:
	$(PY) -m cs2pickem.cli backtest run

live:
	$(PY) -m cs2pickem.cli live update

site:
	$(PY) -m cs2pickem.cli site build

lock:
	$(PY) -m pip freeze --exclude-editable | grep -v '^-e' > requirements.lock
