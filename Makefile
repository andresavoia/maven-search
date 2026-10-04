.PHONY: setup index run test demo eval

PYTHON ?= python3
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

index:
	$(PY) -m app.index_build

run:
	$(PY) -m uvicorn app.server:app --host 127.0.0.1 --port 8000

test:
	$(PY) -m pytest -q

demo:
	$(PY) -m app.demo

eval:
	$(PY) -m app.eval_retrieval
