.PHONY: test install notebook

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PY   := $(if $(wildcard $(ROOT)/.venv/bin/python),$(ROOT)/.venv/bin/python,python3)
export PYTHONNOUSERSITE := 1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD := 1

install:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e "$(ROOT)[dev,plot,notebooks]"

test:
	$(PY) -m pytest -q

notebook:
	$(PY) $(ROOT)/notebooks/_emit.py
