PYTHON ?= python3

.PHONY: install test lint compile selfcheck manifest
install:
	$(PYTHON) -m pip install -e '.[language,dev]'

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check ttt_pt tests scripts/run_config.py

compile:
	$(PYTHON) -m compileall -q ttt_pt scripts analysis tests validation

selfcheck:
	$(PYTHON) scripts/selfcheck.py

manifest:
	$(PYTHON) scripts/source_manifest.py
