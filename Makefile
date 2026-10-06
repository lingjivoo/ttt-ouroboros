PYTHON ?= python3

.PHONY: install install-agents test lint compile selfcheck check suite-dry-run smoke figures

install:
	$(PYTHON) -m pip install -e '.[language,dev]'

install-agents:
	$(PYTHON) -m pip install -e '.[language,agents,dev]'

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check ttt_pt tests scripts/run_config.py scripts/run_paper_suite.py

compile:
	$(PYTHON) -m compileall -q ttt_pt scripts analysis tests figures

selfcheck:
	$(PYTHON) scripts/selfcheck.py --profile language

check: compile test lint

suite-dry-run:
	$(PYTHON) scripts/run_paper_suite.py --suite configs/suites/main_125m.yaml --dry-run

smoke:
	$(PYTHON) scripts/selfcheck.py --profile language --full
	$(PYTHON) scripts/run_paper_suite.py --suite configs/suites/main_125m.yaml --smoke

figures:
	$(PYTHON) figures/make_paper_figures.py
	$(PYTHON) figures/make_commitment_policies.py
	$(PYTHON) figures/make_provenance_noise.py
	$(PYTHON) figures/make_agent_causal_success.py
