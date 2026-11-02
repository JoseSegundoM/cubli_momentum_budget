# Results run bottom-up: tuning -> experiments -> numbers and figures -> paper.
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
export PYTHONPATH := .
export OMP_NUM_THREADS = 1
export OPENBLAS_NUM_THREADS = 1

.PHONY: help setup test numbers table figures supplement paper verify guard tuning experiments

help:
	@echo "setup        virtual environment and dependencies"
	@echo "test         model tests"
	@echo "numbers      paper/numbers.tex from results/"
	@echo "table        Table I (paper/table_main.tex) from results/"
	@echo "figures      figures 2-4 into figures/"
	@echo "supplement   paper/supplement.tex and supplement.pdf"
	@echo "paper        both PDF profiles into paper/ (needs TeX Live)"
	@echo "verify       regenerate numbers, Table I and supplement; fail if they differ"
	@echo "tuning       re-tune every design (hours); OVERWRITES results/tuned_gains.json, needs CONFIRM=1"
	@echo "experiments  Monte-Carlo campaigns E1-E8 (hours); OVERWRITES results/*.csv, needs CONFIRM=1"

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

test:
	$(PY) -m pytest tests -q

numbers:
	$(PY) scripts/make_numbers.py

table:
	$(PY) -c "import sys; sys.path.insert(0, 'scripts'); import fig_results; fig_results.table_main()"

figures:
	$(PY) -c "import sys; sys.path.insert(0, 'scripts'); import fig_results as f; f.fig_sweeps(); f.fig_roa()"
	$(PY) scripts/fig_timeseries.py
	$(PY) scripts/fig_schematic.py

supplement:
	$(PY) scripts/make_supplement.py
	cd paper && pdflatex -interaction=nonstopmode supplement.tex >/dev/null && pdflatex -interaction=nonstopmode supplement.tex >/dev/null

paper:
	bash paper/build_profiles.sh

verify: numbers table
	$(PY) scripts/make_supplement.py
	git diff --stat --exit-code paper/numbers.tex paper/table_main.tex paper/supplement.tex

guard:
	@test "$(CONFIRM)" = 1 || (echo "This overwrites the recorded runs in results/. Run with CONFIRM=1." && exit 1)

tuning: guard
	$(PY) scripts/run_tuning.py

experiments: guard
	$(PY) scripts/run_experiments.py
