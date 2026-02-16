ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON ?= $(ROOT)/venv/bin/python
PYTHON_CHECK := $(shell command -v $(PYTHON) 2>/dev/null)
ifeq ($(PYTHON_CHECK),)
$(error Could not resolve PYTHON='$(PYTHON)'. Set PYTHON to a valid interpreter path.)
endif

PLOT_ENV = MPLBACKEND=Agg MPLCONFIGDIR=/tmp/prism-mpl XDG_CACHE_HOME=/tmp

.PHONY: test smoke-discrete smoke-discrete-iid smoke-discrete-markov smoke-discrete-even smoke-continuous smoke-continuous-psi smoke-all

test:
	@if ! $(PYTHON) -c "import pytest" >/dev/null 2>&1; then \
		echo "pytest is unavailable in $(PYTHON). Install requirements into this interpreter."; \
		exit 1; \
	fi
	PYTHONPATH=$(ROOT)/src $(PYTHON) -m pytest -q -c $(ROOT)/pytest.ini

smoke-discrete-iid:
	cd src && $(PYTHON) -m prism.cli \
		--process iid_bernoulli \
		--reconstructor one_step \
		--ks 1 2 3 \
		--length 4000 \
		--train-frac 0.8 \
		--seeds 0 \
		--save-transitions \
		--show-transitions-for last_2 \
		--outdir ./results/smoke/discrete_iid \
		--force
	cd src && $(PYTHON) -m prism.analysis.summarise --root ./results/smoke/discrete_iid
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.analysis.plot_k \
		--root ./results/smoke/discrete_iid \
		--subsample-step 1 \
		--metrics logloss n_states C_mu_empirical unifilarity_score branch_entropy
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.analysis.phase_diagram --root ./results/smoke/discrete_iid

smoke-discrete-markov:
	cd src && $(PYTHON) -m prism.cli \
		--process markov_order_1 \
		--reconstructor one_step \
		--ks 1 2 3 \
		--length 4000 \
		--train-frac 0.8 \
		--seeds 0 \
		--save-transitions \
		--show-transitions-for last_2 \
		--outdir ./results/smoke/discrete_markov \
		--force
	cd src && $(PYTHON) -m prism.analysis.summarise --root ./results/smoke/discrete_markov
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.analysis.plot_k \
		--root ./results/smoke/discrete_markov \
		--subsample-step 1 \
		--metrics logloss n_states C_mu_empirical unifilarity_score branch_entropy
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.analysis.phase_diagram --root ./results/smoke/discrete_markov

smoke-discrete-even:
	cd src && $(PYTHON) -m prism.cli \
		--process even_process \
		--reconstructor one_step \
		--ks 1 2 3 \
		--length 4000 \
		--train-frac 0.8 \
		--seeds 0 \
		--save-transitions \
		--show-transitions-for last_2 \
		--outdir ./results/smoke/discrete_even \
		--force
	cd src && $(PYTHON) -m prism.analysis.summarise --root ./results/smoke/discrete_even
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.analysis.plot_k \
		--root ./results/smoke/discrete_even \
		--subsample-step 1 \
		--metrics logloss n_states C_mu_empirical unifilarity_score branch_entropy
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.analysis.phase_diagram --root ./results/smoke/discrete_even

smoke-discrete: smoke-discrete-iid smoke-discrete-markov smoke-discrete-even

smoke-continuous:
	cd src && $(PYTHON) -m prism.cli \
		--process linear_gaussian_ssm \
		--reconstructor kalman_iss \
		--ks 1 2 3 \
		--dvs 1 2 \
		--macro-eps 0.25 \
		--macro-bins 3 \
		--length 2500 \
		--train-frac 0.8 \
		--seeds 0 \
		--save-transitions \
		--show-transitions-for iss_d2 \
		--outdir ./results/smoke/continuous \
		--force
	cd src && $(PYTHON) -m prism.analysis.summarise --root ./results/smoke/continuous
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.analysis.plot_k \
		--root ./results/smoke/continuous \
		--subsample-step 1 \
		--dv 1 \
		--metrics logloss gaussian_logloss n_states C_mu_empirical unifilarity_score branch_entropy psi_opt
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.analysis.phase_diagram --root ./results/smoke/continuous --dv 1

smoke-continuous-psi:
	cd src && $(PYTHON) -m prism.cli \
		--process linear_gaussian_ssm \
		--reconstructor kalman_iss \
		--compute-psi \
		--macro-projection psi_opt \
		--psi-optimiser random \
		--psi-restarts 3 \
		--psi-iters 30 \
		--ks 1 2 \
		--dvs 1 2 \
		--length 1800 \
		--train-frac 0.8 \
		--seeds 0 \
		--outdir ./results/smoke/continuous_psi \
		--force
	cd src && $(PYTHON) -m prism.analysis.summarise --root ./results/smoke/continuous_psi
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.analysis.plot_k \
		--root ./results/smoke/continuous_psi \
		--subsample-step 1 \
		--dv 1 \
		--metrics logloss gaussian_logloss n_states C_mu_empirical unifilarity_score branch_entropy psi_opt

smoke-all: smoke-discrete smoke-continuous smoke-continuous-psi
