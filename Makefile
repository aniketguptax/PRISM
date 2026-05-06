ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON ?= $(ROOT)/venv/bin/python
PYTHON_CHECK := $(shell command -v $(PYTHON) 2>/dev/null)
ifeq ($(PYTHON_CHECK),)
$(error Could not resolve PYTHON='$(PYTHON)'. Set PYTHON to a valid interpreter path.)
endif

PLOT_ENV = MPLBACKEND=Agg MPLCONFIGDIR=/tmp/prism-mpl XDG_CACHE_HOME=/tmp

.PHONY: test smoke-discrete smoke-discrete-iid smoke-discrete-markov smoke-discrete-even smoke-continuous smoke-continuous-psi smoke-all block-modular-smoke block-modular-sweep hierarchical-predictive-smoke hierarchical-predictive-sweep hierarchical-predictive-main low-variance-lgssm-smoke low-variance-lgssm-main multiscale-lgssm-smoke multiscale-lgssm-main multiscale-lgssm-robust

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

block-modular-smoke:
	cd src && $(PYTHON) -m prism.experiments.block_modular_recovery \
		--couplings 0.0 0.05 0.20 \
		--seeds 0 \
		--obs-designs random aligned \
		--builders hierarchical_single linear_quantile \
		--eps-macros 0.15 0.25 \
		--length 800 \
		--em-iters 25 \
		--outdir ./results/block_modular_smoke
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.analysis.block_modular_hierarchy \
		--root ./results/block_modular_smoke
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.analysis.block_modular_emergence \
		--root ./results/block_modular_smoke

block-modular-sweep:
	cd src && $(PYTHON) -m prism.experiments.block_modular_recovery \
		--couplings 0.0 0.025 0.05 0.10 0.20 \
		--seeds 0 1 2 \
		--obs-designs random aligned \
		--builders hierarchical_single hierarchical_complete linear_quantile greedy \
		--eps-macros 0.10 0.15 0.25 0.40 \
		--length 4000 \
		--em-iters 50 \
		--outdir ./results/block_modular_sweep
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.analysis.block_modular_hierarchy \
		--root ./results/block_modular_sweep
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.analysis.block_modular_emergence \
		--root ./results/block_modular_sweep

hierarchical-predictive-smoke:
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.experiments.hierarchical_predictive_recovery \
		--noises 0.02 0.16 \
		--seeds 0 \
		--eps-values 0.25 0.30 0.35 0.40 0.45 0.50 \
		--kmeans-ks 3 6 12 \
		--length 8000 \
		--context-len 2 \
		--future-horizon 4 \
		--outdir ./results/hierarchical_predictive_smoke

hierarchical-predictive-sweep:
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.experiments.hierarchical_predictive_recovery \
		--noises 0.02 0.08 0.16 0.28 \
		--seeds 0 1 2 \
		--eps-values 0.25 0.30 0.35 0.40 0.45 0.50 \
		--kmeans-ks 3 6 12 \
		--length 20000 \
		--context-len 2 \
		--future-horizon 4 \
		--outdir ./results/hierarchical_predictive_sweep

hierarchical-predictive-main:
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.experiments.hierarchical_predictive_recovery \
		--noises 0.02 0.05 0.08 0.12 0.16 0.20 \
		--seeds 0 1 2 3 4 \
		--eps-values 0.22 0.25 0.28 0.30 0.32 0.35 0.38 \
		--kmeans-ks 3 6 9 12 18 \
		--length 30000 \
		--context-len 2 \
		--future-horizon 4 \
		--outdir ./results/hierarchical_predictive_main
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.analysis.hierarchical_predictive_figures \
		--root ./results/hierarchical_predictive_main
	cd src && $(PYTHON) -m prism.analysis.hierarchical_predictive_report \
		--root ./results/hierarchical_predictive_main

low-variance-lgssm-smoke:
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.experiments.low_variance_lgssm_recovery \
		--obs-stds 0.25 \
		--seeds 0 1 \
		--eps-values 0.25 0.35 0.50 \
		--kmeans-ks 3 6 12 \
		--length 2500 \
		--em-iters 10 \
		--outdir ./results/low_variance_lgssm_smoke
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.analysis.low_variance_lgssm_figures \
		--root ./results/low_variance_lgssm_smoke

low-variance-lgssm-main:
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.experiments.low_variance_lgssm_recovery \
		--obs-stds 0.15 0.25 0.40 \
		--seeds 0 1 2 3 4 \
		--eps-values 0.25 0.35 0.50 \
		--kmeans-ks 3 6 9 12 18 \
		--length 6000 \
		--em-iters 30 \
		--outdir ./results/low_variance_lgssm_main
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.analysis.low_variance_lgssm_figures \
		--root ./results/low_variance_lgssm_main

multiscale-lgssm-smoke:
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.experiments.multiscale_lgssm_recovery \
		--obs-stds 0.15 \
		--distractor-loadings 2.5 3.0 \
		--seeds 0 1 \
		--kmeans-ks 4 8 12 16 \
		--pca-dims 2 5 \
		--length 1800 \
		--em-iters 8 \
		--history-lens 5 20 \
		--outdir ./results/multiscale_lgssm_smoke

multiscale-lgssm-main:
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.experiments.multiscale_lgssm_recovery \
		--obs-stds 0.12 0.15 0.20 \
		--distractor-loadings 2.5 3.0 3.5 \
		--seeds 0 1 2 3 4 \
		--kmeans-ks 4 8 12 16 24 36 \
		--pca-dims 2 3 5 \
		--length 5000 \
		--em-iters 30 \
		--history-lens 5 20 \
		--outdir ./results/multiscale_lgssm_main

multiscale-lgssm-robust:
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.experiments.multiscale_lgssm_recovery \
		--obs-stds 0.08 0.12 0.15 0.20 0.30 \
		--distractor-loadings 2.0 2.5 3.0 3.5 4.0 \
		--seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 \
		--kmeans-ks 4 8 12 16 24 36 48 \
		--pca-dims 2 3 5 \
		--length 8000 \
		--em-iters 40 \
		--history-lens 5 20 50 100 \
		--outdir ./results/multiscale_lgssm_robust
