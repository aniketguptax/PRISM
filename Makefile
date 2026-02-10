PYTHON ?= python
PLOT_ENV = MPLBACKEND=Agg MPLCONFIGDIR=/tmp/prism-mpl XDG_CACHE_HOME=/tmp

.PHONY: test smoke-discrete smoke-continuous smoke-all

test:
	cd src && $(PYTHON) -m pytest -q

smoke-discrete:
	cd src && $(PYTHON) -m prism.cli \
		--process markov_order_1 \
		--reconstructor one_step \
		--ks 1 2 3 \
		--length 4000 \
		--train-frac 0.8 \
		--seeds 0 \
		--save-transitions \
		--show-transitions-for last_2 \
		--outdir ./results/smoke/discrete \
		--force
	cd src && $(PYTHON) -m prism.analysis.summarise --root ./results/smoke/discrete
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.analysis.plot_k \
		--root ./results/smoke/discrete \
		--subsample-step 1 \
		--metrics logloss n_states C_mu_empirical unifilarity_score branch_entropy
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.analysis.phase_diagram --root ./results/smoke/discrete

smoke-continuous:
	cd src && $(PYTHON) -m prism.cli \
		--process linear_gaussian_ssm \
		--reconstructor kalman_iss \
		--ks 1 2 3 \
		--length 2500 \
		--train-frac 0.8 \
		--seeds 0 \
		--outdir ./results/smoke/continuous \
		--force
	cd src && $(PYTHON) -m prism.analysis.summarise --root ./results/smoke/continuous
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.analysis.plot_k \
		--root ./results/smoke/continuous \
		--subsample-step 1 \
		--metrics logloss gaussian_logloss n_states
	cd src && $(PLOT_ENV) $(PYTHON) -m prism.analysis.phase_diagram --root ./results/smoke/continuous

smoke-all: smoke-discrete smoke-continuous
