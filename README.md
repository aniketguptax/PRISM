# PRISM

Predictive Representations for Inference of Scale and Macrostates.

PRISM supports two pipelines:

- Discrete binary processes with `LastK` representations and one-step merge reconstruction.
- Continuous multivariate time series with Kalman ISS reconstruction plus macrostate
  construction over `(d, d_V)`.

## One-command auto mode

If you want PRISM to pick the configuration automatically from a data file:

```bash
./prism /absolute/path/to/data.csv
# or:
cd src
python -m prism /absolute/path/to/data.csv
```

This runs an auto sweep (with model selection), writes results to `results/auto_*`,
then re-runs the selected best pipeline in `best_pipeline/`.
It also writes `tractability_by_builder.csv` with `N`, memory estimate, and wall-time
for each clustering/coarse-graining builder evaluated in the tractability pass.

### Data file format

For `data.csv`, use a dense numeric matrix with shape `T x p`:

- rows: time points
- columns: observed variables/channels
- no header row
- no index column
- all rows same width
- finite numeric values only (no text, `NaN`, or `inf`)

Both univariate (`p=1`) and multivariate (`p>1`) are supported.

If your file has extra columns, select only the signal columns with:
- `--data-column i` for a single column
- `--data-columns i j k ...` for multiple columns

## Setup

```bash
conda create -n prism39 python=3.9 -y
conda activate prism39
python -m pip install -r requirements.txt
```

Graphviz is required for transition graph rendering:

```bash
brew install graphviz  # macOS
```

## Discrete run

Discrete semantics:
- Reconstruction is fit on the training prefix only.
- Held-out log-loss is evaluated on the test suffix, but each test-time representation
  `z_t = phi_k(x_{1:t})` is computed from the full observed past across the
  train/test boundary (`x_train + x_test`).
- No model parameters are refit on held-out data; unseen test-time contexts back off
  to `p(x_{t+1}=1)=0.5`.
- CLI progress is printed at `INFO` level by default; use `--log-level WARNING`
  (or `ERROR`) for quieter runs.

```bash
cd src
python -m prism.cli \
  --process even_process \
  --reconstructor one_step \
  --ks 2 3 4 5 \
  --seeds 0 1 2 \
  --length 200000 \
  --outdir ./results/even_k_sweep \
  --force
```

Optional transition export:

```bash
python -m prism.cli \
  --process even_process \
  --reconstructor one_step \
  --ks 2 \
  --seeds 0 \
  --length 200000 \
  --save-transitions \
  --show-transitions-for last_2 \
  --outdir ./results/even_transitions \
  --force
```

## Continuous run (Kalman ISS)

Synthetic continuous process:

```bash
cd src
python -m prism.cli \
  --process linear_gaussian_ssm \
  --reconstructor kalman_iss \
  --ks 1 2 3 \
  --dvs 1 2 \
  --macro-eps 0.25 \
  --macro-bins 3 \
  --seeds 0 1 2 \
  --length 5000 \
  --outdir ./results/continuous_iss_sweep \
  --force
```

File-backed continuous process:

```bash
cd src
python -m prism.cli \
  --process continuous_file \
  --data-path /absolute/path/to/series.csv \
  --data-columns 0 1 2 \
  --reconstructor kalman_iss \
  --ks 2 4 \
  --dvs 1 2 \
  --seeds 0 1 \
  --length 10000 \
  --outdir ./results/continuous_file_iss \
  --force
```

### Continuous model selection and macrostate builders

Available macro builders:
- `hierarchical_single` (single-link threshold cut)
- `hierarchical_complete` (complete-link threshold cut)
- `linear_quantile` (linear coarse-graining baseline)
- `greedy` (anchor-binning baseline)

Global aliases are still accepted for compatibility:
`global_single -> hierarchical_single`, `global/global_complete -> hierarchical_complete`.

For blocked-CV model selection across projection mode / macro builder / epsilon grid:

```bash
cd src
python -m prism.cli \
  --process linear_gaussian_ssm \
  --reconstructor kalman_iss \
  --ks 2 \
  --dvs 2 \
  --model-select \
  --selection-score stability \
  --selection-projections pca random psi_opt \
  --selection-builders hierarchical_single hierarchical_complete linear_quantile greedy \
  --selection-macro-eps 0.2 0.25 0.3 \
  --selection-repeats 4 \
  --selection-perturb seed_blocked \
  --selection-block-frac 0.85 \
  --length 5000 \
  --seeds 0 1 \
  --outdir ./results/continuous_iss_selected \
  --force
```

Model-selection metadata is written to `runs.csv`:
`macro_builder`, `model_selection`, `selection_score`,
`selection_value`, `selection_predictive`, `selection_stability`,
variance summaries (`selection_n_states_var`, `selection_unifilarity_var`,
`selection_branch_entropy_var`), and tractability fields
(`macro_build_time_s`, `macro_distance_matrix_mb_est`, `macro_n_pred`).
Outer held-out reporting includes both ISS Gaussian loss (`gaussian_logloss`)
and builder-dependent macro predictive loss (`macro_logloss`).

### ISS Psi optimisation

You can optimise ISS Psi over a macro projection matrix `L` during fitting:

```bash
cd src
python -m prism.cli \
  --process linear_gaussian_ssm \
  --reconstructor kalman_iss \
  --macro-projection psi_opt \
  --compute-psi \
  --psi-optimiser random \
  --psi-restarts 12 \
  --psi-iters 120 \
  --ks 1 2 3 \
  --dvs 1 2 \
  --seeds 0 1 \
  --length 5000 \
  --outdir ./results/continuous_iss_psi \
  --force
```

This writes `psi_opt`, `psi_macro_dim`, and `psi_optimiser` into `runs.csv`.

`--psi-optimiser torch_adam` is also supported for gradient-based optimisation, but requires a local torch install.

## Summaries and figures

```bash
cd src
python -m prism.analysis.summarise --root ../results/even_k_sweep
python -m prism.analysis.plot_k --root ../results/even_k_sweep --metrics logloss n_states unifilarity_score branch_entropy
python -m prism.analysis.phase_diagram --root ../results/even_k_sweep

# Continuous-only analysis (ISS sweeps)
python -m prism.analysis.summarise --root ../results/continuous_iss_sweep
python -m prism.analysis.plot_k --root ../results/continuous_iss_sweep --dv 1 --metrics gaussian_logloss n_states C_mu_empirical psi_opt
python -m prism.analysis.continuous_heatmaps --root ../results/continuous_iss_sweep --shared-scale
python -m prism.analysis.compare_projection_modes --root ../results/continuous_iss_sweep --metrics gaussian_logloss n_states C_mu_empirical psi_opt
```

## Smoke commands

From repository root:

```bash
make smoke-discrete
make smoke-continuous
make smoke-continuous-psi
```

## Tests

```bash
make test
```
