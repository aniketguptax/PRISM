# PRISM

Predictive Representations for Inference of Scale and Macrostates.

PRISM supports two pipelines:

- Discrete binary processes with `LastK` representations and one-step merge reconstruction.
- Continuous scalar time series with Kalman ISS (linear-Gaussian state-space) reconstruction.

## Setup

```bash
conda create -n prism39 python=3.9 -y
conda activate prism39
python -m pip install -r requirements.txt
```

Graphviz is required only for discrete transition rendering:

```bash
brew install graphviz  # macOS
```

## Discrete run (binary baseline)

```bash
cd src
python -m prism.cli \
  --process even_process \
  --reconstructor one_step \
  --ks 2 3 4 5 \
  --seeds 0 1 2 \
  --length 200000 \
  --outdir ../results/even_k_sweep \
  --force
```

Optional transition export (discrete-only):

```bash
python -m prism.cli \
  --process even_process \
  --reconstructor one_step \
  --ks 2 \
  --seeds 0 \
  --length 200000 \
  --save-transitions \
  --show-transitions-for last_2 \
  --outdir ../results/even_transitions \
  --force
```

## Continuous run (Kalman ISS)

Synthetic continuous process:

```bash
cd src
python -m prism.cli \
  --process linear_gaussian_ssm \
  --reconstructor kalman_iss \
  --ks 1 2 3 4 \
  --seeds 0 1 2 \
  --length 5000 \
  --outdir ../results/continuous_iss_sweep \
  --force
```

File-backed continuous process:

```bash
cd src
python -m prism.cli \
  --process continuous_file \
  --data-path /absolute/path/to/series.csv \
  --data-column 0 \
  --reconstructor kalman_iss \
  --ks 2 4 6 \
  --seeds 0 1 \
  --length 10000 \
  --outdir ../results/continuous_file_iss \
  --force
```

`kalman_iss` does not expose discrete symbol-conditioned transitions, so `--save-transitions` is intentionally blocked.

## Summaries and figures

```bash
cd src
python -m prism.analysis.summarise --root ../results/even_k_sweep
python -m prism.analysis.plot_k --root ../results/even_k_sweep --metrics logloss n_states unifilarity_score branch_entropy
python -m prism.analysis.phase_diagram --root ../results/even_k_sweep
```

For continuous runs, phase-diagram plots are skipped automatically when branch/unifilarity metrics are undefined.

## Smoke commands

From repository root:

```bash
make smoke-discrete
make smoke-continuous
```

## Tests

```bash
make test
```
