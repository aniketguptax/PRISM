# PRISM

This is the codebase for my final-year project on predictive-state
representations and macrostates.

PRISM asks whether a chosen representation of a time series induces a useful
finite state machine: one that preserves future-relevant distinctions, has a
tractable transition structure, and remains interpretable at finite sample
size. The repository contains the discrete symbolic pipeline, the continuous
Kalman ISS pipeline, controlled benchmarks, trained-network experiments, and
the neural analysis scripts used in the thesis.

This is a research codebase, *not* a packaged library. Most commands are run from
the repository root with `PYTHONPATH=src`, or from inside `src`.

## Setup

Create a virtual environment and install the Python requirements:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Graph rendering needs the system Graphviz binary:

```bash
brew install graphviz
# or: sudo apt install graphviz
```

If using the Imperial HPC, load Python before using the environment created there:

```bash
module load Python/3.11.3-GCCcore-12.3.0
PYTHONPATH=src "$HOME/venvs/prism/bin/python" -m pytest src/prism/tests
```

For headless plotting, set:

```bash
export MPLBACKEND=Agg
```

For shared machines or PBS jobs, also set thread counts explicitly:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## Basic Checks

From the repository root:

```bash
PYTHONPATH=src venv/bin/python -m pytest src/prism/tests
PYTHONPATH=src venv/bin/python -m prism.cli --help
```

The wrapper script also works after activating the virtual environment:

```bash
./prism --help
```

## Generic CLI

Discrete symbolic example:

```bash
cd src
python -m prism.cli \
  --process even_process \
  --reconstructor one_step \
  --ks 2 3 4 5 \
  --seeds 0 1 2 \
  --length 200000 \
  --outdir ./results/even_process_sweep \
  --force
```

Continuous Kalman ISS example:

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

File-backed continuous example:

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

Input CSVs should be dense numeric matrices with rows as time points and columns as observed variables. Do not include a header row or index column unless the signal columns are selected explicitly with `--data-column` or `--data-columns`.

For a quick file-based auto run:

```bash
./prism /absolute/path/to/data.csv
# or
cd src
python -m prism /absolute/path/to/data.csv
```

This writes an auto sweep under `src/results/auto_*` and then re-runs the selected pipeline in `best_pipeline/`.

## Generic Analysis

For generic CLI outputs:

```bash
cd src
python -m prism.analysis.summarise --root ./results/even_process_sweep
MPLBACKEND=Agg python -m prism.analysis.plot_k \
  --root ./results/even_process_sweep \
  --metrics logloss n_states unifilarity_score branch_entropy
MPLBACKEND=Agg python -m prism.analysis.phase_diagram \
  --root ./results/even_process_sweep
```

## Thesis Experiments

The main controlled results are generated from experiment modules in `src/prism/experiments/`.

Most scripts expect their corresponding result folders to exist already under `src/results/` or `data/results_prism/`. Regeneration is usually a two-step process: run the sweep, then run the report script that summarises it into a CSV, Markdown summary, or figure.

The trained-network experiments live in:

```text
src/prism/experiments/rnn_hidden_state_recovery.py
src/prism/experiments/transformer_hidden_state_recovery.py
src/prism/experiments/even_k_transformer_recovery.py
```

## Notes

- Use `PYTHONPATH=src` when running scripts from the repository root.
- Keep generated data, result folders, logs, and report builds out of git.
- Prefer small smoke runs before large sweeps.
- The code records seeds and configuration files for most generated result folders; use those files when checking or reproducing a run.
