# PRISM

Predictive Representations for Inference of Scale and Macrostates

PRISM is my final-year MEng Computing project.
It implements an experimental pipeline for studying **emergent macrostates** and **predictive state machines** under **representational constraints**.

---

## What this project does

At a high level:

- Generates binary time series from simple stochastic processes
- Applies fixed representations (e.g. last‑k histories)
- Reconstructs macrostates using a simple, explicit algorithm
- Evaluates prediction quality and computational closure properties
- Produces figures and state-machine diagrams

---

## Environment setup

I run everything in **Python 3.9** for dependency stability.

### Create environment
```bash
conda create -n prism39 python=3.9 -y
conda activate prism39
```

### Install dependencies
```bash
python -m pip install -r requirements.txt
```

### Graphviz (required)

Graphviz is needed to render learned state machines.

On macOS:
```bash
brew install graphviz
```

On Ubuntu:
```bash
sudo apt install graphviz
```

On Windows:

Install an up-to-date package from [Graphviz](https://graphviz.org/download/)

---

## Running experiments (CLI)

The main entry point is the CLI:

```bash
python -m prism.cli
```

Example: Even Process with a sweep over `k`:
```bash
python -m prism.cli \
  --process even_process \
  --ks 2 3 4 5 \
  --seeds 0 1 2 3 4 \
  --length 200000 \
  --outdir results/even_k_sweep \
  --force
```

This produces:
- `runs.csv` — raw per-seed results
- `config.json` — run configuration
- State-machine transition files (optionally) 

---

## Saving and visualising transitions

To export reconstructed state machines:

```bash
python -m prism.cli \
  --process even_process \
  --ks 2 \
  --seeds 0 \
  --length 200000 \
  --save-transitions \
  --show-transitions-for last_2
```

This produces `.json`, `.dot`, and `.png` files under:
```
results/<run>/transitions/
```

---

## Summarising results

After a run completes:

```bash
python -m prism.analysis.summarise --root results/even_k_sweep
```

This generates:
- `summary_by_condition.csv`
- `summary_simple.csv`

These are the inputs for all plotting scripts.

---

## Making figures

To generate all standard figures in one go:

```bash
python -m prism.analysis.make_figures \
  --root results/even_k_sweep \
  --subsample-step 1 \
  --metrics branch_entropy unifilarity_score logloss \
  --phase
```

Figures are written to:
```
results/<run>/figures/
```

---

## Frontend

There is also a very simple frontend for running experiments and inspecting outputs:

```bash
streamlit run src/prism/frontend/app.py
```

The frontend is intentionally minimal and mostly wraps the existing CLI and analysis scripts.