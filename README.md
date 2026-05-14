# PRISM

PRISM is the codebase for my final-year project on predictive-state
representations and macrostates. It contains:

- a discrete symbolic pipeline for binary processes;
- a continuous Kalman ISS pipeline for multivariate time series;
- synthetic recovery benchmarks used in the thesis;
- EEG processing scripts for the real-data validation.

The repository is set up as a research codebase rather than a packaged library.
Most commands are run from the repository root or from `src`.

## Layout

```text
src/prism/              Core PRISM package
src/prism/processes/    Synthetic data generators
src/prism/experiments/  Thesis synthetic sweeps
src/prism/analysis/     Figure and summary scripts
data/processing/        EEG experiment and reporting pipeline
hpc/                    Scripts for heavy experimental runs 
```

## Setup

The Makefile expects a Python interpreter at `./venv/bin/python` unless
`PYTHON=...` is supplied.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Graph rendering also needs the system Graphviz binary:

```bash
brew install graphviz      # macOS
# or: sudo apt install graphviz
```

On the Imperial HPC, load Python first and point `make` at the environment you
created there:

```bash
module load Python/3.11.3-GCCcore-12.3.0
PYTHON="$HOME/venvs/prism/bin/python" make test
```

## Basic Checks

```bash
make test
make smoke-discrete
make smoke-continuous
make smoke-continuous-psi
```

The smoke targets write under `src/results/smoke/`.

## Main Thesis Runs

These are the two synthetic runs currently used as the main validation targets.

```bash
make hierarchical-predictive-main
make low-variance-lgssm-main
```

They write:

```text
src/results/hierarchical_predictive_main/
src/results/low_variance_lgssm_main/
```

Each target runs the sweep and then generates the figure/report artefacts used
in the thesis.

## Generic PRISM CLI

The CLI is useful for quick synthetic sweeps and file-backed continuous data.

Discrete example:

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

Input CSVs should be dense numeric matrices with rows as time points and columns
as observed variables. Do not include a header row or index column unless you
select signal columns explicitly with `--data-column` or `--data-columns`.

## Analysis Scripts

For generic CLI runs:

```bash
cd src
python -m prism.analysis.summarise --root ./results/even_k_sweep
MPLBACKEND=Agg python -m prism.analysis.plot_k \
  --root ./results/even_k_sweep \
  --metrics logloss n_states unifilarity_score branch_entropy
MPLBACKEND=Agg python -m prism.analysis.phase_diagram \
  --root ./results/even_k_sweep
```

The thesis-specific Makefile targets already call their own figure/report
scripts, so prefer the `make ...-main` targets for those experiments.

## EEG Dataset

The real-data analysis uses the OpenNeuro `ds001785` EEG dataset from Pereira
et al. (2021), version
[`1.1.1`](https://openneuro.org/datasets/ds001785/versions/1.1.1). It is a
vibrotactile near-threshold detection task: participants received brief
right-thumb stimuli at calibrated intensities, reported whether they detected
the stimulus, and gave confidence ratings. The project uses the available 18
participants with preprocessed EEG and complete behavioural fields.

The local code expects two forms of this data:

- EEGLAB derivatives under `data/ds001785/derivatives/eegprep`, used for channel
  labels and region assignment;
- MATLAB-exported trial arrays under `data/exports_mat`, loaded as
  `trials x time x channels`.

These files are too large for git and are ignored. The subject ids used by the
current analysis are `sub-01` to `sub-07` and `sub-09` to `sub-19`.

## EEG Pipeline

The EEG code lives in `data/processing`. It expects:

- exported subject files under `data/exports_mat`;
- preprocessed EEGLAB files under `data/ds001785/derivatives/eegprep`;
- baseline results under `data/results_baseline` for the final evidence
  comparison.

The current full central-window PRISM run is captured in:

```bash
qsub run_eeg_prism_central_full.pbs
```

For the central-window timecourse, run the array job instead:

```bash
qsub run_eeg_prism_central_timecourse_array.pbs
```

This runs the central PRISM analysis for 18 subjects across overlapping 250 ms
windows stepped every 50 ms from 0-250 to 250-500 ms. After syncing the array outputs back,
summarise the subject-wise timecourse decoder with:

```bash
make eeg-prism-timecourse-decoder
```

To run the matched PRISM region x timecourse sweep:

```bash
qsub run_eeg_prism_region_timecourse_array.pbs
```

This uses the same overlapping 250 ms windows and evaluates all five scalp
regions in each array task.

After syncing the result folders back:

```bash
make eeg-prism-region-timecourse-summary
```

The full central-window script calls `data/processing/run.py prism-region-window`
across all subjects.

After the result folders have been synced back locally, the current report EEG
figures can be regenerated with:

```bash
make eeg-var-timecourse-figure
make eeg-report-figures
```

`eeg-var-timecourse-figure` rebuilds the central VAR time-course figure from
the baseline sweep. `eeg-report-figures` copies the current EEG PDFs into
`report/figures/ch5`; the report folder is still ignored by git.

For local debugging, use the same command with `--max-trials` and a single
subject:

```bash
PYTHONPATH="$PWD/data/processing:$PWD/src" \
python data/processing/run.py prism-region-window \
  --subject sub-01 \
  --export-dir data/exports_mat \
  --derivatives-dir data/ds001785/derivatives/eegprep \
  --outdir data/results_prism/debug_prism_region_window \
  --regions central \
  --rep-dims 4 \
  --projection-modes pca \
  --macro-builder hierarchical_complete \
  --macro-eps 0.25 \
  --macro-bins 3 \
  --em-iters 20 \
  --em-tol 1e-3 \
  --train-start-ms -300 \
  --train-end-ms 0 \
  --test-start-ms 125 \
  --test-end-ms 375 \
  --max-trials 20
```

## Auto Mode

For a quick file-based run where PRISM chooses a configuration:

```bash
./prism /absolute/path/to/data.csv
# or
cd src
python -m prism /absolute/path/to/data.csv
```

This writes an auto sweep under `results/auto_*` and then re-runs the selected
pipeline in `best_pipeline/`.

## Notes

- Use `MPLBACKEND=Agg` on headless machines.
- On shared machines, set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and
  `MKL_NUM_THREADS=1` for batch jobs.
- `psi_opt` supports a random optimiser by default. `torch_adam` is available
  if PyTorch is installed.
- Keep generated result folders out of git. The report should cite generated
  CSVs/figures, but the large data products stay local or on HPC storage.
