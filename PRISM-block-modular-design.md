# Block-modular LGSSM validation

This note fixes the design for the synthetic CE 2.0 validation used in the continuous PRISM chapter.

## Aim

We want a setting where:

- the latent generator has a known modular structure;
- the observation map can either hide or reveal that structure;
- ISS macrostates and PCA+k-means are compared under the same data and scoring;
- the headline result is stated in Hoel CE 2.0 terms, not just recovery ARI.

## Generator

- Latent dimension: `d = 4`, arranged as two `2 x 2` blocks.
- Slow block: diagonal eigenvalues `(0.92, 0.85)`.
- Rotational block: `0.70 * rot(pi / 6)`.
- Cross-block coupling: `epsilon / sqrt(d)` times a dense off-block Gaussian perturbation.
- Observation dimension: `p = 8`.
- Observation designs:
  - `random`: Haar-random orthonormal mixing across both blocks.
  - `aligned`: block-aligned mixing as a positive control.

## Sweep

- Couplings: `{0, 0.025, 0.05, 0.10, 0.20}`
- Seeds: `{0, 1, 2}`
- ISS builders: `hierarchical_single`, `hierarchical_complete`, `linear_quantile`, `greedy`
- Macro tolerances: `{0.10, 0.15, 0.25, 0.40}`
- PCA+k-means baseline: `k in {2, 4, 8}`
- Length: `T = 4000`
- EM iterations: `50`

## Outputs

The sweep writes:

- `recovery.csv`: ARI, unifilarity, branching entropy, and held-out Gaussian log-loss.
- `labels_eps*.npz`: recovered label sequences for ISS and PCA+k-means, plus ground-truth regimes.
- `emergence.csv`: total CE and emergent complexity per run.
- figures under `figures/` for both the recovery and emergence summaries.

## Ground-truth regimes

The generator exposes three regime labels used for evaluation:

- `slow_block`: quantile bin of the slow block.
- `phase_block`: angular bin of the rotational block.
- `joint`: product partition of slow-block and phase-block labels.

The old instantaneous block-attribution label is retained as a check, but it is not the main target.

## Pre-registered endpoints

1. Under random mixing, ISS should show strictly positive `total_ce` at `epsilon in {0, 0.025, 0.05}` for at least two of three seeds.
2. Under random mixing, PCA+k-means should sit materially below ISS at matched coupling.
3. Under aligned mixing, PCA+k-means should approach ISS as a positive control.
4. ISS `total_ce` should decay as coupling increases from `0` to `0.20`.

## Commands

Local smoke run:

```bash
make block-modular-smoke
```

Full sweep:

```bash
make block-modular-sweep
```

The full sweep is intended for the Imperial cluster. The local smoke run is only there to confirm the pipeline and output schema.
