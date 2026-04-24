# Hierarchical Predictive Benchmark: Paired Summary

Hyperparameters are selected separately for each method and noise level by the mean training sweep metric shown in the figure. Paired differences are then computed across the same random seeds at those fixed parameters.

| Metric | Noise | PRISM param | k-means param | PRISM mean | k-means mean | Gain mean | 95% CI | Wins | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Joint ARI | 0.02 | 0.32 | 18 | 0.403 | 0.322 | 0.081 | [0.051, 0.111] | 5/5 | 0.002 |
| Joint ARI | 0.05 | 0.30 | 18 | 0.361 | 0.294 | 0.067 | [0.044, 0.090] | 5/5 | 0.001 |
| Joint ARI | 0.08 | 0.28 | 18 | 0.328 | 0.282 | 0.046 | [0.024, 0.068] | 5/5 | 0.005 |
| Joint ARI | 0.12 | 0.25 | 18 | 0.281 | 0.243 | 0.038 | [0.018, 0.059] | 5/5 | 0.007 |
| Joint ARI | 0.16 | 0.22 | 18 | 0.245 | 0.214 | 0.032 | [0.011, 0.053] | 5/5 | 0.013 |
| Joint ARI | 0.20 | 0.22 | 12 | 0.206 | 0.193 | 0.014 | [-0.012, 0.039] | 3/5 | 0.210 |
| Held-out NLL | 0.02 | 0.22 | 18 | 1.419 | 1.485 | 0.065 | [0.031, 0.100] | 5/5 | 0.006 |
| Held-out NLL | 0.05 | 0.22 | 18 | 1.461 | 1.518 | 0.056 | [0.036, 0.077] | 5/5 | 0.002 |
| Held-out NLL | 0.08 | 0.22 | 18 | 1.495 | 1.539 | 0.044 | [0.035, 0.053] | 5/5 | 0.000 |
| Held-out NLL | 0.12 | 0.22 | 18 | 1.540 | 1.578 | 0.038 | [0.032, 0.044] | 5/5 | 0.000 |
| Held-out NLL | 0.16 | 0.22 | 18 | 1.575 | 1.606 | 0.031 | [0.024, 0.038] | 5/5 | 0.000 |
| Held-out NLL | 0.20 | 0.22 | 18 | 1.617 | 1.639 | 0.022 | [0.018, 0.026] | 5/5 | 0.000 |

Positive joint-ARI gain means PRISM has higher joint-state recovery. Positive NLL gain means PRISM has lower held-out next-symbol negative log-likelihood.
