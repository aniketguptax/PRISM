Figure X. Hierarchical predictive-state recovery benchmark.

The synthetic generator contains a slow coarse regime and a faster fine phase. The
coarse regime changes the future dynamics of the fine phase but is not directly
labelled by instantaneous symbol frequencies, so successful recovery requires
clustering histories by their future predictive distributions. Across five seeds
per noise level, PRISM predictive clustering outperformed raw-history k-means on
best joint hidden-state recovery at every tested noise level, with mean joint ARI
gains ranging from 0.014 to 0.081. PRISM also achieved lower
held-out next-symbol negative log-likelihood at every noise level, with NLL gains
ranging from 0.022 to 0.065. Seed-level comparisons were stable:
PRISM won 28/30 paired joint-ARI comparisons and
30/30 paired held-out-NLL comparisons. The multiscale
path shows that predictive recovery peaks at an intermediate merge tolerance:
overly fine partitions retain many nearly deterministic contexts, while excessive
coarsening collapses the hidden predictive state.
