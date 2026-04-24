## Results Paragraph

To test whether PRISM recovers genuinely predictive hidden structure rather than
simple symbol-frequency clusters, I introduced a hierarchical hidden process in
which a slow coarse regime controls the dynamics of a faster fine phase. The
coarse regime is not directly identified by instantaneous observations; it is
revealed through the future distribution of symbol sequences. Across five random
seeds per noise level, PRISM predictive clustering achieved higher mean joint
hidden-state ARI than raw-history k-means at all tested noise levels, with gains
from 0.014 to 0.081. The paired 95% confidence intervals for
joint ARI remained above zero through emission noise 0.16; at
noise 0.20 the mean gain was still positive but less stable across seeds. PRISM
also achieved lower held-out next-symbol negative log-likelihood at every noise
level, with paired 95% confidence intervals above zero through noise
0.20. The scale-path analysis shows the expected multiscale
structure: very fine partitions are highly unifilar but over-resolved, moderate
merge tolerances maximise joint-state recovery, and aggressive coarsening
collapses the predictive state. This validates the core PRISM claim in a setting
where the target hierarchy is known by construction and cannot be recovered by
clustering raw histories alone.
