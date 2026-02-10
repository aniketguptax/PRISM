"""Shared type aliases used across PRISM."""

from typing import Sequence, Union

# PRISM currently supports scalar discrete (int) and scalar continuous (float)
# observation streams.
Obs = Union[int, float]
LatentState = Union[int, float]
ObsSequence = Sequence[Obs]
