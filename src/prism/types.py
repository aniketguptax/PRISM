from typing import Sequence, Tuple, Union

# Observations are either scalar (discrete or continuous) or fixed-width tuples for
# multivariate continuous streams.
Obs = Union[int, float, Tuple[float, ...]]
LatentState = Union[int, float, Tuple[float, ...]]
ObsSequence = Sequence[Obs]
