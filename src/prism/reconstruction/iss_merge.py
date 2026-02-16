"""Backward-compatible aliases for the Kalman ISS reconstructor module."""

from .kalman_iss import GaussianPredictiveStateModel, KalmanISSReconstructor

# Historical name kept for compatibility with earlier runs/scripts.
KalmanISSGreedyMerge = KalmanISSReconstructor

__all__ = ["GaussianPredictiveStateModel", "KalmanISSReconstructor", "KalmanISSGreedyMerge"]
