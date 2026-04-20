"""Channel-group helpers for region-wise EEG comparisons."""

from __future__ import annotations

import numpy as np


DEFAULT_REGION_ORDER = (
    "all_channels",
    "frontal",
    "central",
    "temporal",
    "parietal",
    "occipital",
)
NON_CORTICAL_LABELS = {"VEOG", "HEOG", "ECG", "EOG"}


def assign_channel_region(label: str) -> str | None:
    label = str(label).strip().upper()
    if not label or label in NON_CORTICAL_LABELS:
        return None

    if label.startswith(("FT", "TP", "T")):
        return "temporal"
    if label.startswith(("PO", "O")):
        return "occipital"
    if label.startswith("P"):
        return "parietal"
    if label.startswith(("CP", "C")):
        return "central"
    if label.startswith(("FP", "AF", "FC", "F")):
        return "frontal"
    return None


def build_channel_groups(
    channel_labels: list[str],
    region_order: tuple[str, ...] = DEFAULT_REGION_ORDER,
) -> dict[str, np.ndarray]:
    channel_labels = [str(label).strip() for label in channel_labels]
    groups: dict[str, np.ndarray] = {}

    for region_name in region_order:
        if region_name == "all_channels":
            groups[region_name] = np.arange(len(channel_labels), dtype=int)
            continue

        idx = [
            chan_idx
            for chan_idx, label in enumerate(channel_labels)
            if assign_channel_region(label) == region_name
        ]
        if idx:
            groups[region_name] = np.asarray(idx, dtype=int)

    return groups


def cortical_channel_indices(channel_labels: list[str]) -> np.ndarray:
    channel_labels = [str(label).strip() for label in channel_labels]
    idx = [
        chan_idx
        for chan_idx, label in enumerate(channel_labels)
        if assign_channel_region(label) is not None
    ]
    return np.asarray(idx, dtype=int)


def build_size_matched_control_groups(
    channel_labels: list[str],
    region_groups: dict[str, np.ndarray],
    *,
    n_draws: int,
    random_state: int,
) -> dict[tuple[str, int], np.ndarray]:
    if n_draws < 0:
        raise ValueError("n_draws must be non-negative")
    if n_draws == 0:
        return {}

    cortical_idx = cortical_channel_indices(channel_labels)
    if cortical_idx.size == 0:
        raise ValueError("No cortical channels were available for size-matched controls")

    control_groups: dict[tuple[str, int], np.ndarray] = {}
    for region_name, region_idx in region_groups.items():
        if region_name == "all_channels":
            continue

        region_idx = np.asarray(region_idx, dtype=int)
        candidate_pool = cortical_idx[~np.isin(cortical_idx, region_idx)]
        if region_idx.size > candidate_pool.size:
            raise ValueError(
                f"Cannot draw a size-matched control for {region_name}: "
                f"need {region_idx.size} channels but only {candidate_pool.size} remain"
            )

        region_seed = int(random_state + sum(ord(char) for char in region_name))
        rng = np.random.default_rng(region_seed)
        for draw_idx in range(1, n_draws + 1):
            sampled_idx = np.sort(
                rng.choice(candidate_pool, size=region_idx.size, replace=False)
            )
            control_groups[(region_name, draw_idx)] = sampled_idx

    return control_groups
