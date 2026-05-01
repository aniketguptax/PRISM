"""Channel-group helpers for region-wise EEG comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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


@dataclass(frozen=True)
class RegionGroup:
    name: str
    indices: np.ndarray
    kind: str = "named_region"
    matched_region_name: str | None = None
    control_draw_idx: int | None = None


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


def select_region_groups(
    channel_labels: list[str],
    requested_regions: Iterable[str],
) -> tuple[RegionGroup, ...]:
    named_groups = build_channel_groups(channel_labels)
    selected_groups = tuple(
        RegionGroup(
            name=region_name,
            indices=named_groups[region_name],
            matched_region_name=region_name,
        )
        for region_name in requested_regions
        if region_name in named_groups
    )
    if not selected_groups:
        raise ValueError(
            "No valid channel groups were selected. Available groups are: "
            + ", ".join(named_groups.keys())
        )
    return selected_groups


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


def build_control_region_groups(
    channel_labels: list[str],
    selected_groups: tuple[RegionGroup, ...],
    *,
    n_draws: int,
    random_state: int,
) -> tuple[RegionGroup, ...]:
    selected_by_name = {group.name: group.indices for group in selected_groups}
    control_groups = build_size_matched_control_groups(
        channel_labels,
        selected_by_name,
        n_draws=n_draws,
        random_state=random_state,
    )
    return tuple(
        RegionGroup(
            name=f"{matched_region_name}_control_{draw_idx:02d}",
            indices=indices,
            kind="size_matched_control",
            matched_region_name=matched_region_name,
            control_draw_idx=int(draw_idx),
        )
        for (matched_region_name, draw_idx), indices in control_groups.items()
    )


def rep_dims_for_group(rep_dims: Iterable[int], group: RegionGroup) -> tuple[int, ...]:
    n_channels = int(group.indices.size)
    return tuple(int(dim) for dim in rep_dims if int(dim) <= n_channels)


def add_region_metadata(
    rows: list[dict],
    group: RegionGroup,
    *,
    include_control_fields: bool = False,
) -> list[dict]:
    for row in rows:
        row["region_name"] = group.name
        row["n_region_channels"] = int(group.indices.size)
        if include_control_fields:
            row["group_kind"] = group.kind
            row["matched_region_name"] = group.matched_region_name or group.name
            row["control_draw_idx"] = (
                np.nan if group.control_draw_idx is None else int(group.control_draw_idx)
            )
    return rows
