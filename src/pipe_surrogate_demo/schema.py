"""Canonical Pipe tensor schema and task registry for the demonstration."""

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

INPUT_CHANNELS = (
    "Rainfall",
    "Cumulative",
    "Length",
    "Diameter",
    "Slope",
    "InletNode_InvertElev",
    "OutletNode_InvertElev",
)
TARGET_NAMES = ("Rate", "Depth", "Vel", "Vol", "Cap")
TARGET_LABELS = {
    "Rate": "flow rate",
    "Depth": "flow depth",
    "Vel": "flow velocity",
    "Vol": "flow volume",
    "Cap": "SWMM link capacity (dimensionless)",
}


def normalize_tasks(tasks: Sequence[str] | None = None) -> tuple[str, ...]:
    """Validate and return selected targets in canonical order."""
    requested = TARGET_NAMES if tasks is None else tuple(tasks)
    if not requested:
        raise ValueError("at least one target task must be selected")
    unknown = sorted(set(requested).difference(TARGET_NAMES))
    if unknown:
        raise ValueError(f"unknown target task(s): {', '.join(unknown)}")
    if len(set(requested)) != len(requested):
        raise ValueError("target tasks must be unique")
    return tuple(name for name in TARGET_NAMES if name in requested)


def all_task_sets() -> tuple[tuple[str, ...], ...]:
    """Return all 31 non-empty task subsets in canonical order."""
    return tuple(
        tuple(task_set)
        for size in range(1, len(TARGET_NAMES) + 1)
        for task_set in combinations(TARGET_NAMES, size)
    )


@dataclass(frozen=True)
class PipeSchema:
    """Shape contract for Pipe inputs and targets.

    The paper data use 396 links.  The demo uses fewer links by default so it
    can run on a CPU in seconds while preserving the same tensor semantics.
    """

    history: int = 6
    horizon: int = 6
    links: int = 12

    @property
    def input_channels(self) -> int:
        return len(INPUT_CHANNELS)

    @property
    def target_channels(self) -> int:
        return len(TARGET_NAMES)

    def validate(self) -> None:
        for name, value in (
            ("history", self.history),
            ("horizon", self.horizon),
            ("links", self.links),
        ):
            minimum = 2 if name == "links" else 1
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}, got {value}")
