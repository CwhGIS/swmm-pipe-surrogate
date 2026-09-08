"""Canonical Pipe tensor schema used by the demonstration."""

from dataclasses import dataclass

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
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
