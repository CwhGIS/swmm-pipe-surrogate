"""Small, reproducible Pipe-surrogate training demonstration."""

from .schema import (
    INPUT_CHANNELS,
    TARGET_NAMES,
    PipeSchema,
    all_task_sets,
    normalize_tasks,
)

__all__ = [
    "INPUT_CHANNELS",
    "TARGET_NAMES",
    "PipeSchema",
    "all_task_sets",
    "normalize_tasks",
]
__version__ = "0.1.0"
