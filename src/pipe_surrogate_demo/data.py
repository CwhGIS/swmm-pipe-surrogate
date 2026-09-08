"""Synthetic data and graph helpers for the clean public demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import PipeSchema


def make_ring_support(links: int) -> torch.Tensor:
    """Return a row-normalized ring adjacency with self-loops."""
    if links < 2:
        raise ValueError("links must be at least 2 for a ring graph")
    adjacency = torch.eye(links, dtype=torch.float32)
    for node in range(links):
        adjacency[node, (node - 1) % links] = 1.0
        adjacency[node, (node + 1) % links] = 1.0
    return adjacency / adjacency.sum(dim=1, keepdim=True)


def make_synthetic_pipe_data(
    samples: int,
    schema: PipeSchema,
    seed: int = 3407,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic toy tensors with the real Pipe tensor layout.

    Returns ``X`` with shape ``(samples, history, links, 7)`` and ``Y`` with
    shape ``(samples, horizon, links, 5)``. Rainfall is dynamic and shared
    across links, cumulative rainfall is its time cumulative, and the five
    geometric channels are fixed per link and broadcast over each sample and
    history. The values are synthetic and do not reproduce paper results.
    """
    if samples < 2:
        raise ValueError("samples must be at least 2")
    schema.validate()
    rng = np.random.default_rng(seed)
    rainfall = rng.uniform(
        0.0, 1.0, size=(samples, schema.history, 1, 1)
    ).astype(np.float32)
    cumulative = np.cumsum(rainfall, axis=1)
    static_attributes = rng.normal(
        0.0, 1.0, size=(schema.links, 5)
    ).astype(np.float32)
    static_attributes = np.broadcast_to(
        static_attributes[None, None, :, :],
        (samples, schema.history, schema.links, 5),
    )
    rainfall = np.broadcast_to(
        rainfall, (samples, schema.history, schema.links, 1)
    )
    cumulative = np.broadcast_to(
        cumulative, (samples, schema.history, schema.links, 1)
    )
    x = np.concatenate((rainfall, cumulative, static_attributes), axis=-1).copy()
    link_position = np.linspace(-1.0, 1.0, schema.links, dtype=np.float32)
    rainfall_last = x[:, -1, :, 0:1]
    cumulative_last = x[:, -1, :, 1:2]
    static_last = x[:, -1, :, 2:].mean(axis=-1, keepdims=True)
    driver = (
        0.35 * rainfall_last
        + 0.20 * cumulative_last
        + 0.05 * static_last
        + 0.03 * link_position[None, :, None]
    )
    target_scales = np.asarray([1.0, 0.7, 0.5, 0.35, 0.2], dtype=np.float32)
    noise = rng.normal(
        0.0,
        0.03,
        size=(samples, schema.horizon, schema.links, schema.target_channels),
    ).astype(np.float32)
    trend = np.linspace(0.0, 0.12, schema.horizon, dtype=np.float32)[None, :, None, None]
    y = driver[:, None, :, :] * target_scales[None, None, None, :]
    y = y + trend * target_scales[None, None, None, :] + noise
    return torch.from_numpy(x), torch.from_numpy(y)


@dataclass
class SyntheticPipeDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """In-memory dataset used by the smoke demo."""

    x: torch.Tensor
    y: torch.Tensor

    @classmethod
    def build(cls, samples: int, schema: PipeSchema, seed: int) -> "SyntheticPipeDataset":
        x, y = make_synthetic_pipe_data(samples, schema, seed)
        return cls(x=x, y=y)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]
