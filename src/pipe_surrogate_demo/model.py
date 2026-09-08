"""Compact GraphWaveNet-style model for the public Pipe demonstration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class TemporalGraphBlock(nn.Module):
    """Causal gated temporal convolution followed by graph propagation."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.dilation = dilation
        self.temporal = nn.Conv2d(
            channels,
            2 * channels,
            kernel_size=(1, 3),
            dilation=(1, dilation),
        )
        self.graph = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        padding = 2 * self.dilation
        temporal = self.temporal(F.pad(x, (padding, 0, 0, 0)))
        gate, filter_value = temporal.chunk(2, dim=1)
        hidden = torch.tanh(gate) * torch.sigmoid(filter_value)
        hidden = torch.einsum("bcnt,nm->bcmt", hidden, support)
        hidden = self.norm(self.graph(hidden))
        return x + hidden


@dataclass(frozen=True)
class ModelConfig:
    """Architecture parameters kept together for a small public API."""

    input_channels: int = 7
    target_channels: int = 5
    horizon: int = 6
    hidden_channels: int = 16
    dilations: tuple[int, ...] = (1, 2)


class GraphWaveNetDemo(nn.Module):
    """Small fixed-support model with the same input/output semantics as Pipe."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.horizon = config.horizon
        self.target_channels = config.target_channels
        self.input_projection = nn.Conv2d(
            config.input_channels, config.hidden_channels, kernel_size=1
        )
        self.blocks = nn.ModuleList(
            TemporalGraphBlock(config.hidden_channels, dilation)
            for dilation in config.dilations
        )
        self.output_projection = nn.Sequential(
            nn.Conv2d(config.hidden_channels, config.hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(
                config.hidden_channels,
                config.target_channels * config.horizon,
                kernel_size=1,
            ),
        )

    def forward(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """Predict ``(batch, horizon, links, target_channels)`` from Pipe ``X``."""
        if x.ndim != 4:
            raise ValueError(f"expected X with 4 dimensions, got {tuple(x.shape)}")
        hidden = self.input_projection(x.permute(0, 3, 2, 1))
        for block in self.blocks:
            hidden = block(hidden, support)
        output = self.output_projection(hidden).mean(dim=-1)
        batch, _, links = output.shape
        output = output.view(batch, self.target_channels, self.horizon, links)
        return output.permute(0, 2, 3, 1)
