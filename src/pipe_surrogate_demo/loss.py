"""Task-composition losses used by the clean public demonstration."""

from __future__ import annotations

import torch
from torch import nn


class TaskwiseMSELoss(nn.Module):
    """Return the unweighted sum of per-task mean squared errors.

    Inputs use the demo layout ``(batch, horizon, links, tasks)``. Averaging
    within each task before summing mirrors the task-wise MSE objective used by
    the paper while keeping the loss independent of the number of links and
    forecast horizons.
    """

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.ndim != 4 or target.ndim != 4:
            raise ValueError("prediction and target must both have four dimensions")
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target shapes differ: "
                f"{tuple(prediction.shape)} != {tuple(target.shape)}"
            )
        per_task_mse = (prediction - target).square().mean(dim=(0, 1, 2))
        return per_task_mse.sum()
