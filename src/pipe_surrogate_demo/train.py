"""CPU-friendly synthetic training entry point for the public demo."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .data import SyntheticPipeDataset, make_ring_support
from .model import GraphWaveNetDemo, ModelConfig
from .schema import PipeSchema


def seed_everything(seed: int) -> None:
    """Seed all local random sources for a repeatable demo run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    support: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    loss_fn = nn.MSELoss()
    losses: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            prediction = model(x.to(device), support)
            losses.append(float(loss_fn(prediction, y.to(device)).item()))
    return float(np.mean(losses))


def _make_loaders(
    config: "DemoConfig", dataset: SyntheticPipeDataset
) -> tuple[DataLoader, DataLoader]:
    """Build deterministic, single-process train and validation loaders."""
    split = max(1, int(0.75 * len(dataset)))
    train_indices = list(range(split))
    val_indices = list(range(split, len(dataset))) or [train_indices[-1]]
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


def _make_model(
    schema: PipeSchema, device_name: str
) -> tuple[GraphWaveNetDemo, torch.Tensor, torch.device]:
    """Create the model and fixed synthetic ring support."""
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    support = make_ring_support(schema.links).to(device)
    model = GraphWaveNetDemo(
        ModelConfig(
            input_channels=schema.input_channels,
            target_channels=schema.target_channels,
            horizon=schema.horizon,
        )
    ).to(device)
    return model, support, device


@dataclass
class EpochContext:
    """Objects shared by the train and validation loops."""

    support: torch.Tensor
    device: torch.device
    optimizer: torch.optim.Optimizer
    loss_fn: nn.Module


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    context: EpochContext,
) -> float:
    """Run one supervised training epoch."""
    model.train()
    losses: list[float] = []
    for x, y in loader:
        context.optimizer.zero_grad(set_to_none=True)
        prediction = model(x.to(context.device), context.support)
        loss = context.loss_fn(prediction, y.to(context.device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        context.optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def _write_outputs(
    config: "DemoConfig", result: dict[str, Any], model: nn.Module
) -> None:
    """Write opt-in demo artifacts below the requested output directory."""
    if config.output_dir is None:
        return
    destination = Path(config.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    torch.save(model.state_dict(), destination / "demo_model.pth")


@dataclass(frozen=True)
class DemoConfig:
    """Runtime options for a deterministic synthetic training run."""

    epochs: int = 2
    samples: int = 24
    links: int = 12
    batch_size: int = 4
    seed: int = 3407
    device: str = "cpu"
    output_dir: str | Path | None = "outputs/demo"


def train_demo(config: DemoConfig | None = None) -> dict[str, Any]:
    """Train the compact model on deterministic synthetic Pipe data."""
    config = config or DemoConfig()
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    seed_everything(config.seed)
    schema = PipeSchema(links=config.links)
    dataset = SyntheticPipeDataset.build(
        samples=config.samples, schema=schema, seed=config.seed
    )
    train_loader, val_loader = _make_loaders(config, dataset)
    model, support, run_device = _make_model(schema, config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    context = EpochContext(support, run_device, optimizer, loss_fn)
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss = _train_epoch(model, train_loader, context)
        val_loss = _evaluate(model, val_loader, context.support, context.device)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch:02d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    result: dict[str, Any] = {
        "seed": config.seed,
        "device": str(run_device),
        "schema": {
            "x": [config.samples, schema.history, schema.links, schema.input_channels],
            "y": [config.samples, schema.horizon, schema.links, schema.target_channels],
        },
        "history": history,
        "best_val_loss": min(item["val_loss"] for item in history),
        "demo_only": True,
    }
    _write_outputs(config, result, model)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--links", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--output-dir", default="outputs/demo")
    return parser.parse_args()


if __name__ == "__main__":
    train_demo(DemoConfig(**vars(_parse_args())))
