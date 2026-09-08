# SWMM Pipe Surrogate

Repository: <https://github.com/CwhGIS/swmm-pipe-surrogate>

## Purpose

This repository provides a small, reproducible synthetic-data demonstration of
the Pipe surrogate-learning workflow. It is intended for code inspection,
installation checks, task-composition examples, and CPU smoke testing.

The demo generates its data at runtime. It does not include the restricted
SWMM time series, production checkpoints, or the numerical results reported in
the study.

## Inputs

The demo uses the following model-ready input tensor:

```text
X: (samples, history=6, links, input_channels=7)
```

The seven input channels are:

```text
Rainfall, Cumulative, Length, Diameter, Slope,
InletNode_InvertElev, OutletNode_InvertElev
```

The default synthetic run uses 24 samples, 12 links, and a six-step history.
The number of samples and links can be changed from the command line.

## Outputs

The model predicts six future steps for a selected subset of five targets:

```text
Y: (samples, horizon=6, links, selected_targets)
```

The canonical target order is:

```text
Rate   - flow rate
Depth  - flow depth
Vel    - flow velocity
Vol    - flow volume
Cap    - SWMM link capacity
```

By default, all five targets are used. A selected task subset changes the last
output dimension. Training uses task-wise summed MSE and keeps the
lowest-validation-loss model. The run writes `metrics.json` and an optional
`demo_model.pth` to the requested output directory.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Running the demo

Run the default five-target example:

```bash
python -m pipe_surrogate_demo.train --epochs 2 --output-dir outputs/demo
```

Run a selected task composition:

```bash
python -m pipe_surrogate_demo.train \
  --tasks Rate Depth Vel \
  --epochs 2 \
  --output-dir outputs/rate-depth-vel
```

List all 31 non-empty task compositions:

```bash
python -m pipe_surrogate_demo.list_task_sets
```

The output directory must be new or empty. The CLI refuses to write into a
non-empty directory. Use `--device cuda` only when the local PyTorch install
provides CUDA support.

Run the smoke tests:

```bash
python -m unittest discover -s tests -v
```
