# SWMM Pipe Surrogate — clean demo

Public repository: https://github.com/CwhGIS/swmm-pipe-surrogate-demo

This repository is a small, reproducible demonstration of the Pipe surrogate
workflow used in the accompanying study. It preserves the Pipe tensor contract
and a GraphWaveNet-style spatiotemporal model, but generates synthetic data at
runtime so the demo is safe to clone and runs on CPU in seconds.

The demo is **not** the paper's trained model and it does not reproduce the
reported SWMM metrics. It is a clean public starting point for code review,
installation checks, and smoke testing. The exact paper checkpoints and raw
SWMM time series are intentionally not included.

## What is included

- seven Pipe input channels and five Pipe targets;
- causal gated temporal convolution plus fixed graph propagation;
- deterministic synthetic data generation;
- CPU and CUDA device selection;
- JSON metrics and an optional demo checkpoint;
- a unit-level smoke test with reproducibility checks.

## Tensor contract

The paper data use the following layouts:

```text
X: (samples, history=6, links=396, input_channels=7)
Y: (samples, horizon=6, links=396, target_channels=5)
```

Input channels:

```text
Rainfall, Cumulative, Length, Diameter, Slope,
InletNode_InvertElev, OutletNode_InvertElev
```

Targets:

```text
Rate, Depth, Vel, Vol, Cap
```

The demo defaults to 12 links to keep the smoke run lightweight. The `--links`
option can be changed without changing the tensor semantics.

## Quick start

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pipe_surrogate_demo.train --epochs 2 --output-dir outputs/demo
```

The command writes `outputs/demo/metrics.json` and
`outputs/demo/demo_model.pth`. Generated outputs are ignored by Git.

Run the smoke test:

```bash
python -m unittest discover -s tests -v
```

Use CUDA only when the local PyTorch installation provides it:

```bash
python -m pipe_surrogate_demo.train --device cuda --epochs 2
```

## Layout

```text
swmm_pipe_surrogate_demo/
├── src/pipe_surrogate_demo/
│   ├── data.py       # synthetic data and graph fixture
│   ├── model.py      # compact GraphWaveNet-style model
│   ├── schema.py     # Pipe tensor contract
│   └── train.py      # CLI and deterministic training loop
├── tests/test_smoke.py
├── data/README.md
├── configs/pipe_schema.json
├── DESIGN.md
├── pyproject.toml
└── requirements.txt
```

## Transition to the paper data

The full study pipeline should be added as a separate, reviewed layer. It must
accept explicit paths for train/validation/test tensors, topology, scalers,
checkpoints, and output directories. Do not reintroduce workstation-specific
paths into this demo. A full-data release should also state the data licence,
event split, target scaling, seed, and checkpoint provenance.

## Release notes

This is a GitHub-only clean demo release. For a formal archival snapshot, add
an author-approved `LICENSE`, a complete `CITATION.cff`, and pinned dependency
versions. A DOI is optional and is not required for this demo. Keep raw
SWMM/geospatial inputs and large checkpoints in a separately documented
research-data record when their distribution terms permit it.
