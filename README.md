# SWMM Pipe Surrogate — clean demo

Public repository: https://github.com/CwhGIS/swmm-pipe-surrogate-demo

This repository is a small, reproducible synthetic-data demonstration of the
Pipe tensor contract and a compact GraphWaveNet-style execution path associated
with the accompanying study. It is not the production implementation used to
generate the reported results. Synthetic data are generated at runtime so the
demo is safe to clone and runs on CPU in seconds.

The demo is **not** the paper's production model and it does not reproduce the
reported SWMM metrics. It is a clean public starting point for code inspection,
installation checks, task-composition examples, and smoke testing. The exact
paper checkpoints and raw SWMM time series are intentionally not included.

## What is included

- seven Pipe input channels and five Pipe targets;
- causal gated temporal convolution plus fixed graph propagation;
- deterministic synthetic data generation;
- CPU and CUDA device selection;
- JSON metrics and an optional demo checkpoint;
- 31 non-empty target subsets with task-wise summed MSE;
- a unit-level smoke test with reproducibility checks.

## Paper/demo correspondence

| Component | Paper production path | Clean demo |
|---|---|---|
| Input projection | 1x1 convolution | 1x1 convolution |
| Graph-temporal blocks | 6 blocks | 2 compact blocks |
| Residual channels | 64 | 16 |
| Dilations | `[1, 2, 4, 8, 1, 2]` | `(1, 2)` |
| Fixed supports | directed support and reverse support | synthetic ring support |
| Diffusion order | 2 | single propagation |
| Adaptive adjacency | rank-16 learned support | not included |
| Skip channels | 128 | no separate skip path |
| Dropout | 0.1 | not included |
| Task compositions | all 31 non-empty subsets | all 31 subsets for synthetic runs |

The correspondence is intentionally limited to the tensor contract, task-set
enumeration, and a compact execution path. The demo does not claim architectural
or numerical reproduction of the paper model.

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

| Demo key | Manuscript variable |
|---|---|
| `Rate` | flow rate |
| `Depth` | flow depth |
| `Vel` | flow velocity |
| `Vol` | flow volume |
| `Cap` | SWMM link capacity (dimensionless; not asserted to be `Q/Q_full`) |

The synthetic tensors are generated directly in model-ready space. The paper's
log, z-score, min-max, identity-capacity preprocessing, and inverse-scaling
procedures are not implemented here.

The demo defaults to 12 links to keep the smoke run lightweight. The `--links`
option can be changed without changing the tensor semantics.

The synthetic generator keeps rainfall dynamic and shared across links,
computes cumulative rainfall along history, and keeps the five geometric
channels fixed per link and broadcast over samples and history. The graph is a
synthetic ring fixture; it does not reproduce the directed conduit topology of
the study.

## Task-composition demonstration

The paper evaluates all non-empty subsets of five target tasks:

```text
5 STL + 25 selective-MTL + 1 Full-MTL = 31 subsets
```

List them with:

```bash
python -m pipe_surrogate_demo.list_task_sets
```

Run a selected synthetic subset with task-wise summed MSE:

```bash
python -m pipe_surrogate_demo.train --tasks Rate Depth Vel --epochs 2
```

The demo checkpoint is selected by lowest validation loss, following the
selection principle used by the paper. The demo remains synthetic and does not
produce paper-level metrics.

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
`outputs/demo/demo_model.pth`. Generated outputs are ignored by Git. For
checkpoint safety, the CLI refuses to write into a non-empty output directory;
use a new demo output directory for each run.

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
│   ├── loss.py       # task-wise summed MSE
│   ├── model.py      # compact GraphWaveNet-style model
│   ├── list_task_sets.py # all 31 non-empty target subsets
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
