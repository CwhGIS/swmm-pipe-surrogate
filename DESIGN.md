# Design record

## Scope

This module is a clean demonstration boundary for the Pipe surrogate code. It
is deliberately independent of the workstation-specific paper tree and has no
network, filesystem-data, or checkpoint-download side effects.

## Scientific contract

- Input tensor layout and channel order match the Pipe schema: seven channels,
  six historical time steps, and one value per link.
- Output tensor layout and target-channel order match the Pipe schema: five
  targets over six horizons.
- Synthetic rainfall is dynamic and shared across links; cumulative rainfall is
  its history-wise cumulative; geometric channels are fixed per link and
  broadcast over samples and history.
- The default graph has 12 synthetic links; the paper graph has 396 links.
- Synthetic values are generated deterministically and are not paper results.

## Architecture

```text
X (B,T,N,C)
  -> input projection
  -> causal gated temporal blocks
  -> fixed row-normalized graph propagation
  -> temporal pooling and output projection
  -> Y_hat (B,H,N,C_target)
```

The model is intentionally smaller than the archived production training code.
It demonstrates the data contract and execution path without bundling the
study's checkpoints or raw SWMM outputs.

The synthetic ring support is a graph fixture only; it does not reproduce the
directed conduit topology used by the study. The demo supports all 31 non-empty
target subsets and uses an unweighted sum of per-task MSE values. It selects the
checkpoint with the lowest validation loss, but it does not implement the
paper's preprocessing, inverse scaling, or paper-level evaluation export.

## Design decisions

| Decision | Rationale |
|---|---|
| Synthetic data at runtime | Keeps the public demo small and avoids redistributing raw inputs. |
| CPU default | Makes installation and CI usable without a GPU. |
| Explicit `--device` | Prevents accidental CUDA assumptions. |
| `num_workers=0` | Avoids platform-specific multiprocessing failures in a demo. |
| No `torch.load` of external files | Avoids untrusted checkpoint deserialization. |
| Output directory is explicit | Keeps generated artifacts out of source control. |
| Non-empty output directories are rejected | Prevents accidental checkpoint overwrite. |
| Task subset is explicit | Makes the 31 non-empty composition space inspectable. |
| Best validation checkpoint | Mirrors the paper's checkpoint-selection principle. |
| Fixed seed and deterministic cuDNN flags | Makes smoke results reproducible. |

## Security boundary

The demo accepts only typed scalar CLI arguments and writes only below the
requested output directory. It does not execute shell commands, download
files, evaluate user code, or load arbitrary checkpoints. Real-data adapters
must validate paths, shapes, finite values, and file provenance before use.

## Known limitations

- The synthetic generator is not a SWMM simulator.
- The compact model is not the exact archived 31-combination production model;
  only the task-set enumeration and loss principle are demonstrated.
- No uncertainty calibration, inverse scaling, or paper-level metric export is
  included here.
- Author-approved license, citation metadata, and pinned environment versions
  remain release tasks for a formal archival snapshot.

## Change history

### 2026-09-08 — Initial clean demo

Created an isolated Pipe schema, synthetic fixture, compact model, deterministic
training CLI, smoke test, and release documentation. No canonical paper files
were modified.

### 2026-09-08 — Method-correspondence layer

Added explicit target-subset enumeration, task-wise summed MSE, static/dynamic
synthetic channel semantics, validation-checkpoint selection, target mapping,
and paper/demo correspondence documentation. The public boundary remains
synthetic-only.
