# Data boundary

The clean demo generates synthetic tensors at runtime; no SWMM input, raw
simulation output, geospatial map, scaler, or trained paper checkpoint is
stored in this directory.

For a separately approved full-data release, document the following before
adding files:

- train/validation/test event split and sample counts;
- `X` and `Y` shapes and channel order;
- topology matrix provenance and licence;
- target scaling and inverse-transform rules;
- checkpoint hashes and training seed;
- restrictions on raw SWMM and geospatial inputs.
