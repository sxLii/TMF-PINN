# TMF-PINN

Physics-informed neural networks for modeling **transient mixed flows in sewer networks**.

## Overview

TMF-PINN combines governing equations, initial and boundary conditions, and sparse observations to reconstruct transient hydraulic states, including water depth and velocity.

The repository contains three model variants:

| Variant | Description |
| --- | --- |
| `PINN_standard` | Standard PINN baseline |
| `PINN_lite` | Lightweight TMF-PINN variant |
| `PINN_full` | Full TMF-PINN model with additional flow-regime modeling |

## Repository structure

```text
TMF-PINN/
├── data/       # FVM and SWMM data
├── models/     # PINN model implementations
├── train/      # Training entry points and utilities
└── results/    # Saved predictions and evaluation notebook
```

## Associated article

This repository accompanies the following article:

<https://www.sciencedirect.com/science/article/pii/S2589914724000562>

If you use this repository or its results in your work, please cite the associated article.

## Data

The `.mat` data files are large, requiring approximately **4 GB x 12 in total**. Because of their size, complete data handling and FVM data-generation code will be added in a future update.

## Environment and setup

Environment configuration and usage instructions are available in the project issue:

<https://github.com/sxLii/TMF-PINN/issues/1>

## Status

This repository is under active development. Training scripts, model implementations, and result files are provided for research and reproducibility purposes.
