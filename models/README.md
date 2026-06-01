# models/

This directory stores **trained model weights and serialised checkpoints**.

It is intentionally **not committed to git** (see `.gitignore`).
Use the W&B artifact links printed at the end of each training cell to
re-download weights when needed.

## Layout

```
models/
├── autoencoder/
│   └── ae_model.pt           ← trained autoencoder weights (torch)
├── hgt/
│   ├── hgt_model.pt          ← trained HGT weights (torch)
│   └── hgt_results.pkl       ← TrainResult (history + metrics + model ref)
└── knn/
    └── knn_cf_val_sweep_results_nbrs.npz   ← pre-computed neighbour matrix
```

## What goes here vs. `data/final/models/`

| Artefact | Location |
|---|---|
| PyTorch `.pt` weights | `models/<model>/` |
| Pickle checkpoints | `models/<model>/` |
| Neighbour index `.npz` | `models/knn/` |
| Metrics CSVs / JSON | `data/final/models/<model>/` |
| Plots / figures | `data/final/models/<model>/` |
| Qualitative analysis CSVs | `data/final/models/<model>/` |

## Reproducibility

All weights are logged as **W&B artifacts** by the training cells.
To restore a checkpoint without retraining:

```python
import wandb
api = wandb.Api()
artifact = api.artifact("music-recommender-system/hgt_model:latest")
artifact.download(root="models/hgt/")
```
