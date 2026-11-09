# Stage 3.5: ORL / CIFAR Experiments

## Experimental design

Each of the six conditions starts from a fresh model initialization with seed 42. Variants change only the hidden-layer width or the three local pooling layers relative to the corresponding baseline.

| Dataset | Condition | Hidden dimension | Local pooling |
|---|---|---:|---|
| ORL | `baseline` | 256 | MaxPool |
| ORL | `hidden_dim` | 128 | MaxPool |
| ORL | `avg_pool` | 256 | AvgPool |
| CIFAR-10 | `baseline` | 128 | MaxPool |
| CIFAR-10 | `hidden_dim` | 256 | MaxPool |
| CIFAR-10 | `avg_pool` | 128 | AvgPool |

All models use three Conv–BatchNorm–ReLU–Pool blocks with channels 32/64/128, followed by fully connected layers and dropout. ORL retains `AdaptiveAvgPool2d((7, 6))` in every condition.

| Training setting | ORL | CIFAR-10 |
|---|---|---|
| Epochs | 50 | 10 |
| Batch size | 32 | 64 |
| Weight decay | 0.0001 | 0 |
| Training augmentation | RandomAffine: rotation ±5°, translation up to 3% per axis, scale 0.95–1.05 | None |

Shared settings: Adam, learning rate 0.001, CrossEntropyLoss, dropout 0.3, and seed 42. The supplied train/test splits are unchanged. Train and test metrics are evaluated after each epoch in evaluation mode without augmentation. Saved weights and reported results use the **final epoch**, not the epoch with the highest test accuracy.

This is a single-seed comparison, not evidence of statistical superiority. Test curves are for monitoring, not checkpoint or hyperparameter selection; further tuning requires a separate validation split.

## Data and execution

Place the course-provided pickle files at these paths, preserving capitalization and using no file extension:

```text
data/stage_3_data/ORL
data/stage_3_data/CIFAR
```

Only load trusted files: unpickling can execute arbitrary code. Course data and generated model weights are not committed. Use `--data-dir PATH` for another data directory.

Run from the repository root using the uv environment described in the README:

```sh
uv run python -m local_code.stage_3_code.run_experiments --dry-run
uv run python -m local_code.stage_3_code.run_experiments --check-data
uv run python -m local_code.stage_3_code.run_experiments
```

- `--dry-run` prints all six configurations without loading data or training.
- `--check-data` validates files, splits, labels, and image shapes without training. Run it separately from `--dry-run`.
- With no arguments, the runner trains all six conditions using the fixed settings above.
- `--device auto` is the default. On this Apple Silicon Mac it selects CPU for ORL and MPS for CIFAR; CUDA is preferred when available. ORL's adaptive pooling is unsupported on MPS, so explicit ORL `--device mps` is rejected. Use `--device cpu` to run both datasets on CPU. Results may differ across devices.
- `--datasets orl` or `--datasets cifar` runs one dataset. `--epochs 1` overrides the epoch count for a smoke test, not the fixed comparison.

## Outputs and results

Each training invocation creates a unique `result/stage_3_experiments/<UTC timestamp>/` directory. Runs start from scratch and do not overwrite previous results. `--output-dir PATH` changes the parent directory.

| File | Contents |
|---|---|
| `suite.json` | Experiment plan, devices, completion status, and epoch-override flag |
| `summary.csv` / `summary.md` | Final metrics, differences from baseline, and learning-curve links |
| `<dataset>_<condition>/config.json` | Training settings, seed, device, parameter count, data hashes, source hashes, and Git commit |
| `<dataset>_<condition>/history.csv` | Per-epoch train/test losses and metrics |
| `<dataset>_<condition>/metrics.json` | Final-epoch evaluation metrics |
| `<dataset>_<condition>/model.pth` | Final-epoch model state dictionary; excluded from Git |
| `<dataset>_<condition>/confusion_matrix.csv` | Test confusion matrix: rows are true labels, columns are predictions |
| `<dataset>_<condition>/learning_curves.png` | Train/test loss and accuracy curves |

Accuracy and macro precision/recall/F1 in JSON and history files use a 0–1 scale. Summary accuracy uses percent; `delta_accuracy_pp` is the difference from the dataset's baseline in percentage points. Macro metrics average over all classes, with undefined values set to zero. Confusion-matrix labels are 0–39 for ORL (original labels 1–40) and 0–9 for CIFAR. In `history.csv`, `optimization_loss` is measured during training; `train_loss` is evaluated after the epoch without augmentation.

See the [recorded six-condition results and learning curves](../../result/stage_3_experiments/20261109T002416_269852Z/summary.md). Final test accuracy for `baseline` / `hidden_dim` / `avg_pool` was **97.50% / 100.00% / 97.50%** on ORL and **77.49% / 76.45% / 76.21%** on CIFAR-10.
