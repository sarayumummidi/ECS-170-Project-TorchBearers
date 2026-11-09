# ECS 170 Project

## Environment Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

**macOS / Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Clone this repo, then run:
```bash
uv sync --locked
```

## Running Code

```bash
uv run python -m path.to.file
```

## Stage 3.5: ORL / CIFAR Experiments

See the [experiment guide](local_code/stage_3_code/EXPERIMENTS.md) for the six fixed comparisons, training settings, and recorded results.

Place the trusted course-provided `ORL` and `CIFAR` pickle files in `data/stage_3_data/`, then run from the repository root:

```bash
uv run python -m local_code.stage_3_code.run_experiments --dry-run
uv run python -m local_code.stage_3_code.run_experiments --check-data
uv run python -m local_code.stage_3_code.run_experiments
```

`--dry-run` prints the plan without loading data or training; `--check-data` validates the data without training. Training creates a unique directory under `result/stage_3_experiments/` for configurations, metrics, curves, and comparison tables. Course data and generated model weights are not committed.