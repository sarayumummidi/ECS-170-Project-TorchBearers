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