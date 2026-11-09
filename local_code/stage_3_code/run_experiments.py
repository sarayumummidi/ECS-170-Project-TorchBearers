"""Fixed Stage 3.5 comparisons; run from the repository root with python -m."""

import argparse
import csv
import hashlib
import json
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

if __package__:
    from .dataset_loader import ImageDataset
    from .Method_CNN_CIFAR import Method_CNN_CIFAR
    from .Method_CNN_ORL import Method_CNN_ORL
else:
    from dataset_loader import ImageDataset
    from Method_CNN_CIFAR import Method_CNN_CIFAR
    from Method_CNN_ORL import Method_CNN_ORL


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ("baseline", "hidden_dim", "avg_pool")
DEFAULTS = {
    "orl": dict(
        epochs=50,
        batch_size=32,
        hidden_dim=256,
        weight_decay=0.0001,
        num_classes=40,
        input_shape=(1, 112, 92),
    ),
    "cifar": dict(
        epochs=10,
        batch_size=64,
        hidden_dim=128,
        weight_decay=0.0,
        num_classes=10,
        input_shape=(3, 32, 32),
    ),
}
HISTORY_FIELDS = (
    "epoch",
    "optimization_loss",
    "train_loss",
    "train_accuracy",
    "test_loss",
    "test_accuracy",
    "elapsed_seconds",
)
SUMMARY_FIELDS = (
    "dataset",
    "experiment",
    "seed",
    "epochs",
    "batch_size",
    "learning_rate",
    "hidden_dim",
    "pooling",
    "parameter_count",
    "train_accuracy_pct",
    "test_accuracy_pct",
    "delta_accuracy_pp",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "test_loss",
    "duration_seconds",
    "run_dir",
)


class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.RandomAffine(
            degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.transform(image), label


def make_config(dataset, experiment, seed=42, epochs=None):
    config = dict(DEFAULTS[dataset])
    config.update(
        dataset=dataset,
        experiment=experiment,
        seed=seed,
        learning_rate=0.001,
        optimizer="Adam",
        loss="CrossEntropyLoss",
        pooling="max",
        dropout=0.3,
        conv_channels=[32, 64, 128],
        kernel_size=3,
        padding=1,
        conv_stride=1,
        pool_size=2,
        batch_normalization=True,
        augmentation=(
            dict(degrees=5, translate=[0.03, 0.03], scale=[0.95, 1.05])
            if dataset == "orl"
            else None
        ),
        preprocessing="float32 / 255; HWC to CHW; ORL: first channel, labels minus 1",
        checkpoint="final epoch (no best-test selection)",
        metric_scale="accuracy, precision, recall, F1 in [0, 1]; summary accuracy in percent",
        epoch_override=epochs is not None,
    )
    if experiment == "hidden_dim":
        config["hidden_dim"] = 128 if dataset == "orl" else 256
    elif experiment == "avg_pool":
        config["pooling"] = "avg"
    elif experiment != "baseline":
        raise ValueError(f"Unknown experiment: {experiment}")
    if epochs is not None:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        config["epochs"] = epochs
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(True)


def choose_device(requested, dataset):
    if requested == "auto":
        # The existing ORL adaptive pool (14x11 -> 7x6) is unsupported on MPS.
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if dataset != "orl" and torch.backends.mps.is_available()
            else "cpu"
        )
    if requested == "mps" and dataset == "orl":
        raise ValueError(
            "ORL's existing adaptive pooling is unsupported on MPS. "
            "Use --device auto (ORL: CPU, CIFAR: MPS on Apple Silicon) or --device cpu."
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; choose --device cpu or mps")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; choose --device cpu")
    return torch.device(requested)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_datasets(data_dir, names):
    missing = [
        data_dir / name.upper()
        for name in names
        if not (data_dir / name.upper()).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing course-provided pickle files (not image folders):\n"
            + "\n".join(str(path) for path in missing)
            + "\nPlace the instructor's ORL/CIFAR files there, or use --data-dir."
        )
    loaded = {}
    for name in names:
        path = data_dir / name.upper()
        # pickle can execute code: only load the instructor's trusted data files.
        splits = {
            split: ImageDataset(str(path), split, name) for split in ("train", "test")
        }
        metadata = {"filename": path.name, "sha256": file_sha256(path)}
        for split, dataset in splits.items():
            if len(dataset) == 0:
                raise ValueError(f"{name}/{split}: empty split")
            counts = [0] * DEFAULTS[name]["num_classes"]
            shapes = set()
            for sample in dataset.samples:
                label = int(sample["label"]) - (1 if name == "orl" else 0)
                if not 0 <= label < len(counts):
                    raise ValueError(f"{name}/{split}: invalid label {sample['label']}")
                counts[label] += 1
                shapes.add(tuple(np.asarray(sample["image"]).shape))
            if len(shapes) != 1:
                raise ValueError(f"{name}/{split}: inconsistent image shapes {shapes}")
            image, _ = dataset[0]
            expected = DEFAULTS[name]["input_shape"]
            # ORL uses adaptive pooling, so its spatial size is not fixed.
            valid_shape = (
                (image.ndim == 3 and image.shape[0] == 1 and min(image.shape[1:]) >= 8)
                if name == "orl"
                else tuple(image.shape) == expected
            )
            if not valid_shape:
                raise ValueError(
                    f"{name}/{split}: unexpected input shape {tuple(image.shape)}"
                )
            metadata[split] = dict(
                samples=len(dataset), input_shape=list(image.shape), label_counts=counts
            )
        loaded[name] = (splits, metadata)
    return loaded


def source_info():
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "--no-optional-locks", "--no-pager", *args],
                cwd=PROJECT_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return "unavailable"

    paths = [
        Path(__file__),
        Path(__file__).with_name("dataset_loader.py"),
        Path(__file__).with_name("cnn_models.py"),
        Path(__file__).with_name("Method_CNN_ORL.py"),
        Path(__file__).with_name("Method_CNN_CIFAR.py"),
    ]
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "source_modified": bool(
            git("status", "--porcelain", "--", "local_code/stage_3_code")
        ),
        "source_sha256": {
            str(p.relative_to(PROJECT_ROOT)): file_sha256(p) for p in paths
        },
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }


def write_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    labels, predictions = [], []
    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images.to(device))
            loss_sum += criterion(outputs, targets.to(device)).item() * targets.size(0)
            labels.extend(targets.tolist())
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())
    return (
        {
            "loss": loss_sum / len(labels),
            "accuracy": float(accuracy_score(labels, predictions)),
        },
        labels,
        predictions,
    )


def final_metrics(scores, labels, predictions, num_classes):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )
    return dict(
        scores,
        macro_precision=float(precision),
        macro_recall=float(recall),
        macro_f1=float(f1),
    )


def save_curves(path, history, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = [row["epoch"] for row in history]
    for split in ("train", "test"):
        axes[0].plot(epochs, [row[f"{split}_loss"] for row in history], label=split)
        axes[1].plot(
            epochs, [100 * row[f"{split}_accuracy"] for row in history], label=split
        )
    for axis, ylabel in zip(axes, ("Cross-entropy loss", "Accuracy (%)")):
        axis.set(xlabel="Epoch", ylabel=ylabel)
        axis.grid(True)
        axis.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_run(config, splits, metadata, device, run_dir, provenance):
    run_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(config["seed"])
    model_class = Method_CNN_ORL if config["dataset"] == "orl" else Method_CNN_CIFAR
    model = model_class(hidden_dim=config["hidden_dim"], pooling=config["pooling"]).to(
        device
    )
    config = dict(
        config,
        device=str(device),
        deterministic_algorithms=True,
        parameter_count=sum(p.numel() for p in model.parameters()),
        data=metadata,
        provenance=provenance,
    )
    write_json(run_dir / "config.json", config)
    # Independent shuffle RNG keeps batch order fixed even when model width changes.
    train_data = (
        AugmentedDataset(splits["train"]) if config["augmentation"] else splits["train"]
    )
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    eval_loaders = {
        split: DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            generator=torch.Generator().manual_seed(config["seed"]),
            num_workers=0,
        )
        for split, dataset in splits.items()
    }
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    history = []
    started = time.perf_counter()
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            loss_sum = 0.0
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), targets)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * targets.size(0)
            train_scores, _, _ = evaluate(
                model, eval_loaders["train"], criterion, device
            )
            test_scores, labels, predictions = evaluate(
                model, eval_loaders["test"], criterion, device
            )
            row = dict(
                epoch=epoch,
                optimization_loss=loss_sum / len(train_data),
                train_loss=train_scores["loss"],
                train_accuracy=train_scores["accuracy"],
                test_loss=test_scores["loss"],
                test_accuracy=test_scores["accuracy"],
                elapsed_seconds=time.perf_counter() - started,
            )
            history.append(row)
            writer.writerow(row)
            stream.flush()
            print(
                f"{config['dataset']}/{config['experiment']} [{epoch}/{config['epochs']}] "
                f"train loss={row['train_loss']:.4f} acc={100 * row['train_accuracy']:.2f}% | "
                f"test loss={row['test_loss']:.4f} acc={100 * row['test_accuracy']:.2f}%",
                flush=True,
            )
    duration = time.perf_counter() - started
    metrics = dict(
        train=train_scores,
        test=final_metrics(test_scores, labels, predictions, config["num_classes"]),
        duration_seconds=duration,
        epoch=config["epochs"],
    )
    write_json(run_dir / "metrics.json", metrics)

    torch.save(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        run_dir / "model.pth",
    )
    matrix = confusion_matrix(
        labels, predictions, labels=list(range(config["num_classes"]))
    )
    np.savetxt(run_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    save_curves(
        run_dir / "learning_curves.png",
        history,
        f"{config['dataset'].upper()} / {config['experiment']} / seed {config['seed']}",
    )
    return dict(
        dataset=config["dataset"],
        experiment=config["experiment"],
        seed=config["seed"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        hidden_dim=config["hidden_dim"],
        pooling=config["pooling"],
        parameter_count=config["parameter_count"],
        train_accuracy_pct=100 * metrics["train"]["accuracy"],
        test_accuracy_pct=100 * metrics["test"]["accuracy"],
        delta_accuracy_pp=None,
        test_macro_precision=metrics["test"]["macro_precision"],
        test_macro_recall=metrics["test"]["macro_recall"],
        test_macro_f1=metrics["test"]["macro_f1"],
        test_loss=metrics["test"]["loss"],
        duration_seconds=duration,
        run_dir=run_dir.name,
    )


def write_summary(suite_dir, rows):
    baselines = {
        row["dataset"]: row["test_accuracy_pct"]
        for row in rows
        if row["experiment"] == "baseline"
    }
    for row in rows:
        baseline = baselines.get(row["dataset"])
        row["delta_accuracy_pp"] = (
            None if baseline is None else row["test_accuracy_pct"] - baseline
        )
    with (suite_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Stage 3.5 results",
        "",
        "Final epoch, not the best test epoch. Accuracy is percent; delta is percentage points.",
        "",
        "| Dataset | Experiment | Hidden | Pool | Train acc (%) | Test acc (%) | Delta (pp) | Macro F1 | Seconds |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = (
            "—"
            if row["delta_accuracy_pp"] is None
            else f"{row['delta_accuracy_pp']:+.2f}"
        )
        lines.append(
            f"| {row['dataset'].upper()} | {row['experiment']} | {row['hidden_dim']} | {row['pooling']} | "
            f"{row['train_accuracy_pct']:.2f} | {row['test_accuracy_pct']:.2f} | {delta} | "
            f"{row['test_macro_f1']:.4f} | {row['duration_seconds']:.1f} |"
        )
    lines += ["", "## Learning curves", ""]
    for row in rows:
        lines += [
            f"### {row['dataset'].upper()} / {row['experiment']}",
            "",
            f"![Learning curves]({row['run_dir']}/learning_curves.png)",
            "",
        ]

    (suite_dir / "summary.md").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DEFAULTS), default=list(DEFAULTS)
    )
    parser.add_argument(
        "--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS)
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "data" / "stage_3_data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "result" / "stage_3_experiments",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override BOTH datasets' epochs, e.g. 1 for a smoke test",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without loading data or training",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Validate the course files without training",
    )
    args = parser.parse_args(argv)
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    if len(set(args.datasets)) != len(args.datasets) or len(
        set(args.experiments)
    ) != len(args.experiments):
        parser.error("Duplicate datasets/experiments are not allowed")
    return args


def main(argv=None):
    args = parse_args(argv)
    plan = [
        make_config(name, experiment, args.seed, args.epochs)
        for name in args.datasets
        for experiment in args.experiments
    ]
    print("Fixed comparisons (each run starts from scratch):", flush=True)
    for config in plan:
        print(
            f"  {config['dataset']}/{config['experiment']}: epochs={config['epochs']} "
            f"batch={config['batch_size']} lr={config['learning_rate']} "
            f"hidden={config['hidden_dim']} pooling={config['pooling']} seed={config['seed']}",
            flush=True,
        )
    if args.dry_run:
        return
    loaded = load_datasets(args.data_dir, args.datasets)
    if args.check_data:
        print(
            json.dumps(
                {name: metadata for name, (_, metadata) in loaded.items()}, indent=2
            )
        )
        return
    devices = {name: choose_device(args.device, name) for name in args.datasets}
    provenance = source_info()
    suite_dir = args.output_dir / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    suite_dir.mkdir(parents=True, exist_ok=False)
    suite = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        status="running",
        devices={name: str(device) for name, device in devices.items()},
        plan=plan,
        provenance=provenance,
        epoch_override=args.epochs is not None,
    )
    write_json(suite_dir / "suite.json", suite)

    print(f"Devices: {suite['devices']}\nResults: {suite_dir}", flush=True)
    rows = []
    try:
        for config in plan:
            splits, metadata = loaded[config["dataset"]]
            row = train_run(
                config,
                splits,
                metadata,
                devices[config["dataset"]],
                suite_dir / f"{config['dataset']}_{config['experiment']}",
                provenance,
            )
            rows.append(row)
            write_summary(suite_dir, rows)
    except BaseException as error:
        suite.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
            completed_runs=len(rows),
        )
        write_json(suite_dir / "suite.json", suite)
        raise
    suite.update(status="completed", completed_runs=len(rows))
    write_json(suite_dir / "suite.json", suite)
    print(f"Done. Report table and curves: {suite_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
