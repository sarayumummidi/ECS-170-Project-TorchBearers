"""Runner tests use tiny synthetic pickles, never the course data."""

import csv
import io
import json
import pickle
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import run_experiments as runner


def synthetic_data(directory):
    rng = np.random.default_rng(170)
    for name, shape in (("orl", (112, 92, 3)), ("cifar", (32, 32, 3))):
        data = {
            split: [
                {
                    "image": rng.integers(0, 256, shape, dtype=np.uint8),
                    "label": index % 2 + (1 if name == "orl" else 0),
                }
                for index in range(size)
            ]
            for split, size in (("train", 4), ("test", 2))
        }
        with (directory / name.upper()).open("wb") as stream:
            pickle.dump(data, stream)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous_threads = torch.get_num_threads()
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        cls.addClassCleanup(torch.use_deterministic_algorithms, previous_deterministic)
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()
        self.output_dir = self.root / "results"

    def run_main(self, *arguments):
        with redirect_stdout(io.StringIO()):
            runner.main(
                [
                    "--data-dir",
                    str(self.data_dir),
                    "--output-dir",
                    str(self.output_dir),
                    *arguments,
                ]
            )

    def test_only_one_factor_changes(self):
        for name in runner.DEFAULTS:
            baseline = runner.make_config(name, "baseline")
            for experiment, changed_key in (
                ("hidden_dim", "hidden_dim"),
                ("avg_pool", "pooling"),
            ):
                variant = runner.make_config(name, experiment)
                changed = {key for key in baseline if baseline[key] != variant[key]}
                self.assertEqual(changed, {"experiment", changed_key})
            self.assertEqual(baseline["seed"], 42)
            self.assertFalse(baseline["epoch_override"])

    def test_cli_and_missing_data(self):
        self.run_main("--dry-run")
        self.assertFalse(self.output_dir.exists())
        with self.assertRaisesRegex(
            FileNotFoundError, "Missing course-provided pickle"
        ):
            self.run_main("--check-data")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            for args in (
                ("--epochs", "0"),
                ("--seed", "-1"),
                ("--datasets", "orl", "orl"),
                ("--experiments", "baseline", "baseline"),
            ):
                with self.subTest(args=args), self.assertRaises(SystemExit):
                    runner.parse_args(args)

    def test_preflight_metadata_and_invalid_labels(self):
        synthetic_data(self.data_dir)
        loaded = runner.load_datasets(self.data_dir, ["orl", "cifar"])
        for name, (splits, metadata) in loaded.items():
            self.assertEqual(len(metadata["sha256"]), 64)
            self.assertEqual(metadata["filename"], name.upper())
            self.assertNotIn("path", metadata)
            self.assertEqual(metadata["train"]["samples"], 4)
            self.assertEqual(metadata["test"]["samples"], 2)
            self.assertEqual(
                len(metadata["train"]["label_counts"]),
                runner.DEFAULTS[name]["num_classes"],
            )
            self.assertEqual(splits["train"][0][1].item(), 0)
        with (self.data_dir / "ORL").open("rb") as stream:
            bad = pickle.load(stream)
        bad["test"][0]["label"] = 0
        with (self.data_dir / "ORL").open("wb") as stream:
            pickle.dump(bad, stream)
        with self.assertRaisesRegex(ValueError, "invalid label"):
            runner.load_datasets(self.data_dir, ["orl"])

    def test_device_selection(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.backends.mps, "is_available", return_value=True),
        ):
            self.assertEqual(str(runner.choose_device("auto", "orl")), "cpu")
            self.assertEqual(str(runner.choose_device("auto", "cifar")), "mps")
            with self.assertRaisesRegex(ValueError, "ORL's existing adaptive pooling"):
                runner.choose_device("mps", "orl")
        with patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(str(runner.choose_device("auto", "orl")), "cuda")

    def test_full_cpu_suite_and_reproducibility(self):
        synthetic_data(self.data_dir)
        self.run_main("--device", "cpu", "--epochs", "1")
        first_suite = next(self.output_dir.iterdir())
        suite = json.loads((first_suite / "suite.json").read_text())
        self.assertEqual(suite["status"], "completed")
        self.assertEqual(suite["completed_runs"], 6)
        self.assertTrue(suite["epoch_override"])
        with (first_suite / "summary.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 6)
        for row in rows:
            run_dir = first_suite / row["run_dir"]
            for filename in (
                "config.json",
                "metrics.json",
                "history.csv",
                "model.pth",
                "confusion_matrix.csv",
                "learning_curves.png",
            ):
                self.assertTrue((run_dir / filename).is_file(), filename)
            config = json.loads((run_dir / "config.json").read_text())
            self.assertEqual(config["epochs"], 1)
            self.assertEqual(config["device"], "cpu")
            self.assertTrue(config["epoch_override"])
            if row["experiment"] == "baseline":
                self.assertEqual(float(row["delta_accuracy_pp"]), 0.0)
            matrix = np.loadtxt(run_dir / "confusion_matrix.csv", delimiter=",")
            self.assertEqual(
                matrix.shape, (config["num_classes"], config["num_classes"])
            )
            self.assertEqual(matrix.sum(), 2)
        self.assertIn(
            "orl_baseline/learning_curves.png", (first_suite / "summary.md").read_text()
        )
        self.assertFalse(any(first_suite.glob("*.py")))
        self.assertNotIn("status", suite["provenance"])
        self.run_main(
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--datasets",
            "orl",
            "--experiments",
            "baseline",
        )
        second_suite = next(
            path for path in self.output_dir.iterdir() if path != first_suite
        )
        for filename in ("metrics.json", "model.pth"):
            if filename == "metrics.json":
                original = json.loads(
                    (first_suite / "orl_baseline" / filename).read_text()
                )
                repeated = json.loads(
                    (second_suite / "orl_baseline" / filename).read_text()
                )
                self.assertEqual(original["test"], repeated["test"])
                self.assertEqual(original["train"], repeated["train"])
            else:
                original = torch.load(
                    first_suite / "orl_baseline" / filename, weights_only=True
                )
                repeated = torch.load(
                    second_suite / "orl_baseline" / filename, weights_only=True
                )
                self.assertTrue(
                    all(torch.equal(original[key], repeated[key]) for key in original)
                )

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_cifar_mps_suite(self):
        synthetic_data(self.data_dir)
        self.run_main("--datasets", "cifar", "--device", "mps", "--epochs", "1")
        suite_dir = next(self.output_dir.iterdir())
        suite = json.loads((suite_dir / "suite.json").read_text())
        self.assertEqual(suite["status"], "completed")
        self.assertEqual(suite["completed_runs"], 3)
        self.assertEqual(suite["devices"], {"cifar": "mps"})

    def test_failure_is_recorded(self):
        synthetic_data(self.data_dir)
        with patch.object(
            runner, "train_run", side_effect=ValueError("synthetic failure")
        ):
            with self.assertRaisesRegex(ValueError, "synthetic failure"):
                self.run_main("--device", "cpu", "--epochs", "1")
        suite_dir = next(self.output_dir.iterdir())
        suite = json.loads((suite_dir / "suite.json").read_text())
        self.assertEqual(suite["status"], "failed")
        self.assertEqual(suite["completed_runs"], 0)
        self.assertIn("synthetic failure", suite["error"])


if __name__ == "__main__":
    unittest.main()
