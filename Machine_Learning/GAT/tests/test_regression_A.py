"""Synthetic HDF5 and end-to-end checks for single-task A regression."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tpc_GAT"))
sys.path.insert(0, str(ROOT / "results"))
from task_spec import get_task_spec, get_target_transform, read_A_targets
from tpc_graph_dataset import TPCGraphDataset
from utils import validate_epoch
from plot_A_result import A_to_Emax, binned_stats, primary_radius, residual_summary


def write_fixture(path):
    """Write 25 events with two electrons and three connected hits each."""
    n = 25
    with h5py.File(path, "w") as f:
        fields = dict(hit_start=np.arange(n) * 3, hit_count=np.full(n, 3),
                      primary_start=np.arange(n) * 2, primary_count=np.full(n, 2),
                      n_primaries=np.full(n, 2), n_hits=np.full(n, 3),
                      primary_origin_z=np.linspace(-500, 500, n),
                      total_energy_filtered=np.linspace(1000, 2400, n),
                      id=np.tile([1, 21, 22, 23, 24], 5))
        for name, values in fields.items():
            f[f"events/{name}"] = values
        E1 = np.linspace(0, 100, n)
        f["primaries/primary_energy"] = np.column_stack([E1, 100 - E1]).ravel()
        f["primaries/primary_particle_name"] = np.full(n * 2, b"e-")
        for name, values in dict(x=[0., 1., 2.], y=[0., 0., 1.], energy=[1., 2., 3.],
                                 log_energy=[0., 0.69, 1.10], module_id=[0, 0, 0]).items():
            f[f"hits/{name}"] = np.tile(values, n)
        f["stats/feature_names"] = np.asarray([b"x", b"y", b"energy", b"log_energy"])
        for name, values in dict(mean=[0., 0., 2., 0.6], std=[1., 1., 1., 1.],
                                 min=[0., 0., 1., 0.], max=[2., 1., 3., 1.1]).items():
            f[f"stats/{name}"] = values


class RegressionATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "events.h5"
        write_fixture(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_labels_transform_and_invalid_primaries(self):
        with h5py.File(self.path, "r") as f:
            A, E1, E2 = read_A_targets(f)
        np.testing.assert_allclose(A, np.abs(E1 - E2) / (E1 + E2), atol=1e-7)
        self.assertEqual(float(A[12]), 0.)
        self.assertEqual(float(A[0]), 1.)
        transform = get_target_transform("regression_A")
        np.testing.assert_allclose(transform.inverse(transform.forward(A)), A, atol=1e-7)
        self.assertEqual(get_task_spec("regression_A").num_outputs, 1)
        with self.assertRaises(ValueError):
            get_target_transform("regression_A", "minus_one_one")
        for path, value in (("events/primary_count", 1), ("events/primary_start", -1),
                            ("events/primary_start", 10000),
                            ("primaries/primary_particle_name", b"g"),
                            ("primaries/primary_energy", -1.), ("primaries/primary_energy", np.nan)):
            with self.subTest(path=path, value=value):
                write_fixture(self.path)
                with h5py.File(self.path, "r+") as f:
                    f[path][0] = value
                    with self.assertRaises(ValueError):
                        read_A_targets(f)
        write_fixture(self.path)
        with h5py.File(self.path, "r+") as f:
            f["primaries/primary_energy"][:2] = 0
            with self.assertRaises(ValueError):
                read_A_targets(f)

    def test_Z_transform_compatibility(self):
        values = np.array([-845., 0., 845.], dtype=np.float32)
        for mode, expected in (("minus_one_one", [-1., 0., 1.]),
                               ("zero_one", [0., 0.5, 1.]), ("none", values)):
            transform = get_target_transform("regression_z", mode)
            np.testing.assert_allclose(transform.forward(values), expected)
            np.testing.assert_allclose(transform.inverse(transform.forward(values)), values)

    def test_coordinate_features_without_statistics(self):
        with h5py.File(self.path, "r+") as f:
            f["hits/z"] = np.tile([-850., 0., 850.], 25)
            for axis in ("x", "y"):
                f[f"events/primary_origin_{axis}"] = np.zeros(25)
            del f["stats/feature_names"]
            f["stats/feature_names"] = np.asarray([b"x", b"y", b"z", b"energy", b"log_energy"])
            for key, value in (("mean", 100.), ("std", 20.), ("min", -850.), ("max", 850.)):
                values = np.insert(f[f"stats/{key}"][:], 2, value)
                del f[f"stats/{key}"]
                f[f"stats/{key}"] = values

        expected = {}
        for mode in ("standard", "minmax", "standard_minmax", "none"):
            for vertex_relative in (False, True):
                dataset = TPCGraphDataset(str(self.path), task="regression_A",
                                          node_feature_names=("x", "y", "z", "energy", "log_energy"),
                                          feature_norm_mode=mode, vertex_relative=vertex_relative)
                try:
                    expected[mode, vertex_relative] = dataset[0].x.numpy().copy()
                    np.testing.assert_allclose(expected[mode, vertex_relative][:, 2], [-1., 0., 1.])
                finally:
                    dataset.close()

        for removed in (("z",), ("x", "y")):
            with h5py.File(self.path, "r+") as f:
                names = f["stats/feature_names"][:]
                keep = np.array([name.decode() not in removed for name in names])
                for key in ("feature_names", "mean", "std", "min", "max"):
                    values = f[f"stats/{key}"][:][keep]
                    del f[f"stats/{key}"]
                    f[f"stats/{key}"] = values
            for (mode, vertex_relative), values in expected.items():
                with self.subTest(removed=removed, mode=mode, vertex_relative=vertex_relative):
                    dataset = TPCGraphDataset(str(self.path), task="regression_A",
                                              node_feature_names=("x", "y", "z", "energy", "log_energy"),
                                              feature_norm_mode=mode, vertex_relative=vertex_relative)
                    try:
                        np.testing.assert_array_equal(dataset[0].x.numpy(), values)
                    finally:
                        dataset.close()

    def test_missing_hits_and_noncoordinate_statistics_still_fail(self):
        with self.assertRaisesRegex(KeyError, "hits/z"):
            TPCGraphDataset(str(self.path), task="regression_A", node_feature_names=("x", "y", "z"))
        with h5py.File(self.path, "r+") as f:
            names = f["stats/feature_names"][:]
            keep = names != b"energy"
            for key in ("feature_names", "mean", "std", "min", "max"):
                values = f[f"stats/{key}"][:][keep]
                del f[f"stats/{key}"]
                f[f"stats/{key}"] = values
        with self.assertRaisesRegex(KeyError, "energy.*stats/feature_names"):
            TPCGraphDataset(str(self.path), task="regression_A")

    def test_filtered_alignment_and_validation_units(self):
        with h5py.File(self.path, "r+") as f:
            f["events/hit_count"][1] = 0
        dataset = TPCGraphDataset(str(self.path), task="regression_A")
        try:
            self.assertEqual(len(dataset), 24)
            self.assertEqual(dataset[1].event_index.item(), 2)
            self.assertAlmostEqual(dataset[1].E1.item(), 100 / 12, places=5)
            batch = next(iter(DataLoader(dataset, batch_size=3)))
            self.assertEqual(batch.E1.numel(), 3)
            np.testing.assert_array_equal(batch.event_index.numpy(), [0, 2, 3])

            class OffsetModel(torch.nn.Module):
                def forward(self, data):
                    return data.y.view(-1) + 1

            result = validate_epoch(OffsetModel(), torch.device("cpu"),
                                    DataLoader(dataset, batch_size=5), torch.nn.SmoothL1Loss(),
                                    task="regression_A")
            self.assertAlmostEqual(result["mae"], 0.2663, places=6)
            self.assertAlmostEqual(result["loss"], 0.5, places=6)
            self.assertNotIn("mae_mm", result)
            self.assertGreater(result["out_of_range_fraction"], 0)

            class VariableOffsetModel(torch.nn.Module):
                def forward(self, data):
                    return data.y.view(-1) + data.event_index.float() / 10

            duplicated = Subset(dataset, [0, 0, 1, 2, 3, 4])
            result = validate_epoch(VariableOffsetModel(), torch.device("cpu"),
                                    DataLoader(duplicated, batch_size=4), torch.nn.SmoothL1Loss(),
                                    task="regression_A")
            errors = np.array([0, 2, 3, 4, 5]) / 10
            self.assertEqual(len(result["preds"]), 5)
            self.assertAlmostEqual(result["loss"], float(np.mean(errors ** 2 / 2)), places=6)
        finally:
            dataset.close()
        with self.assertRaises(ValueError):
            TPCGraphDataset(str(self.path), task="regression_A", event_feature_keys=["E1"])
        with self.assertRaises(ValueError):
            TPCGraphDataset(str(self.path), task="regression_A", event_feature_keys=["primary_origin_z"])

    def test_plot_statistics_keep_boundaries_and_unclipped_predictions(self):
        metrics = residual_summary([0., 1.], [-0.2, 1.2])
        self.assertAlmostEqual(metrics["mae"], 0.2)
        self.assertEqual(metrics["out_of_range_fraction"], 1.)
        rows = binned_stats(np.array([0., 1.]), np.array([-0.2, 0.2]), np.linspace(0, 1, 3))
        np.testing.assert_array_equal(rows[:, 0], [1, 1])

    def test_Emax_conversion_and_primary_radius(self):
        np.testing.assert_allclose(A_to_Emax(np.array([0., 0.5, 1.]), 2457.),
                                   [1228.5, 1842.75, 2457.])
        with self.assertRaises(ValueError):
            A_to_Emax([0.5], 0.)
        np.testing.assert_allclose(primary_radius([[3., 4., 10.], [5., 12., -10.]]), [5., 13.])
        with self.assertRaises(ValueError):
            primary_radius([3., 4., 10.])

    def test_training_outputs_and_legacy_tasks(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        for task in ("regression_A", "regression_z", "binary_id", "multiclass_id"):
            with self.subTest(task=task):
                out = Path(self.temp.name) / task
                command = [sys.executable, str(ROOT / "tpc_GAT/run.py"), "--infile", str(self.path),
                           "--outdir", str(out), "--task", task, "--epochs", "2", "--batch_size", "4",
                           "--num_workers", "0", "--hidden_channels", "8", "--num_layers", "1",
                           "--heads", "1", "--head_hidden_channels", "8", "--log_interval", "1"]
                result = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                with np.load(out / "training_history.npz") as history:
                    self.assertEqual(len(history["lr"]), 10)
                    self.assertEqual(len(history["train_loss"]), 2)
                    self.assertTrue(np.all(np.isfinite(history["val_loss"])))
                with np.load(out / "split_indices.npz") as split:
                    self.assertEqual(len(split["train_event_index"]), 20)
                    self.assertEqual(len(split["val_event_index"]), 5)
                    self.assertEqual(np.intersect1d(split["train_event_index"], split["val_event_index"]).size, 0)
                with np.load(out / "best_evaluation.npz") as evaluation:
                    self.assertEqual(len(evaluation["event_index"]), 5)
                    if task == "regression_z":
                        self.assertIn("mae_mm", evaluation.files)
                    if task == "regression_A":
                        self.assertNotIn("mae_mm", evaluation.files)
                        np.testing.assert_allclose(evaluation["preds"], evaluation["preds_scaled"] * 0.2663 + 0.4113)
                        self.assertEqual(evaluation["E1"].shape, (5,))
                if task == "regression_A":
                    plot = subprocess.run([sys.executable, str(ROOT / "results/plot_A_result.py"),
                                           "--indir", str(out)], env=env, capture_output=True, text=True)
                    self.assertEqual(plot.returncode, 0, plot.stdout + plot.stderr)
                    self.assertIn('"E_max"', plot.stdout)
                    self.assertGreater((out / "A_all_plots.pdf").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
