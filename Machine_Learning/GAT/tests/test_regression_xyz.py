"""Synthetic checks for joint vertex regression, event identity and plotting."""

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
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tpc_GAT"))
sys.path.insert(0, str(ROOT / "results"))
from gat_model import TPCGATModel
from task_spec import get_target_transform, vertex_metrics
from tpc_graph_dataset import TPCGraphDataset
from utils import validate_epoch
from plot_vertex_result import binned_stats, load_run


def write_fixture(path):
    """Write distinct three-dimensional vertices and four hits per event."""
    n = 25
    vertices = np.column_stack([np.linspace(-300, 300, n),
                                150 * np.cos(np.arange(n)), np.linspace(-800, 800, n)])
    with h5py.File(path, "w") as f:
        f["events/hit_start"] = np.arange(n) * 4
        f["events/hit_count"] = np.full(n, 4)
        for j, axis in enumerate("xyz"):
            f[f"events/primary_origin_{axis}"] = vertices[:, j].astype(np.float32)
            f[f"hits/{axis}"] = (vertices[:, j, None] + [-1., 0., 1., 2.]).ravel().astype(np.float32)
        for key, values in (("energy", [1., 2., 3., 4.]),
                            ("log_energy", np.log1p([1., 2., 3., 4.])), ("module_id", [0, 0, 0, 0])):
            f[f"hits/{key}"] = np.tile(values, n)
        f["stats/feature_names"] = np.asarray([b"energy", b"log_energy"])
        for key, values in (("mean", [2.5, 1.]), ("std", [1., 1.]),
                            ("min", [1., 0.]), ("max", [4., 2.])):
            f[f"stats/{key}"] = values


class VertexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "events.h5"
        write_fixture(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, **kwargs):
        return TPCGraphDataset(str(self.path), task="regression_xyz",
                               node_feature_names=("x", "y", "z", "energy", "log_energy"), **kwargs)

    def test_shapes_centroids_and_backward(self):
        with h5py.File(self.path, "r+") as f:
            f["events/hit_count"][1] = 0
        data = self.dataset()
        try:
            batch = next(iter(DataLoader(data, batch_size=3)))
            self.assertEqual(tuple(batch.y.shape), (3, 3))
            np.testing.assert_array_equal(batch.event_index, [0, 2, 3])
            np.testing.assert_allclose(batch.y * 845, batch.target_raw, atol=1e-4)
            np.testing.assert_allclose(batch.hit_centroid, batch.target_raw + 0.5, atol=1e-4)
            np.testing.assert_allclose(batch.energy_centroid, batch.target_raw + 1., atol=1e-4)
            model = TPCGATModel(in_channels=5, hidden_channels=8, num_layers=1,
                                heads=1, head_hidden_channels=8, task="regression_xyz")
            output = model(batch)
            self.assertEqual(tuple(output.shape), (3, 3))
            torch.nn.functional.smooth_l1_loss(output, batch.y).backward()
            self.assertTrue(torch.isfinite(model.head.layers[-1].weight.grad).all())
            for task in ("regression_z", "regression_A", "regression_theta"):
                scalar = TPCGATModel(in_channels=5, hidden_channels=8, num_layers=1, heads=1, task=task)
                self.assertEqual(tuple(scalar(batch).shape), (3,))
        finally:
            data.close()

    def test_validation_deduplicates_whole_vertices(self):
        data = self.dataset()

        class OffsetModel(torch.nn.Module):
            def forward(self, batch):
                return batch.y + torch.tensor([3., -4., 12.]) / 845

        try:
            result = validate_epoch(OffsetModel(), torch.device("cpu"),
                                    DataLoader(Subset(data, [0, 0, 2, 4, 6]), batch_size=3),
                                    torch.nn.SmoothL1Loss(), task="regression_xyz")
            self.assertEqual(result["preds"].shape, (4, 3))
            self.assertEqual(result["metadata"]["hit_centroid"].shape, (4, 3))
            np.testing.assert_array_equal(result["metadata"]["event_index"], [0, 2, 4, 6])
            np.testing.assert_allclose(result["mae_xyz"], [3., 4., 12.], atol=1e-4)
            self.assertAlmostEqual(result["r68_mm"], 13., places=3)
            self.assertAlmostEqual(result["rmse_3d_mm"], 13., places=3)
            self.assertAlmostEqual(result["loss"], (9 + 16 + 144) / (6 * 845 ** 2), places=7)
        finally:
            data.close()

    def test_invalid_truth_and_leaking_inputs(self):
        with self.assertRaisesRegex(ValueError, "vertex_relative"):
            self.dataset(vertex_relative=True)
        for key in ("primary_origin_x", "primary_origin_y", "primary_origin_z", "primary_origin",
                    "primary_dir_x", "g4_sensitive_volume_energy"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.dataset(event_feature_keys=[key])
        for value in (np.nan, 900.):
            write_fixture(self.path)
            with h5py.File(self.path, "r+") as f:
                f["events/primary_origin_x"][0] = value
            with self.assertRaises(ValueError):
                self.dataset()
        write_fixture(self.path)
        with h5py.File(self.path, "r+") as f:
            del f["events/primary_origin_y"]
        with self.assertRaisesRegex(KeyError, "primary_origin_y"):
            self.dataset()

    def test_metric_and_profile_definitions(self):
        metrics = vertex_metrics([[0., 0., 0.]], [[3., -4., 12.]])
        self.assertEqual(metrics["r68_mm"], 13.)
        self.assertEqual(metrics["rmse_3d_mm"], 13.)
        for invalid in (np.zeros(3), np.zeros((0, 3)), np.full((1, 3), np.nan)):
            with self.assertRaises(ValueError):
                vertex_metrics(invalid, invalid)
        rows = binned_stats([0., 1.], [3., 4.], [0., 0.5, 1.])
        np.testing.assert_array_equal(rows[:, 0], [1, 1])
        values = np.array([[-300., 100., -800.], [300., -100., 800.]])
        transform = get_target_transform("regression_xyz")
        np.testing.assert_allclose(transform.inverse(transform.forward(values)), values, atol=1e-4)

    def test_training_and_plot_outputs(self):
        out = Path(self.temp.name) / "run"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        command = [sys.executable, str(ROOT / "tpc_GAT/run.py"), "--infile", str(self.path),
                   "--outdir", str(out), "--task", "regression_xyz", "--epochs", "2",
                   "--batch_size", "4", "--num_workers", "0", "--hidden_channels", "8",
                   "--num_layers", "1", "--heads", "1", "--head_hidden_channels", "8",
                   "--features", "x", "y", "z", "energy", "log_energy"]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        config, evaluation, history, split = load_run(out)
        self.assertEqual(evaluation["preds"].shape, (5, 3))
        self.assertEqual(history["val_mae_xyz"].shape, (2, 3))
        self.assertEqual(split["train_targets"].shape, (20, 3))
        np.testing.assert_allclose(evaluation["preds"], evaluation["preds_scaled"] * 845, atol=1e-4)
        plot = subprocess.run([sys.executable, str(ROOT / "results/plot_vertex_result.py"),
                               "--indir", str(out)], env=env, capture_output=True, text=True)
        self.assertEqual(plot.returncode, 0, plot.stdout + plot.stderr)
        self.assertGreater((out / "vertex_all_plots.pdf").stat().st_size, 1000)
        summary = json.loads((out / "vertex_metrics.json").read_text())
        self.assertIn("Energy centroid", summary["metrics"])
        np.testing.assert_allclose(summary["metrics"]["GAT"]["mae_xyz"], evaluation["mae_xyz"])
        # Reject accidental event/coordinate flattening in persisted results.
        evaluation["preds_scaled"] = evaluation["preds_scaled"].ravel()
        np.savez_compressed(out / "best_evaluation.npz", **evaluation)
        with self.assertRaises(ValueError):
            load_run(out)


if __name__ == "__main__":
    unittest.main()
