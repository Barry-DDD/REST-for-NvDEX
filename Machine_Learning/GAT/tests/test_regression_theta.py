"""Synthetic checks for opening-angle CDF regression."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tpc_GAT"))
sys.path.insert(0, str(ROOT / "results"))
from task_spec import fit_cdf_transform, get_target_transform, read_theta_targets
from tpc_graph_dataset import TPCGraphDataset
from plot_theta_result import cdf_forward
from test_regression_A import write_fixture


def write_theta_fixture(path):
    """Add two finite electron directions to the shared graph fixture."""
    write_fixture(path)
    n = 25
    cos_theta = np.linspace(-1.0, 1.0, n)
    with h5py.File(path, "r+") as h5:
        h5["primaries/primary_dir_x"] = np.column_stack([np.ones(n), cos_theta]).ravel()
        h5["primaries/primary_dir_y"] = np.column_stack([np.zeros(n), np.sqrt(1.0 - cos_theta**2)]).ravel()
        h5["primaries/primary_dir_z"] = np.zeros(n * 2)


class RegressionThetaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "events.h5"
        write_theta_fixture(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_targets_and_training_only_cdf(self):
        with h5py.File(self.path, "r") as h5:
            targets = read_theta_targets(h5)
        np.testing.assert_allclose(targets, np.linspace(-1.0, 1.0, 25), atol=1e-6)
        transform = fit_cdf_transform(targets[:20])
        mapped = transform.forward(targets[:20])
        self.assertTrue(np.all(np.diff(mapped) > 0))
        np.testing.assert_allclose(transform.inverse(mapped), targets[:20], atol=1e-6)
        self.assertEqual(get_target_transform("regression_theta").mode, "cdf")
        with self.assertRaises(ValueError):
            get_target_transform("regression_theta", "standard")

        dataset = TPCGraphDataset(str(self.path), task="regression_theta")
        dataset.set_target_transform(transform)
        try:
            self.assertAlmostEqual(dataset[5].target_raw.item(), targets[5], places=6)
            self.assertAlmostEqual(dataset[5].y.item(), mapped[5], places=6)
        finally:
            dataset.close()
        with self.assertRaises(ValueError):
            TPCGraphDataset(str(self.path), task="regression_theta", event_feature_keys=["primary_dir_x"])

    def test_invalid_direction_and_end_to_end_outputs(self):
        with h5py.File(self.path, "r+") as h5:
            h5["primaries/primary_dir_x"][0] = np.nan
            with self.assertRaises(ValueError):
                read_theta_targets(h5)
        write_theta_fixture(self.path)
        out = Path(self.temp.name) / "theta"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        command = [sys.executable, str(ROOT / "tpc_GAT/run.py"), "--infile", str(self.path),
                   "--outdir", str(out), "--task", "regression_theta", "--epochs", "2",
                   "--batch_size", "4", "--num_workers", "0", "--hidden_channels", "8",
                   "--num_layers", "1", "--heads", "1", "--head_hidden_channels", "8"]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with np.load(out / "split_indices.npz") as split, np.load(out / "best_evaluation.npz") as evaluation:
            self.assertIn("cdf_values", split.files)
            self.assertIn("cdf_probabilities", evaluation.files)
            np.testing.assert_allclose(evaluation["labels_scaled"],
                                       cdf_forward(evaluation["labels"], split["cdf_values"], split["cdf_probabilities"]), atol=1e-6)
        plot = subprocess.run([sys.executable, str(ROOT / "results/plot_theta_result.py"), "--indir", str(out)],
                              env=env, capture_output=True, text=True)
        self.assertEqual(plot.returncode, 0, plot.stdout + plot.stderr)
        self.assertGreater((out / "theta_all_plots.pdf").stat().st_size, 1000)


    def test_angle_mode_targets_and_end_to_end_outputs(self):
        transform = get_target_transform("regression_theta", "angle")
        self.assertEqual((transform.mode, transform.unit), ("angle", "deg"))
        np.testing.assert_allclose(transform.forward(np.asarray([0.0, 90.0, 180.0])), [-1.0, 0.0, 1.0])
        np.testing.assert_allclose(transform.inverse(np.asarray([-1.0, 1.0])), [0.0, 180.0])
        dataset = TPCGraphDataset(str(self.path), task="regression_theta", target_mode="angle")
        try:
            expected = np.degrees(np.arccos(np.linspace(-1.0, 1.0, 25)))
            self.assertEqual(dataset.target_key, "theta_deg")
            self.assertAlmostEqual(dataset[5].target_raw.item(), expected[5], places=3)
            self.assertAlmostEqual(dataset[5].y.item(), (expected[5] - 90.0) / 90.0, places=5)
        finally:
            dataset.close()

        out = Path(self.temp.name) / "theta_angle"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        command = [sys.executable, str(ROOT / "tpc_GAT/run.py"), "--infile", str(self.path),
                   "--outdir", str(out), "--task", "regression_theta", "--target_mode", "angle",
                   "--epochs", "2", "--batch_size", "4", "--num_workers", "0", "--hidden_channels", "8",
                   "--num_layers", "1", "--heads", "1", "--head_hidden_channels", "8"]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with np.load(out / "split_indices.npz") as split, np.load(out / "best_evaluation.npz") as evaluation:
            self.assertNotIn("cdf_values", split.files)
            self.assertEqual(str(evaluation["target_mode"]), "angle")
            self.assertTrue(np.all((evaluation["labels"] >= 0) & (evaluation["labels"] <= 180)))
        plot = subprocess.run([sys.executable, str(ROOT / "results/plot_theta_result.py"), "--indir", str(out)],
                              env=env, capture_output=True, text=True)
        self.assertEqual(plot.returncode, 0, plot.stdout + plot.stderr)
        self.assertGreater((out / "theta_all_plots.pdf").stat().st_size, 1000)

if __name__ == "__main__":
    unittest.main()
