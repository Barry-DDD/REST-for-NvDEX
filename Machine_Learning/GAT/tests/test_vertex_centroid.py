"""Checks for truth-free hit-centroid frames in A and theta regression."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tpc_GAT"))
from tpc_graph_dataset import TPCGraphDataset
from utils import validate_epoch
from test_regression_theta import write_theta_fixture


class CentroidFrameTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "events.h5"
        write_theta_fixture(self.path)
        with h5py.File(self.path, "r+") as f:
            f["hits/y"][:] = np.tile([-1., 0., 1.], 25)
            f["hits/z"] = np.tile([10., 20., 30.], 25)
            for axis in "xyz":
                key = f"events/primary_origin_{axis}"
                if key in f:
                    del f[key]

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, task="regression_A", **kwargs):
        return TPCGraphDataset(str(self.path), task=task, **kwargs)

    def test_centroid_values_features_and_zero_direction(self):
        for task in ("regression_A", "regression_theta"):
            plain = self.dataset(task)
            data = self.dataset(task, vertex_centroid=True)
            try:
                # Fail if the centroid input path ever tries to access the truth vertex.
                def fail_truth_read(_):
                    raise AssertionError("Centroid mode read the truth vertex")
                data._read_vertex = fail_truth_read
                graph = data[0]
                self.assertEqual(graph.x.shape, (3, 11))
                np.testing.assert_allclose(graph.x[:, :4], plain[0].x)
                np.testing.assert_allclose(graph.frame_vertex, [[1., 0., 20.]])
                delta = np.array([[-1., -1., -10.], [0., 0., 0.], [1., 1., 10.]])
                scaled = delta / [445., 445., 850.]
                expected = np.column_stack([scaled, np.linalg.norm(scaled, axis=1),
                                             delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-6)])
                np.testing.assert_allclose(graph.x[:, -7:], expected, atol=1e-7)
                self.assertTrue(torch.isfinite(graph.x).all())
                self.assertEqual(data.vertex_source, "centroid")
                # The unequal hit energies must not move the unweighted reference point.
                self.assertNotAlmostEqual(graph.frame_vertex[0, 0].item(), 4. / 3.)
            finally:
                plain.close()
                data.close()

    def test_translation_and_truth_independence(self):
        data = self.dataset(vertex_centroid=True)
        before = data[0]
        data.close()
        shift = np.array([20., -30., 100.])
        with h5py.File(self.path, "r+") as f:
            for j, axis in enumerate("xyz"):
                f[f"hits/{axis}"][:] += shift[j]
                f[f"events/primary_origin_{axis}"] = np.full(25, 123456.)
        data = self.dataset(vertex_centroid=True)
        try:
            after = data[0]
            np.testing.assert_allclose(after.frame_vertex, before.frame_vertex + shift, atol=1e-6)
            np.testing.assert_allclose(after.x[:, -7:], before.x[:, -7:], atol=1e-7)
            np.testing.assert_array_equal(after.edge_index, before.edge_index)
            np.testing.assert_allclose(after.edge_attr, before.edge_attr)
        finally:
            data.close()

    def test_batch_validation_and_empty_event_alignment(self):
        with h5py.File(self.path, "r+") as f:
            f["events/hit_count"][1] = 0
        data = self.dataset(vertex_centroid=True)
        class Echo(torch.nn.Module):
            def forward(self, batch):
                return batch.y.view(-1)
        try:
            result = validate_epoch(Echo(), torch.device("cpu"), DataLoader(data, batch_size=5),
                                    torch.nn.SmoothL1Loss(), task="regression_A")
            self.assertEqual(result["metadata"]["frame_vertex"].shape, (24, 3))
            np.testing.assert_array_equal(result["metadata"]["event_index"][:3], [0, 2, 3])
            np.testing.assert_allclose(result["metadata"]["frame_vertex"], np.tile([1., 0., 20.], (24, 1)))
        finally:
            data.close()
        data = self.dataset(vertex_centroid=True, drop_empty_events=False)
        try:
            np.testing.assert_allclose(data[1].frame_vertex, [[0., 0., 0.]])
            self.assertTrue(torch.isfinite(data[1].x).all())
        finally:
            data.close()

    def test_missing_invalid_inputs_and_option_conflict(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.dataset(vertex_relative=True, vertex_centroid=True)
        with self.assertRaisesRegex(ValueError, "supported only"):
            self.dataset("regression_xyz", vertex_centroid=True)
        # Existing truth mode continues to require all truth vertex coordinates.
        with self.assertRaisesRegex(KeyError, "primary_origin"):
            self.dataset(vertex_relative=True)
        with h5py.File(self.path, "r+") as f:
            f["hits/z"][0] = np.nan
        data = self.dataset(vertex_centroid=True)
        try:
            with self.assertRaisesRegex(ValueError, "finite"):
                data[0]
        finally:
            data.close()
        with h5py.File(self.path, "r+") as f:
            del f["hits/z"]
        with self.assertRaisesRegex(KeyError, "hits/z"):
            self.dataset(vertex_centroid=True)

    def test_cli_training_and_saved_frame(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        for task, mode in (("regression_A", "standard"), ("regression_theta", "cdf"),
                           ("regression_theta", "angle")):
            with self.subTest(task=task, mode=mode):
                out = Path(self.temp.name) / (task + mode)
                command = [sys.executable, str(ROOT / "tpc_GAT/run.py"), "--infile", str(self.path),
                           "--outdir", str(out), "--task", task, "--target_mode", mode,
                           "--vertex_centroid", "--features", "x", "y", "z", "energy", "log_energy",
                           "--epochs", "1", "--batch_size", "4", "--num_workers", "0",
                           "--hidden_channels", "8", "--num_layers", "1", "--heads", "1",
                           "--head_hidden_channels", "8"]
                result = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                config = json.loads((out / "training_config.json").read_text())
                self.assertEqual(config["vertex_source"], "centroid")
                self.assertTrue(config["vertex_centroid"])
                self.assertFalse(config["vertex_relative"])
                self.assertEqual(config["input_channels"], 12)
                with np.load(out / "best_evaluation.npz", allow_pickle=False) as evaluation:
                    self.assertEqual(evaluation["preds"].shape, (5,))
                    self.assertTrue(np.isfinite(evaluation["preds"]).all())
                    np.testing.assert_allclose(evaluation["frame_vertex"], np.tile([1., 0., 20.], (5, 1)))
                conflict = subprocess.run(command + ["--vertex_relative"], env=env, capture_output=True, text=True)
                self.assertEqual(conflict.returncode, 2)
                self.assertIn("not allowed with argument", conflict.stderr)


if __name__ == "__main__":
    unittest.main()
