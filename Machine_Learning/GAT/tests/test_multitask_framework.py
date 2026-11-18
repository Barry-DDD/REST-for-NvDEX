from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "feature_extraction"))
sys.path.insert(0, str(REPO_ROOT / "tpc_GAT"))


class RootToHdf5EventIdTests(unittest.TestCase):
    def _hit_block(self):
        return {
            "event_hit_start": np.asarray([0, 2], dtype=np.int64),
            "event_hit_count": np.asarray([2, 3], dtype=np.int32),
            "n_hits_original": np.asarray([2, 3], dtype=np.int32),
            "total_energy_filtered": np.asarray([10.0, 20.0], dtype=np.float32),
        }

    def test_keeps_optional_event_id_when_present(self):
        from root_to_hdf5 import build_event_metadata

        events = [
            {"id": 1, "n_hits": 2, "n_primaries": 1, "n_tracks": 1},
            {"id": 21, "n_hits": 3, "n_primaries": 1, "n_tracks": 1},
        ]

        metadata = build_event_metadata(events, self._hit_block())

        np.testing.assert_array_equal(metadata["id"], np.asarray([1, 21], dtype=np.int32))

    def test_omits_event_id_when_absent(self):
        from root_to_hdf5 import build_event_metadata

        events = [
            {"n_hits": 2, "n_primaries": 1, "n_tracks": 1},
            {"n_hits": 3, "n_primaries": 1, "n_tracks": 1},
        ]

        metadata = build_event_metadata(events, self._hit_block())

        self.assertNotIn("id", metadata)


class TaskSpecTests(unittest.TestCase):
    def test_binary_id_mapping(self):
        from task_spec import get_task_spec

        spec = get_task_spec("binary_id")

        self.assertEqual(spec.num_outputs, 2)
        np.testing.assert_array_equal(
            spec.map_raw_labels(np.asarray([1, 21, 22, 23, 24])),
            np.asarray([1, 0, 0, 0, 0]),
        )

    def test_multiclass_id_mapping(self):
        from task_spec import get_task_spec

        spec = get_task_spec("multiclass_id")

        self.assertEqual(spec.num_outputs, 5)
        np.testing.assert_array_equal(
            spec.map_raw_labels(np.asarray([1, 21, 22, 23, 24])),
            np.asarray([0, 1, 2, 3, 4]),
        )

    def test_unknown_event_id_is_rejected(self):
        from task_spec import get_task_spec

        spec = get_task_spec("binary_id")

        with self.assertRaisesRegex(ValueError, "Unsupported id"):
            spec.map_raw_labels(np.asarray([99]))


class ModelHeadTests(unittest.TestCase):
    def test_multitask_model_output_shapes(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        from gat_model import TPCGATModel

        regression_model = TPCGATModel(
            in_channels=4,
            hidden_channels=16,
            num_layers=1,
            heads=1,
            task="regression_z",
        )
        binary_model = TPCGATModel(
            in_channels=4,
            hidden_channels=16,
            num_layers=1,
            heads=1,
            task="binary_id",
        )

        self.assertEqual(regression_model.head(torch.zeros(3, regression_model.embedding_dim)).shape, (3,))
        self.assertEqual(binary_model.head(torch.zeros(3, binary_model.embedding_dim)).shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
