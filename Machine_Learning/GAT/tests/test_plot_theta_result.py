"""Check that the radial plotting cut preserves event alignment."""
from pathlib import Path
import sys
import unittest
import tempfile

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results"))
from plot_theta_result import electron_energy_fraction, validation_views


class ThetaPlotSelectionTests(unittest.TestCase):
    def test_energy_fraction_uses_event_identity_and_radial_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.h5"
            with h5py.File(path, "w") as source:
                source["events/primary_start"] = [0, 2, 4]
                source["events/primary_count"] = [2, 2, 2]
                source["primaries/primary_energy"] = [5., 5., 1., 9., 8., 2.]
            evaluation = {"labels": np.zeros(3), "event_index": np.array([2, 0, 1])}
            fraction = electron_energy_fraction(evaluation, path)
            np.testing.assert_allclose(fraction, [.8, .5, .9])
            evaluation["electron_energy_fraction"] = fraction
            evaluation["primary_origin"] = np.array([[0., 0., 0.], [351., 0., 0.], [350., 0., 0.]])
            selected = list(validation_views(evaluation))[1][1]
            np.testing.assert_allclose(selected["electron_energy_fraction"], [.8, .9])
            with h5py.File(path, "r+") as source:
                source["events/primary_count"][0] = 1
            with self.assertRaises(ValueError):
                electron_energy_fraction({"labels": np.zeros(1), "event_index": np.array([0])}, path)

    def test_saved_energy_fraction_endpoints_and_invalid_energies(self):
        evaluation = {"labels": np.zeros(3), "E1": np.array([1., 0., 3.]), "E2": np.array([1., 2., 1.])}
        np.testing.assert_allclose(electron_energy_fraction(evaluation, None), [.5, 1., .75])
        evaluation["E1"][0] = -1
        with self.assertRaises(ValueError):
            electron_energy_fraction(evaluation, None)
        evaluation["E1"][0] = 0
        evaluation["E2"][0] = 0
        with self.assertRaises(ValueError):
            electron_energy_fraction(evaluation, None)

    def test_inclusive_radius_and_aligned_metadata(self):
        evaluation = {
            "labels": np.array([10., 20., 30., 40.]),
            "preds": np.array([11., 22., 33., 44.]),
            "primary_origin": np.array([[210., 280., 800.], [351., 0., 0.],
                                        [0., 0., -800.], [-350., 0., 0.]]),
            "event_index": np.array([41, 7, 19, 82]),
            "n_hit": np.array([101, 202, 303, 404]),
            "target_mode": np.asarray("angle"),
        }
        original = {k: v.copy() for k, v in evaluation.items()}
        full, selected = list(validation_views(evaluation))
        self.assertIs(full[1], evaluation)
        self.assertIn("N=3/4", selected[0])
        np.testing.assert_array_equal(selected[1]["event_index"], [41, 19, 82])
        np.testing.assert_array_equal(selected[1]["labels"], [10, 30, 40])
        np.testing.assert_array_equal(selected[1]["preds"], [11, 33, 44])
        np.testing.assert_array_equal(selected[1]["n_hit"], [101, 303, 404])
        self.assertEqual(selected[1]["target_mode"].item(), "angle")
        for key in evaluation:
            np.testing.assert_array_equal(evaluation[key], original[key])

    def test_missing_or_empty_selection_keeps_full_plots(self):
        evaluation = {"labels": np.array([10., 20.])}
        self.assertEqual(len(list(validation_views(evaluation))), 1)
        evaluation["primary_origin"] = np.array([[351., 0., 0.], [0., 400., 0.]])
        self.assertEqual(len(list(validation_views(evaluation))), 1)
        evaluation["primary_origin"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            list(validation_views(evaluation))


if __name__ == "__main__":
    unittest.main()
