from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "results"))


class PlotPidResultStatsTests(unittest.TestCase):
    def test_binary_roc_curve_and_auc_are_computed_from_signal_scores(self):
        from plot_pid_result import compute_roc_curve

        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        scores = np.asarray([0.1, 0.4, 0.35, 0.8], dtype=np.float64)

        fpr, tpr, auc = compute_roc_curve(labels, scores)

        self.assertAlmostEqual(auc, 0.75)
        np.testing.assert_allclose(fpr[[0, -1]], [0.0, 1.0])
        np.testing.assert_allclose(tpr[[0, -1]], [0.0, 1.0])

    def test_confusion_matrices_include_binary_and_raw_id_views(self):
        from plot_pid_result import compute_binary_confusion, compute_raw_id_prediction_matrix

        labels = np.asarray([1, 1, 0, 0, 0, 0], dtype=np.int64)
        predicted = np.asarray([1, 0, 0, 1, 0, 1], dtype=np.int64)
        raw_labels = np.asarray([1, 1, 21, 22, 23, 24], dtype=np.int64)

        binary = compute_binary_confusion(labels, predicted)
        raw_id_matrix, row_names = compute_raw_id_prediction_matrix(raw_labels, predicted)

        np.testing.assert_array_equal(binary, np.asarray([[2, 2], [1, 1]], dtype=np.int64))
        self.assertEqual(row_names, ["2nubb_ex", "2nubb_gs", "co60", "u238", "th232"])
        np.testing.assert_array_equal(
            raw_id_matrix,
            np.asarray(
                [
                    [1, 1],
                    [1, 0],
                    [0, 1],
                    [1, 0],
                    [0, 1],
                ],
                dtype=np.int64,
            ),
        )

    def test_efficiency_profile_uses_binomial_errors(self):
        from plot_pid_result import compute_efficiency_profile

        energy = np.asarray([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
        passed = np.asarray([True, False, True, True])
        bins = np.asarray([0.0, 2.0, 4.0], dtype=np.float64)

        profile = compute_efficiency_profile(energy, passed, bins)

        np.testing.assert_allclose(profile["centers"], [1.0, 3.0])
        np.testing.assert_array_equal(profile["counts"], [2, 2])
        np.testing.assert_allclose(profile["efficiency"], [0.5, 1.0])
        np.testing.assert_allclose(profile["error"], [np.sqrt(0.5 * 0.5 / 2.0), 0.0])


if __name__ == "__main__":
    unittest.main()
