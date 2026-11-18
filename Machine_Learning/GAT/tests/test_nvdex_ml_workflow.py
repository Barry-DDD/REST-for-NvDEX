"""Exercise workflow decisions, rejected data, real tiny CPU training, and review."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills/nvdex-ml-workflow/scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "tpc_GAT"))

from workflow_config import resolve, training_arguments, write_json, load_json
from workflow_data import inspect, inventory
from workflow import prepare, execute
from workflow_results import check_run, review


def fixture(path):
    """Write hand-checkable two-electron events and all five registered raw IDs."""
    n = 25
    cosine = np.linspace(-1, 1, n)
    with h5py.File(path, "w") as h5:
        events = dict(hit_start=np.arange(n)*3, hit_count=np.full(n, 3),
                      primary_start=np.arange(n)*2, primary_count=np.full(n, 2),
                      primary_origin_x=np.linspace(-10, 10, n), primary_origin_y=np.zeros(n),
                      primary_origin_z=np.linspace(-500, 500, n), id=np.tile([1, 21, 22, 23, 24], 5),
                      n_hits=np.full(n, 3), total_energy_filtered=np.full(n, 100))
        for key, value in events.items():
            h5[f"events/{key}"] = value
        for key, value in dict(x=[0., 1., 2.], y=[0., 0., 1.], z=[0., 1., 2.],
                               energy=[1., 2., 3.], log_energy=[0., np.log(2), np.log(3)], module_id=[0, 0, 0]).items():
            h5[f"hits/{key}"] = np.tile(value, n)
        for key, value in dict(primary_energy=np.column_stack([np.linspace(0, 100, n), np.linspace(100, 0, n)]).ravel(),
                               primary_dir_x=np.column_stack([np.ones(n), cosine]).ravel(),
                               primary_dir_y=np.column_stack([np.zeros(n), np.sqrt(1-cosine**2)]).ravel(),
                               primary_dir_z=np.zeros(n*2), primary_particle_name=np.full(n*2, b"e-")).items():
            h5[f"primaries/{key}"] = value
        h5["stats/feature_names"] = np.array([b"energy", b"log_energy"])
        for key, value in dict(mean=[2., .6], std=[1., 1.], min=[1., 0.], max=[3., 1.1]).items():
            h5[f"stats/{key}"] = value


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nvdex-workflow-")
        self.folder = Path(self.temp.name)
        self.path = self.folder / "events sample.h5"
        fixture(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def request(self, preset="theta-angle-v1", vertex="truth"):
        return dict(preset=preset, infile=str(self.path), vertex_source=vertex,
                    parameters=dict(batch_size=4, num_workers=0, epochs=2, hidden_channels=8,
                                    num_layers=1, heads=1, head_hidden_channels=8, radius_mm=4.))

    def test_missing_decisions_and_exact_user_overrides(self):
        response = resolve(dict(task="regression_theta", infile=str(self.path)), ROOT)
        self.assertEqual(response["status"], "needs_input")
        self.assertEqual({x["field"] for x in response["questions"]},
                         {"vertex_source", "target_mode", "parameters.features", "parameters.radius_mm"})
        config = resolve(self.request(), ROOT)
        arguments = training_arguments(config, self.folder / "output")
        self.assertIn("--vertex_relative", arguments)
        self.assertNotIn("--vertex_centroid", arguments)
        self.assertEqual(arguments[arguments.index("--radius_mm")+1], "4.0")
        self.assertEqual(config["parameter_origins"]["radius_mm"], "request")

    def test_unknown_and_conflicting_parameters_fail(self):
        cases = [dict(task="made_up"), dict(task="regression_A"), dict(vertex_source="unknown"),
                 dict(parameters={"radius_mm": float("nan")}), dict(parameters={"batch_size": True}),
                 dict(parameters={"event_features": ["primary_origin_x"]}),
                 dict(resources={"partition": "gpu\n#SBATCH --gres=gpu:100"})]
        for change in cases:
            request = self.request()
            request.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                resolve(request, ROOT)
        with self.assertRaises(ValueError):
            resolve(self.request("xyz-v1", "truth"), ROOT)

    def test_hand_calculated_targets_and_actual_graphs(self):
        report, values = inspect(resolve(self.request(), ROOT), sample_events=3)
        self.assertEqual(report["usable_events"], 25)
        np.testing.assert_allclose(values["target"][[0, 12, 24]], [180, 90, 0], atol=1e-4)
        self.assertTrue(all(row["edges"] == 6 for row in report["graph_sample"]))
        self.assertFalse(report["metadata_presence"]["g4_sensitive_volume_energy"])
        self.assertEqual(inventory(self.path)["status"], "inventory_only")
        report_a, values_a = inspect(resolve(self.request("A-v1", "none"), ROOT), sample_events=2)
        np.testing.assert_allclose(values_a["target"][[0, 12, 24]], [1, 0, 1])
        self.assertEqual(report_a["vertex_source"], "none")

    def test_missing_coordinates_stats_and_truth_have_distinct_outcomes(self):
        inspect(resolve(self.request(), ROOT), sample_events=2)
        with h5py.File(self.path, "r+") as h5:
            del h5["events/primary_origin_x"]
        with self.assertRaises(KeyError):
            inspect(resolve(self.request(), ROOT), sample_events=2)
        inspect(resolve(self.request(vertex="centroid"), ROOT), sample_events=2)
        with h5py.File(self.path, "r+") as h5:
            del h5["hits/z"]
        with self.assertRaises(KeyError):
            inspect(resolve(self.request(vertex="centroid"), ROOT), sample_events=2)
        fixture(self.path)
        with h5py.File(self.path, "r+") as h5:
            h5["stats/feature_names"][0] = b"other"
        with self.assertRaises(KeyError):
            inspect(resolve(self.request(), ROOT), sample_events=2)

    def test_invalid_events_are_rejected_without_filtering(self):
        with h5py.File(self.path, "r+") as h5:
            h5["events/hit_start"][1] = 0
        with self.assertRaisesRegex(ValueError, "overlap"):
            inspect(resolve(self.request(), ROOT), sample_events=2)
        fixture(self.path)
        with h5py.File(self.path, "r+") as h5:
            h5["primaries/primary_dir_x"][0] = np.nan
        with self.assertRaises(ValueError):
            inspect(resolve(self.request(), ROOT), sample_events=2)
        fixture(self.path)
        with h5py.File(self.path, "r+") as h5:
            del h5["events/id"]
            h5["events/id"] = np.full(25, 1.5)
        with self.assertRaisesRegex(ValueError, "integers"):
            inspect(resolve(self.request("binary-v1", "none"), ROOT), sample_events=2)

    def test_all_present_loader_metadata_and_stats_are_checked(self):
        with h5py.File(self.path, "r+") as h5:
            del h5["events/id"]
            h5["events/id"] = np.r_[np.ones(24), np.nan]
        with self.assertRaisesRegex(ValueError, "events/id"):
            inspect(resolve(self.request(), ROOT), sample_events=1)
        fixture(self.path)
        with h5py.File(self.path, "r+") as h5:
            del h5["events/n_hits"]
            h5["events/n_hits"] = [3]
        with self.assertRaisesRegex(ValueError, "events/n_hits"):
            inspect(resolve(self.request(), ROOT), sample_events=1)
        for stat, value in (("std", -1.), ("min", 100.)):
            fixture(self.path)
            with h5py.File(self.path, "r+") as h5:
                h5[f"stats/{stat}"][0] = value
            with self.assertRaisesRegex(ValueError, "statistics"):
                inspect(resolve(self.request(), ROOT), sample_events=1)

    def test_preparation_is_reviewable_and_never_submits(self):
        config = resolve(self.request(), ROOT)
        outdir = self.folder / "prepared bundle"
        result = prepare(config, outdir, ROOT, sample_events=2)
        self.assertEqual(result["status"], "prepared")
        subprocess.run(["bash", "-n", str(outdir / "job.sh")], check=True)
        self.assertGreater((outdir / "input_checks.pdf").stat().st_size, 1000)
        self.assertFalse((outdir / "results").exists())
        with self.assertRaises(ValueError):
            prepare(config, outdir, ROOT, sample_events=2)
        previous = os.environ.pop("SLURM_JOB_ID", None)
        try:
            with self.assertRaisesRegex(ValueError, "SLURM"):
                execute(outdir, ROOT)
        finally:
            if previous is not None:
                os.environ["SLURM_JOB_ID"] = previous

    def test_all_registered_tasks_tiny_cpu_training_and_review(self):
        presets = ["theta-angle-v1", "theta-cdf-v1", "A-v1", "xyz-v1", "z-v1", "binary-v1", "multiclass-v1"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        for preset in presets:
            with self.subTest(preset=preset):
                vertex = "truth" if preset.startswith("theta") else "none"
                config = resolve(self.request(preset, vertex), ROOT)
                output = self.folder / preset
                inspect(config, sample_events=2)
                manifest = None
                if preset == "theta-angle-v1":
                    bundle = self.folder / "manifest_case"
                    prepare(config, bundle, ROOT, sample_events=2)
                    manifest = load_json(bundle / "plan.json")
                    output.mkdir()
                    write_json(output / "workflow_manifest.json", manifest)
                command = [sys.executable, str(ROOT / "tpc_GAT/run.py"), *training_arguments(config, output)]
                completed = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                _, evaluation, _, _, report = check_run(output, ROOT)
                self.assertEqual(report["status"], "passed")
                self.assertNotIn("g4_sensitive_volume_energy", evaluation)
                review(output, output / "review", ROOT, plots=preset in {"theta-angle-v1", "A-v1", "multiclass-v1"})
                if preset == "theta-angle-v1":
                    self.assertGreater((output / "review/workflow_summary.pdf").stat().st_size, 1000)
                    self.assertFalse(report["warnings"][0].startswith("Legacy"))
                    changed_manifest = json.loads(json.dumps(manifest))
                    first_source = next(iter(changed_manifest["sources"]))
                    changed_manifest["sources"][first_source] = "changed"
                    write_json(output / "workflow_manifest.json", changed_manifest)
                    with self.assertRaisesRegex(ValueError, "sources changed"):
                        check_run(output, ROOT)
                    write_json(output / "workflow_manifest.json", manifest)
                    config_file = output / "training_config.json"
                    runtime_config = load_json(config_file)
                    changed_config = dict(runtime_config, radius_mm=8.)
                    write_json(config_file, changed_config)
                    with self.assertRaisesRegex(ValueError, "radius_mm"):
                        check_run(output, ROOT)
                    write_json(config_file, runtime_config)
                    original_stat = self.path.stat()
                    os.utime(self.path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns+2_000_000_000))
                    self.assertNotEqual(self.path.stat().st_mtime_ns, manifest["inspection"]["input"]["mtime_ns"])
                    with self.assertRaisesRegex(ValueError, "Input fingerprint"):
                        check_run(output, ROOT)
                    os.utime(self.path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
                    source = output / "best_evaluation.npz"
                    with np.load(source) as archive:
                        changed = dict(archive)
                    original = changed["event_index"].copy()
                    changed["event_index"][0] = changed["event_index"][1]
                    np.savez_compressed(source, **changed)
                    with self.assertRaisesRegex(ValueError, "duplicate"):
                        check_run(output, ROOT)
                    changed["event_index"] = original
                    changed["preds"][0] += 1
                    np.savez_compressed(source, **changed)
                    with self.assertRaisesRegex(ValueError, "Inverse"):
                        check_run(output, ROOT)

    def test_variable_primary_energy_sum_has_no_fixed_q_emax_plots(self):
        with h5py.File(self.path, "r+") as h5:
            h5["primaries/primary_energy"][1::2] += np.linspace(1, 30, 25)
        config = resolve(self.request("A-v1", "none"), ROOT)
        output = self.folder / "variable_sum_A"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        completed = subprocess.run([sys.executable, str(ROOT / "tpc_GAT/run.py"), *training_arguments(config, output)],
                                   env=env, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        report = review(output, output / "review", ROOT)
        self.assertFalse(report["emax_fixed_sum_conversion_allowed"])
        self.assertFalse((output / "review/validated_artifacts").exists())
        self.assertTrue((output / "review/workflow_summary.pdf").exists())


if __name__ == "__main__":
    unittest.main()
