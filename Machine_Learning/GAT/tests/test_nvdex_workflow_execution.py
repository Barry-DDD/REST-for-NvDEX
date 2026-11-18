"""Verify portable launchers using real, tiny CPU execution and trusted setup scripts."""

import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_nvdex_ml_workflow import ROOT, fixture
from workflow import prepare, execute
from workflow_config import IHEP, load_json, resolve
from workflow_execution import execution_environment, setup_fingerprint, shell_environment_lines


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nvdex-execution-")
        self.folder = Path(self.temp.name)
        self.path = self.folder / "events.h5"
        fixture(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def request(self, **execution):
        return dict(deployment="current", preset="theta-angle-v1", infile=str(self.path),
                    vertex_source="truth", execution=dict(backend="direct", device="cpu", **execution),
                    parameters=dict(batch_size=4, num_workers=0, epochs=2, hidden_channels=8,
                                    num_layers=1, heads=1, head_hidden_channels=8, radius_mm=4.))

    def test_default_ihep_and_current_deployments(self):
        request = self.request()
        current = resolve(request, ROOT)
        self.assertEqual(current["python"], sys.executable)
        self.assertEqual(current["resources"], {})
        self.assertEqual(current["execution"]["backend"], "direct")
        request.pop("deployment")
        request.pop("execution")
        default = resolve(request, ROOT)
        self.assertEqual(default["python"], IHEP["python"])
        self.assertEqual(default["execution"]["backend"], "slurm")
        self.assertEqual(default["resources"]["account"], "junogpu")

    def test_invalid_execution_combinations_fail(self):
        for execution in ({"backend":"unknown"}, {"backend":"slurm","device":"cpu"},
                          {"backend":"slurm","device":"cuda","gpu":"0"},
                          {"backend":"direct","device":"cuda","gpu":"0,1"},
                          {"backend":"direct","device":"cpu","gpu":"0"}, {"setup_script":42}):
            request = self.request()
            request["execution"].update(execution)
            with self.subTest(execution=execution), self.assertRaises(ValueError):
                resolve(request, ROOT)
        request = self.request()
        request["resources"] = {"partition":"gpu"}
        with self.assertRaisesRegex(ValueError, "do not apply"):
            resolve(request, ROOT)

    def test_device_selection_does_not_override_slurm_assignment(self):
        request = self.request()
        request["execution"] = dict(backend="direct", device="cuda", gpu="3")
        self.assertEqual(execution_environment(resolve(request, ROOT))["CUDA_VISIBLE_DEVICES"], "3")
        request["execution"] = dict(backend="slurm", device="cuda")
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="7"):
            self.assertEqual(execution_environment(resolve(request, ROOT))["CUDA_VISIBLE_DEVICES"], "7")

    def test_slurm_setup_preserves_allocated_gpu_visibility(self):
        setup = self.folder / "gpu_setup.sh"
        setup.write_text("export CUDA_VISIBLE_DEVICES=0\n")
        request = self.request()
        request["execution"] = dict(backend="slurm", device="cuda", setup_script=str(setup))
        config = resolve(request, ROOT)
        body = "\n".join([*shell_environment_lines(config, setup_fingerprint(config), ROOT),
                           'printf "%s" "${CUDA_VISIBLE_DEVICES-unset}"'])
        for initial, expected in (("7", "7"), (None, "unset")):
            env = dict(os.environ)
            env.pop("CUDA_VISIBLE_DEVICES", None)
            if initial is not None:
                env["CUDA_VISIBLE_DEVICES"] = initial
            result = subprocess.run(["bash", "-c", body], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, expected)

    def test_interpreter_symlinks_are_not_silently_substituted(self):
        alias = self.folder / "another_env/bin/python"
        alias.parent.mkdir(parents=True)
        alias.symlink_to(sys.executable)
        request = self.request()
        request["python"] = str(alias)
        self.assertEqual(alias.resolve(), Path(sys.executable).resolve())
        with self.assertRaisesRegex(ValueError, "same Python"):
            prepare(resolve(request, ROOT), self.folder / "alias_bundle", ROOT, sample_events=2)

    def test_missing_or_failing_setup_does_not_prepare(self):
        request = self.request(setup_script=str(self.folder / "missing.sh"))
        with self.assertRaisesRegex(ValueError, "missing"):
            prepare(resolve(request, ROOT), self.folder / "missing_bundle", ROOT, sample_events=2)
        setup = self.folder / "failed.sh"
        setup.write_text("exit 9\n")
        with self.assertRaisesRegex(ValueError, "probe failed"):
            prepare(resolve(self.request(setup_script=str(setup)), ROOT), self.folder / "failed_bundle", ROOT, sample_events=2)
        self.assertFalse((self.folder / "failed_bundle").exists())

    def test_setup_drift_is_rejected_before_sourcing(self):
        marker = self.folder / "calls.txt"
        setup = self.folder / "setup.sh"
        setup.write_text("printf setup >> " + shlex.quote(str(marker)) + "\n")
        bundle = self.folder / "drift_bundle"
        prepare(resolve(self.request(setup_script=str(setup)), ROOT), bundle, ROOT, sample_events=2)
        self.assertEqual(marker.read_text(), "setup")
        setup.write_text(setup.read_text() + "# Changed after preparation.\n")
        result = subprocess.run(["bash", str(bundle / "run.sh")], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("setup script changed", result.stderr)
        self.assertEqual(marker.read_text(), "setup")
        self.assertFalse((bundle / "launch.lock").exists())

    def test_direct_cuda_never_falls_back_to_cpu(self):
        request = self.request()
        request["execution"] = dict(backend="direct", device="cuda")
        config = resolve(request, ROOT)
        bundle = self.folder / "cuda_bundle"
        prepare(config, bundle, ROOT, sample_events=2)
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "No CPU fallback"):
                execute(bundle, ROOT)
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=2):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                execute(bundle, ROOT)
        self.assertFalse((bundle / "launch.lock").exists())

    def test_generated_direct_launcher_runs_full_pipeline_without_slurm(self):
        marker = self.folder / "source calls.txt"
        setup = self.folder / "setup 'quoted'.sh"
        setup.write_text("echo environment-ready\nunset NVDEX_TEST_UNSET_REFERENCE\nexport NVDEX_ACTIVATED_VALUE=prefix:$NVDEX_TEST_UNSET_REFERENCE\nprintf setup >> "
                         + shlex.quote(str(marker)) + "\n")
        bundle = self.folder / "direct 'quoted' bundle"
        config = resolve(self.request(setup_script=str(setup)), ROOT)
        prepared = prepare(config, bundle, ROOT, sample_events=2)
        self.assertEqual(prepared["backend"], "direct")
        self.assertNotIn("submit_command", prepared)
        self.assertFalse((bundle / "job.sh").exists())
        subprocess.run(["bash", "-n", str(bundle / "run.sh")], check=True)
        self.assertFalse((bundle / "results").exists())
        env = dict(os.environ, PYTHONWARNINGS="ignore", OMP_NUM_THREADS="1")
        env.pop("SLURM_JOB_ID", None)
        completed = subprocess.run(["bash", str(bundle / "run.sh")], env=env, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        status = load_json(bundle / "results/direct/workflow_status.json")
        self.assertEqual((status["state"], status["backend"], status["device"]), ("complete", "direct", "cpu"))
        self.assertIsNone(status["job_id"])
        self.assertEqual(marker.read_text(), "setupsetup")
        self.assertTrue((bundle / "results/direct/workflow_review/workflow_summary.pdf").exists())
        again = subprocess.run(["bash", str(bundle / "run.sh")], env=env, capture_output=True, text=True)
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(load_json(bundle / "results/direct/workflow_status.json")["state"], "complete")


if __name__ == "__main__":
    unittest.main()
