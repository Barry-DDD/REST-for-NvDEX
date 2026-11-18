"""Shared environment checks and shell setup for direct and SLURM launchers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

from workflow_config import interpreter_path, runtime_versions


def setup_fingerprint(config):
    path = config["execution"]["setup_script"]
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Environment setup script is missing: {path}")
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def check_setup_fingerprint(plan):
    if setup_fingerprint(plan["config"]) != plan.get("setup_script"):
        raise ValueError("Environment setup script changed; prepare a fresh bundle.")


def execution_environment(config):
    env = dict(os.environ)
    execution = config["execution"]
    if execution["device"] == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif execution["backend"] == "direct" and execution["gpu"] is not None:
        env["CUDA_VISIBLE_DEVICES"] = execution["gpu"]
    env.setdefault("OMP_NUM_THREADS", "1")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def verify_setup_environment(config, root):
    """Probe the exact selected interpreter after sourcing the user's trusted script."""
    setup = setup_fingerprint(config)
    if setup is None:
        return None
    module_dir = str(Path(__file__).resolve().parent)
    code = ("import sys,json; sys.path.insert(0," + repr(module_dir) + "); "
            "from workflow_config import runtime_versions; "
            "print('NVDEX_ENV_PROBE='+json.dumps({'python':sys.executable,'environment':runtime_versions()}))")
    body = "\n".join([*shell_environment_lines(config, setup, root), 'exec "$@"'])
    command = ["bash", "--noprofile", "--norc", "-c", body,
               "nvdex-env-check", config["python"], "-c", code]
    try:
        completed = subprocess.run(command, cwd=root, env=execution_environment(config),
                                   capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as error:
        raise ValueError("Environment setup probe exceeded 60 seconds; use a noninteractive setup script.") from error
    if completed.returncode:
        raise ValueError("Environment setup probe failed: " + completed.stderr[-1500:])
    probes = [line.split("=", 1)[1] for line in completed.stdout.splitlines() if line.startswith("NVDEX_ENV_PROBE=")]
    if len(probes) != 1:
        raise ValueError("The setup script did not produce a valid runtime probe.")
    probe = json.loads(probes[0])
    if interpreter_path(probe["python"]) != interpreter_path(config["python"]) or probe["environment"] != runtime_versions():
        raise ValueError("Setup changes the prepared runtime. Source it first, then prepare using that environment's Python.")
    if setup_fingerprint(config) != setup:
        raise ValueError("Environment setup script changed during its probe.")
    return setup


def shell_environment_lines(config, setup, root):
    """Check source content before executing it, then constrain device selection."""
    lines = ["set -euo pipefail", "cd " + shlex.quote(str(root))]
    execution = config["execution"]
    if execution["backend"] == "slurm":
        lines += ['nvdex_allocated_cuda_present="${CUDA_VISIBLE_DEVICES+x}"',
                  'nvdex_allocated_cuda_value="${CUDA_VISIBLE_DEVICES-}"']
    if setup is not None:
        lines += [f"SETUP_SCRIPT={shlex.quote(setup['path'])}",
                  'setup_hash=$(sha256sum < "$SETUP_SCRIPT")',
                  f'if [[ "${{setup_hash:0:64}}" != {shlex.quote(setup["sha256"])} ]]; then',
                  '  echo "Environment setup script changed; prepare a fresh bundle." >&2',
                  "  exit 1", "fi", "set +u", 'source "$SETUP_SCRIPT"',
                  "nvdex_setup_status=$?", "set -euo pipefail",
                  'if [[ "$nvdex_setup_status" != 0 ]]; then exit "$nvdex_setup_status"; fi',
                  "export NVDEX_WORKFLOW_SETUP_HASH=" + shlex.quote(setup["sha256"])]
    if execution["backend"] == "slurm":
        lines += ['if [[ "$nvdex_allocated_cuda_present" == x ]]; then',
                  '  export CUDA_VISIBLE_DEVICES="$nvdex_allocated_cuda_value"',
                  "else", "  unset CUDA_VISIBLE_DEVICES", "fi"]
    if execution["device"] == "cpu":
        lines.append("export CUDA_VISIBLE_DEVICES=''")
    elif execution["backend"] == "direct" and execution["gpu"] is not None:
        lines.append("export CUDA_VISIBLE_DEVICES=" + shlex.quote(execution["gpu"]))
    lines += ['export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"', "export PYTHONDONTWRITEBYTECODE=1"]
    return lines
