"""Resolve explicit experiments against the repository's registered tasks."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys


VERSION = 1
DEPLOYMENT_PATH = Path(__file__).resolve().parents[1] / "assets/ihep-deployment.json"
IHEP = json.loads(DEPLOYMENT_PATH.read_text(encoding="utf-8"))
COMMON = dict(epochs=30, batch_size=64, learning_rate=3e-4, weight_decay=5e-5,
              features=["x", "y", "z", "energy", "log_energy"], radius_mm=2.0,
              feature_norm_mode="standard", xy_abs_max=445.0, z_abs_max=850.0,
              target_abs_max=845.0, event_features=[], keep_empty_events=False,
              hidden_channels=128, num_layers=3, heads=4, head_hidden_channels=128,
              dropout=0.1, activation="gelu", num_workers=4, grad_clip=1.0,
              seed=42, val_fraction=0.2, log_interval=50)
PRESETS = {
    "theta-angle-v1": ("regression_theta", "angle"),
    "theta-cdf-v1": ("regression_theta", "cdf"),
    "A-v1": ("regression_A", "standard"),
    "xyz-v1": ("regression_xyz", "minus_one_one"),
    "z-v1": ("regression_z", "minus_one_one"),
    "binary-v1": ("binary_id", "none"),
    "multiclass-v1": ("multiclass_id", "none"),
}
RESOURCE_DEFAULTS = IHEP["resources"]
CODE_FILES = ["tpc_GAT/task_spec.py", "tpc_GAT/tpc_graph_dataset.py",
              "tpc_GAT/gat_model.py", "tpc_GAT/run.py", "tpc_GAT/utils.py"]


def target_label(task, mode, component=0):
    """Use physical axis labels rather than implementation-specific component names."""
    if task == "regression_theta":
        return "Opening angle theta [deg]" if mode == "angle" else "cos(theta)"
    if task == "regression_A":
        return "Energy asymmetry A"
    if task in {"regression_z", "regression_xyz"}:
        axis = "xyz"[component] if task == "regression_xyz" else "z"
        return f"Vertex {axis} [mm]"
    return "Raw event class ID"


def project_root(value=None):
    """Find the project independently of the installed skill location."""
    candidates = [Path(value)] if value else [Path.cwd(), *Path.cwd().parents, Path(IHEP["project"])]
    for candidate in candidates:
        if (candidate / "tpc_GAT/task_spec.py").is_file():
            root = candidate.resolve()
            sys.path.insert(0, str(root / "tpc_GAT"))
            return root
    raise ValueError("Specify --project pointing to the NvDEx GAT repository.")


def load_json(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_keys)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def interpreter_path(value):
    """Keep venv symlinks distinct instead of equating their underlying binaries."""
    return os.path.normcase(os.path.abspath(str(value)))


def source_hashes(root):
    """Fingerprint the exact training and workflow sources without requiring Git."""
    paths = [root / name for name in CODE_FILES]
    paths += sorted(Path(__file__).parent.glob("*.py"))
    paths.append(DEPLOYMENT_PATH)
    paths += [root / "results" / name for name in
              ("plot_A_result.py", "plot_theta_result.py", "plot_vertex_result.py")]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def runtime_versions():
    from importlib.metadata import PackageNotFoundError, version
    try:
        packages = {name: version(name) for name in
                    ("numpy", "h5py", "torch", "torch-geometric", "matplotlib", "scienceplots", "scikit-learn")}
    except PackageNotFoundError as error:
        raise ValueError(f"Missing Python dependency: {error}. Activate a compatible ML environment.") from error
    return dict(python=sys.version, executable=interpreter_path(sys.executable),
                prefix=sys.prefix, base_prefix=sys.base_prefix, packages=packages)


def resolve(request, root):
    """Return a complete configuration or a structured list of missing decisions."""
    import math
    import re
    from task_spec import SUPPORTED_TASKS, get_target_transform

    if not isinstance(request, dict):
        raise ValueError("The request must be a JSON object.")
    allowed = {"schema_version", "preset", "task", "infile", "vertex_source", "target_mode",
               "parameters", "resources", "python", "deployment", "execution"}
    if set(request) - allowed:
        raise ValueError(f"Unknown request keys: {sorted(set(request) - allowed)}")
    if type(request.get("schema_version", VERSION)) is not int or request.get("schema_version", VERSION) != VERSION:
        raise ValueError("Unsupported request schema_version.")
    deployment = request.get("deployment", "ihep")
    if deployment not in ("ihep", "current"):
        raise ValueError("deployment must be ihep or current.")
    execution = copy.deepcopy(IHEP["execution"])
    if deployment == "current":
        execution["backend"] = "direct"
    overrides = request.get("execution", {})
    if not isinstance(overrides, dict) or set(overrides) - set(execution):
        raise ValueError("Unknown execution options; use backend, device, gpu, setup_script.")
    execution.update(overrides)
    if execution["backend"] not in ("direct", "slurm") or execution["device"] not in ("cuda", "cpu"):
        raise ValueError("execution.backend must be direct/slurm and device must be cuda/cpu.")
    if execution["backend"] == "slurm" and (execution["device"] != "cuda" or execution["gpu"] is not None):
        raise ValueError("SLURM assigns the CUDA GPU; do not set execution.gpu or select CPU.")
    if execution["gpu"] is not None:
        if not isinstance(execution["gpu"], str) or not re.fullmatch(r"[0-9]+", execution["gpu"]) or execution["device"] != "cuda":
            raise ValueError("execution.gpu must be one CUDA device index string, for example '0'.")
    setup = execution["setup_script"]
    if setup is not None:
        if not isinstance(setup, str) or not setup or any(c in setup for c in "\n\r\x00"):
            raise ValueError("setup_script must be a path without control characters.")
        setup_path = Path(setup).expanduser()
        execution["setup_script"] = str((setup_path if setup_path.is_absolute() else root/setup_path).resolve())
    preset = request.get("preset")
    if preset is not None and (not isinstance(preset, str) or preset not in PRESETS):
        raise ValueError(f"Unknown preset: {preset}")
    base_task, base_mode = PRESETS[preset] if preset else (None, None)
    task = request.get("task", base_task)
    if task is not None and task not in SUPPORTED_TASKS:
        raise ValueError(f"Unregistered task: {task}")
    if preset and task != base_task:
        raise ValueError("The explicit task conflicts with the preset.")
    parameters = copy.deepcopy(request.get("parameters", {}))
    if not isinstance(parameters, dict) or set(parameters) - set(COMMON):
        raise ValueError("parameters contains unsupported training options.")
    missing = []
    if task is None:
        missing.append(dict(field="task", choices=list(SUPPORTED_TASKS)))
    if not request.get("infile"):
        missing.append(dict(field="infile", reason="Provide the server HDF5 path."))
    vertex = request.get("vertex_source")
    if vertex is None and task in {"regression_A", "regression_theta"}:
        missing.append(dict(field="vertex_source", choices=["truth", "centroid", "none"],
                            reason="Choose whether truth information is available to this experiment."))
    elif vertex is None:
        vertex = "none"
    if vertex is not None and vertex not in {"truth", "centroid", "none"}:
        raise ValueError("vertex_source must be truth, centroid, or none.")
    if vertex == "centroid" and task not in {"regression_A", "regression_theta"}:
        raise ValueError("centroid is supported only for A/theta.")
    if vertex == "truth" and task not in {"regression_A", "regression_theta"}:
        raise ValueError("This workflow permits truth vertex inputs only for explicit A/theta studies.")
    mode = request.get("target_mode", base_mode)
    if task == "regression_theta" and mode is None:
        missing.append(dict(field="target_mode", choices=["angle", "cdf"]))
    if not preset:
        for name in ("features", "radius_mm"):
            if name not in parameters:
                missing.append(dict(field=f"parameters.{name}", reason="Choose explicitly or name a preset."))
    if missing:
        return dict(status="needs_input", questions=missing)
    for name in ("infile", "python"):
        if name in request and (not isinstance(request[name], str) or not request[name] or any(c in request[name] for c in "\n\r\x00")):
            raise ValueError(f"{name} must be a nonempty path string without control characters.")
    params = copy.deepcopy(COMMON)
    if task in {"binary_id", "multiclass_id", "regression_z"}:
        params["features"] = ["x", "y", "energy", "log_energy"]
    params.update(parameters)
    for name, default in COMMON.items():
        value = params[name]
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean.")
        elif isinstance(default, int):
            if type(value) is not int or (value < 0 if name in {"num_workers", "seed"} else value <= 0):
                raise ValueError(f"Invalid integer parameter {name}.")
        elif isinstance(default, float):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite numeric data.")
            if name not in {"weight_decay", "grad_clip", "dropout"} and value <= 0:
                raise ValueError(f"{name} must be positive.")
        elif isinstance(default, list):
            if not isinstance(value, list) or any(not isinstance(x, str) or not x or "/" in x for x in value):
                raise ValueError(f"{name} must be a list of field names.")
            if len(set(value)) != len(value):
                raise ValueError(f"Duplicate entries in {name}.")
    if not params["features"] or not 0 < params["val_fraction"] < 1 or not 0 <= params["dropout"] < 1:
        raise ValueError("Invalid features, val_fraction, or dropout.")
    if params["weight_decay"] < 0 or params["activation"] not in {"relu", "gelu", "silu"}:
        raise ValueError("Invalid weight_decay or activation.")
    if params["feature_norm_mode"] not in {"standard", "minmax", "standard_minmax", "none"}:
        raise ValueError("Unknown feature_norm_mode.")
    if any(x.startswith(("primary_", "g4_")) or x == "id" for x in params["event_features"]):
        raise ValueError("Truth/label fields cannot be event input features in this workflow.")
    transform = get_target_transform(task, mode, params["target_abs_max"])
    if task in {"binary_id", "multiclass_id"} and mode not in (None, "none"):
        raise ValueError("Classification requires target_mode=none.")
    infile = Path(request["infile"]).expanduser()
    if not infile.is_absolute():
        infile = root / infile
    python = Path(request.get("python", IHEP["python"] if deployment == "ihep" else sys.executable)).expanduser()
    if not python.is_absolute():
        raise ValueError("python must be an absolute executable path on the server.")
    resources = copy.deepcopy(RESOURCE_DEFAULTS)
    overrides = request.get("resources", {})
    if not isinstance(overrides, dict) or set(overrides) - set(resources):
        raise ValueError("Unknown resource options; the first version supports one GPU on one node.")
    resources.update(overrides)
    if execution["backend"] == "direct" and overrides:
        raise ValueError("SLURM resources do not apply to direct execution; remove resources.")
    if execution["backend"] == "slurm" and (type(resources["cpus"]) is not int or resources["cpus"] < params["num_workers"] + 1):
        raise ValueError("resources.cpus must cover one main process plus num_workers.")
    for name in ("partition", "account", "qos", "memory", "time"):
        if not isinstance(resources[name], str) or not re.fullmatch(r"[A-Za-z0-9_:.-]+", resources[name]):
            raise ValueError(f"Unsafe or invalid SLURM resource: {name}")
    if not re.fullmatch(r"[1-9][0-9]*[KMGTPkmgpt]?", resources["memory"]):
        raise ValueError("memory must be a positive SLURM memory amount, for example 64G.")
    if not re.fullmatch(r"(?:[0-9]+-)?[0-9]{1,3}:[0-5][0-9]:[0-5][0-9]", resources["time"]):
        raise ValueError("time must use HH:MM:SS or D-HH:MM:SS.")
    origins = {key: ("request" if key in parameters else f"preset:{preset}" if preset else "workflow-default")
               for key in params}
    return dict(status="resolved", schema_version=VERSION, deployment=deployment, execution=execution, preset=preset, task=task,
                target_mode=transform.mode, vertex_source=vertex, infile=str(infile.resolve()),
                python=str(python), parameters=params, resources=resources if execution["backend"] == "slurm" else {}, parameter_origins=origins,
                choice_origins={key: "request" if key in request else f"preset:{preset}" if preset and key in {"task", "target_mode"}
                                else f"deployment:{deployment}" if key == "python" else "workflow-default"
                                for key in ("task", "target_mode", "vertex_source", "python")})


def training_arguments(config, outdir):
    """Build argv without shell interpolation or implicit training defaults."""
    result = ["--infile", config["infile"], "--outdir", str(outdir), "--task", config["task"],
              "--target_mode", config["target_mode"]]
    for name, value in config["parameters"].items():
        if isinstance(value, bool):
            if value:
                result.append(f"--{name}")
        elif isinstance(value, list):
            if value:
                result.extend([f"--{name}", *value])
        else:
            result.extend([f"--{name}", str(value)])
    if config["vertex_source"] != "none":
        result.append("--vertex_relative" if config["vertex_source"] == "truth" else "--vertex_centroid")
    return result
