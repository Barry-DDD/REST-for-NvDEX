"""Task definitions for TPC graph learning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


REGRESSION_TASK = "regression_z"
SUPPORTED_TASKS = (REGRESSION_TASK, "regression_xyz", "regression_A", "regression_theta", "binary_id", "multiclass_id")
VERTEX_TARGET_KEYS = tuple(f"primary_origin_{axis}" for axis in "xyz")
VERTEX_METRIC_KEYS = ("mae_xyz", "rmse_xyz", "bias_xyz", "residual_std_xyz", "mean_distance_mm",
                      "median_distance_mm", "r68_mm", "r95_mm", "rmse_3d_mm")
BINARY_ID_MAP = {1: 1, 21: 0, 22: 0, 23: 0, 24: 0}
MULTICLASS_ID_MAP = {1: 0, 21: 1, 22: 2, 23: 3, 24: 4}
THETA_DEG_CENTER = 90.0


@dataclass(frozen=True)
class TaskSpec:
    """Describe target handling, output shape, loss, and metrics for one task."""

    name: str
    task_type: str
    target_key: str
    num_outputs: int
    label_map: dict[int, int] | None = None
    positive_label: int | None = None

    @property
    def is_classification(self) -> bool:
        return self.task_type == "classification"

    @property
    def is_regression(self) -> bool:
        return self.task_type == "regression"

    def build_loss(self):
        """Return the default criterion for this task."""
        import torch.nn as nn

        if self.is_classification:
            return nn.CrossEntropyLoss()
        return nn.SmoothL1Loss()

    def map_raw_labels(self, raw_values):
        """Map raw HDF5 target values to train labels."""
        raw_values = np.asarray(raw_values).reshape(-1)
        if self.label_map is None:
            return raw_values.astype(np.float32, copy=False)

        labels = np.empty(raw_values.shape[0], dtype=np.int64)
        for i, value in enumerate(raw_values):
            raw_id = int(value)
            if raw_id not in self.label_map:
                allowed = ", ".join(str(k) for k in sorted(self.label_map))
                raise ValueError(f"Unsupported id {raw_id}. Supported ids for {self.name}: {allowed}.")
            labels[i] = int(self.label_map[raw_id])
        return labels


def get_task_spec(task: str) -> TaskSpec:
    """Return a task definition from its CLI name."""
    task = str(task)
    if task == "regression_z":
        return TaskSpec(
            name=task,
            task_type="regression",
            target_key="primary_origin_z",
            num_outputs=1,
        )
    if task == "regression_xyz":
        return TaskSpec(name=task, task_type="regression", target_key="primary_origin", num_outputs=3)
    if task == "regression_A":
        return TaskSpec(name=task, task_type="regression", target_key="A", num_outputs=1)
    if task == "regression_theta":
        return TaskSpec(name=task, task_type="regression", target_key="cos_theta", num_outputs=1)
    if task == "binary_id":
        return TaskSpec(
            name=task,
            task_type="classification",
            target_key="id",
            num_outputs=2,
            label_map=BINARY_ID_MAP,
            positive_label=1,
        )
    if task == "multiclass_id":
        return TaskSpec(
            name=task,
            task_type="classification",
            target_key="id",
            num_outputs=5,
            label_map=MULTICLASS_ID_MAP,
        )
    raise ValueError(f"task must be one of: {', '.join(SUPPORTED_TASKS)}.")


@dataclass(frozen=True)
class TargetTransform:
    """Affine target transform shared by the dataset and validation."""

    mode: str
    offset: float
    scale: float
    unit: str
    cdf_values: np.ndarray | None = None
    cdf_probabilities: np.ndarray | None = None

    def forward(self, value):
        if self.mode == "cdf":
            return np.asarray(np.interp(value, self.cdf_values, self.cdf_probabilities,
                                        left=0.0, right=1.0), dtype=np.float32)
        value = (np.asarray(value, dtype=np.float32) - self.offset) / self.scale
        if self.mode == "minus_one_one":
            value = np.clip(value, -1.0, 1.0)
        elif self.mode == "zero_one":
            value = np.clip(value, 0.0, 1.0)
        return value

    def inverse(self, value):
        # Arithmetic works on both NumPy arrays and torch tensors.
        if self.mode == "cdf":
            return np.asarray(np.interp(value, self.cdf_probabilities, self.cdf_values),
                              dtype=np.float32)
        return value * self.scale + self.offset


def get_target_transform(task="regression_z", target_mode=None, target_abs_max=845.0):
    """Resolve task defaults and reject incompatible target modes."""
    spec = get_task_spec(task)
    if task == "regression_A":
        if target_mode not in (None, "standard"):
            raise ValueError("regression_A requires target_mode=standard.")
        return TargetTransform("standard", 0.4113, 0.2663, "dimensionless")
    if task == "regression_theta":
        if target_mode == "angle":
            # Opening angle in degrees, mapped linearly from [0, 180] to [-1, 1].
            return TargetTransform("angle", THETA_DEG_CENTER, THETA_DEG_CENTER, "deg")
        if target_mode not in (None, "cdf"):
            raise ValueError("regression_theta requires target_mode=cdf or angle.")
        return TargetTransform("cdf", 0.0, 1.0, "dimensionless",
                               np.asarray([-1.0, 1.0]), np.asarray([0.0, 1.0]))
    if spec.is_classification:
        return TargetTransform("none", 0.0, 1.0, "class")
    mode = target_mode or "minus_one_one"
    if not np.isfinite(target_abs_max) or target_abs_max <= 0:
        raise ValueError("target_abs_max must be finite and positive.")
    if mode == "minus_one_one":
        return TargetTransform(mode, 0.0, float(target_abs_max), "mm")
    if mode == "zero_one":
        return TargetTransform(mode, -float(target_abs_max), 2.0 * target_abs_max, "mm")
    if mode == "none":
        return TargetTransform(mode, 0.0, 1.0, "mm")
    raise ValueError("Vertex target_mode must be minus_one_one, zero_one, or none.")


def read_vertex_targets(h5):
    """Read finite truth vertices in fixed x, y, z order without filling missing axes."""
    n_events = h5["events/hit_count"].shape[0]
    columns = []
    for key in VERTEX_TARGET_KEYS:
        path = f"events/{key}"
        if path not in h5:
            raise KeyError(f"regression_xyz requires {path}.")
        values = np.asarray(h5[path], dtype=np.float32)
        if values.shape != (n_events,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{path} must contain one finite value per event.")
        columns.append(values)
    return np.stack(columns, axis=1)


def vertex_metrics(labels, preds):
    """Return per-axis errors and Euclidean vertex-distance metrics in mm."""
    labels, preds = np.asarray(labels, dtype=np.float64), np.asarray(preds, dtype=np.float64)
    if labels.ndim != 2 or labels.shape[1] != 3 or labels.shape[0] == 0 or preds.shape != labels.shape:
        raise ValueError("Vertex labels and predictions must have matching nonempty [N, 3] shapes.")
    if not np.all(np.isfinite(labels)) or not np.all(np.isfinite(preds)):
        raise ValueError("Vertex labels and predictions must be finite.")
    residual = preds - labels
    distance = np.linalg.norm(residual, axis=1)
    return dict(mae_xyz=np.mean(np.abs(residual), axis=0),
                rmse_xyz=np.sqrt(np.mean(residual ** 2, axis=0)),
                bias_xyz=np.mean(residual, axis=0), residual_std_xyz=np.std(residual, axis=0),
                mean_distance_mm=float(np.mean(distance)), median_distance_mm=float(np.median(distance)),
                r68_mm=float(np.quantile(distance, 0.68)), r95_mm=float(np.quantile(distance, 0.95)),
                rmse_3d_mm=float(np.sqrt(np.mean(distance ** 2))))


def theta_target_key(target_mode):
    """Return the raw target name used by regression_theta for a target mode."""
    return "theta_deg" if target_mode == "angle" else "cos_theta"


def cos_theta_to_deg(cos_theta):
    """Convert opening-angle cosines to degrees in [0, 180]."""
    return np.degrees(np.arccos(np.clip(np.asarray(cos_theta, dtype=np.float64), -1.0, 1.0))).astype(np.float32)


def fit_cdf_transform(values):
    """Fit a monotonic empirical CDF using mid-ranks for repeated values."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.all(np.isfinite(values) & (values >= -1.0) & (values <= 1.0)):
        raise ValueError("CDF fitting requires at least two finite cos(theta) values in [-1, 1].")
    unique, counts = np.unique(values, return_counts=True)
    probabilities = (np.cumsum(counts) - 0.5 * counts) / values.size
    return TargetTransform("cdf", 0.0, 1.0, "dimensionless",
                           unique.astype(np.float64), probabilities.astype(np.float64))


def read_A_targets(h5):
    """Validate all events and derive A from exactly two primary electrons."""
    paths = ("events/primary_start", "events/primary_count",
             "primaries/primary_energy", "primaries/primary_particle_name")
    for path in paths:
        if path not in h5:
            raise KeyError(f"regression_A requires {path}.")
        if h5[path].ndim != 1:
            raise ValueError(f"{path} must be one-dimensional.")
    starts = np.asarray(h5[paths[0]])
    counts = np.asarray(h5[paths[1]])
    n_events = h5["events/hit_count"].shape[0]
    if starts.size != n_events or counts.size != n_events:
        raise ValueError("Primary offsets/counts must match the event count.")
    if starts.dtype.kind not in "iu" or counts.dtype.kind not in "iu":
        raise ValueError("Primary offsets/counts must be integers.")
    energies = np.asarray(h5[paths[2]], dtype=np.float64)
    names = np.asarray(h5[paths[3]]).astype(str)
    if names.size != energies.size:
        raise ValueError("Primary names and energies have different lengths.")
    valid = (counts == 2) & (starts >= 0) & (starts <= energies.size - 2)
    if not np.all(valid):
        raise ValueError(f"Invalid two-primary count/offset at event rows {np.flatnonzero(~valid)[:10].tolist()}.")
    starts = starts.astype(np.int64)
    E1, E2 = energies[starts], energies[starts + 1]
    valid = ((names[starts] == "e-") & (names[starts + 1] == "e-")
             & np.isfinite(E1) & np.isfinite(E2) & (E1 >= 0) & (E2 >= 0)
             & np.isfinite(E1 + E2) & ((E1 + E2) > 0))
    if not np.all(valid):
        raise ValueError(f"Invalid primary electron names/energies at event rows {np.flatnonzero(~valid)[:10].tolist()}.")
    A = np.abs(E1 - E2) / (E1 + E2)
    if not np.all(np.isfinite(A) & (A >= 0) & (A <= 1)):
        raise ValueError("A must be finite and within [0, 1].")
    return A.astype(np.float32), E1.astype(np.float32), E2.astype(np.float32)


def read_theta_targets(h5):
    """Validate two-electron directions and derive their opening-angle cosine."""
    paths = ("events/primary_start", "events/primary_count",
             "primaries/primary_dir_x", "primaries/primary_dir_y", "primaries/primary_dir_z",
             "primaries/primary_particle_name")
    for path in paths:
        if path not in h5:
            raise KeyError(f"regression_theta requires {path}.")
        if h5[path].ndim != 1:
            raise ValueError(f"{path} must be one-dimensional.")
    starts = np.asarray(h5[paths[0]])
    counts = np.asarray(h5[paths[1]])
    n_events = h5["events/hit_count"].shape[0]
    direction_size = h5[paths[2]].shape[0]
    if starts.size != n_events or counts.size != n_events:
        raise ValueError("Primary offsets/counts must match the event count.")
    if starts.dtype.kind not in "iu" or counts.dtype.kind not in "iu":
        raise ValueError("Primary offsets/counts must be integers.")
    valid = (counts == 2) & (starts >= 0) & (starts <= direction_size - 2)
    if not np.all(valid):
        raise ValueError(f"Invalid two-primary count/offset at event rows {np.flatnonzero(~valid)[:10].tolist()}.")
    starts = starts.astype(np.int64)
    names = np.asarray(h5[paths[5]]).astype(str)
    components = [np.asarray(h5[path], dtype=np.float64) for path in paths[2:5]]
    if any(values.size != direction_size for values in components) or names.size != direction_size:
        raise ValueError("Primary names and direction components have different lengths.")
    d1 = np.stack([values[starts] for values in components], axis=1)
    d2 = np.stack([values[starts + 1] for values in components], axis=1)
    n1, n2 = np.linalg.norm(d1, axis=1), np.linalg.norm(d2, axis=1)
    valid = ((names[starts] == "e-") & (names[starts + 1] == "e-")
             & np.all(np.isfinite(d1), axis=1) & np.all(np.isfinite(d2), axis=1)
             & np.isfinite(n1) & np.isfinite(n2) & (n1 > 0) & (n2 > 0))
    if not np.all(valid):
        raise ValueError(f"Invalid primary electron names/directions at event rows {np.flatnonzero(~valid)[:10].tolist()}.")
    cos_theta = np.einsum("ij,ij->i", d1, d2) / (n1 * n2)
    return np.clip(cos_theta, -1.0, 1.0).astype(np.float32)
