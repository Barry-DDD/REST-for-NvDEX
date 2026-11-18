"""
TPC hit graph dataset for regression and event classification.

The input HDF5 file follows the schema produced by tpc_feature_extraction.
Each returned event is a PyG Data graph with variable-size hit nodes, radius
edges in the global xy plane, edge attributes, regression target, and event
metadata for later evaluation.
"""

from __future__ import annotations

from collections.abc import Iterable

import h5py
import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from task_spec import (cos_theta_to_deg, get_task_spec, get_target_transform, read_A_targets,
                       read_theta_targets, read_vertex_targets, theta_target_key, VERTEX_TARGET_KEYS)


class TPCData(Data):
    """PyG Data subclass that keeps scalar metadata as plain metadata."""

    def __inc__(self, key, value, *args, **kwargs):
        if key in {"event_index"}:
            return 0
        return super().__inc__(key, value, *args, **kwargs)


class TPCGraphDataset(Dataset):
    """
    Dataset for variable-hit TPC HDF5 files.

    Returned Data fields:
        x: normalized node features, default [x, y, energy, log_energy]
        pos: raw xy hit coordinates in mm, shape [n_hit, 2]
        edge_index: directed radius graph edges, shape [2, n_edge]
        edge_attr: [dx/r, dy/r, distance/r, dE, same_module]
        y: transformed target or classification label, shape [1] or [1, 3] for XYZ
        target_raw: raw task target, shape [1] or [1, 3] for XYZ
        event-level and primary-level metadata for later evaluation
    """

    DEFAULT_NODE_FEATURES = ("x", "y", "energy", "log_energy")
    # Extra node features expressed in the vertex frame (appended when
    # truth or centroid frame is enabled): normalized displacement (v_dx, v_dy, v_dz),
    # normalized radial distance (v_r), and the unit direction from the vertex
    # to the hit (v_ux, v_uy, v_uz). Intended for A / theta reconstruction.
    VERTEX_RELATIVE_FEATURES = ("v_dx", "v_dy", "v_dz", "v_r", "v_ux", "v_uy", "v_uz")
    EVENT_INT_KEYS = (
        "n_hits",
        "n_hits_original",
        "n_primaries",
        "n_tracks",
        "id",
    )
    EVENT_FLOAT_KEYS = (
        "total_energy",
        "total_energy_filtered",
        "g4_total_deposited_energy",
        "g4_sensitive_volume_energy",
    )
    PRIMARY_FLOAT_KEYS = (
        "primary_energy",
        "primary_dir_x",
        "primary_dir_y",
        "primary_dir_z",
    )

    def __init__(
        self,
        h5_path: str,
        node_feature_names: Iterable[str] = DEFAULT_NODE_FEATURES,
        radius_mm: float = 20.0,
        task: str = "regression_z",
        target_key: str = "primary_origin_z",
        target_mode: str | None = None,
        target_abs_max: float = 845.0,
        xy_abs_max: float = 445.0,
        z_abs_max: float = 850.0,
        feature_norm_mode: str = "standard",
        event_feature_keys: Iterable[str] = (),
        vertex_relative: bool = False,
        drop_empty_events: bool = True,
        eps: float = 1e-6,
        transform=None,
        vertex_centroid: bool = False,
    ):
        super().__init__(None, transform)
        self.file_path = str(h5_path)
        self.node_feature_names = tuple(node_feature_names)
        self.vertex_relative = bool(vertex_relative)
        self.vertex_centroid = bool(vertex_centroid)
        if self.vertex_relative and self.vertex_centroid:
            raise ValueError("vertex_relative and vertex_centroid are mutually exclusive.")
        self.use_vertex_frame = self.vertex_relative or self.vertex_centroid
        self.vertex_source = "centroid" if self.vertex_centroid else "truth" if self.vertex_relative else "none"
        self.vertex_relative_dim = len(self.VERTEX_RELATIVE_FEATURES) if self.use_vertex_frame else 0
        # Total width of the per-node feature vector produced by the dataset.
        # Used by run.py to size the model's input layer.
        self.node_feature_dim = len(self.node_feature_names) + self.vertex_relative_dim
        self.radius_mm = float(radius_mm)
        self.task_spec = get_task_spec(task)
        self.task = self.task_spec.name
        if self.vertex_centroid and self.task not in {"regression_A", "regression_theta"}:
            raise ValueError("vertex_centroid is supported only for regression_A / regression_theta.")
        if self.task == "regression_xyz" and self.vertex_relative:
            raise ValueError("regression_xyz forbids vertex_relative truth inputs.")
        self.target_key = self.task_spec.target_key if self.task != "regression_z" else str(target_key)
        self.target_transform = get_target_transform(self.task, target_mode, target_abs_max)
        self.target_mode = self.target_transform.mode
        if self.task == "regression_theta":
            self.target_key = theta_target_key(self.target_mode)
        self.target_abs_max = float(target_abs_max)
        self.xy_abs_max = float(xy_abs_max)
        self.z_abs_max = float(z_abs_max)
        self.feature_norm_mode = str(feature_norm_mode)
        self.event_feature_keys = tuple(event_feature_keys)
        self.drop_empty_events = bool(drop_empty_events)
        self.eps = float(eps)
        self.h5file = None

        if self.radius_mm <= 0.0:
            raise ValueError("radius_mm must be positive.")
        if self.target_abs_max <= 0.0:
            raise ValueError("target_abs_max must be positive.")
        if self.xy_abs_max <= 0.0:
            raise ValueError("xy_abs_max must be positive.")
        if self.z_abs_max <= 0.0:
            raise ValueError("z_abs_max must be positive.")
        self._validate_event_features()

        with h5py.File(self.file_path, "r") as h5:
            self.total_events = self._validate_h5(h5)
            self.stats_feature_names = self._decode_string_array(h5["stats/feature_names"][...])
            self.stats_index = {name: i for i, name in enumerate(self.stats_feature_names)}
            self.feature_mean = np.asarray(h5["stats/mean"], dtype=np.float32).reshape(-1)
            self.feature_std = np.asarray(h5["stats/std"], dtype=np.float32).reshape(-1)
            self.feature_min = np.asarray(h5["stats/min"], dtype=np.float32).reshape(-1)
            self.feature_max = np.asarray(h5["stats/max"], dtype=np.float32).reshape(-1)
            self._validate_feature_selection(h5)
            self._validate_vertex_inputs(h5)
            self.event_feature_dim = self._infer_event_feature_dim(h5)
            self.event_indices, self.hit_counts = self._build_event_index(h5)
            if self.task == "regression_xyz":
                self.raw_targets = read_vertex_targets(h5)[self.event_indices]
                if self.target_mode != "none" and np.any(np.abs(self.raw_targets) > self.target_abs_max):
                    raise ValueError("Vertex targets exceed target_abs_max; increase the scale to avoid clipping truth.")
            elif self.task == "regression_A":
                all_targets, E1, E2 = read_A_targets(h5)
                self.raw_targets = all_targets[self.event_indices]
                self.E1, self.E2 = E1[self.event_indices], E2[self.event_indices]
            elif self.task == "regression_theta":
                cos_theta = read_theta_targets(h5)[self.event_indices]
                self.raw_targets = cos_theta_to_deg(cos_theta) if self.target_mode == "angle" else cos_theta
            else:
                self.raw_targets = np.asarray(h5[f"events/{self.target_key}"][self.event_indices])
            self.labels = self._build_labels(self.raw_targets)

        self._prepare_feature_normalization_constants()
        self.edge_energy_scale = self._feature_std_or_one("energy")

        print("[TPCGraphDataset] Dataset initialized:")
        print(f"  File: {self.file_path}")
        print(f"  Total events in file: {self.total_events}")
        print(f"  Usable events: {len(self.event_indices)}")
        print(f"  Dropped empty-hit events: {self.total_events - len(self.event_indices)}")
        print(f"  Node features: {self.node_feature_names}")
        print(f"  Vertex frame: {self.vertex_source} (+{self.vertex_relative_dim} dims, total node dim {self.node_feature_dim})")
        print(f"  XY coordinate scale: +/- {self.xy_abs_max:.3f} mm -> [-1, 1]")
        print(f"  Z coordinate scale: +/- {self.z_abs_max:.3f} mm -> [-1, 1]")
        print(f"  Non-XY feature normalization mode: {self.feature_norm_mode}")
        print(f"  Event feature keys: {self.event_feature_keys}")
        print(f"  Event feature dimension: {self.event_feature_dim}")
        print(f"  Radius graph: {self.radius_mm:.3f} mm")
        print(f"  Task: {self.task}")
        if self.task_spec.is_classification:
            print(f"  Target: {self.target_key}, classes={self.task_spec.num_outputs}")
        else:
            print(f"  Target: {self.target_key}, mode={self.target_mode}, offset={self.target_transform.offset}, scale={self.target_transform.scale}, unit={self.target_transform.unit}")
        if len(self.hit_counts) > 0:
            print(
                "  Hit count min/median/mean/max: "
                f"{int(np.min(self.hit_counts))}/"
                f"{float(np.median(self.hit_counts)):.1f}/"
                f"{float(np.mean(self.hit_counts)):.1f}/"
                f"{int(np.max(self.hit_counts))}"
            )

    def _validate_h5(self, h5):
        """Validate the required HDF5 groups and datasets; return total event count."""
        for group_name in ("events", "hits", "stats"):
            if group_name not in h5:
                raise KeyError(f"HDF5 file does not contain group '{group_name}'.")
        derived_target = self.task in {"regression_A", "regression_theta", "regression_xyz"}
        for key in (("hit_start", "hit_count") if derived_target else ("hit_start", "hit_count", self.target_key)):
            if f"events/{key}" not in h5:
                raise KeyError(f"HDF5 file does not contain required dataset 'events/{key}'.")
        for key in ("x", "y", "energy", "log_energy", "module_id"):
            if f"hits/{key}" not in h5:
                raise KeyError(f"HDF5 file does not contain required dataset 'hits/{key}'.")
        if self.task == "regression_xyz" and "hits/z" not in h5:
            raise KeyError("regression_xyz requires hits/z for vertex baselines.")
        for key in ("feature_names", "mean", "std", "min", "max"):
            if f"stats/{key}" not in h5:
                raise KeyError(f"HDF5 file does not contain required dataset 'stats/{key}'.")

        total_events = int(h5["events/hit_count"].shape[0])
        if h5["events/hit_start"].shape[0] != total_events:
            raise ValueError("events/hit_start and events/hit_count have different lengths.")
        if not derived_target and h5[f"events/{self.target_key}"].shape[0] != total_events:
            raise ValueError(f"events/{self.target_key} length does not match event count.")
        return total_events

    def _validate_event_features(self):
        """Prevent labels and truth targets from being used as input features."""
        forbidden = {"id", self.target_key, self.task_spec.target_key}
        if self.task == "regression_xyz":
            forbidden.update(VERTEX_TARGET_KEYS)
            forbidden.update(key for key in self.event_feature_keys if key.startswith(("primary_", "g4_")))
        if self.task in {"regression_A", "regression_theta"}:
            forbidden.update({"E1", "E2", "primary_energy", "primary_start", "primary_count", "n_primaries"})
            forbidden.update(key for key in self.event_feature_keys if key.startswith(("primary_", "g4_")))
        leaked = sorted(forbidden.intersection(self.event_feature_keys))
        if leaked:
            raise ValueError(f"event_features cannot include task target fields: {leaked}.")

    def _validate_feature_selection(self, h5):
        """Validate selected node features against hits and stats."""
        for name in self.node_feature_names:
            if f"hits/{name}" not in h5:
                raise KeyError(f"Selected node feature 'hits/{name}' is missing.")
            if name not in ("x", "y", "z") and name not in self.stats_index:
                raise KeyError(f"Selected node feature '{name}' is missing from stats/feature_names.")
        n_stats = len(self.stats_feature_names)
        for stat_name, values in (
            ("mean", self.feature_mean),
            ("std", self.feature_std),
            ("min", self.feature_min),
            ("max", self.feature_max),
        ):
            if values.shape[0] != n_stats:
                raise ValueError(f"stats/{stat_name} length does not match stats/feature_names.")

    def _validate_vertex_inputs(self, h5):
        """Validate datasets required to build vertex-relative node features."""
        if not self.use_vertex_frame:
            return
        if "hits/z" not in h5:
            raise KeyError("Vertex-frame features require dataset 'hits/z'.")
        if self.vertex_centroid:
            return
        for axis in ("x", "y", "z"):
            path = f"events/primary_origin_{axis}"
            if path not in h5:
                raise KeyError(f"vertex_relative=True requires dataset '{path}'.")

    def _infer_event_feature_dim(self, h5):
        """Infer the flat event-level feature dimension."""
        dim = 0
        for key in self.event_feature_keys:
            path = f"events/{key}"
            if path not in h5:
                raise KeyError(f"Requested event feature '{path}' is missing.")
            sample_shape = np.asarray(h5[path][0]).shape
            dim += int(np.prod(sample_shape)) if sample_shape else 1
        return dim

    @staticmethod
    def _decode_string_array(values):
        """Decode an HDF5 string array to a Python string list."""
        decoded = []
        for value in values:
            if isinstance(value, bytes):
                decoded.append(value.decode("utf-8"))
            else:
                decoded.append(str(value))
        return decoded

    def _build_event_index(self, h5):
        """Build an index of events with at least one positive-energy hit."""
        hit_counts = np.asarray(h5["events/hit_count"], dtype=np.int64).reshape(-1)
        if self.drop_empty_events:
            event_indices = np.nonzero(hit_counts > 0)[0].astype(np.int64)
        else:
            event_indices = np.arange(hit_counts.shape[0], dtype=np.int64)
        if event_indices.size == 0:
            raise ValueError("No usable events were found. All events have zero filtered hits.")
        return event_indices, hit_counts[event_indices].astype(np.int64)

    def _build_labels(self, raw_targets):
        """Build cached train labels for splitters and diagnostics."""
        if self.task_spec.is_classification:
            return self.task_spec.map_raw_labels(raw_targets)
        return np.asarray(self.target_transform.forward(raw_targets), dtype=np.float32)

    def set_target_transform(self, target_transform):
        """Replace a fitted target transform before DataLoaders are created."""
        self.target_transform = target_transform
        self.target_mode = target_transform.mode
        self.labels = self._build_labels(self.raw_targets)

    def _prepare_feature_normalization_constants(self):
        """Precompute normalization constants for selected node features."""
        self.sel_mean = np.zeros(len(self.node_feature_names), dtype=np.float32)
        self.sel_std = np.ones(len(self.node_feature_names), dtype=np.float32)
        self.sel_min = np.zeros(len(self.node_feature_names), dtype=np.float32)
        self.sel_max = np.ones(len(self.node_feature_names), dtype=np.float32)
        for col, name in enumerate(self.node_feature_names):
            # Coordinates use fixed scales and do not require HDF5 statistics.
            if name in ("x", "y", "z"):
                continue
            idx = self.stats_index[name]
            self.sel_mean[col] = self.feature_mean[idx]
            self.sel_std[col] = self.feature_std[idx]
            self.sel_min[col] = self.feature_min[idx]
            self.sel_max[col] = self.feature_max[idx]
        self.sel_std = np.where(np.abs(self.sel_std) < self.eps, 1.0, self.sel_std).astype(np.float32)
        self.sel_range = np.where(np.abs(self.sel_max - self.sel_min) < self.eps, 1.0, self.sel_max - self.sel_min).astype(np.float32)
        # Precomputed constants for standard_minmax mode (standardized-space min/max/range).
        self.sel_min_std = ((self.sel_min - self.sel_mean) / self.sel_std).astype(np.float32)
        self.sel_max_std = ((self.sel_max - self.sel_mean) / self.sel_std).astype(np.float32)
        self.sel_range_std = np.where(
            np.abs(self.sel_max_std - self.sel_min_std) < self.eps,
            1.0,
            self.sel_max_std - self.sel_min_std,
        ).astype(np.float32)

    def len(self):
        return len(self.event_indices)

    def __len__(self):
        return len(self.event_indices)

    @property
    def file(self):
        """Open the HDF5 file lazily per DataLoader worker."""
        if self.h5file is None:
            self.h5file = h5py.File(self.file_path, "r")
        return self.h5file

    def get(self, idx):
        """Load one event and return a PyG Data object."""
        file_idx = int(self.event_indices[idx])
        start = int(self.file["events/hit_start"][file_idx])
        count = int(self.file["events/hit_count"][file_idx])

        hits = self._load_hits(start, count)
        if count == 0:
            hits = self._dummy_hits()

        raw_xy = np.stack([hits["x"], hits["y"]], axis=1).astype(np.float32, copy=False)
        raw_energy = hits["energy"].astype(np.float32, copy=False)
        module_id = hits["module_id"].astype(np.int64, copy=False)
        if self.vertex_centroid:
            vertex = self._geometric_center(hits)
        else:
            vertex = self._read_vertex(file_idx) if self.vertex_relative else None
        node_features = self._load_node_features(hits, vertex)
        edge_index, edge_attr = self._build_edges(raw_xy, raw_energy, module_id)

        raw_target_value = self.raw_targets[idx]
        raw_target = raw_target_value if self.task_spec.is_regression else int(raw_target_value)
        if self.task_spec.is_classification:
            label = int(self.task_spec.map_raw_labels(np.asarray([raw_target]))[0])
            y = torch.tensor([label], dtype=torch.long)
            target_raw = torch.tensor([raw_target], dtype=torch.long)
        else:
            y = torch.as_tensor(np.asarray(self.labels[idx]).copy(), dtype=torch.float32).reshape(-1)
            target_raw = torch.as_tensor(np.asarray(raw_target).copy(), dtype=torch.float32).reshape(-1)
            if self.task_spec.num_outputs > 1:
                y, target_raw = y.view(1, -1), target_raw.view(1, -1)
        data = TPCData(
            x=torch.from_numpy(node_features).float(),
            pos=torch.from_numpy(raw_xy).float(),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            target_raw=target_raw,
            event_index=torch.tensor([file_idx], dtype=torch.long),
            n_hit=torch.tensor([count], dtype=torch.long),
        )
        if self.use_vertex_frame:
            data.frame_vertex = torch.from_numpy(vertex).float().view(1, 3)
        if self.task == "regression_A":
            data.E1 = torch.tensor([self.E1[idx]], dtype=torch.float32)
            data.E2 = torch.tensor([self.E2[idx]], dtype=torch.float32)
        if self.task == "regression_xyz":
            xyz = np.column_stack([hits[axis] for axis in "xyz"]).astype(np.float64)
            weights = np.asarray(hits["energy"], dtype=np.float64)
            if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(weights)):
                raise ValueError(f"Nonfinite vertex baseline inputs at event {file_idx}.")
            center = xyz.mean(axis=0)
            energy_center = np.average(xyz, axis=0, weights=weights) if weights.sum() > 0 else center
            data.hit_centroid = torch.tensor(center, dtype=torch.float32).view(1, 3)
            data.energy_centroid = torch.tensor(energy_center, dtype=torch.float32).view(1, 3)
        self._attach_event_metadata(data, file_idx)
        self._attach_primary_metadata(data, file_idx)
        if self.event_feature_dim > 0:
            data.event_features = torch.from_numpy(self._load_event_features(file_idx)).float().view(1, -1)
        return data

    def _load_hits(self, start, count):
        """Load selected hit fields for one event."""
        if count <= 0:
            return {}
        needed = set(self.node_feature_names) | {"x", "y", "energy", "module_id"}
        if self.use_vertex_frame or self.task == "regression_xyz":
            needed |= {"z"}
        return {
            name: np.asarray(self.file[f"hits/{name}"][start : start + count])
            for name in needed
        }

    def _dummy_hits(self):
        """Create one zero-valued hit for explicitly retained empty events."""
        float_names = set(self.node_feature_names) | {"x", "y", "energy"}
        if self.use_vertex_frame or self.task == "regression_xyz":
            float_names |= {"z"}
        return {
            name: np.zeros(1, dtype=np.float32) for name in float_names
        } | {"module_id": np.zeros(1, dtype=np.int64)}

    def _load_node_features(self, hits, vertex=None):
        """Stack and normalize selected hit-level node features.

        When a vertex frame is enabled, vertex-frame features (normalized
        displacement, radial distance, and unit direction from the vertex to
        each hit) are appended as extra columns.
        """
        features = np.stack([np.asarray(hits[name], dtype=np.float32) for name in self.node_feature_names], axis=1)
        normalized = np.empty_like(features, dtype=np.float32)
        for col, name in enumerate(self.node_feature_names):
            values = features[:, col]
            if name in ("x", "y"):
                normalized[:, col] = values / self.xy_abs_max
            elif name == "z":
                normalized[:, col] = values / self.z_abs_max
            else:
                normalized[:, col] = self._normalize_non_xy_feature(values, col)
        features = normalized
        if self.use_vertex_frame:
            vertex_features = self._compute_vertex_relative_features(hits, vertex)
            features = np.concatenate([features, vertex_features], axis=1)
        return features.astype(np.float32, copy=False)

    @staticmethod
    def _geometric_center(hits):
        """Return the unweighted mean of retained hit coordinates in raw mm."""
        xyz = np.column_stack([hits[axis] for axis in "xyz"]).astype(np.float64)
        if xyz.shape[0] == 0 or not np.all(np.isfinite(xyz)):
            raise ValueError("The geometric center requires finite, nonempty XYZ hits.")
        return xyz.mean(axis=0).astype(np.float32)

    def _read_vertex(self, idx):
        """Read the truth primary vertex (origin) for one event, in mm."""
        return np.asarray(
            [
                float(self._read_event_scalar("primary_origin_x", idx, 0.0)),
                float(self._read_event_scalar("primary_origin_y", idx, 0.0)),
                float(self._read_event_scalar("primary_origin_z", idx, 0.0)),
            ],
            dtype=np.float32,
        )

    def _compute_vertex_relative_features(self, hits, vertex):
        """Build vertex-frame node features for one event.

        Returns an array of shape [n_hit, 7]:
          [v_dx, v_dy, v_dz, v_r, v_ux, v_uy, v_uz]
        where displacement components are normalized by the xy/z scales, v_r is
        the norm of that normalized displacement, and (v_ux, v_uy, v_uz) is the
        unit direction from the vertex to the hit (from raw mm displacement).
        """
        if vertex is None:
            raise ValueError("vertex_relative features require a vertex vector.")
        x = np.asarray(hits["x"], dtype=np.float32)
        y = np.asarray(hits["y"], dtype=np.float32)
        z = np.asarray(hits["z"], dtype=np.float32)
        raw_dx = x - float(vertex[0])
        raw_dy = y - float(vertex[1])
        raw_dz = z - float(vertex[2])

        # Normalized displacement (same scales as the x/y/z node features).
        v_dx = raw_dx / self.xy_abs_max
        v_dy = raw_dy / self.xy_abs_max
        v_dz = raw_dz / self.z_abs_max
        v_r = np.sqrt(v_dx * v_dx + v_dy * v_dy + v_dz * v_dz)

        # Unit direction from the raw mm displacement (guard the vertex hit).
        raw_norm = np.sqrt(raw_dx * raw_dx + raw_dy * raw_dy + raw_dz * raw_dz)
        safe_norm = np.maximum(raw_norm, self.eps)
        v_ux = raw_dx / safe_norm
        v_uy = raw_dy / safe_norm
        v_uz = raw_dz / safe_norm

        return np.stack([v_dx, v_dy, v_dz, v_r, v_ux, v_uy, v_uz], axis=1).astype(np.float32, copy=False)

    def _normalize_non_xy_feature(self, values, col):
        """Normalize one selected non-coordinate hit feature."""
        if self.feature_norm_mode == "standard":
            return (values - self.sel_mean[col]) / self.sel_std[col]
        if self.feature_norm_mode == "minmax":
            values = (values - self.sel_min[col]) / self.sel_range[col]
            return np.clip(values, 0.0, 1.0)
        if self.feature_norm_mode == "standard_minmax":
            values_std = (values - self.sel_mean[col]) / self.sel_std[col]
            values = (values_std - self.sel_min_std[col]) / self.sel_range_std[col]
            return np.clip(values, 0.0, 1.0)
        if self.feature_norm_mode == "none":
            return values
        raise ValueError("feature_norm_mode must be one of: 'standard', 'minmax', 'standard_minmax', 'none'.")

    def _feature_std_or_one(self, name):
        """Return a stable global feature std for derived feature scaling."""
        if name not in self.stats_index:
            return 1.0
        value = float(self.feature_std[self.stats_index[name]])
        if abs(value) < self.eps:
            return 1.0
        return value

    def _build_edges(self, xy, energy, module_id):
        """Build directed radius edges and edge attributes for one event."""
        n_hits = int(xy.shape[0])
        if n_hits <= 1:
            empty_index = torch.empty((2, 0), dtype=torch.long)
            empty_attr = torch.empty((0, 5), dtype=torch.float32)
            return empty_index, empty_attr

        delta = xy[None, :, :] - xy[:, None, :]
        distance = np.linalg.norm(delta, axis=2)
        mask = (distance <= self.radius_mm) & (distance > 0.0)
        src, dst = np.nonzero(mask)

        if src.size == 0:
            empty_index = torch.empty((2, 0), dtype=torch.long)
            empty_attr = torch.empty((0, 5), dtype=torch.float32)
            return empty_index, empty_attr

        dxy = xy[dst] - xy[src]
        dist = distance[src, dst].astype(np.float32)
        d_energy = (energy[dst] - energy[src]).astype(np.float32)
        same_module = (module_id[src] == module_id[dst]).astype(np.float32)

        edge_attr = np.stack(
            [
                dxy[:, 0] / self.radius_mm,
                dxy[:, 1] / self.radius_mm,
                dist / self.radius_mm,
                d_energy / self.edge_energy_scale,
                same_module,
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        edge_index = np.stack([src, dst], axis=0).astype(np.int64, copy=False)
        return torch.from_numpy(edge_index).long(), torch.from_numpy(edge_attr).float()

    def _scale_target(self, value):
        """Transform a raw target into training space."""
        return float(self.target_transform.forward(value))

    def inverse_scale_target(self, value):
        """Convert model output to the task's raw target units."""
        return self.target_transform.inverse(np.asarray(value, dtype=np.float32))

    def _read_event_scalar(self, key, idx, default=0.0):
        """Read an event-level scalar from /events."""
        path = f"events/{key}"
        if path not in self.file:
            return default
        return self.file[path][idx]

    def _attach_event_metadata(self, data, idx):
        """Attach event-level metadata tensors to a Data object."""
        for key in self.EVENT_INT_KEYS:
            value = int(self._read_event_scalar(key, idx, 0))
            setattr(data, key, torch.tensor([value], dtype=torch.long))
        for key in self.EVENT_FLOAT_KEYS:
            value = float(self._read_event_scalar(key, idx, 0.0))
            setattr(data, key, torch.tensor([value], dtype=torch.float32))
        data.primary_origin = torch.tensor(
            [
                float(self._read_event_scalar("primary_origin_x", idx, 0.0)),
                float(self._read_event_scalar("primary_origin_y", idx, 0.0)),
                float(self._read_event_scalar("primary_origin_z", idx, 0.0)),
            ],
            dtype=torch.float32,
        ).view(1, 3)

    def _attach_primary_metadata(self, data, idx):
        """Attach variable-length primary-particle metadata when available."""
        if "events/primary_start" not in self.file or "events/primary_count" not in self.file:
            data.primary_particle_name = []
            return

        start = int(self.file["events/primary_start"][idx])
        count = int(self.file["events/primary_count"][idx])
        data.primary_start = torch.tensor([start], dtype=torch.long)
        data.primary_count = torch.tensor([count], dtype=torch.long)

        if count <= 0 or "primaries" not in self.file:
            data.primary_particle_name = []
            return

        for key in self.PRIMARY_FLOAT_KEYS:
            path = f"primaries/{key}"
            if path in self.file:
                values = np.asarray(self.file[path][start : start + count], dtype=np.float32)
            else:
                values = np.zeros(count, dtype=np.float32)
            setattr(data, key, torch.from_numpy(values).float())

        if "primaries/primary_particle_name" in self.file:
            raw_names = self.file["primaries/primary_particle_name"][start : start + count]
            data.primary_particle_name = self._decode_string_array(raw_names)
        else:
            data.primary_particle_name = []

    def _load_event_features(self, idx):
        """Load and flatten configured event-level features.

        NOTE: These values are returned raw (no dataset-level normalization).
        The model's event_mlp applies LayerNorm per sample, which normalizes
        within the feature vector but not across the dataset distribution.
        If features have very different scales (e.g., n_hits vs total_energy),
        consider pre-normalizing them before passing via --event_features.
        """
        values = []
        for key in self.event_feature_keys:
            values.append(np.asarray(self.file[f"events/{key}"][idx], dtype=np.float32).reshape(-1))
        return np.concatenate(values).astype(np.float32, copy=False)

    def close(self):
        """Close the HDF5 file handle."""
        if getattr(self, "h5file", None) is not None:
            self.h5file.close()
            self.h5file = None

    def __del__(self):
        """Ensure the HDF5 file is closed when the dataset object is destroyed."""
        self.close()
