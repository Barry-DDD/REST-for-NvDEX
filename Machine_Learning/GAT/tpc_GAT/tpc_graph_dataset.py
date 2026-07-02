"""
TPC hit graph dataset for primary_origin_z regression.

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

from task_spec import get_task_spec


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
        edge_attr: [dx/r, dy/r, distance/r, dE, same_module], plus dz/std_z when z is selected
        y: scaled primary_origin_z target, shape [1]
        target_raw: raw primary_origin_z in mm, shape [1]
        event-level and primary-level metadata for later evaluation
    """

    DEFAULT_NODE_FEATURES = ("x", "y", "energy", "log_energy")
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
        target_mode: str = "minus_one_one",
        target_abs_max: float = 845.0,
        xy_abs_max: float = 445.0,
        feature_norm_mode: str = "standard",
        event_feature_keys: Iterable[str] = (),
        drop_empty_events: bool = True,
        eps: float = 1e-6,
        transform=None,
    ):
        super().__init__(None, transform)
        self.file_path = str(h5_path)
        self.node_feature_names = tuple(node_feature_names)
        self.radius_mm = float(radius_mm)
        self.task_spec = get_task_spec(task)
        self.task = self.task_spec.name
        self.target_key = self.task_spec.target_key if self.task_spec.is_classification else str(target_key)
        self.target_mode = str(target_mode)
        self.target_abs_max = float(target_abs_max)
        self.xy_abs_max = float(xy_abs_max)
        self.feature_norm_mode = str(feature_norm_mode)
        self.event_feature_keys = tuple(event_feature_keys)
        self.drop_empty_events = bool(drop_empty_events)
        self.eps = float(eps)
        self.h5file = None
        self.use_z_edges = "z" in self.node_feature_names
        self.edge_dim = 6 if self.use_z_edges else 5

        if self.radius_mm <= 0.0:
            raise ValueError("radius_mm must be positive.")
        if self.target_abs_max <= 0.0:
            raise ValueError("target_abs_max must be positive.")
        if self.xy_abs_max <= 0.0:
            raise ValueError("xy_abs_max must be positive.")
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
            self.event_feature_dim = self._infer_event_feature_dim(h5)
            self.event_indices, self.hit_counts = self._build_event_index(h5)
            self.raw_targets = np.asarray(h5[f"events/{self.target_key}"][self.event_indices])
            self.labels = self._build_labels(self.raw_targets)

        self._prepare_feature_normalization_constants()
        self.edge_energy_scale = self._feature_std_or_one("energy")
        self.edge_z_scale = self._feature_std_or_one("z")

        print("[TPCGraphDataset] Dataset initialized:")
        print(f"  File: {self.file_path}")
        print(f"  Total events in file: {self.total_events}")
        print(f"  Usable events: {len(self.event_indices)}")
        print(f"  Dropped empty-hit events: {self.total_events - len(self.event_indices)}")
        print(f"  Node features: {self.node_feature_names}")
        print(f"  XY coordinate scale: +/- {self.xy_abs_max:.3f} mm -> [-1, 1]")
        print(f"  Non-XY feature normalization mode: {self.feature_norm_mode}")
        print(f"  Event feature keys: {self.event_feature_keys}")
        print(f"  Event feature dimension: {self.event_feature_dim}")
        print(f"  Radius graph: {self.radius_mm:.3f} mm")
        print(f"  Edge feature dimension: {self.edge_dim}")
        if self.use_z_edges:
            print(f"  Edge dz scale: stats/std(z)={self.edge_z_scale:.6g}")
        print(f"  Task: {self.task}")
        if self.task_spec.is_classification:
            print(f"  Target: {self.target_key}, classes={self.task_spec.num_outputs}")
        else:
            print(f"  Target: {self.target_key}, mode={self.target_mode}, abs_max={self.target_abs_max}")
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
        for key in ("hit_start", "hit_count", self.target_key):
            if f"events/{key}" not in h5:
                raise KeyError(f"HDF5 file does not contain required dataset 'events/{key}'.")
        for key in ("x", "y", "energy", "log_energy", "module_id"):
            if f"hits/{key}" not in h5:
                raise KeyError(f"HDF5 file does not contain required dataset 'hits/{key}'.")
        for key in ("feature_names", "mean", "std", "min", "max"):
            if f"stats/{key}" not in h5:
                raise KeyError(f"HDF5 file does not contain required dataset 'stats/{key}'.")

        total_events = int(h5["events/hit_count"].shape[0])
        if h5["events/hit_start"].shape[0] != total_events:
            raise ValueError("events/hit_start and events/hit_count have different lengths.")
        if h5[f"events/{self.target_key}"].shape[0] != total_events:
            raise ValueError(f"events/{self.target_key} length does not match event count.")
        return total_events

    def _validate_event_features(self):
        """Prevent labels and truth targets from being used as input features."""
        forbidden = {"id", self.target_key, self.task_spec.target_key}
        leaked = sorted(forbidden.intersection(self.event_feature_keys))
        if leaked:
            raise ValueError(f"event_features cannot include task target fields: {leaked}.")

    def _validate_feature_selection(self, h5):
        """Validate selected node features against hits and stats."""
        for name in self.node_feature_names:
            if f"hits/{name}" not in h5:
                raise KeyError(f"Selected node feature 'hits/{name}' is missing.")
            if name not in self.stats_index:
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
        return np.asarray([self._scale_target(float(value)) for value in raw_targets], dtype=np.float32)

    def _prepare_feature_normalization_constants(self):
        """Precompute normalization constants for selected node features."""
        selected = np.asarray([self.stats_index[name] for name in self.node_feature_names], dtype=np.int64)
        self.sel_mean = self.feature_mean[selected].astype(np.float32)
        self.sel_std = np.where(np.abs(self.feature_std[selected]) < self.eps, 1.0, self.feature_std[selected]).astype(np.float32)
        self.sel_min = self.feature_min[selected].astype(np.float32)
        self.sel_max = self.feature_max[selected].astype(np.float32)
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
        raw_z = np.asarray(hits["z"], dtype=np.float32) if self.use_z_edges else None
        raw_energy = hits["energy"].astype(np.float32, copy=False)
        module_id = hits["module_id"].astype(np.int64, copy=False)
        node_features = self._load_node_features(hits)
        edge_index, edge_attr = self._build_edges(raw_xy, raw_z, raw_energy, module_id)

        raw_target_value = self.file[f"events/{self.target_key}"][file_idx]
        raw_target = float(raw_target_value) if self.task_spec.is_regression else int(raw_target_value)
        if self.task_spec.is_classification:
            label = int(self.task_spec.map_raw_labels(np.asarray([raw_target]))[0])
            y = torch.tensor([label], dtype=torch.long)
            target_raw = torch.tensor([raw_target], dtype=torch.long)
        else:
            y = torch.tensor([self._scale_target(float(raw_target))], dtype=torch.float32)
            target_raw = torch.tensor([float(raw_target)], dtype=torch.float32)
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
        self._attach_event_metadata(data, file_idx)
        self._attach_primary_metadata(data, file_idx)
        if self.event_feature_dim > 0:
            data.event_features = torch.from_numpy(self._load_event_features(file_idx)).float().view(1, -1)
        return data

    def _load_hits(self, start, count):
        """Load selected hit fields for one event."""
        if count <= 0:
            return {}
        return {
            name: np.asarray(self.file[f"hits/{name}"][start : start + count])
            for name in set(self.node_feature_names) | {"x", "y", "energy", "module_id"}
        }

    def _dummy_hits(self):
        """Create one zero-valued hit for explicitly retained empty events."""
        return {
            name: np.zeros(1, dtype=np.float32)
            for name in set(self.node_feature_names) | {"x", "y", "energy"}
        } | {"module_id": np.zeros(1, dtype=np.int64)}

    def _load_node_features(self, hits):
        """Stack and normalize selected hit-level node features."""
        features = np.stack([np.asarray(hits[name], dtype=np.float32) for name in self.node_feature_names], axis=1)
        normalized = np.empty_like(features, dtype=np.float32)
        for col, name in enumerate(self.node_feature_names):
            values = features[:, col]
            if name in ("x", "y"):
                normalized[:, col] = values / self.xy_abs_max
            else:
                normalized[:, col] = self._normalize_non_xy_feature(values, col)
        features = normalized
        return features.astype(np.float32, copy=False)

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

    def _build_edges(self, xy, z, energy, module_id):
        """Build directed radius edges and edge attributes for one event."""
        n_hits = int(xy.shape[0])
        if n_hits <= 1:
            empty_index = torch.empty((2, 0), dtype=torch.long)
            empty_attr = torch.empty((0, self.edge_dim), dtype=torch.float32)
            return empty_index, empty_attr

        delta = xy[None, :, :] - xy[:, None, :]
        distance = np.linalg.norm(delta, axis=2)
        mask = (distance <= self.radius_mm) & (distance > 0.0)
        src, dst = np.nonzero(mask)

        if src.size == 0:
            empty_index = torch.empty((2, 0), dtype=torch.long)
            empty_attr = torch.empty((0, self.edge_dim), dtype=torch.float32)
            return empty_index, empty_attr

        dxy = xy[dst] - xy[src]
        dist = distance[src, dst].astype(np.float32)
        d_energy = (energy[dst] - energy[src]).astype(np.float32)
        same_module = (module_id[src] == module_id[dst]).astype(np.float32)

        columns = [
            dxy[:, 0] / self.radius_mm,
            dxy[:, 1] / self.radius_mm,
            dist / self.radius_mm,
        ]
        if self.use_z_edges:
            dz = (z[dst] - z[src]).astype(np.float32)
            columns.append(dz / self.edge_z_scale)
        columns.extend(
            [
                d_energy / self.edge_energy_scale,
                same_module,
            ]
        )
        edge_attr = np.stack(columns, axis=1).astype(np.float32, copy=False)
        edge_index = np.stack([src, dst], axis=0).astype(np.int64, copy=False)
        return torch.from_numpy(edge_index).long(), torch.from_numpy(edge_attr).float()

    def _scale_target(self, value):
        """Scale primary_origin_z to the configured training range."""
        if self.target_mode == "minus_one_one":
            return float(np.clip(value / self.target_abs_max, -1.0, 1.0))
        if self.target_mode == "zero_one":
            scaled = (value + self.target_abs_max) / (2.0 * self.target_abs_max)
            return float(np.clip(scaled, 0.0, 1.0))
        if self.target_mode == "none":
            return float(value)
        raise ValueError("target_mode must be one of: 'minus_one_one', 'zero_one', 'none'.")

    def inverse_scale_target(self, value):
        """Convert a scaled model output back to primary_origin_z in mm."""
        value = np.asarray(value, dtype=np.float32)
        if self.target_mode == "minus_one_one":
            return value * self.target_abs_max
        if self.target_mode == "zero_one":
            return value * (2.0 * self.target_abs_max) - self.target_abs_max
        if self.target_mode == "none":
            return value
        raise ValueError("target_mode must be one of: 'minus_one_one', 'zero_one', 'none'.")

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
        if self.h5file is not None:
            self.h5file.close()
            self.h5file = None

    def __del__(self):
        """Ensure the HDF5 file is closed when the dataset object is destroyed."""
        self.close()
