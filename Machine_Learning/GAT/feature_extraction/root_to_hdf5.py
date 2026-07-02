#!/usr/bin/env python3
"""
Convert processed ROOT HitTree files into HDF5 files for graph and point-cloud ML.

The input ROOT files are expected to be produced by extract_hits_for_ml_hier_hex.C.
The output schema stores hit-level data as flat arrays and stores event offsets so
each event can be reconstructed without padding.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "1.0"
DEFAULT_TREE_NAME = "HitTree"

HIT_FLOAT_FIELDS = [
    "x",
    "y",
    "z",
    "energy",
    "log_energy",
    "pad_center_x",
    "pad_center_y",
    "residual_x",
    "residual_y",
    "nearest_pad_distance",
]
HIT_INT_FIELDS = [
    "module_id",
    "module_col",
    "module_row",
    "module_q",
    "module_r",
    "local_col",
    "local_row",
    "local_q",
    "local_r",
    "q",
    "r",
    "valid_geometry",
]
EVENT_INT_FIELDS = ["n_hits", "n_primaries", "n_tracks"]
OPTIONAL_EVENT_INT_FIELDS = ["id"]
EVENT_FLOAT_FIELDS = [
    "total_energy",
    "primary_origin_x",
    "primary_origin_y",
    "primary_origin_z",
    "g4_total_deposited_energy",
    "g4_sensitive_volume_energy",
]
PRIMARY_FLOAT_FIELDS = ["primary_energy", "primary_dir_x", "primary_dir_y", "primary_dir_z"]
PRIMARY_STRING_FIELDS = ["primary_particle_name"]
STATS_FIELDS = ["x", "y", "z", "local_q", "local_r", "module_id", "energy", "log_energy"]


class RunningStats:
    """Track population statistics for ML normalization features."""

    def __init__(self, names: Iterable[str]) -> None:
        self.names = list(names)
        n_features = len(self.names)
        self.count = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.m2 = np.zeros(n_features, dtype=np.float64)
        self.min = np.full(n_features, np.inf, dtype=np.float64)
        self.max = np.full(n_features, -np.inf, dtype=np.float64)

    def update(self, columns: dict[str, np.ndarray]) -> None:
        arrays = [np.asarray(columns[name], dtype=np.float64) for name in self.names]
        if not arrays or arrays[0].size == 0:
            return

        stacked = np.column_stack(arrays)
        batch_count = stacked.shape[0]
        batch_mean = stacked.mean(axis=0)
        batch_m2 = ((stacked - batch_mean) ** 2).sum(axis=0)
        batch_min = stacked.min(axis=0)
        batch_max = stacked.max(axis=0)

        if self.count == 0:
            self.count = int(batch_count)
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            total_count = self.count + batch_count
            delta = batch_mean - self.mean
            self.mean = self.mean + delta * batch_count / total_count
            self.m2 = self.m2 + batch_m2 + delta * delta * self.count * batch_count / total_count
            self.count = int(total_count)

        self.min = np.minimum(self.min, batch_min)
        self.max = np.maximum(self.max, batch_max)

    def snapshot(self) -> dict[str, Any]:
        if self.count == 0:
            nan_values = np.full(len(self.names), np.nan, dtype=np.float64)
            return {
                "names": np.asarray(self.names, dtype=object),
                "count": 0,
                "mean": nan_values.copy(),
                "std": nan_values.copy(),
                "min": nan_values.copy(),
                "max": nan_values.copy(),
            }

        variance = self.m2 / self.count
        return {
            "names": np.asarray(self.names, dtype=object),
            "count": int(self.count),
            "mean": self.mean.copy(),
            "std": np.sqrt(variance),
            "min": self.min.copy(),
            "max": self.max.copy(),
        }


def _to_numpy(value: Any, dtype: np.dtype | type) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=dtype)
    return np.asarray(value, dtype=dtype)


def _event_scalar(event: dict[str, Any], name: str, default: Any = 0) -> Any:
    value = event.get(name, default)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return value.item()
    return value


def _hit_vector(event: dict[str, Any], name: str, dtype: np.dtype | type, n_hits: int) -> np.ndarray:
    if name not in event or event[name] is None:
        warnings.warn(f"Branch {name!r} absent in ROOT file; filling {n_hits} hits with zeros.", stacklevel=3)
        return np.zeros(n_hits, dtype=dtype)
    values = np.asarray(event[name], dtype=dtype)
    if values.size != n_hits:
        raise ValueError(f"Hit branch {name!r} has {values.size} values, expected {n_hits}.")
    return values


def build_filtered_hit_block(events: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Return flattened positive-energy hits plus per-event hit offsets."""

    hit_columns: dict[str, list[np.ndarray]] = {field: [] for field in HIT_FLOAT_FIELDS + HIT_INT_FIELDS}
    event_hit_start: list[int] = []
    event_hit_count: list[int] = []
    event_n_hits_original: list[int] = []
    event_total_energy_filtered: list[float] = []
    total_hits = 0

    for event in events:
        energy = _to_numpy(event.get("energy"), np.float32)
        keep = energy > 0
        kept_count = int(keep.sum())

        event_hit_start.append(total_hits)
        event_hit_count.append(kept_count)
        event_n_hits_original.append(int(_event_scalar(event, "n_hits", energy.size)))
        event_total_energy_filtered.append(float(energy[keep].sum(dtype=np.float64)))

        if kept_count:
            for field in HIT_FLOAT_FIELDS:
                if field == "log_energy":
                    values = np.log1p(energy[keep]).astype(np.float32)
                else:
                    values = _hit_vector(event, field, np.float32, energy.size)[keep]
                hit_columns[field].append(values.astype(np.float32, copy=False))

            for field in HIT_INT_FIELDS:
                values = _hit_vector(event, field, np.int32, energy.size)[keep]
                hit_columns[field].append(values.astype(np.int32, copy=False))

        total_hits += kept_count

    block: dict[str, np.ndarray] = {
        "event_hit_start": np.asarray(event_hit_start, dtype=np.int64),
        "event_hit_count": np.asarray(event_hit_count, dtype=np.int32),
        "n_hits_original": np.asarray(event_n_hits_original, dtype=np.int32),
        "total_energy_filtered": np.asarray(event_total_energy_filtered, dtype=np.float32),
    }

    for field in HIT_FLOAT_FIELDS:
        block[field] = (
            np.concatenate(hit_columns[field]).astype(np.float32, copy=False)
            if hit_columns[field]
            else np.asarray([], dtype=np.float32)
        )
    for field in HIT_INT_FIELDS:
        block[field] = (
            np.concatenate(hit_columns[field]).astype(np.int32, copy=False)
            if hit_columns[field]
            else np.asarray([], dtype=np.int32)
        )

    return block


def load_root_events(input_path: Path, tree_name: str) -> list[dict[str, Any]]:
    try:
        import uproot
    except ImportError as exc:
        raise SystemExit("Missing dependency: install uproot before converting ROOT files.") from exc

    branches = (
        EVENT_INT_FIELDS
        + OPTIONAL_EVENT_INT_FIELDS
        + EVENT_FLOAT_FIELDS
        + HIT_FLOAT_FIELDS
        + HIT_INT_FIELDS
        + PRIMARY_FLOAT_FIELDS
        + PRIMARY_STRING_FIELDS
    )
    with uproot.open(input_path) as root_file:
        if tree_name not in root_file:
            available = ", ".join(root_file.keys())
            raise KeyError(f"Tree {tree_name!r} not found in {input_path}. Available keys: {available}")
        tree = root_file[tree_name]
        existing_branches = [name for name in branches if name in tree.keys()]
        missing_required = [
            name for name in ["energy", "x", "y", "local_q", "local_r", "module_id"]
            if name not in existing_branches
        ]
        if missing_required:
            raise KeyError(f"Input tree is missing required branches: {missing_required}")

        arrays = tree.arrays(existing_branches, library="np")
        n_events = tree.num_entries

    events: list[dict[str, Any]] = []
    for index in range(n_events):
        event: dict[str, Any] = {name: arrays[name][index] for name in existing_branches}
        events.append(event)
    return events


def build_event_metadata(events: list[dict[str, Any]], hit_block: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    metadata: dict[str, np.ndarray] = {
        "hit_start": hit_block["event_hit_start"],
        "hit_count": hit_block["event_hit_count"],
        "n_hits_original": hit_block["n_hits_original"],
        "total_energy_filtered": hit_block["total_energy_filtered"],
    }

    for field in EVENT_INT_FIELDS:
        metadata[field] = np.asarray([int(_event_scalar(event, field, 0)) for event in events], dtype=np.int32)
    for field in OPTIONAL_EVENT_INT_FIELDS:
        if any(field in event for event in events):
            metadata[field] = np.asarray([int(_event_scalar(event, field, 0)) for event in events], dtype=np.int32)
    for field in EVENT_FLOAT_FIELDS:
        metadata[field] = np.asarray([float(_event_scalar(event, field, 0.0)) for event in events], dtype=np.float32)

    return metadata


def build_primary_block(events: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    primary_start: list[int] = []
    primary_count: list[int] = []
    total_primaries = 0
    float_columns: dict[str, list[np.ndarray]] = {field: [] for field in PRIMARY_FLOAT_FIELDS}
    names: list[str] = []

    for i, event in enumerate(events):
        raw_names = event.get("primary_particle_name")
        # atleast_1d guards against uproot returning a scalar string (TString branch)
        # instead of an array; iterating a 0-d object array via tolist() yields characters.
        name_array = (
            np.atleast_1d(np.asarray(raw_names, dtype=object))
            if raw_names is not None
            else np.asarray([], dtype=object)
        )
        count = int(name_array.size)

        primary_start.append(total_primaries)
        primary_count.append(count)

        if count:
            names.extend(str(n) for n in name_array.tolist())
            for field in PRIMARY_FLOAT_FIELDS:
                arr = _to_numpy(event.get(field), np.float32)
                if arr.size != count:
                    raise ValueError(
                        f"Event {i}: primary field {field!r} has {arr.size} entries, expected {count}."
                    )
                float_columns[field].append(arr.astype(np.float32, copy=False))

        total_primaries += count

    block: dict[str, np.ndarray] = {
        "event_primary_start": np.asarray(primary_start, dtype=np.int64),
        "event_primary_count": np.asarray(primary_count, dtype=np.int32),
        "primary_particle_name": np.asarray(names, dtype=object),
    }

    for field in PRIMARY_FLOAT_FIELDS:
        block[field] = (
            np.concatenate(float_columns[field]).astype(np.float32, copy=False)
            if float_columns[field]
            else np.asarray([], dtype=np.float32)
        )
    return block


def write_hdf5(
    output_path: Path,
    input_path: Path,
    tree_name: str,
    events: list[dict[str, Any]],
    hit_block: dict[str, np.ndarray],
    event_metadata: dict[str, np.ndarray],
    primary_block: dict[str, np.ndarray],
    stats: RunningStats,
) -> None:
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit("Missing dependency: install h5py before writing HDF5 files.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_path, "w") as h5:
        h5.attrs["schema_version"] = SCHEMA_VERSION
        h5.attrs["source_root"] = str(input_path)
        h5.attrs["tree_name"] = tree_name
        h5.attrs["filter"] = "energy > 0"
        h5.attrs["hit_feature_names"] = json.dumps(["x", "y", "z", "energy", "log_energy"])
        h5.attrs["graph_coordinate_names"] = json.dumps(["x", "y", "local_q", "local_r", "module_id"])

        events_group = h5.create_group("events")
        for name, values in event_metadata.items():
            events_group.create_dataset(name, data=values, compression="gzip")
        events_group.create_dataset("primary_start", data=primary_block["event_primary_start"], compression="gzip")
        events_group.create_dataset("primary_count", data=primary_block["event_primary_count"], compression="gzip")

        hits_group = h5.create_group("hits")
        for field in HIT_FLOAT_FIELDS + HIT_INT_FIELDS:
            hits_group.create_dataset(field, data=hit_block[field], compression="gzip")

        primaries_group = h5.create_group("primaries")
        primaries_group.create_dataset(
            "primary_particle_name",
            data=primary_block["primary_particle_name"],
            dtype=string_dtype,
            compression="gzip",
        )
        for field in PRIMARY_FLOAT_FIELDS:
            primaries_group.create_dataset(field, data=primary_block[field], compression="gzip")

        stats_snapshot = stats.snapshot()
        stats_group = h5.create_group("stats")
        stats_group.attrs["count"] = stats_snapshot["count"]
        stats_group.attrs["std_definition"] = "population"
        stats_group.create_dataset("feature_names", data=stats_snapshot["names"], dtype=string_dtype)
        for name in ["mean", "std", "min", "max"]:
            stats_group.create_dataset(name, data=stats_snapshot[name])

        n_hits_after = int(hit_block["event_hit_count"].sum(dtype=np.int64))
        n_primaries = int(primary_block["event_primary_count"].sum(dtype=np.int64))

        summary = h5.create_group("summary")
        summary.attrs["n_events"] = len(events)
        summary.attrs["n_hits_after_filter"] = n_hits_after
        summary.attrs["n_hits_before_filter"] = int(hit_block["n_hits_original"].sum(dtype=np.int64))
        summary.attrs["n_primaries"] = n_primaries


def convert_root_to_hdf5(input_path: Path, output_path: Path, tree_name: str = DEFAULT_TREE_NAME) -> dict[str, int]:
    events = load_root_events(input_path, tree_name)
    hit_block = build_filtered_hit_block(events)
    event_metadata = build_event_metadata(events, hit_block)
    primary_block = build_primary_block(events)

    stats = RunningStats(STATS_FIELDS)
    stats.update({field: hit_block[field] for field in STATS_FIELDS})

    write_hdf5(output_path, input_path, tree_name, events, hit_block, event_metadata, primary_block, stats)
    return {
        "n_events": len(events),
        "n_hits_before_filter": int(hit_block["n_hits_original"].sum(dtype=np.int64)),
        "n_hits_after_filter": int(hit_block["event_hit_count"].sum(dtype=np.int64)),
        "n_primaries": int(primary_block["event_primary_count"].sum(dtype=np.int64)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert processed HitTree ROOT files into ML-ready HDF5.")
    parser.add_argument("input", type=Path, help="Input processed ROOT file produced by extract_hits_for_ml_hier_hex.C.")
    parser.add_argument("output", type=Path, help="Output HDF5 file.")
    parser.add_argument("--tree", default=DEFAULT_TREE_NAME, help=f"Input TTree name. Default: {DEFAULT_TREE_NAME}.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = convert_root_to_hdf5(args.input, args.output, args.tree)
    print(f"Wrote {args.output}")
    print(
        "Events={n_events} hits_before={n_hits_before_filter} "
        "hits_after={n_hits_after_filter} primaries={n_primaries}".format(**summary)
    )


if __name__ == "__main__":
    main()

#python3 root_to_hdf5.py processed_Xe_sim_10.root processed_Xe_sim_10.h5 --tree HitTree
