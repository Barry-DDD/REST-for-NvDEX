#!/usr/bin/env python3
"""Merge per-file ML HDF5 outputs into one training HDF5 file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


SCHEMA_VERSION = "1.0"


def merge_event_offsets(starts: list[np.ndarray], counts: list[np.ndarray]) -> np.ndarray:
    shifted: list[np.ndarray] = []
    running_total = 0
    for file_starts, file_counts in zip(starts, counts):
        shifted.append(np.asarray(file_starts, dtype=np.int64) + running_total)
        running_total += int(np.asarray(file_counts, dtype=np.int64).sum())
    return np.concatenate(shifted) if shifted else np.asarray([], dtype=np.int64)


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit("Missing dependency: install h5py before merging HDF5 files.") from exc
    return h5py


def _decode_string_array(arr: np.ndarray) -> list[str]:
    return [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in arr]


def _validate_input_files(input_files: list) -> None:
    """Raise ValueError if any file is schema-incompatible with the first file."""
    if len(input_files) < 2:
        return

    ref = input_files[0]
    ref_version = ref.attrs.get("schema_version", "")
    ref_feat = _decode_string_array(ref["stats/feature_names"][...])

    for idx, h5 in enumerate(input_files[1:], 1):
        fname = getattr(h5, "filename", f"file[{idx}]")

        ver = h5.attrs.get("schema_version", "")
        if ver != ref_version:
            raise ValueError(f"{fname}: schema_version {ver!r} != reference {ref_version!r}")

        for group_name in ("events", "hits", "primaries"):
            if group_name not in ref:
                continue
            if group_name not in h5:
                raise ValueError(f"{fname}: group {group_name!r} is missing")
            ref_keys = sorted(ref[group_name].keys())
            file_keys = sorted(h5[group_name].keys())
            if ref_keys != file_keys:
                raise ValueError(
                    f"{fname}: group {group_name!r} datasets differ.\n"
                    f"  reference : {ref_keys}\n  this file : {file_keys}"
                )
            for name in ref_keys:
                path = f"{group_name}/{name}"
                ref_dtype = ref[path].dtype
                file_dtype = h5[path].dtype
                if ref_dtype != file_dtype:
                    raise ValueError(
                        f"{fname}: dataset {path!r} dtype {file_dtype} != reference {ref_dtype}"
                    )

        file_feat = _decode_string_array(h5["stats/feature_names"][...])
        if file_feat != ref_feat:
            raise ValueError(
                f"{fname}: stats/feature_names {file_feat} != reference {ref_feat}"
            )


def _concat_datasets(files: list, dataset_path: str) -> np.ndarray:
    arrays = []
    for h5 in files:
        if dataset_path not in h5:
            raise ValueError(
                f"Dataset {dataset_path!r} missing from {getattr(h5, 'filename', repr(h5))}"
            )
        arrays.append(h5[dataset_path][...])
    if not arrays:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(arrays)


def _write_group_datasets(
    output_group: object,
    reference_group: object,
    files: list,
    skip: set[str] | None = None,
) -> None:
    skip = skip or set()
    for name in reference_group.keys():
        if name in skip:
            continue
        path = f"{reference_group.name}/{name}"
        output_group.create_dataset(name, data=_concat_datasets(files, path), compression="gzip")


def _write_stats(output_file: object, input_files: list) -> None:
    h5py = _require_h5py()
    string_dtype = h5py.string_dtype(encoding="utf-8")

    names = _decode_string_array(input_files[0]["stats/feature_names"][...])
    count = 0
    mean = np.zeros(len(names), dtype=np.float64)
    m2 = np.zeros(len(names), dtype=np.float64)
    min_values = np.full(len(names), np.inf, dtype=np.float64)
    max_values = np.full(len(names), -np.inf, dtype=np.float64)

    for h5 in input_files:
        stats = h5["stats"]
        file_count = int(stats.attrs["count"])
        if file_count == 0:
            continue
        file_mean = stats["mean"][...].astype(np.float64)
        file_std = stats["std"][...].astype(np.float64)
        # reconstruct m2 from stored population std: m2 = std^2 * count
        file_m2 = (file_std ** 2) * file_count

        if count == 0:
            count = file_count
            mean = file_mean
            m2 = file_m2
        else:
            total_count = count + file_count
            delta = file_mean - mean
            mean = mean + delta * file_count / total_count
            m2 = m2 + file_m2 + delta * delta * count * file_count / total_count
            count = total_count

        min_values = np.minimum(min_values, stats["min"][...])
        max_values = np.maximum(max_values, stats["max"][...])

    stats_group = output_file.create_group("stats")
    stats_group.attrs["count"] = int(count)
    stats_group.attrs["std_definition"] = "population"
    stats_group.create_dataset("feature_names", data=np.asarray(names, dtype=object), dtype=string_dtype)
    if count == 0:
        nan_values = np.full(len(names), np.nan, dtype=np.float64)
        for k in ("mean", "std", "min", "max"):
            stats_group.create_dataset(k, data=nan_values)
    else:
        stats_group.create_dataset("mean", data=mean)
        stats_group.create_dataset("std", data=np.sqrt(m2 / count))
        stats_group.create_dataset("min", data=min_values)
        stats_group.create_dataset("max", data=max_values)


def merge_hdf5_files(input_paths: list[Path], output_path: Path) -> dict[str, int]:
    h5py = _require_h5py()
    if not input_paths:
        raise ValueError("At least one input HDF5 file is required.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_files = [h5py.File(path, "r") for path in input_paths]
    try:
        _validate_input_files(input_files)

        with h5py.File(output_path, "w") as out:
            out.attrs["schema_version"] = SCHEMA_VERSION
            out.attrs["source_hdf5_files"] = json.dumps([str(path) for path in input_paths])
            out.attrs["filter"] = input_files[0].attrs.get("filter", "energy > 0")

            event_group = out.create_group("events")
            hit_starts = [h5["events/hit_start"][...] for h5 in input_files]
            hit_counts = [h5["events/hit_count"][...] for h5 in input_files]
            primary_starts = [h5["events/primary_start"][...] for h5 in input_files]
            primary_counts = [h5["events/primary_count"][...] for h5 in input_files]
            _write_group_datasets(
                event_group,
                input_files[0]["events"],
                input_files,
                skip={"hit_start", "primary_start"},
            )
            event_group.create_dataset(
                "hit_start", data=merge_event_offsets(hit_starts, hit_counts), compression="gzip"
            )
            event_group.create_dataset(
                "primary_start",
                data=merge_event_offsets(primary_starts, primary_counts),
                compression="gzip",
            )

            hits_group = out.create_group("hits")
            _write_group_datasets(hits_group, input_files[0]["hits"], input_files)

            primaries_group = out.create_group("primaries")
            _write_group_datasets(primaries_group, input_files[0]["primaries"], input_files)

            _write_stats(out, input_files)

            summary = out.create_group("summary")
            summary.attrs["n_input_files"] = len(input_paths)
            summary.attrs["n_events"] = int(sum(h5["summary"].attrs["n_events"] for h5 in input_files))
            summary.attrs["n_hits_after_filter"] = int(
                sum(h5["summary"].attrs["n_hits_after_filter"] for h5 in input_files)
            )
            summary.attrs["n_hits_before_filter"] = int(
                sum(h5["summary"].attrs["n_hits_before_filter"] for h5 in input_files)
            )
            summary.attrs["n_primaries"] = int(sum(h5["summary"].attrs["n_primaries"] for h5 in input_files))

            return {
                "n_input_files": int(summary.attrs["n_input_files"]),
                "n_events": int(summary.attrs["n_events"]),
                "n_hits_after_filter": int(summary.attrs["n_hits_after_filter"]),
                "n_hits_before_filter": int(summary.attrs["n_hits_before_filter"]),
                "n_primaries": int(summary.attrs["n_primaries"]),
            }
    finally:
        for h5 in input_files:
            h5.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-file ML HDF5 outputs into one file.")
    parser.add_argument("output", type=Path, help="Merged output HDF5 file.")
    parser.add_argument("inputs", type=Path, nargs="+", help="Input HDF5 files produced by root_to_hdf5.py.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = merge_hdf5_files(args.inputs, args.output)
    print(f"Wrote {args.output}")
    print(
        "Files={n_input_files} events={n_events} hits_before={n_hits_before_filter} "
        "hits_after={n_hits_after_filter} primaries={n_primaries}".format(**summary)
    )


if __name__ == "__main__":
    main()

# python3 merge_hdf5.py all_events.h5 ../rebuildEvents/hdf5/processed_Xe_sim_*.h5
