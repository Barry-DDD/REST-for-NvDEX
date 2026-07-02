"""Quick statistics for TPC hit graphs and GAT radius selection."""

from __future__ import annotations

import argparse

import h5py
import numpy as np


def summarize(name, arr, unit=""):
    """Print standard summary statistics for a 1D array."""
    arr = np.asarray(arr, dtype=np.float64)
    suffix = f" {unit}" if unit else ""
    print(f"\n[{name}]  (n={arr.size})")
    if arr.size == 0:
        print("  empty")
        return
    pcts = np.percentile(arr, [1, 5, 50, 95, 99])
    print(f"  min      = {arr.min():.4f}{suffix}")
    print(f"  max      = {arr.max():.4f}{suffix}")
    print(f"  mean     = {arr.mean():.4f}{suffix}")
    print(f"  std      = {arr.std():.4f}{suffix}")
    print(f"  p01/p05  = {pcts[0]:.4f} / {pcts[1]:.4f}{suffix}")
    print(f"  median   = {pcts[2]:.4f}{suffix}")
    print(f"  p95/p99  = {pcts[3]:.4f} / {pcts[4]:.4f}{suffix}")


def nearest_neighbor_distances(xy):
    """Return the nearest-neighbor xy distance for each hit in one event."""
    if xy.shape[0] < 2:
        return np.asarray([], dtype=np.float32)
    delta = xy[None, :, :] - xy[:, None, :]
    distance = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(distance, np.inf)
    return np.min(distance, axis=1).astype(np.float32)


def radius_edge_count(xy, radius_mm):
    """Return directed radius-graph edge count for one event."""
    if xy.shape[0] < 2:
        return 0
    delta = xy[None, :, :] - xy[:, None, :]
    distance = np.linalg.norm(delta, axis=2)
    mask = (distance <= radius_mm) & (distance > 0.0)
    return int(np.count_nonzero(mask))


def scan_file(h5_path, n_events, radii_mm, seed):
    """Scan a TPC HDF5 file and return graph statistics."""
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r") as h5:
        hit_count = np.asarray(h5["events/hit_count"], dtype=np.int64)
        n_total = int(hit_count.shape[0])
        n_sample = min(int(n_events), n_total)
        indices = rng.choice(n_total, size=n_sample, replace=False)
        indices.sort()

        hit_counts = []
        nn_dists = []
        x_values = []
        y_values = []
        energies = []
        log_energies = []
        targets = []
        event_ids = []
        edge_counts = {float(radius): [] for radius in radii_mm}
        has_event_id = "events/id" in h5

        for idx in indices:
            start = int(h5["events/hit_start"][idx])
            count = int(h5["events/hit_count"][idx])
            hit_counts.append(count)
            targets.append(float(h5["events/primary_origin_z"][idx]))
            if has_event_id:
                event_ids.append(int(h5["events/id"][idx]))
            if count <= 0:
                for radius in edge_counts:
                    edge_counts[radius].append(0)
                continue

            x = np.asarray(h5["hits/x"][start : start + count], dtype=np.float32)
            y = np.asarray(h5["hits/y"][start : start + count], dtype=np.float32)
            xy = np.stack([x, y], axis=1)
            x_values.append(x)
            y_values.append(y)
            nn = nearest_neighbor_distances(xy)
            if nn.size:
                nn_dists.append(nn)
            energies.append(np.asarray(h5["hits/energy"][start : start + count], dtype=np.float32))
            log_energies.append(np.asarray(h5["hits/log_energy"][start : start + count], dtype=np.float32))

            for radius in edge_counts:
                edge_counts[radius].append(radius_edge_count(xy, radius))

    return {
        "n_total": n_total,
        "n_sample": n_sample,
        "hit_counts": np.asarray(hit_counts, dtype=np.int64),
        "nn_dists": np.concatenate(nn_dists) if nn_dists else np.asarray([], dtype=np.float32),
        "x_values": np.concatenate(x_values) if x_values else np.asarray([], dtype=np.float32),
        "y_values": np.concatenate(y_values) if y_values else np.asarray([], dtype=np.float32),
        "energies": np.concatenate(energies) if energies else np.asarray([], dtype=np.float32),
        "log_energies": np.concatenate(log_energies) if log_energies else np.asarray([], dtype=np.float32),
        "targets": np.asarray(targets, dtype=np.float32),
        "event_ids": np.asarray(event_ids, dtype=np.int32),
        "edge_counts": {radius: np.asarray(values, dtype=np.int64) for radius, values in edge_counts.items()},
    }


def print_id_distribution(event_ids):
    """Print classification label counts when /events/id is available."""
    if event_ids.size == 0:
        print("\n[event id distribution]")
        print("  /events/id not found")
        return

    values, counts = np.unique(event_ids, return_counts=True)
    print("\n[event id distribution]")
    for value, count in zip(values, counts):
        print(f"  id={int(value):>2}: {int(count)}")

    signal = int(np.count_nonzero(event_ids == 1))
    background = int(np.count_nonzero(np.isin(event_ids, [21, 22, 23, 24])))
    other = int(event_ids.size - signal - background)
    print("\n[binary classification distribution]")
    print(f"  signal(id=1): {signal}")
    print(f"  background(id=21/22/23/24): {background}")
    if other:
        print(f"  unsupported/other ids: {other}")


def main():
    """Run the statistics scan."""
    parser = argparse.ArgumentParser(description="Scan TPC HDF5 hit graph statistics.")
    parser.add_argument("h5_path", type=str, help="Input merged TPC HDF5 file.")
    parser.add_argument("--n-events", type=int, default=1000, help="Number of events to scan.")
    parser.add_argument("--radii-mm", default="15,20,25", type=str, help="Comma-separated radius candidates in mm.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    radii = [float(item) for item in args.radii_mm.split(",") if item.strip()]
    stats = scan_file(args.h5_path, args.n_events, radii, args.seed)

    print("=" * 72)
    print(f"File: {args.h5_path}")
    print(f"Total events: {stats['n_total']}, scanned: {stats['n_sample']}")
    print("Hit filter: energy > 0 was already applied during HDF5 creation")
    print("=" * 72)

    summarize("hit count per event", stats["hit_counts"])
    summarize("hit x coordinate", stats["x_values"], "mm")
    summarize("hit y coordinate", stats["y_values"], "mm")
    summarize("nearest-neighbor xy distance", stats["nn_dists"], "mm")
    summarize("hit energy", stats["energies"])
    summarize("hit log_energy", stats["log_energies"])
    summarize("primary_origin_z", stats["targets"], "mm")
    print_id_distribution(stats["event_ids"])

    print("\n[directed edge count per event]")
    for radius, values in stats["edge_counts"].items():
        summarize(f"radius {radius:g} mm", values)

    print("\n--- Suggestions ---")
    if stats["nn_dists"].size:
        nn_med = float(np.median(stats["nn_dists"]))
        print(f"Median 1-NN xy distance: {nn_med:.3f} mm")
        print("Reasonable radius candidates: {:.1f}, {:.1f}, {:.1f} mm".format(2.0 * nn_med, 3.0 * nn_med, 4.0 * nn_med))
    if stats["x_values"].size and stats["y_values"].size:
        x_abs = float(np.max(np.abs(stats["x_values"])))
        y_abs = float(np.max(np.abs(stats["y_values"])))
        xy_abs = max(x_abs, y_abs)
        print(f"Observed coordinate bounds: max(|x|)={x_abs:.3f} mm, max(|y|)={y_abs:.3f} mm")
        print(f"If using one shared xy scale, a data-observed lower bound is {xy_abs:.3f} mm.")
    if stats["hit_counts"].size:
        med_hits = int(np.median(stats["hit_counts"]))
        print(f"Median hit count: {med_hits}")
        print("For about 20 hits/event, a lightweight GAT with 2-4 layers and hidden_channels 64-128 is a reasonable first pass.")


if __name__ == "__main__":
    main()


# python event_stats.py  /public/home/liuz1/work/26.03.18_rest/test_example/feature_extraction/all_events.h5  --n-events 1000 --radii-mm 12,24,36
# # radius 12 mm:  38 edges
  # radius 24 mm: 100 edges
  # radius 36 mm: 154 edges

    
