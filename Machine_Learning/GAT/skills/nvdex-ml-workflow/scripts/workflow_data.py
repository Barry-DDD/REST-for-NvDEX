"""Read-only HDF5 checks and bounded graph diagnostics using the real loader."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import h5py
import numpy as np


def fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def finite_array(values, name, shape=None):
    values = np.asarray(values)
    if shape is not None and values.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {values.shape}.")
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite numeric values.")
    return values


def indices(values, name, upper=None):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu" or np.any(values < 0):
        raise ValueError(f"{name} must contain nonnegative integer indices.")
    if upper is not None and np.any(values >= upper):
        raise ValueError(f"{name} exceeds the source event range.")
    return values.astype(np.int64)


def source_targets(h5, task, mode):
    """Use registered target readers; never derive alternative physics labels."""
    from task_spec import cos_theta_to_deg, get_task_spec, read_A_targets, read_theta_targets, read_vertex_targets

    n = len(h5["events/hit_count"])
    spec = get_task_spec(task)
    if task == "regression_A":
        return read_A_targets(h5)[0]
    if task == "regression_theta":
        cosine = read_theta_targets(h5)
        return cos_theta_to_deg(cosine) if mode == "angle" else cosine
    if task == "regression_xyz":
        return read_vertex_targets(h5)
    target = finite_array(h5[f"events/{spec.target_key}"][:], spec.target_key, (n,))
    if spec.is_classification:
        if np.any(target != np.floor(target)):
            raise ValueError("Classification raw IDs must be integers; truncation is forbidden.")
        spec.map_raw_labels(target)
    return target


def inventory(path):
    """List schema fields without loading hit tables or claiming training readiness."""
    before = fingerprint(path)
    with h5py.File(path, "r") as h5:
        fields = {}
        for group in ("events", "hits", "primaries", "stats"):
            if group in h5:
                fields[group] = {name: dict(shape=list(ds.shape), dtype=str(ds.dtype))
                                 for name, ds in h5[group].items() if isinstance(ds, h5py.Dataset)}
        attrs = {name: str(value) for name, value in h5.attrs.items()}
    if before != fingerprint(path):
        raise ValueError("Input changed during inventory.")
    return dict(status="inventory_only", input=before, fields=fields, attributes=attrs,
                note="Field presence does not establish target validity or training readiness.")


def inspect(config, sample_events=24, max_graph_hits=2048):
    """Check all selected scalar values and offsets, then sample actual CPU graphs."""
    from task_spec import get_task_spec
    from tpc_graph_dataset import TPCGraphDataset

    if sample_events < 1 or max_graph_hits < 1:
        raise ValueError("Diagnostic sample limits must be positive.")
    path, p = config["infile"], config["parameters"]
    before = fingerprint(path)
    warnings = ["HDF5 stats are read as stored; their training-only provenance is not established.",
                "File identity uses path/size/mtime, not a full content hash.",
                "The existing runner splits event rows; independence of related source events is not established."]
    values_for_plots = {}
    with h5py.File(path, "r") as h5:
        counts = indices(h5["events/hit_count"][:], "hit_count")
        starts = indices(h5["events/hit_start"][:], "hit_start")
        n = len(counts)
        if n == 0 or starts.shape != counts.shape:
            raise ValueError("Empty dataset or mismatched event offsets/counts.")
        total_hits = len(h5["hits/x"])
        if np.any(starts > total_hits) or np.any(counts > total_hits - starts):
            raise ValueError("Hit offsets/counts exceed the flat hit table.")
        occupied = np.flatnonzero(counts)
        order = occupied[np.argsort(starts[occupied])]
        if len(order) > 1 and np.any(starts[order[1:]] < starts[order[:-1]] + counts[order[:-1]]):
            raise ValueError("Event hit slices overlap.")
        fields = set(p["features"]) | {"x", "y", "energy", "log_energy", "module_id"}
        if config["vertex_source"] != "none" or config["task"] == "regression_xyz":
            fields.add("z")
        for field in fields:
            ds = h5[f"hits/{field}"]
            if ds.shape != (total_hits,):
                raise ValueError(f"hits/{field} has an inconsistent shape.")
            for start in range(0, total_hits, 1_000_000):
                chunk = finite_array(ds[start:start + 1_000_000], f"hits/{field}")
                if field == "energy" and np.any(chunk <= 0):
                    raise ValueError("The current HDF5 contract requires positive-energy stored hits.")
                if field == "module_id" and np.any(chunk != np.floor(chunk)):
                    raise ValueError("module_id must contain integer values.")
        target = source_targets(h5, config["task"], config["target_mode"])
        finite_array(target, "raw targets", (n, 3) if config["task"] == "regression_xyz" else (n,))
        if config["task"] in {"regression_z", "regression_xyz"} and config["target_mode"] != "none":
            if np.any(np.abs(target) > p["target_abs_max"]):
                raise ValueError("Vertex targets exceed target_abs_max; refusing clipped truth labels.")
        required_events = list(p["event_features"])
        if config["vertex_source"] == "truth":
            required_events += [f"primary_origin_{a}" for a in "xyz"]
        for field in required_events:
            ds = h5[f"events/{field}"]
            if not ds.shape or ds.shape[0] != n:
                raise ValueError(f"events/{field} does not align with event rows.")
            finite_array(ds[:], f"events/{field}")
        presence = {name: f"events/{name}" in h5 for name in
                    (*TPCGraphDataset.EVENT_INT_KEYS, *TPCGraphDataset.EVENT_FLOAT_KEYS,
                     *(f"primary_origin_{a}" for a in "xyz"))}
        for field, present in presence.items():
            if present:
                data = finite_array(h5[f"events/{field}"][:], f"events/{field}", (n,))
                if field in TPCGraphDataset.EVENT_INT_KEYS and np.any(data != np.floor(data)):
                    raise ValueError(f"events/{field} must contain integer metadata.")
        if "events/primary_start" in h5 and "events/primary_count" in h5:
            primary_starts = indices(h5["events/primary_start"][:], "primary_start")
            primary_counts = indices(h5["events/primary_count"][:], "primary_count")
            if primary_starts.shape != (n,) or primary_counts.shape != (n,):
                raise ValueError("Primary offsets/counts must align with all event rows.")
            for field in (*TPCGraphDataset.PRIMARY_FLOAT_KEYS, "primary_particle_name"):
                key = f"primaries/{field}"
                if key in h5:
                    ds = h5[key]
                    if ds.ndim != 1 or np.any(primary_starts > len(ds)) or np.any(primary_counts > len(ds)-primary_starts):
                        raise ValueError(f"Primary slices exceed {key}.")
                    if field != "primary_particle_name":
                        finite_array(ds[:], key)
        stat_names = [x.decode() if isinstance(x, bytes) else str(x) for x in h5["stats/feature_names"][:]]
        if len(set(stat_names)) != len(stat_names):
            raise ValueError("Duplicate stats/feature_names.")
        for name in ("mean", "std", "min", "max"):
            finite_array(h5[f"stats/{name}"][:], f"stats/{name}", (len(stat_names),))
        if np.any(h5["stats/std"][:] < 0) or np.any(h5["stats/min"][:] > h5["stats/max"][:]):
            raise ValueError("HDF5 statistics have negative standard deviations or reversed extrema.")
        usable = np.arange(n) if p["keep_empty_events"] else occupied
        if len(usable) < 2:
            raise ValueError("At least two usable events are required.")
        spec = get_task_spec(config["task"])
        if spec.is_classification:
            classes = spec.map_raw_labels(target[usable])
            unique, class_counts = np.unique(classes, return_counts=True)
            if len(unique) != spec.num_outputs or np.any(class_counts < 2):
                raise ValueError("Every registered class needs at least two usable events for train/validation.")
            nval = sum(min(int(c) - 1, max(1, int(round(c * p["val_fraction"])))) for c in class_counts)
        else:
            nval = max(1, int(round(len(usable) * p["val_fraction"])))
        if len(usable) - nval < p["batch_size"]:
            raise ValueError("The training loader would have zero full batches; reduce batch_size explicitly.")
        if config["task"] == "regression_theta" and config["target_mode"] == "cdf" and len(usable) - nval < 2:
            raise ValueError("CDF fitting requires at least two training events.")
        values_for_plots["target"] = target[usable]
        values_for_plots["hit_count"] = counts[usable]

    with contextlib.redirect_stdout(io.StringIO()):
        dataset = TPCGraphDataset(path, node_feature_names=p["features"], radius_mm=p["radius_mm"],
                    task=config["task"], target_mode=config["target_mode"], target_abs_max=p["target_abs_max"],
                    xy_abs_max=p["xy_abs_max"], z_abs_max=p["z_abs_max"], feature_norm_mode=p["feature_norm_mode"],
                    event_feature_keys=p["event_features"], vertex_relative=config["vertex_source"] == "truth",
                    vertex_centroid=config["vertex_source"] == "centroid", drop_empty_events=not p["keep_empty_events"])
    sampled = np.sort(np.random.default_rng(p["seed"]).choice(len(dataset), min(sample_events, len(dataset)), replace=False))
    graph_rows, skipped = [], []
    hit_values = {name: [] for name in p["features"]}
    try:
        for row in sampled:
            if dataset.hit_counts[row] > max_graph_hits:
                skipped.append(int(dataset.event_indices[row]))
                continue
            graph = dataset[int(row)]
            for key in ("x", "edge_attr", "y"):
                finite_array(getattr(graph, key).numpy(), key)
            degree = np.bincount(graph.edge_index[0].numpy(), minlength=graph.num_nodes)
            graph_rows.append(dict(event_index=int(dataset.event_indices[row]), nodes=int(graph.num_nodes),
                                   edges=int(graph.edge_index.shape[1]), isolated_fraction=float(np.mean(degree == 0))))
            event = int(dataset.event_indices[row])
            start, count = int(starts[event]), int(counts[event])
            for name in hit_values:
                hit_values[name].append(dataset.file[f"hits/{name}"][start:start+count])
        if not graph_rows:
            raise ValueError("No sampled graph fits max_graph_hits; inspect high-multiplicity events before proceeding.")
        node_dim = dataset.node_feature_dim
    finally:
        dataset.close()
    if skipped:
        warnings.append(f"Skipped graph construction above max_graph_hits for event rows {skipped}.")
    if any(row["isolated_fraction"] > 0.5 for row in graph_rows):
        warnings.append("Some sampled graphs have >50% isolated nodes before GAT self-loops; review radius_mm.")
    if before != fingerprint(path):
        raise ValueError("Input changed during inspection.")
    values_for_plots["sampled_hits"] = {name: np.concatenate(parts) for name, parts in hit_values.items()}
    report = dict(status="passed", scope="full selected scalar/offset scan plus bounded sampled CPU graphs",
                  input=before, task=config["task"], target_mode=config["target_mode"],
                  vertex_source=config["vertex_source"], total_events=n, usable_events=len(usable),
                  dropped_empty_events=n-len(usable), node_feature_dim=node_dim,
                  metadata_presence=presence, graph_sample=graph_rows, warnings=warnings)
    return report, values_for_plots


def plot_inspection(report, values, output):
    """Produce input distributions and graph diagnostics with the project style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401
    from matplotlib.backends.backend_pdf import PdfPages
    from workflow_config import target_label

    plt.style.use(["science", "no-latex", "grid"])
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 15, "axes.titlesize": 16})
    with PdfPages(output) as pdf:
        target = np.asarray(values["target"])
        columns = [target] if target.ndim == 1 else list(target.T)
        for axis, value in enumerate(columns):
            fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
            ax.hist(value, bins=50, histtype="step")
            ax.set(xlabel=target_label(report["task"], report["target_mode"], axis),
                   ylabel="Events", title=f"Input targets, N={len(value)}")
            pdf.savefig(fig)
            plt.close(fig)
        for name, data in values.get("sampled_hits", {}).items():
            fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
            ax.hist(data, bins=50, histtype="step")
            unit = "mm" if name in {"x", "y", "z"} else "stored units"
            ax.set(xlabel=f"Hit {name} [{unit}]", ylabel="Hits in sampled events", title="Sampled input feature distribution")
            pdf.savefig(fig)
            plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        axes[0].hist(values["hit_count"], bins=40, histtype="step")
        axes[0].set(xlabel="Stored hits per event", ylabel="Events")
        axes[1].scatter([x["nodes"] for x in report["graph_sample"]], [x["edges"] for x in report["graph_sample"]])
        axes[1].set(xlabel="Nodes", ylabel="Directed edges before self-loops", title="Sampled actual loader graphs")
        pdf.savefig(fig)
        plt.close(fig)
