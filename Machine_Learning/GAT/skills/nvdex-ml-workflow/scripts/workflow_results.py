"""Validate saved results against source events before metrics or plotting."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

from workflow_config import load_json, write_json, source_hashes, runtime_versions, canonical_hash, target_label
from workflow_data import finite_array, fingerprint, indices, source_targets


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def assert_close(actual, expected, name):
    if np.shape(actual) != np.shape(expected) or not np.allclose(actual, expected, rtol=1e-5, atol=2e-4):
        raise ValueError(f"{name} does not match its authoritative source.")


def regression_metrics(labels, preds):
    residual = np.asarray(preds, dtype=np.float64) - labels
    return dict(mae=float(np.mean(np.abs(residual))), rmse=float(np.sqrt(np.mean(residual**2))),
                bias=float(np.mean(residual)), residual_std=float(np.std(residual)))


def check_run(indir, root):
    """Validate complete artifacts, exact validation membership, labels, and transforms."""
    from task_spec import get_task_spec, get_target_transform, fit_cdf_transform, vertex_metrics, VERTEX_TARGET_KEYS

    indir = Path(indir).resolve()
    config = load_json(indir / "training_config.json")
    evaluation = load_npz(indir / "best_evaluation.npz")
    history = load_npz(indir / "training_history.npz")
    split = load_npz(indir / "split_indices.npz")
    if not (indir / "best_model.pt").is_file() or config.get("best_epoch", 0) < 1:
        raise ValueError("Training is incomplete: a best model and final training metadata are required.")
    task, mode = config["task"], config["target_mode"]
    spec = get_task_spec(task)
    if str(evaluation["task"]) != task or str(evaluation["target_mode"]) != mode:
        raise ValueError("Task/target mode disagree across result artifacts.")
    for key in ("train_loss", "val_loss"):
        values = finite_array(history[key], key)
        if values.shape != (config["epochs"],):
            raise ValueError("Training history does not cover every configured epoch.")
    warnings = []
    source_path = Path(config["infile"])
    if not source_path.is_absolute():
        source_path = root / source_path
    before = fingerprint(source_path)
    manifest_path = indir / "workflow_manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if canonical_hash(manifest["config"]) != manifest["config_hash"]:
            raise ValueError("Prepared configuration hash mismatch.")
        if "execution" in manifest["config"]:
            from workflow_execution import check_setup_fingerprint
            check_setup_fingerprint(manifest)
        if before != manifest["inspection"]["input"]:
            raise ValueError("Input fingerprint changed since preflight.")
        if source_hashes(root) != manifest["sources"]:
            raise ValueError("Training/workflow sources changed since preparation; use the recorded version.")
        if runtime_versions() != manifest["environment"]:
            warnings.append("The review Python/package versions differ from the prepared environment.")
        expected = manifest["config"]
        for key in ("task", "target_mode", "infile"):
            if config[key] != expected[key]:
                raise ValueError(f"Runtime {key} differs from the prepared experiment.")
        actual_vertex = config.get("vertex_source", "truth" if config.get("vertex_relative") else "none")
        if actual_vertex != expected["vertex_source"]:
            raise ValueError("Runtime vertex_source differs from preparation.")
        for key, value in expected["parameters"].items():
            if config.get(key) != value:
                raise ValueError(f"Runtime parameter {key} differs from preparation.")
    else:
        warnings.append("Legacy run: source labels are cross-checked, but the original input/code identity was not recorded.")
    with h5py.File(source_path, "r") as source:
        raw = source_targets(source, task, mode)
        n = len(raw)
        train = indices(split["train_event_index"], "train_event_index", n)
        val = indices(split["val_event_index"], "val_event_index", n)
        ids = indices(evaluation["event_index"], "event_index", n)
        for key, array in (("train", train), ("validation", val), ("evaluation", ids)):
            if not len(array) or len(np.unique(array)) != len(array):
                raise ValueError(f"Empty or duplicate {key} event indices.")
        if np.intersect1d(train, val).size or not np.array_equal(np.sort(ids), np.sort(val)):
            raise ValueError("Train/validation overlap or wrong evaluation membership.")
        usable = np.arange(n) if config.get("keep_empty_events", False) else np.flatnonzero(source["events/hit_count"][:] > 0)
        if not np.array_equal(np.sort(np.concatenate([train, val])), usable):
            raise ValueError("Split does not cover the configured usable source events.")
        assert_close(split["train_targets"], raw[train], "Training raw labels")
        assert_close(split["val_targets"], raw[val], "Validation raw labels")
        energy_sum = None
        if task == "regression_A":
            from task_spec import read_A_targets
            _, e1, e2 = read_A_targets(source)
            assert_close(evaluation["E1"].reshape(-1), e1[ids], "E1 metadata")
            assert_close(evaluation["E2"].reshape(-1), e2[ids], "E2 metadata")
            energy_sum = (e1.astype(np.float64) + e2.astype(np.float64))[usable]
        for field in ("n_hits", "n_hits_original", "total_energy", "total_energy_filtered",
                      "n_primaries", "n_tracks", "id", "g4_total_deposited_energy", "g4_sensitive_volume_energy"):
            source_key = f"events/{field}"
            if source_key not in source:
                evaluation.pop(field, None)
                warnings.append(f"Diagnostic {field} is absent in source; loader defaults are excluded.")
            elif field in evaluation:
                assert_close(evaluation[field].reshape(-1), source[source_key][:][ids], field)
        origin_keys = [f"events/primary_origin_{a}" for a in "xyz"]
        if all(key in source for key in origin_keys):
            origin = np.column_stack([source[key][:][ids] for key in origin_keys])
            finite_array(origin, "primary_origin")
            if "primary_origin" in evaluation:
                assert_close(evaluation["primary_origin"], origin, "primary_origin")
        else:
            evaluation.pop("primary_origin", None)
            warnings.append("Complete truth vertex is unavailable; R/Z truth diagnostics are skipped.")
        if "n_hit" in evaluation:
            assert_close(evaluation["n_hit"].reshape(-1), source["events/hit_count"][:][ids], "n_hit")
    if fingerprint(source_path) != before:
        raise ValueError("Source changed during result validation.")
    labels = finite_array(evaluation["labels"], "labels")
    report = dict(status="passed", task=task, target_mode=mode, source=before, count=len(ids),
                  vertex_source=config.get("vertex_source", "truth" if config.get("vertex_relative") else "none"),
                  warnings=warnings)
    if spec.is_classification:
        truth = spec.map_raw_labels(raw[ids])
        assert_close(labels, truth, "Mapped classification labels")
        assert_close(evaluation["raw_labels"], raw[ids], "Raw classification labels")
        probs = finite_array(evaluation["probabilities"], "probabilities", (len(ids), spec.num_outputs))
        logits = finite_array(evaluation["logits"], "logits", probs.shape)
        expected_probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        expected_probs /= expected_probs.sum(axis=1, keepdims=True)
        assert_close(probs, expected_probs, "Softmax probabilities")
        prediction = indices(evaluation["predicted_class"], "predicted_class", spec.num_outputs)
        assert_close(prediction, probs.argmax(axis=1), "Predicted class")
        confusion = np.zeros((spec.num_outputs, spec.num_outputs), dtype=np.int64)
        np.add.at(confusion, (truth, prediction), 1)
        assert_close(evaluation["confusion_matrix"], confusion, "Confusion matrix")
        recall = np.divide(np.diag(confusion), confusion.sum(axis=1), out=np.zeros(spec.num_outputs), where=confusion.sum(axis=1)>0)
        precision = np.divide(np.diag(confusion), confusion.sum(axis=0), out=np.zeros(spec.num_outputs), where=confusion.sum(axis=0)>0)
        report.update(accuracy=float(np.mean(prediction == truth)), confusion_matrix=confusion.tolist(),
                      class_precision=[float(x) if count else None for x, count in zip(precision, confusion.sum(axis=0))],
                      class_recall=[float(x) if count else None for x, count in zip(recall, confusion.sum(axis=1))],
                      class_support=confusion.sum(axis=1).tolist(), label_map=spec.label_map)
        if spec.num_outputs == 2:
            from utils import _binary_roc_auc
            report["roc_auc"] = float(_binary_roc_auc(truth, probs[:, spec.positive_label])) if len(np.unique(truth)) == 2 else None
    else:
        shape = (len(ids), 3) if spec.num_outputs == 3 else (len(ids),)
        finite_array(labels, "labels", shape)
        preds = finite_array(evaluation["preds"], "preds", shape)
        assert_close(labels, raw[ids], "Evaluation truth labels")
        transform = get_target_transform(task, mode, config["target_abs_max"])
        if mode == "cdf":
            transform = fit_cdf_transform(raw[train])
            for key, expected_array in (("cdf_values", transform.cdf_values), ("cdf_probabilities", transform.cdf_probabilities)):
                assert_close(split[key], expected_array, f"Training-only {key}")
                assert_close(evaluation[key], expected_array, f"Evaluation {key}")
        for key, expected_value in (("target_offset", transform.offset), ("target_scale", transform.scale)):
            if not np.isclose(config[key], expected_value) or not np.isclose(float(evaluation[key]), expected_value):
                raise ValueError(f"Incorrect {key}.")
        if config["target_unit"] != transform.unit or str(evaluation["target_unit"]) != transform.unit:
            raise ValueError("Target units disagree with the registered transform.")
        finite_array(evaluation["preds_scaled"], "preds_scaled", shape)
        assert_close(evaluation["labels_scaled"], transform.forward(labels), "Forward target transform")
        assert_close(preds, transform.inverse(evaluation["preds_scaled"]), "Inverse prediction transform")
        report.update(unit=transform.unit, model=regression_metrics(labels, preds),
                      training_mean_baseline=regression_metrics(labels, np.broadcast_to(np.mean(raw[train], axis=0), shape)))
        limits = (0, 1) if task == "regression_A" else (0, 180) if mode == "angle" else (-1, 1) if task == "regression_theta" else None
        if limits:
            report["out_of_range_fraction"] = float(np.mean((preds < limits[0]) | (preds > limits[1])))
        if task == "regression_xyz":
            if list(evaluation["target_names"]) != list(VERTEX_TARGET_KEYS):
                raise ValueError("Incorrect XYZ target order.")
            report["vertex_metrics"] = {k: np.asarray(v).tolist() for k, v in vertex_metrics(labels, preds).items()}
        if task == "regression_A":
            constant = bool(np.allclose(energy_sum, energy_sum[0], rtol=1e-6, atol=1e-5))
            report.update(emax_fixed_sum_conversion_allowed=constant,
                          primary_energy_sum=dict(min=float(energy_sum.min()), max=float(energy_sum.max())))
            if not constant:
                warnings.append("E1+E2 varies across usable events: fixed-Q Emax plots are disabled; only A is evaluated.")
    return config, evaluation, history, split, report


def summary_plot(config, evaluation, history, output):
    """Cover all registered tasks, including multiclass without binary assumptions."""
    from task_spec import get_task_spec
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401
    from matplotlib.backends.backend_pdf import PdfPages

    plt.style.use(["science", "no-latex", "grid"])
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 15, "axes.titlesize": 16})
    spec = get_task_spec(config["task"])
    with PdfPages(output) as pdf:
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for key in ("train_loss", "val_loss"):
            ax.plot(np.arange(len(history[key])) + 1, history[key], label=key)
        ax.set(xlabel="Epoch", ylabel="Loss", title=f"{spec.name}: training history")
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)
        if spec.is_classification:
            fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
            matrix = evaluation["confusion_matrix"]
            im = ax.imshow(matrix, cmap="Blues")
            for i, row in enumerate(matrix):
                for j, value in enumerate(row):
                    ax.text(j, i, str(value), ha="center", va="center")
            ax.set(xticks=range(spec.num_outputs), yticks=range(spec.num_outputs), xlabel="Predicted mapped class",
                   ylabel="True mapped class", title=f"Validation confusion; raw ID map: {spec.label_map}")
            fig.colorbar(im, ax=ax, label="Events")
            pdf.savefig(fig)
            plt.close(fig)
            from sklearn.metrics import roc_curve, auc
            fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
            for cls in range(spec.num_outputs):
                binary = evaluation["labels"] == cls
                if len(np.unique(binary)) < 2:
                    continue
                fpr, tpr, _ = roc_curve(binary, evaluation["probabilities"][:, cls])
                ax.plot(fpr, tpr, label=f"Class {cls} vs rest, AUC={auc(fpr, tpr):.4f}")
            ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Validation ROC")
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)
        else:
            labels, preds = evaluation["labels"], evaluation["preds"]
            columns = [(labels, preds)] if labels.ndim == 1 else list(zip(labels.T, preds.T))
            for component, (truth, prediction) in enumerate(columns):
                fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
                im = axes[0].hexbin(truth, prediction, gridsize=35, mincnt=1, bins="log")
                low, high = float(min(truth.min(), prediction.min())), float(max(truth.max(), prediction.max()))
                axes[0].plot([low, high], [low, high], "r--")
                unit = config["target_unit"]
                label = target_label(config["task"], config["target_mode"], component)
                axes[0].set(xlabel=f"True {label}", ylabel=f"Predicted {label}", title="Validation response")
                fig.colorbar(im, ax=axes[0], label="Events")
                axes[1].hist(prediction-truth, bins=50, histtype="step")
                axes[1].set(xlabel=f"Predicted - true [{unit}]", ylabel="Events")
                pdf.savefig(fig)
                plt.close(fig)


def review(indir, outdir, root, plots=True):
    """Write a new review directory; original artifacts are never overwritten."""
    config, evaluation, history, split, report = check_run(indir, root)
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    write_json(outdir / "metrics.json", report)
    if plots:
        summary_plot(config, evaluation, history, outdir / "workflow_summary.pdf")
        adapters = {"regression_A": "plot_A_result.py", "regression_theta": "plot_theta_result.py",
                    "regression_xyz": "plot_vertex_result.py"}
        use_adapter = config["task"] in adapters and not (
            config["task"] == "regression_A" and not report["emax_fixed_sum_conversion_allowed"])
        if use_adapter:
            staged = outdir / "validated_artifacts"
            staged.mkdir()
            write_json(staged / "training_config.json", config)
            for name, arrays in (("best_evaluation", evaluation), ("training_history", history), ("split_indices", split)):
                np.savez_compressed(staged / f"{name}.npz", **arrays)
            command = [sys.executable, str(root / "results" / adapters[config["task"]]), "--indir", str(staged)]
            with (outdir / "plot.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    lines = ["# NvDEx workflow review", "", f"Task: `{report['task']}`; mode: `{report['target_mode']}`; "
             f"vertex: `{report['vertex_source']}`; validation events: {report['count']}.", "",
             "Artifact/source consistency checks passed. This does not establish reconstruction quality or physical closure.",
             "", "Machine-readable measurements: `metrics.json`. Generated plots require visual inspection.", "",
             "## Limitations", "", *[f"- {message}" for message in report["warnings"]], "",
             "## Scientific interpretation", "", "Have the assistant cite measured metrics, plots and sample counts; "
             "separate observations, hypotheses and proposed tests. Do not treat validation-tuned thresholds as independent test performance.", ""]
    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return report
