"""Plot TPC GAT binary PID training and validation results."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


RAW_ID_NAMES = {
    1: "2nubb_ex",
    21: "2nubb_gs",
    22: "co60",
    23: "u238",
    24: "th232",
}
BACKGROUND_IDS = (21, 22, 23, 24)
RAW_ID_ORDER = (1, 21, 22, 23, 24)
DEFAULT_TRUE_ENERGY_MIN = 0.0
DEFAULT_TRUE_ENERGY_MAX = 1580.0
DEFAULT_RECONSTRUCTED_ENERGY_MIN = 1000.0
DEFAULT_RECONSTRUCTED_ENERGY_MAX = 1580.0
LEGEND_FONTSIZE = 11
plt = None
PdfPages = None


def require_matplotlib():
    """Import and configure matplotlib only when plotting is requested."""
    global plt, PdfPages
    if plt is not None and PdfPages is not None:
        return plt, PdfPages

    import matplotlib as mpl
    import matplotlib.pyplot as pyplot
    from matplotlib.backends.backend_pdf import PdfPages as PdfPagesClass

    try:
        import scienceplots  # noqa: F401

        pyplot.style.use(["science"])
    except Exception:
        pyplot.style.use("default")

    mpl.rcParams["text.usetex"] = False
    pyplot.rcParams["mathtext.fontset"] = "stix"
    pyplot.rcParams["font.family"] = "STIXGeneral"

    plt = pyplot
    PdfPages = PdfPagesClass
    return plt, PdfPages


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot TPC GAT binary PID results.")
    parser.add_argument("--indir", type=str, required=True, help="Directory containing best_evaluation.npz and training_history.npz.")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory. If not set, use --indir.")
    parser.add_argument("--eval_file", type=str, default="best_evaluation.npz", help="Evaluation npz file name.")
    parser.add_argument("--history_file", type=str, default="training_history.npz", help="Training history npz file name.")
    parser.add_argument("--config_file", type=str, default="training_config.json", help="Training config json file name.")
    parser.add_argument("--output_pdf", type=str, default="pid_all_plots.pdf", help="Output PDF file name.")
    parser.add_argument("--energy_bins", type=int, default=10, help="Number of bins for each efficiency profile.")
    parser.add_argument("--true_energy_min", type=float, default=DEFAULT_TRUE_ENERGY_MIN, help="Minimum total_energy for true-energy efficiency profiles.")
    parser.add_argument("--true_energy_max", type=float, default=DEFAULT_TRUE_ENERGY_MAX, help="Maximum total_energy for true-energy efficiency profiles.")
    parser.add_argument("--reconstructed_energy_min", type=float, default=DEFAULT_RECONSTRUCTED_ENERGY_MIN, help="Minimum g4_sensitive_volume_energy for reconstructed-energy efficiency profiles.")
    parser.add_argument("--reconstructed_energy_max", type=float, default=DEFAULT_RECONSTRUCTED_ENERGY_MAX, help="Maximum g4_sensitive_volume_energy for reconstructed-energy efficiency profiles.")
    return parser.parse_args()


def load_history(history_path):
    """Load structured training history from npz."""
    if not os.path.exists(history_path):
        print(f"Warning: {history_path} not found. Skipping learning curves.")
        return None
    data = np.load(history_path)
    return {key: data[key] for key in data.files}


def load_config(config_path):
    """Load optional training config JSON."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def scalar_from_history(history, key, default=None):
    """Read a scalar from a history dictionary."""
    if history is None or key not in history:
        return default
    value = np.asarray(history[key]).reshape(-1)
    if value.size == 0:
        return default
    return value[0].item()


def check_eval_keys(eval_data, required_keys):
    """Raise a clear error when required evaluation fields are missing."""
    available_keys = set(eval_data.files)
    missing_keys = [key for key in required_keys if key not in available_keys]
    if missing_keys:
        raise KeyError(f"Missing keys {missing_keys}. Available keys are: {sorted(available_keys)}")


def compute_roc_curve(labels, scores):
    """Compute binary ROC curve and trapezoidal AUC without sklearn."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid = np.isfinite(scores) & ((labels == 0) | (labels == 1))
    labels = labels[valid]
    scores = scores[valid]
    n_pos = int(np.count_nonzero(labels == 1))
    n_neg = int(np.count_nonzero(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]), float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]

    tpr = [0.0]
    fpr = [0.0]
    tp = 0
    fp = 0
    index = 0
    while index < sorted_labels.size:
        end = index + 1
        while end < sorted_labels.size and sorted_scores[end] == sorted_scores[index]:
            end += 1
        group = sorted_labels[index:end]
        tp += int(np.count_nonzero(group == 1))
        fp += int(np.count_nonzero(group == 0))
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)
        index = end

    if fpr[-1] != 1.0 or tpr[-1] != 1.0:
        fpr.append(1.0)
        tpr.append(1.0)
    fpr_arr = np.asarray(fpr, dtype=np.float64)
    tpr_arr = np.asarray(tpr, dtype=np.float64)
    if hasattr(np, "trapezoid"):
        auc = float(np.trapezoid(tpr_arr, fpr_arr))
    else:
        auc = float(np.trapz(tpr_arr, fpr_arr))
    return fpr_arr, tpr_arr, auc


def compute_background_rejection_curve(labels, scores):
    """Compute signal efficiency and background rejection from ROC coordinates."""
    fpr, tpr, auc = compute_roc_curve(labels, scores)
    return tpr, 1.0 - fpr, auc


def compute_significance_scan(labels, scores, thresholds=None):
    """Scan score thresholds and compute efficiency-based relative significance."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape[0] != scores.shape[0]:
        raise ValueError(f"labels length {labels.shape[0]} != scores length {scores.shape[0]}.")

    valid = np.isfinite(scores) & ((labels == 0) | (labels == 1))
    labels = labels[valid]
    scores = scores[valid]
    if thresholds is None:
        if scores.size == 0:
            thresholds = np.asarray([], dtype=np.float64)
        else:
            thresholds = np.unique(scores)[::-1]
    else:
        thresholds = np.asarray(thresholds, dtype=np.float64).reshape(-1)
        thresholds = thresholds[np.isfinite(thresholds)]

    signal_counts = np.zeros(thresholds.shape, dtype=np.int64)
    background_counts = np.zeros(thresholds.shape, dtype=np.int64)
    signal_efficiency = np.full(thresholds.shape, np.nan, dtype=np.float64)
    background_acceptance = np.full(thresholds.shape, np.nan, dtype=np.float64)
    significance = np.full(thresholds.shape, np.nan, dtype=np.float64)
    total_signal = int(np.count_nonzero(labels == 1))
    total_background = int(np.count_nonzero(labels == 0))
    for i, threshold in enumerate(thresholds):
        selected = scores >= threshold
        signal_counts[i] = int(np.count_nonzero(selected & (labels == 1)))
        background_counts[i] = int(np.count_nonzero(selected & (labels == 0)))
        if total_signal > 0:
            signal_efficiency[i] = signal_counts[i] / total_signal
        if total_background > 0:
            background_acceptance[i] = background_counts[i] / total_background
        if background_acceptance[i] > 0.0:
            significance[i] = signal_efficiency[i] / np.sqrt(background_acceptance[i])

    finite = np.isfinite(significance)
    if np.any(finite):
        finite_indices = np.flatnonzero(finite)
        best_index = int(finite_indices[np.argmax(significance[finite])])
        best_threshold = float(thresholds[best_index])
        best_significance = float(significance[best_index])
    else:
        best_index = -1
        best_threshold = float("nan")
        best_significance = float("nan")

    return {
        "thresholds": thresholds,
        "signal_counts": signal_counts,
        "background_counts": background_counts,
        "signal_efficiency": signal_efficiency,
        "background_acceptance": background_acceptance,
        "significance": significance,
        "best_index": best_index,
        "best_threshold": best_threshold,
        "best_significance": best_significance,
    }


def compute_predictions_from_threshold(signal_scores, threshold):
    """Convert signal scores to binary predictions with a score threshold."""
    signal_scores = np.asarray(signal_scores, dtype=np.float64).reshape(-1)
    return (signal_scores >= float(threshold)).astype(np.int64)


def compute_threshold_for_signal_efficiency(labels, signal_scores, target_efficiency):
    """Find the threshold whose selected signal fraction is closest to a target."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    signal_scores = np.asarray(signal_scores, dtype=np.float64).reshape(-1)
    if labels.shape[0] != signal_scores.shape[0]:
        raise ValueError(f"labels length {labels.shape[0]} != signal_scores length {signal_scores.shape[0]}.")
    target_efficiency = float(target_efficiency)
    if target_efficiency < 0.0 or target_efficiency > 1.0:
        raise ValueError("target_efficiency must be between 0 and 1.")

    signal_mask = (labels == 1) & np.isfinite(signal_scores)
    signal_only_scores = signal_scores[signal_mask]
    total_signal = int(signal_only_scores.size)
    if total_signal == 0:
        return {
            "threshold": float("nan"),
            "signal_efficiency": float("nan"),
            "signal_count": 0,
            "total_signal": 0,
        }

    thresholds = np.unique(signal_only_scores)[::-1]
    selected_counts = np.asarray([np.count_nonzero(signal_only_scores >= threshold) for threshold in thresholds], dtype=np.int64)
    efficiencies = selected_counts / total_signal
    distance = np.abs(efficiencies - target_efficiency)
    best_distance = np.min(distance)
    candidate_indices = np.flatnonzero(np.isclose(distance, best_distance))
    best_index = int(candidate_indices[0])
    return {
        "threshold": float(thresholds[best_index]),
        "signal_efficiency": float(efficiencies[best_index]),
        "signal_count": int(selected_counts[best_index]),
        "total_signal": total_signal,
    }


def compute_threshold_performance(labels, signal_scores, threshold):
    """Compute signal efficiency and background rejection for one threshold."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    signal_scores = np.asarray(signal_scores, dtype=np.float64).reshape(-1)
    if labels.shape[0] != signal_scores.shape[0]:
        raise ValueError(f"labels length {labels.shape[0]} != signal_scores length {signal_scores.shape[0]}.")

    valid = np.isfinite(signal_scores) & ((labels == 0) | (labels == 1))
    labels = labels[valid]
    signal_scores = signal_scores[valid]
    selected = signal_scores >= float(threshold)

    signal_total = int(np.count_nonzero(labels == 1))
    background_total = int(np.count_nonzero(labels == 0))
    signal_count = int(np.count_nonzero(selected & (labels == 1)))
    background_count = int(np.count_nonzero(selected & (labels == 0)))
    signal_efficiency = signal_count / signal_total if signal_total > 0 else float("nan")
    background_acceptance = background_count / background_total if background_total > 0 else float("nan")
    background_rejection = 1.0 - background_acceptance if np.isfinite(background_acceptance) else float("nan")

    return {
        "signal_efficiency": signal_efficiency,
        "background_rejection": background_rejection,
        "background_acceptance": background_acceptance,
        "signal_count": signal_count,
        "background_count": background_count,
        "signal_total": signal_total,
        "background_total": background_total,
    }


def compute_background_acceptance_scan(labels, raw_labels, signal_scores, target_signal_efficiencies):
    """Compute per-background signal-like acceptance at target signal efficiencies."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    raw_labels = np.asarray(raw_labels, dtype=np.int64).reshape(-1)
    signal_scores = np.asarray(signal_scores, dtype=np.float64).reshape(-1)
    if labels.shape[0] != raw_labels.shape[0] or labels.shape[0] != signal_scores.shape[0]:
        raise ValueError("labels, raw_labels, and signal_scores must have the same length.")

    target_signal_efficiencies = np.asarray(target_signal_efficiencies, dtype=np.float64).reshape(-1)
    thresholds = np.full(target_signal_efficiencies.shape, np.nan, dtype=np.float64)
    actual_signal_efficiencies = np.full(target_signal_efficiencies.shape, np.nan, dtype=np.float64)
    acceptance_by_id = {background_id: np.full(target_signal_efficiencies.shape, np.nan, dtype=np.float64) for background_id in BACKGROUND_IDS}

    for i, target_efficiency in enumerate(target_signal_efficiencies):
        threshold_result = compute_threshold_for_signal_efficiency(labels, signal_scores, target_efficiency)
        threshold = threshold_result["threshold"]
        thresholds[i] = threshold
        actual_signal_efficiencies[i] = threshold_result["signal_efficiency"]
        if not np.isfinite(threshold):
            continue
        for background_id in BACKGROUND_IDS:
            mask = (raw_labels == background_id) & np.isfinite(signal_scores)
            total = int(np.count_nonzero(mask))
            if total > 0:
                accepted = int(np.count_nonzero(mask & (signal_scores >= threshold)))
                acceptance_by_id[background_id][i] = accepted / total

    return {
        "target_signal_efficiencies": target_signal_efficiencies,
        "signal_efficiencies": actual_signal_efficiencies,
        "thresholds": thresholds,
        "acceptance_by_id": acceptance_by_id,
    }


def format_probability_cell(value):
    """Format probability cells without hiding small nonzero probabilities."""
    value = float(value)
    if not np.isfinite(value):
        return "--"
    if value == 0.0:
        return "0"
    if abs(value) < 0.005:
        return f"{value:.1e}"
    return f"{value:.2f}"


def compute_binary_confusion(labels, predicted_class):
    """Return row=true, column=predicted binary confusion matrix."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    predicted_class = np.asarray(predicted_class, dtype=np.int64).reshape(-1)
    matrix = np.zeros((2, 2), dtype=np.int64)
    for label, pred in zip(labels, predicted_class):
        if 0 <= int(label) <= 1 and 0 <= int(pred) <= 1:
            matrix[int(label), int(pred)] += 1
    return matrix


def compute_raw_id_prediction_matrix(raw_labels, predicted_class):
    """Return a 5x2 truth-id versus binary-prediction matrix."""
    raw_labels = np.asarray(raw_labels, dtype=np.int64).reshape(-1)
    predicted_class = np.asarray(predicted_class, dtype=np.int64).reshape(-1)
    matrix = np.zeros((len(RAW_ID_ORDER), 2), dtype=np.int64)
    id_to_row = {raw_id: row for row, raw_id in enumerate(RAW_ID_ORDER)}
    for raw_id, pred in zip(raw_labels, predicted_class):
        if int(raw_id) in id_to_row and 0 <= int(pred) <= 1:
            matrix[id_to_row[int(raw_id)], int(pred)] += 1
    return matrix, [RAW_ID_NAMES[raw_id] for raw_id in RAW_ID_ORDER]


def normalize_confusion_rows(matrix):
    """Normalize each true-class row and preserve empty rows as NaN."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {matrix.shape}.")
    row_totals = np.sum(matrix, axis=1, keepdims=True)
    scores = np.full(matrix.shape, np.nan, dtype=np.float64)
    np.divide(matrix, row_totals, out=scores, where=row_totals != 0.0)
    return scores


def get_efficiency_plot_specs(
    true_energy_min=DEFAULT_TRUE_ENERGY_MIN,
    true_energy_max=DEFAULT_TRUE_ENERGY_MAX,
    reconstructed_energy_min=DEFAULT_RECONSTRUCTED_ENERGY_MIN,
    reconstructed_energy_max=DEFAULT_RECONSTRUCTED_ENERGY_MAX,
):
    """Return metadata for the true- and reconstructed-energy efficiency pages."""
    return [
        {
            "key": "total_energy",
            "title": "True Energy [keV]",
            "energy_min": float(true_energy_min),
            "energy_max": float(true_energy_max),
        },
        {
            "key": "g4_sensitive_volume_energy",
            "title": "Reconstructed Energy [keV]",
            "energy_min": float(reconstructed_energy_min),
            "energy_max": float(reconstructed_energy_max),
        },
    ]


def get_efficiency_y_minimum(energy_key):
    """Return the y-axis lower bound for an efficiency profile."""
    if energy_key == "g4_sensitive_volume_energy":
        return 0.4
    return 0.0


def make_energy_bins(energy, n_bins, energy_min=None, energy_max=None):
    """Build linearly spaced energy bins from finite event energies."""
    energy = np.asarray(energy, dtype=np.float64).reshape(-1)
    energy = energy[np.isfinite(energy)]
    if energy.size == 0:
        return None
    lower = float(np.min(energy)) if energy_min is None else float(energy_min)
    upper = float(np.max(energy)) if energy_max is None else float(energy_max)
    if not np.isfinite(lower) or not np.isfinite(upper):
        return None
    if np.isclose(lower, upper):
        lower -= 0.5
        upper += 0.5
    if lower > upper:
        raise ValueError("energy_min must be smaller than energy_max.")
    return np.linspace(lower, upper, int(n_bins) + 1)


def compute_efficiency_profile(energy, passed, bins):
    """Compute binomial efficiency and statistical error in energy bins."""
    energy = np.asarray(energy, dtype=np.float64).reshape(-1)
    passed = np.asarray(passed, dtype=bool).reshape(-1)
    bins = np.asarray(bins, dtype=np.float64).reshape(-1)
    if energy.shape[0] != passed.shape[0]:
        raise ValueError(f"energy length {energy.shape[0]} != passed length {passed.shape[0]}.")

    centers = 0.5 * (bins[:-1] + bins[1:])
    half_widths = 0.5 * (bins[1:] - bins[:-1])
    efficiency = np.full(centers.shape, np.nan, dtype=np.float64)
    error = np.full(centers.shape, np.nan, dtype=np.float64)
    counts = np.zeros(centers.shape, dtype=np.int64)
    passed_counts = np.zeros(centers.shape, dtype=np.int64)

    for i in range(centers.size):
        if i == centers.size - 1:
            mask = (energy >= bins[i]) & (energy <= bins[i + 1])
        else:
            mask = (energy >= bins[i]) & (energy < bins[i + 1])
        mask &= np.isfinite(energy)
        counts[i] = int(np.count_nonzero(mask))
        passed_counts[i] = int(np.count_nonzero(passed[mask]))
        if counts[i] > 0:
            efficiency[i] = passed_counts[i] / counts[i]
            error[i] = np.sqrt(efficiency[i] * (1.0 - efficiency[i]) / counts[i])

    return {
        "centers": centers,
        "half_widths": half_widths,
        "counts": counts,
        "passed": passed_counts,
        "efficiency": efficiency,
        "error": error,
    }


def compute_histogram_counts(energy, bins, mask):
    """Count finite energy values inside histogram bins after applying a mask."""
    energy = np.asarray(energy, dtype=np.float64).reshape(-1)
    bins = np.asarray(bins, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if energy.shape[0] != mask.shape[0]:
        raise ValueError(f"energy length {energy.shape[0]} != mask length {mask.shape[0]}.")
    counts, _ = np.histogram(energy[mask & np.isfinite(energy)], bins=bins)
    return counts


def plot_learning_curves(history_path, config_path, pdf):
    """Plot loss and accuracy versus epoch."""
    require_matplotlib()
    history = load_history(history_path)
    if history is None:
        return
    if "train_loss" not in history or "val_loss" not in history:
        print(f"Warning: {history_path} does not contain train_loss/val_loss. Skipping learning curves.")
        return

    config = load_config(config_path)
    train_loss = np.asarray(history["train_loss"], dtype=np.float64)
    val_loss = np.asarray(history["val_loss"], dtype=np.float64)
    epochs = np.arange(1, train_loss.size + 1)
    best_epoch = config.get("best_epoch", scalar_from_history(history, "best_epoch", None))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax1.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", lw=1.8)
    ax1.plot(epochs, val_loss, label="Val Loss", color="#d62728", lw=1.8, linestyle="--")
    if best_epoch is not None:
        ax1.axvline(float(best_epoch), color="black", linestyle=":", lw=1.2, label=f"Best Epoch ({int(best_epoch)})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(fontsize=LEGEND_FONTSIZE)

    if "train_accuracy" in history and "val_accuracy" in history:
        ax2.plot(epochs, history["train_accuracy"], label="Train Accuracy", color="#1f77b4", lw=1.8)
        ax2.plot(epochs, history["val_accuracy"], label="Val Accuracy", color="#d62728", lw=1.8, linestyle="--")
        if best_epoch is not None:
            ax2.axvline(float(best_epoch), color="black", linestyle=":", lw=1.2, label=f"Best Epoch ({int(best_epoch)})")
        ax2.set_ylim(0.0, 1.05)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.legend(fontsize=LEGEND_FONTSIZE)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "Accuracy history not found", ha="center", va="center")

    fig.suptitle("PID Training History", y=1.02)
    plt.tight_layout()
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def plot_roc_curves(raw_labels, labels, signal_scores, pdf):
    """Plot overall and per-background signal-efficiency versus background-rejection curves."""
    require_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.2))

    signal_efficiency, background_rejection, auc = compute_background_rejection_curve(labels, signal_scores)
    ax.plot(signal_efficiency, background_rejection, lw=2.0, color="black", label=f"signal vs all bkg, AUC={auc:.4f}")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    raw_labels = np.asarray(raw_labels, dtype=np.int64).reshape(-1)
    signal_scores = np.asarray(signal_scores, dtype=np.float64).reshape(-1)
    for color, background_id in zip(colors, BACKGROUND_IDS):
        mask = (raw_labels == 1) | (raw_labels == background_id)
        if np.count_nonzero(raw_labels[mask] == 1) == 0 or np.count_nonzero(raw_labels[mask] == background_id) == 0:
            continue
        binary_labels = (raw_labels[mask] == 1).astype(np.int64)
        signal_eff_i, rejection_i, auc_i = compute_background_rejection_curve(binary_labels, signal_scores[mask])
        ax.plot(signal_eff_i, rejection_i, lw=1.6, color=color, label=f"signal vs {RAW_ID_NAMES[background_id]}, AUC={auc_i:.4f}")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Signal Efficiency")
    ax.set_ylabel("Background Rejection Efficiency")
    ax.set_title("Signal Efficiency vs Background Rejection")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="lower left")
    plt.tight_layout()
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def plot_significance_scan(labels, signal_scores, pdf):
    """Plot relative significance versus PID score threshold and mark the maximum."""
    require_matplotlib()
    scan = compute_significance_scan(labels, signal_scores)

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.0))
    thresholds = scan["thresholds"]
    significance = scan["significance"]
    valid = np.isfinite(significance)
    if np.any(valid):
        ax.plot(thresholds[valid], significance[valid], color="black", lw=1.8)
        best_index = scan["best_index"]
        best_threshold = scan["best_threshold"]
        best_significance = scan["best_significance"]
        best_signal = scan["signal_counts"][best_index]
        best_background = scan["background_counts"][best_index]
        performance = compute_threshold_performance(labels, signal_scores, best_threshold)
        ax.scatter([best_threshold], [best_significance], color="#d62728", zorder=3)
        ax.axvline(best_threshold, color="#d62728", linestyle="--", lw=1.2)
        ax.annotate(
            f"max={best_significance:.3f}\nt={best_threshold:.4f}\nS={best_signal}, B={best_background}"
            f"\nSig eff={performance['signal_efficiency']:.3f}"
            f"\nBkg acc={performance['background_acceptance']:.3g}"
            f"\nBkg rej={performance['background_rejection']:.3f}",
            xy=(best_threshold, best_significance),
            xytext=(10, -10),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d62728", "alpha": 0.85},
        )
    else:
        ax.text(0.5, 0.5, "No finite S/sqrt(B) points", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Signal Score Threshold")
    ax.set_ylabel(r"Relative Significance ($\epsilon_S/\sqrt{\epsilon_B}$)")
    ax.set_title("Relative Significance Scan")
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    pdf.savefig(fig, dpi=300)
    plt.close(fig)
    return scan


def plot_background_acceptance_scan(labels, raw_labels, signal_scores, pdf):
    """Plot background acceptance versus selected signal efficiency."""
    require_matplotlib()
    target_signal_efficiencies = np.asarray([0.01, 0.05] + list(np.arange(0.10, 0.91, 0.10)) + [0.95], dtype=np.float64)
    scan = compute_background_acceptance_scan(labels, raw_labels, signal_scores, target_signal_efficiencies)

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.2))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    for color, background_id in zip(colors, BACKGROUND_IDS):
        acceptance = scan["acceptance_by_id"][background_id]
        valid = np.isfinite(scan["signal_efficiencies"]) & np.isfinite(acceptance) & (acceptance > 0.0)
        if not np.any(valid):
            continue
        ax.plot(
            100.0 * scan["signal_efficiencies"][valid],
            acceptance[valid],
            marker="o",
            lw=1.8,
            color=color,
            label=RAW_ID_NAMES[background_id],
        )

    ax.set_xlabel("Signal Efficiency [%]")
    ax.set_ylabel("Background Acceptance")
    ax.set_title("Background Acceptance vs Signal Efficiency")
    ax.set_xlim(0.0, 100.0)
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def _annotate_heatmap(ax, matrix, fmt="{:d}", fontsize=12):
    """Annotate a heatmap with cell values."""
    matrix = np.asarray(matrix)
    finite_values = matrix[np.isfinite(matrix)]
    threshold = 0.5 * np.max(finite_values) if finite_values.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if np.isfinite(value) and value > threshold else "black"
            if callable(fmt):
                text = fmt(value)
            else:
                text = fmt.format(value) if np.isfinite(value) else "--"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=fontsize)


def plot_confusion_matrices(labels, raw_labels, predicted_class, pdf, title_suffix=""):
    """Plot count and row-normalized binary PID confusion heatmaps."""
    require_matplotlib()
    binary = compute_binary_confusion(labels, predicted_class)
    raw_id_matrix, row_names = compute_raw_id_prediction_matrix(raw_labels, predicted_class)
    binary_scores = normalize_confusion_rows(binary)
    raw_id_scores = normalize_confusion_rows(raw_id_matrix)

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 9.2), gridspec_kw={"height_ratios": [1.0, 1.25]})
    ax1, ax2, ax3, ax4 = axes.ravel()
    im1 = ax1.imshow(binary, cmap="Blues")
    _annotate_heatmap(ax1, binary)
    ax1.set_xticks([0, 1], ["Pred Background", "Pred Signal"], rotation=25, ha="right")
    ax1.set_yticks([0, 1], ["True Background", "True Signal"])
    ax1.set_title(f"Binary Confusion Matrix: Events{title_suffix}")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(binary_scores, cmap="Blues", vmin=0.0, vmax=1.0)
    _annotate_heatmap(ax2, binary_scores, fmt=format_probability_cell)
    ax2.set_xticks([0, 1], ["Pred Background", "Pred Signal"], rotation=25, ha="right")
    ax2.set_yticks([0, 1], ["True Background", "True Signal"])
    ax2.set_title(f"Binary Confusion Matrix: P(pred | true){title_suffix}")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    im3 = ax3.imshow(raw_id_matrix, cmap="Greens", aspect="auto")
    _annotate_heatmap(ax3, raw_id_matrix)
    ax3.set_xticks([0, 1], ["Pred Background", "Pred Signal"], rotation=25, ha="right")
    ax3.set_yticks(np.arange(len(row_names)), row_names)
    ax3.set_title(f"Truth Type vs Binary Prediction: Events{title_suffix}")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    im4 = ax4.imshow(raw_id_scores, cmap="Greens", vmin=0.0, vmax=1.0, aspect="auto")
    _annotate_heatmap(ax4, raw_id_scores, fmt=format_probability_cell)
    ax4.set_xticks([0, 1], ["Pred Background", "Pred Signal"], rotation=25, ha="right")
    ax4.set_yticks(np.arange(len(row_names)), row_names)
    ax4.set_title(f"Truth Type vs Binary Prediction: P(pred | true){title_suffix}")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    plt.tight_layout()
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def plot_reconstructed_energy_histograms(raw_labels, predicted_class, reconstructed_energy, bins, pdf, title_suffix=""):
    """Plot reconstructed-energy distributions before and after binary PID."""
    require_matplotlib()
    raw_labels = np.asarray(raw_labels, dtype=np.int64).reshape(-1)
    predicted_class = np.asarray(predicted_class, dtype=np.int64).reshape(-1)
    reconstructed_energy = np.asarray(reconstructed_energy, dtype=np.float64).reshape(-1)
    if raw_labels.shape[0] != reconstructed_energy.shape[0]:
        raise ValueError("raw_labels and reconstructed_energy must have the same length.")
    if predicted_class.shape[0] != reconstructed_energy.shape[0]:
        raise ValueError("predicted_class and reconstructed_energy must have the same length.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=False)
    truth_styles = [
        (1, "#d62728", "signal: 2nubb_ex"),
        (21, "#1f77b4", "background: 2nubb_gs"),
        (22, "#ff7f0e", "background: co60"),
        (23, "#2ca02c", "background: u238"),
        (24, "#9467bd", "background: th232"),
    ]
    for raw_id, color, label in truth_styles:
        counts = compute_histogram_counts(reconstructed_energy, bins, raw_labels == raw_id)
        ax1.stairs(counts, bins, color=color, linestyle="-", linewidth=1.6, label=label)
    total_background_counts = compute_histogram_counts(
        reconstructed_energy,
        bins,
        np.isin(raw_labels, BACKGROUND_IDS),
    )
    ax1.stairs(total_background_counts, bins, color="black", linestyle="--", linewidth=1.8, label="total background")
    ax1.set_xlabel("Reconstructed Energy [keV]")
    ax1.set_ylabel("Events")
    ax1.set_title("Truth Event-Type Distributions")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(fontsize=LEGEND_FONTSIZE)

    signal_counts = compute_histogram_counts(reconstructed_energy, bins, predicted_class == 1)
    background_counts = compute_histogram_counts(reconstructed_energy, bins, predicted_class == 0)
    ax2.stairs(signal_counts, bins, color="#d62728", linestyle="-", linewidth=1.8, label="identified as signal")
    ax2.stairs(background_counts, bins, color="black", linestyle="-", linewidth=1.8, label="identified as background")
    ax2.set_xlabel("Reconstructed Energy [keV]")
    ax2.set_ylabel("Events")
    ax2.set_title(f"PID-Classified Event Distributions{title_suffix}")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(fontsize=LEGEND_FONTSIZE)

    fig.suptitle("Reconstructed-Energy PID Distributions")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def _plot_efficiency_series(ax, profile, label, color):
    """Draw one efficiency profile with binomial errors."""
    valid = np.isfinite(profile["efficiency"])
    if not np.any(valid):
        return
    ax.errorbar(
        profile["centers"][valid],
        profile["efficiency"][valid],
        yerr=profile["error"][valid],
        fmt="o",
        ms=4,
        capsize=2,
        label=label,
        color=color,
    )


def plot_efficiency_profiles(raw_labels, predicted_class, energy, bins, energy_title, pdf, title_suffix="", y_min=0.0):
    """Plot paired total and component PID efficiencies versus one energy definition."""
    require_matplotlib()
    raw_labels = np.asarray(raw_labels, dtype=np.int64).reshape(-1)
    predicted_class = np.asarray(predicted_class, dtype=np.int64).reshape(-1)
    energy = np.asarray(energy, dtype=np.float64).reshape(-1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)
    signal_mask = raw_labels == 1
    total_background_mask = np.isin(raw_labels, BACKGROUND_IDS)
    if np.any(signal_mask):
        signal_profile = compute_efficiency_profile(energy[signal_mask], predicted_class[signal_mask] == 1, bins)
        _plot_efficiency_series(ax1, signal_profile, "signal classified as signal", "#d62728")
    if np.any(total_background_mask):
        background_profile = compute_efficiency_profile(
            energy[total_background_mask],
            predicted_class[total_background_mask] == 0,
            bins,
        )
        _plot_efficiency_series(ax1, background_profile, "all backgrounds classified as background", "#1f77b4")
    ax1.set_xlabel(energy_title)
    ax1.set_ylabel("Correct Classification Efficiency")
    ax1.set_title("PID Efficiency: Signal and Total Background")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(fontsize=LEGEND_FONTSIZE)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    for color, background_id in zip(colors, BACKGROUND_IDS):
        mask = raw_labels == background_id
        if not np.any(mask):
            continue
        profile = compute_efficiency_profile(energy[mask], predicted_class[mask] == 0, bins)
        _plot_efficiency_series(ax2, profile, f"{RAW_ID_NAMES[background_id]} classified as background", color)
    ax2.set_xlabel(energy_title)
    ax2.set_title("PID Efficiency: Background Components")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(fontsize=LEGEND_FONTSIZE)
    ax1.set_ylim(float(y_min), 1.05)
    fig.suptitle(f"PID Efficiency vs {energy_title}{title_suffix}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def print_eval_summary(eval_data, labels, raw_labels, predicted_class):
    """Print a short summary of the loaded PID evaluation file."""
    binary = compute_binary_confusion(labels, predicted_class)
    raw_id_matrix, row_names = compute_raw_id_prediction_matrix(raw_labels, predicted_class)
    accuracy = float(np.mean(np.asarray(labels) == np.asarray(predicted_class))) if np.asarray(labels).size else 0.0

    print("========== TPC GAT PID Evaluation Summary ==========")
    print(f"Number of events: {np.asarray(labels).reshape(-1).size}")
    print(f"Available keys: {sorted(eval_data.files)}")
    print(f"Binary accuracy: {accuracy:.5f}")
    print("Binary confusion matrix rows=true, cols=pred [background, signal]:")
    print(binary)
    print("Truth type vs binary prediction rows=true id, cols=pred [background, signal]:")
    for name, row in zip(row_names, raw_id_matrix):
        print(f"  {name:>9}: {row}")
    print("====================================================")


def main():
    """Plot all PID result figures."""
    args = parse_args()

    indir = args.indir
    outdir = args.outdir if args.outdir is not None else indir
    os.makedirs(outdir, exist_ok=True)

    eval_path = os.path.join(indir, args.eval_file)
    history_path = os.path.join(indir, args.history_file)
    config_path = os.path.join(indir, args.config_file)
    output_pdf_path = os.path.join(outdir, args.output_pdf)

    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")

    eval_data = np.load(eval_path)
    efficiency_specs = get_efficiency_plot_specs(
        true_energy_min=args.true_energy_min,
        true_energy_max=args.true_energy_max,
        reconstructed_energy_min=args.reconstructed_energy_min,
        reconstructed_energy_max=args.reconstructed_energy_max,
    )
    check_eval_keys(
        eval_data,
        ["probabilities", "predicted_class", "labels", "raw_labels"] + [spec["key"] for spec in efficiency_specs],
    )
    probabilities = np.asarray(eval_data["probabilities"], dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape [n_events, >=2].")
    signal_scores = probabilities[:, 1]
    predicted_class = np.asarray(eval_data["predicted_class"], dtype=np.int64).reshape(-1)
    labels = np.asarray(eval_data["labels"], dtype=np.int64).reshape(-1)
    raw_labels = np.asarray(eval_data["raw_labels"], dtype=np.int64).reshape(-1)
    efficiency_energies = {spec["key"]: np.asarray(eval_data[spec["key"]], dtype=np.float64).reshape(-1) for spec in efficiency_specs}

    n_events = labels.shape[0]
    for name, values in (
        ("predicted_class", predicted_class),
        ("raw_labels", raw_labels),
        ("signal_scores", signal_scores),
    ):
        if values.shape[0] != n_events:
            raise ValueError(f"{name} length {values.shape[0]} != labels length {n_events}.")
    for energy_key, energy_values in efficiency_energies.items():
        if energy_values.shape[0] != n_events:
            raise ValueError(f"{energy_key} length {energy_values.shape[0]} != labels length {n_events}.")

    print_eval_summary(eval_data, labels, raw_labels, predicted_class)

    _, pdf_pages_class = require_matplotlib()
    with pdf_pages_class(output_pdf_path) as pdf:
        print("Plotting learning curves...")
        plot_learning_curves(history_path, config_path, pdf)

        print("Plotting signal-efficiency versus background-rejection curves...")
        plot_roc_curves(raw_labels, labels, signal_scores, pdf)

        print("Plotting relative significance threshold scan...")
        significance_scan = plot_significance_scan(labels, signal_scores, pdf)
        best_threshold = significance_scan["best_threshold"]
        if np.isfinite(best_threshold):
            best_predicted_class = compute_predictions_from_threshold(signal_scores, best_threshold)
            threshold_suffix = f" (t={best_threshold:.4f}, max relative significance)"
        else:
            best_predicted_class = None
            threshold_suffix = ""

        print("Plotting background acceptance scan...")
        plot_background_acceptance_scan(labels, raw_labels, signal_scores, pdf)

        print("Plotting reconstructed-energy histograms...")
        reconstructed_spec = next(spec for spec in efficiency_specs if spec["key"] == "g4_sensitive_volume_energy")
        reconstructed_energy = efficiency_energies[reconstructed_spec["key"]]
        reconstructed_bins = make_energy_bins(
            reconstructed_energy,
            args.energy_bins,
            reconstructed_spec["energy_min"],
            reconstructed_spec["energy_max"],
        )
        if reconstructed_bins is None:
            raise ValueError("No finite g4_sensitive_volume_energy values are available for histogram plots.")
        plot_reconstructed_energy_histograms(
            raw_labels,
            best_predicted_class if best_predicted_class is not None else predicted_class,
            reconstructed_energy,
            reconstructed_bins,
            pdf,
            title_suffix=threshold_suffix,
        )

        if best_predicted_class is not None:
            print("Plotting best-relative-significance-threshold confusion matrices...")
            plot_confusion_matrices(labels, raw_labels, best_predicted_class, pdf, title_suffix=threshold_suffix)

            print("Plotting best-relative-significance-threshold efficiency profiles...")
            for spec in efficiency_specs:
                energy = efficiency_energies[spec["key"]]
                bins = make_energy_bins(energy, args.energy_bins, spec["energy_min"], spec["energy_max"])
                if bins is None:
                    raise ValueError(f"No finite {spec['key']} values are available for efficiency plots.")
                plot_efficiency_profiles(
                    raw_labels,
                    best_predicted_class,
                    energy,
                    bins,
                    spec["title"],
                    pdf,
                    title_suffix=threshold_suffix,
                    y_min=get_efficiency_y_minimum(spec["key"]),
                )

        else:
            print("Warning: no finite best relative-significance threshold. Skipping best-threshold confusion and efficiency plots.")

        print("Plotting fixed-signal-efficiency confusion matrices...")
        for target_efficiency in (0.10, 0.05, 0.01):
            threshold_result = compute_threshold_for_signal_efficiency(labels, signal_scores, target_efficiency)
            threshold = threshold_result["threshold"]
            if not np.isfinite(threshold):
                print(f"Warning: no finite threshold for signal efficiency {target_efficiency:.0%}. Skipping.")
                continue
            target_predicted_class = compute_predictions_from_threshold(signal_scores, threshold)
            fixed_suffix = (
                f" (target signal eff={target_efficiency:.0%}, "
                f"actual={threshold_result['signal_efficiency']:.3f}, t={threshold:.4f})"
            )
            plot_confusion_matrices(labels, raw_labels, target_predicted_class, pdf, title_suffix=fixed_suffix)

    print(f"All PID plots successfully saved to: {output_pdf_path}")


if __name__ == "__main__":
    main()


# python plot_pid_result.py --indir results_binary_id --output_pdf pid_all_plots.pdf
