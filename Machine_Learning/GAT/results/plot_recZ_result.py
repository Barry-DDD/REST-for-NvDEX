"""Plot TPC GAT regression training and validation results."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


try:
    import scienceplots  # noqa: F401

    plt.style.use(["science"])
except Exception:
    plt.style.use("default")

mpl.rcParams["text.usetex"] = False
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot TPC GAT primary_origin_z regression results.")
    parser.add_argument("--indir", type=str, required=True, help="Directory containing best_evaluation.npz and training_history.npz.")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory. If not set, use --indir.")
    parser.add_argument("--eval_file", type=str, default="best_evaluation.npz", help="Evaluation npz file name.")
    parser.add_argument("--history_file", type=str, default="training_history.npz", help="Training history npz file name.")
    parser.add_argument("--config_file", type=str, default="training_config.json", help="Training config json file name.")
    parser.add_argument("--output_pdf", type=str, default="all_plots.pdf", help="Output PDF file name.")
    parser.add_argument("--scatter_size", type=float, default=7.0, help="Marker size for z reconstruction scatter plot.")
    parser.add_argument("--scatter_alpha", type=float, default=0.45, help="Marker alpha for z reconstruction scatter plot.")
    parser.add_argument("--residual_bins", type=int, default=80, help="Number of bins for residual histogram.")
    parser.add_argument("--profile_bins", type=int, default=10, help="Number of bins for residual profile plots.")
    parser.add_argument(
        "--sensitive_energy_min",
        type=float,
        default=2443.0,
        help="Apply g4_sensitive_volume_energy > this threshold to evaluation plots. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--sensitive_energy_max",
        type=float,
        default=float("inf"),
        help="Apply g4_sensitive_volume_energy < this threshold to evaluation plots. Default: no upper bound.",
    )
    return parser.parse_args()


def load_history(history_path):
    """Load structured training history from npz or legacy npy."""
    if not os.path.exists(history_path):
        print(f"Warning: {history_path} not found. Skipping learning curves.")
        return None

    if history_path.endswith(".npy"):
        return np.load(history_path, allow_pickle=True).item()

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


def scaled_mae_to_cm(values, target_abs_max, target_mode):
    """Convert scaled-space MAE values to centimeters."""
    values = np.asarray(values, dtype=np.float64)
    if target_mode == "zero_one":
        return values * (2.0 * float(target_abs_max)) / 10.0
    if target_mode == "none":
        return values / 10.0
    return values * float(target_abs_max) / 10.0


def plot_learning_curves(history_path, config_path, pdf):
    """Plot loss and regression metric curves versus epoch."""
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
    target_abs_max = config.get("target_abs_max", scalar_from_history(history, "target_abs_max", 845.0))
    target_mode = config.get("target_mode", "minus_one_one")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax1.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", lw=1.8)
    ax1.plot(epochs, val_loss, label="Val Loss", color="#d62728", lw=1.8, linestyle="--")
    if best_epoch is not None:
        ax1.axvline(float(best_epoch), color="black", linestyle=":", lw=1.2, label=f"Best Epoch ({int(best_epoch)})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("SmoothL1 Loss")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend()

    plotted_metric = False
    if "train_scaled_mae" in history:
        train_mae_cm = scaled_mae_to_cm(history["train_scaled_mae"], target_abs_max, target_mode)
        ax2.plot(epochs, train_mae_cm, label="Train MAE", color="#1f77b4", lw=1.8)
        plotted_metric = True
    if "val_mae_mm" in history:
        val_mae_cm = np.asarray(history["val_mae_mm"], dtype=np.float64) / 10.0
        ax2.plot(epochs, val_mae_cm, label="Val MAE", color="#d62728", lw=1.8, linestyle="--")
        plotted_metric = True

    if plotted_metric:
        random_guess_mae_cm = float(target_abs_max) / 20.0
        ax2.axhline(random_guess_mae_cm, color="gray", linestyle=":", lw=1.2, label=f"Random Guess ({random_guess_mae_cm:.2f} cm)")
        if best_epoch is not None:
            ax2.axvline(float(best_epoch), color="black", linestyle=":", lw=1.2, label=f"Best Epoch ({int(best_epoch)})")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MAE [cm]")
        ax2.grid(True, linestyle="--", alpha=0.35)
        lines = ax2.get_lines()
        labels = [line.get_label() for line in lines]
        ax2.legend(lines, labels, loc="best")
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "Regression metric history not found", ha="center", va="center")

    fig.suptitle("Training History", y=1.02)
    plt.tight_layout()
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def check_eval_keys(eval_data, required_keys):
    """Check whether all required keys are available in the evaluation file."""
    available_keys = set(eval_data.files)
    missing_keys = [key for key in required_keys if key not in available_keys]
    if missing_keys:
        raise KeyError(f"Missing keys {missing_keys}. Available keys are: {sorted(available_keys)}")


def finite_pair(labels_cm, preds_cm):
    """Return finite true and reconstructed z arrays."""
    labels_cm = np.asarray(labels_cm, dtype=np.float64).reshape(-1)
    preds_cm = np.asarray(preds_cm, dtype=np.float64).reshape(-1)
    valid = np.isfinite(labels_cm) & np.isfinite(preds_cm)
    return labels_cm[valid], preds_cm[valid]


def plot_z_reconstruction(labels_mm, preds_mm, pdf, scatter_size=7.0, scatter_alpha=0.45, residual_bins=80):
    """Plot z_rec vs z_true and z residual distribution."""
    z_true_cm, z_rec_cm = finite_pair(np.asarray(labels_mm) / 10.0, np.asarray(preds_mm) / 10.0)
    residual_cm = z_rec_cm - z_true_cm
    mean_cm = float(np.mean(residual_cm)) if residual_cm.size else 0.0
    std_cm = float(np.std(residual_cm)) if residual_cm.size else 0.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))

    ax1.scatter(z_true_cm, z_rec_cm, s=scatter_size, alpha=scatter_alpha, color="#1f77b4", edgecolors="none")
    if z_true_cm.size:
        low = float(min(np.min(z_true_cm), np.min(z_rec_cm)))
        high = float(max(np.max(z_true_cm), np.max(z_rec_cm)))
    else:
        low, high = -85.0, 85.0
    margin = 0.04 * max(high - low, 1.0)
    low -= margin
    high += margin
    ax1.plot([low, high], [low, high], color="black", linestyle="--", lw=1.2, label="Ideal")
    ax1.set_xlim(low, high)
    ax1.set_ylim(low, high)
    ax1.set_xlabel(r"$z_\mathrm{true}$ [cm]")
    ax1.set_ylabel(r"$z_\mathrm{rec}$ [cm]")
    ax1.set_title(r"$z_\mathrm{rec}$ vs $z_\mathrm{true}$")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend()

    ax2.hist(
        residual_cm,
        bins=residual_bins,
        histtype="stepfilled",
        alpha=0.65,
        edgecolor="#1f4e79",
        color="#1f77b4",
        label=rf"mean = {mean_cm:.3f} cm" + "\n" + rf"std = {std_cm:.3f} cm",
    )
    ax2.axvline(0.0, color="black", linestyle="--", lw=1.1)
    ax2.axvline(mean_cm, color="#d62728", linestyle="-", lw=1.2)
    ax2.set_xlabel(r"$z_\mathrm{rec} - z_\mathrm{true}$ [cm]")
    ax2.set_ylabel("Events")
    ax2.set_title("Residual")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend()

    plt.tight_layout()
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def make_profile_bins(values, n_bins):
    """Return robust linearly spaced bin edges for profile plots."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if np.isclose(min_value, max_value):
        min_value -= 0.5
        max_value += 0.5
    return np.linspace(min_value, max_value, int(n_bins) + 1)


def binned_residual_stats(x_values, residual_cm, n_bins):
    """Compute residual mean and std in bins of x_values."""
    x_values = np.asarray(x_values, dtype=np.float64).reshape(-1)
    residual_cm = np.asarray(residual_cm, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x_values) & np.isfinite(residual_cm)
    x_values = x_values[valid]
    residual_cm = residual_cm[valid]
    bins = make_profile_bins(x_values, n_bins)
    if bins is None:
        return None

    centers = 0.5 * (bins[:-1] + bins[1:])
    half_widths = 0.5 * (bins[1:] - bins[:-1])
    means = np.full(centers.shape, np.nan, dtype=np.float64)
    stds = np.full(centers.shape, np.nan, dtype=np.float64)
    counts = np.zeros(centers.shape, dtype=np.int64)

    for i in range(len(centers)):
        if i == len(centers) - 1:
            mask = (x_values >= bins[i]) & (x_values <= bins[i + 1])
        else:
            mask = (x_values >= bins[i]) & (x_values < bins[i + 1])
        values = residual_cm[mask]
        counts[i] = values.size
        if values.size > 0:
            means[i] = float(np.mean(values))
            stds[i] = float(np.std(values))

    return {
        "centers": centers,
        "half_widths": half_widths,
        "means": means,
        "stds": stds,
        "counts": counts,
    }


def plot_residual_profile(x_values, residual_cm, x_label, title_prefix, pdf, n_bins=10):
    """Plot residual mean and std in bins of one event-level variable."""
    stats = binned_residual_stats(x_values, residual_cm, n_bins)
    if stats is None:
        print(f"Warning: no finite values found for {title_prefix}. Skipping residual profile.")
        return

    centers = stats["centers"]
    half_widths = stats["half_widths"]
    means = stats["means"]
    stds = stats["stds"]
    counts = stats["counts"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))

    ax1.errorbar(centers, means, xerr=half_widths, fmt="o", color="#1f77b4", capsize=2, label="Residual Mean")
    ax1.axhline(0.0, color="black", linestyle="--", lw=1.1)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(r"mean($z_\mathrm{rec} - z_\mathrm{true}$) [cm]")
    ax1.set_title(f"{title_prefix}: Residual Mean")
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2.errorbar(centers, stds, xerr=half_widths, fmt="o", color="#1f77b4", capsize=2, label="Residual Std")
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(r"std($z_\mathrm{rec} - z_\mathrm{true}$) [cm]")
    ax2.set_title(f"{title_prefix}: Residual Std")
    ax2.grid(True, linestyle="--", alpha=0.35)

    for ax, values in [(ax1, means), (ax2, stds)]:
        finite = np.isfinite(values)
        for x, y, count in zip(centers[finite], values[finite], counts[finite]):
            ax.annotate(str(int(count)), (x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7)

    plt.tight_layout()
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def eval_field(eval_data, primary_key, fallback_key=None):
    """Load one evaluation field with an optional fallback key."""
    if primary_key in eval_data.files:
        return np.asarray(eval_data[primary_key]).reshape(-1)
    if fallback_key is not None and fallback_key in eval_data.files:
        return np.asarray(eval_data[fallback_key]).reshape(-1)
    return None


def apply_mask(values, mask):
    """Apply a one-dimensional event mask to a flattened array."""
    if values is None:
        return None
    values = np.asarray(values).reshape(-1)
    if values.shape[0] != mask.shape[0]:
        raise ValueError(f"Cannot apply mask: value length {values.shape[0]} != mask length {mask.shape[0]}.")
    return values[mask]


def build_eval_mask(eval_data, labels_mm, min_threshold, max_threshold):
    """Build the event-level mask for evaluation plots."""
    n_events = np.asarray(labels_mm).reshape(-1).shape[0]
    mask = np.ones(n_events, dtype=bool)
    sensitive_energy = eval_field(eval_data, "g4_sensitive_volume_energy")
    use_min = min_threshold is not None and min_threshold > 0
    use_max = max_threshold is not None and np.isfinite(max_threshold)
    if use_min or use_max:
        if sensitive_energy is None:
            print("Warning: g4_sensitive_volume_energy not found. Evaluation plots are not energy-filtered.")
        else:
            if sensitive_energy.shape[0] != n_events:
                raise ValueError(
                    "g4_sensitive_volume_energy length does not match labels length: "
                    f"{sensitive_energy.shape[0]} vs {n_events}"
                )
            if use_min:
                mask &= sensitive_energy > float(min_threshold)
            if use_max:
                mask &= sensitive_energy < float(max_threshold)
            lower_text = f"> {min_threshold:g}" if use_min else "> -inf"
            upper_text = f"< {max_threshold:g}" if use_max else "< inf"
            print(
                "Evaluation plot filter: "
                f"{lower_text} keV and {upper_text} keV keeps {int(mask.sum())}/{n_events} events."
            )
    return mask


def print_eval_summary(labels_mm, preds_mm, eval_data):
    """Print a short summary of the loaded evaluation file."""
    labels_cm, preds_cm = finite_pair(np.asarray(labels_mm) / 10.0, np.asarray(preds_mm) / 10.0)
    residual_cm = preds_cm - labels_cm
    print("========== TPC GAT Evaluation Summary ==========")
    print(f"Number of finite events: {labels_cm.size}")
    print(f"Available keys: {sorted(eval_data.files)}")
    if residual_cm.size:
        print(f"Residual mean: {np.mean(residual_cm):.4f} cm")
        print(f"Residual std:  {np.std(residual_cm):.4f} cm")
        print(f"Residual MAE:  {np.mean(np.abs(residual_cm)):.4f} cm")
        print(f"Residual RMSE: {np.sqrt(np.mean(residual_cm**2)):.4f} cm")
    print("================================================")


def main():
    """Plot all requested result figures."""
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
    check_eval_keys(eval_data, ["labels", "preds"])
    labels_mm_all = np.asarray(eval_data["labels"]).reshape(-1)
    preds_mm_all = np.asarray(eval_data["preds"]).reshape(-1)
    eval_mask = build_eval_mask(eval_data, labels_mm_all, args.sensitive_energy_min, args.sensitive_energy_max)
    labels_mm = labels_mm_all[eval_mask]
    preds_mm = preds_mm_all[eval_mask]
    residual_cm = np.asarray(preds_mm, dtype=np.float64).reshape(-1) / 10.0 - np.asarray(labels_mm, dtype=np.float64).reshape(-1) / 10.0

    print_eval_summary(labels_mm, preds_mm, eval_data)

    with PdfPages(output_pdf_path) as pdf:
        print("Plotting learning curves...")
        plot_learning_curves(history_path, config_path, pdf)

        print("Plotting z reconstruction...")
        plot_z_reconstruction(
            labels_mm,
            preds_mm,
            pdf,
            scatter_size=args.scatter_size,
            scatter_alpha=args.scatter_alpha,
            residual_bins=args.residual_bins,
        )

        print("Plotting residual profile vs z_true...")
        z_true_cm = np.asarray(labels_mm, dtype=np.float64).reshape(-1) / 10.0
        plot_residual_profile(
            z_true_cm,
            residual_cm,
            r"$z_\mathrm{true}$ [cm]",
            r"$z_\mathrm{true}$",
            pdf,
            n_bins=args.profile_bins,
        )

        n_hits = apply_mask(eval_field(eval_data, "n_hits", fallback_key="n_hit"), eval_mask)
        if n_hits is not None:
            print("Plotting residual profile vs n_hits...")
            plot_residual_profile(n_hits, residual_cm, "n_hits", "n_hits", pdf, n_bins=args.profile_bins)
        else:
            print("Warning: n_hits/n_hit not found. Skipping residual profile vs n_hits.")

        sensitive_energy = apply_mask(eval_field(eval_data, "g4_sensitive_volume_energy"), eval_mask)
        if sensitive_energy is not None:
            print("Plotting residual profile vs g4_sensitive_volume_energy...")
            plot_residual_profile(
                sensitive_energy,
                residual_cm,
                "g4_sensitive_volume_energy [keV]",
                "g4_sensitive_volume_energy",
                pdf,
                n_bins=args.profile_bins,
            )
        else:
            print("Warning: g4_sensitive_volume_energy not found. Skipping residual profile.")

    print(f"All plots successfully saved to: {output_pdf_path}")


if __name__ == "__main__":
    main()

#python plot_result.py --indir results_r24_elecsim_12mm --output_pdf elecsim_12mm_r24_test_Ecut.pdf --sensitive_energy_min 2443 --sensitive_energy_max 2473
# python plot_result.py --indir results_r24 --output_pdf all_plots.pdf --sensitive_energy_min 0 --sensitive_energy_max inf

