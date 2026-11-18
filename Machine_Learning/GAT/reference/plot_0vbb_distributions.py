#!/usr/bin/env python3
"""Plot 0vbb kinematic distributions from a processed HDF5 file.

Distributions:
  1. g4_sensitive_volume_energy
  2. E1 / Q_0vbb
  3. E_max / Q_0vbb            (E_max = max(E1, E2))
  4. A = |E1 - E2| / (E1 + E2)
  5. E_max / (E1 + E2)
  6. E1 + E2
  7. y = (A - mean(A)) / std(A)   (standardized asymmetry)
  8. Opening angle theta between the two primary electrons [deg]
  9. cos(theta) of the opening angle

All plots are written to a single PDF page in a 5x2 layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Q-value of Xe-136 double-beta decay in MeV
Q_0VBB_MEV = 2.457


def load_data(h5_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        starts = f["events/primary_start"][:]
        counts = f["events/primary_count"][:]
        all_e_kev = f["primaries/primary_energy"][:]
        g4_sv = f["events/g4_sensitive_volume_energy"][:]
        dx = f["primaries/primary_dir_x"][:]
        dy = f["primaries/primary_dir_y"][:]
        dz = f["primaries/primary_dir_z"][:]

    # Keep only two-primary events (0vbb signal).
    mask = counts == 2
    i0 = starts[mask]
    i1 = i0 + 1

    # Convert keV -> MeV so ratios with Q_0vbb are unit-consistent.
    e1_mev = all_e_kev[i0] / 1000.0
    e2_mev = all_e_kev[i1] / 1000.0

    # Opening angle from primary direction vectors. Renormalize defensively
    # in case stored vectors are not exactly unit length.
    d1 = np.stack([dx[i0], dy[i0], dz[i0]], axis=1)
    d2 = np.stack([dx[i1], dy[i1], dz[i1]], axis=1)
    n1 = np.linalg.norm(d1, axis=1)
    n2 = np.linalg.norm(d2, axis=1)
    valid = (n1 > 0) & (n2 > 0)
    cos_theta = np.full(d1.shape[0], np.nan)
    cos_theta[valid] = np.einsum("ij,ij->i", d1[valid], d2[valid]) / (
        n1[valid] * n2[valid]
    )
    # Clip to guard against tiny numerical overshoot outside [-1, 1].
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(cos_theta))

    return {
        "g4_sv_energy": g4_sv[mask],
        "E1": e1_mev,
        "E2": e2_mev,
        "cos_theta": cos_theta,
        "theta_deg": theta_deg,
        "n_total": int(counts.size),
        "n_kept": int(mask.sum()),
        "n_valid_angle": int(valid.sum()),
    }


def plot_distributions(data: dict[str, np.ndarray], pdf_path: Path) -> None:
    e1 = data["E1"]
    e2 = data["E2"]
    e_sum = e1 + e2
    e_max = np.maximum(e1, e2)
    asym = np.abs(e1 - e2) / e_sum

    fig, axes = plt.subplots(5, 2, figsize=(11, 20.5))

    ax = axes[0, 0]
    ax.hist(data["g4_sv_energy"], bins=100, histtype="step", color="C0")
    ax.set_xlabel("Deposited Energy [keV]")
    ax.set_ylabel("Events")
    ax.set_yscale("log")
    ax.set_title("Geant4 sensitive volume deposited energy")

    ax = axes[0, 1]
    ax.hist(e1 / Q_0VBB_MEV, bins=100, range=(0.0, 1.0), histtype="step", color="C1")
    ax.set_xlabel(r"$E_1 / Q_{0\nu\beta\beta}$")
    ax.set_ylabel("Events")
    ax.set_title(rf"$E_1 / Q_{{0\nu\beta\beta}}$   ($Q={Q_0VBB_MEV}$ MeV)")

    ax = axes[1, 0]
    ax.hist(e_max / Q_0VBB_MEV, bins=100, range=(0.5, 1.0), histtype="step", color="C2")
    ax.set_xlabel(r"$E_{\max} / Q_{0\nu\beta\beta}$")
    ax.set_ylabel("Events")
    ax.set_title(r"$\max(E_1,E_2) / Q_{0\nu\beta\beta}$")

    ax = axes[1, 1]
    ax.hist(asym, bins=100, range=(0.0, 1.0), histtype="step", color="C3")
    ax.set_xlabel(r"$|E_1 - E_2| / (E_1 + E_2)$")
    ax.set_ylabel("Events")
    ax.set_title("Primary electron energy asymmetry")

    ax = axes[2, 0]
    ax.hist(e_max / e_sum, bins=100, range=(0.5, 1.0), histtype="step", color="C4")
    ax.set_xlabel(r"$E_{\max} / (E_1 + E_2)$")
    ax.set_ylabel("Events")
    ax.set_title(r"$\max(E_1,E_2) / (E_1 + E_2)$")

    ax = axes[2, 1]
    # Generator-level 0vbb: E1+E2 is a delta at Q_0vbb. Fix range so the
    # bar is visible; without an explicit range matplotlib autoscales to a
    # wide default window when all values are identical.
    ax.hist(e_sum, bins=60, range=(0.0, 3.0), histtype="step", color="C5")
    ax.set_xlabel(r"$E_1 + E_2$ [MeV]")
    ax.set_ylabel("Events")
    ax.set_title("Primary total energy")

    mu_A = float(asym.mean())
    sigma_A = float(asym.std())
    y = (asym - mu_A) / sigma_A

    ax = axes[3, 0]
    ax.hist(y, bins=100, histtype="step", color="C6")
    ax.set_xlabel(r"$y = (A - \mu)/\sigma$,   $A = |E_1-E_2|/(E_1+E_2)$")
    ax.set_ylabel("Events")
    ax.set_title("Standardized primary energy asymmetry")
    ax.text(
        0.97,
        0.95,
        rf"$\mu = {mu_A:.4f}$" + "\n" + rf"$\sigma = {sigma_A:.4f}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )

    # Opening angle theta between the two primary electrons.
    theta = data["theta_deg"]
    cos_t = data["cos_theta"]
    theta_valid = theta[np.isfinite(theta)]
    cos_valid = cos_t[np.isfinite(cos_t)]

    ax = axes[3, 1]
    ax.hist(theta_valid, bins=90, range=(0.0, 180.0), histtype="step", color="C7")
    ax.set_xlabel(r"Opening angle $\theta$ [deg]")
    ax.set_ylabel("Events")
    ax.set_title(r"Opening angle between the two primary $e^-$")
    ax.text(
        0.97,
        0.95,
        rf"$\langle\theta\rangle = {theta_valid.mean():.2f}^\circ$"
        + "\n"
        + rf"n = {theta_valid.size}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )

    ax = axes[4, 0]
    ax.hist(cos_valid, bins=100, range=(-1.0, 1.0), histtype="step", color="C8")
    ax.set_xlabel(r"$\cos\theta$")
    ax.set_ylabel("Events")
    ax.set_title(r"$\cos\theta$ of the two-electron opening angle")
    ax.text(
        0.97,
        0.95,
        rf"$\langle\cos\theta\rangle = {cos_valid.mean():.4f}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )

    # Hide the unused 10th slot in the 5x2 grid.
    axes[4, 1].axis("off")

    fig.suptitle(
        f"0vbb kinematic distributions - {data['n_kept']} two-primary events "
        f"(of {data['n_total']} total)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(pdf_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 0vbb kinematic distributions.")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("output/processed_0vbb_above1Mev.h5"),
        help="Input HDF5 file (default: %(default)s).",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("output/plots_0vbb_above1MeV.pdf"),
        help="Output PDF file (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data(args.input)
    plot_distributions(data, args.output)
    print(f"Wrote {args.output}")
    print(f"  events kept (n_primaries==2): {data['n_kept']} / {data['n_total']}")
    print(f"  events with valid opening angle: {data['n_valid_angle']}")


if __name__ == "__main__":
    main()
