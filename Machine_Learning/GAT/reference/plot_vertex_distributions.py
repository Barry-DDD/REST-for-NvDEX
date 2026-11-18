#!/usr/bin/env python3
"""Plot the primary-vertex (electron origin) truth distributions from an HDF5 file.

The vertex is stored once per event in the ``events`` group as the common
origin of the primary electrons:

  * events/primary_origin_x
  * events/primary_origin_y
  * events/primary_origin_z

Three 1-D histograms (x, y, z) are written to a single PDF page in a 2x2 layout
(the 4th slot shows summary statistics).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_vertices(h5_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        x = f["events/primary_origin_x"][:]
        y = f["events/primary_origin_y"][:]
        z = f["events/primary_origin_z"][:]
    return {"x": x, "y": y, "z": z, "n": int(x.size)}


def _hist_axis(ax: plt.Axes, values: np.ndarray, label: str, color: str) -> None:
    # A degenerate (single-valued) coordinate, e.g. x == y == 0 for a line
    # source along z, would make matplotlib autoscale to a wide default
    # window. Pad the range explicitly so the bar stays visible.
    vmin = float(values.min())
    vmax = float(values.max())
    if vmin == vmax:
        pad = 1.0 if vmin == 0.0 else abs(vmin) * 0.1
        hist_range = (vmin - pad, vmax + pad)
    else:
        hist_range = (vmin, vmax)

    ax.hist(values, bins=100, range=hist_range, histtype="step", color=color)
    ax.set_xlabel(f"Vertex {label} [mm]")
    ax.set_ylabel("Events")
    ax.set_title(f"Primary vertex {label}")
    ax.text(
        0.97,
        0.95,
        rf"$\mu = {values.mean():.3f}$"
        + "\n"
        + rf"$\sigma = {values.std():.3f}$"
        + "\n"
        + rf"min $= {vmin:.3f}$"
        + "\n"
        + rf"max $= {vmax:.3f}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )


def plot_vertices(data: dict[str, np.ndarray], pdf_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    _hist_axis(axes[0, 0], data["x"], "x", "C0")
    _hist_axis(axes[0, 1], data["y"], "y", "C1")
    _hist_axis(axes[1, 0], data["z"], "z", "C2")

    # Summary slot.
    ax = axes[1, 1]
    ax.axis("off")
    lines = [f"Events: {data['n']}", ""]
    for key in ("x", "y", "z"):
        v = data[key]
        lines.append(
            f"{key}: mean={v.mean():.3f}  std={v.std():.3f}  "
            f"range=[{v.min():.3f}, {v.max():.3f}] mm"
        )
    ax.text(
        0.02,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        family="monospace",
        fontsize=10,
    )

    fig.suptitle(
        f"Primary vertex (electron origin) distributions - {data['n']} events",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(pdf_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot primary-vertex x/y/z truth distributions."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("input/0vbb_nodiff_2mm.h5"),
        help="Input HDF5 file (default: %(default)s).",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("input/vertex_distributions.pdf"),
        help="Output PDF file (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_vertices(args.input)
    plot_vertices(data, args.output)
    print(f"Wrote {args.output}")
    print(f"  events: {data['n']}")
    for key in ("x", "y", "z"):
        v = data[key]
        print(
            f"  {key}: mean={v.mean():.3f} std={v.std():.3f} "
            f"range=[{v.min():.3f}, {v.max():.3f}]"
        )


if __name__ == "__main__":
    main()
