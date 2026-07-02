import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Data
pitch = np.array([2, 3, 6, 8, 10, 12])

mean = np.array([
    -1.313,
     0.618,
     0.037,
    -0.962,
    -3.255,
    -1.622
])

std = np.array([
     3.090,
     3.517,
     5.669,
     9.062,
    11.851,
    15.612
])

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

# Left: mean
ax = axes[0]
ax.scatter(pitch, mean, s=80)

ax.axhline(
    y=0,
    linestyle="--",
    linewidth=1.5,
    color="black"
)

for x, y in zip(pitch, mean):
    ax.annotate(
        f"{y:.2f}",
        (x, y),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=11
    )

ax.set_ylim(-4.5, 1.5)

ax.set_xlim(1, 14)
ax.set_xlabel("Pitch [mm]")
ax.set_ylabel(r"Mean [cm]")
ax.set_title(r"Mean of ($z_{\rm rec} - z_{\rm true}$)")
ax.grid(True, alpha=0.3)

# Right: std
ax = axes[1]
ax.scatter(pitch, std, s=80)
ax.plot(pitch, std, linewidth=1.5)

for x, y in zip(pitch, std):
    ax.annotate(
        f"{y:.2f}",
        (x, y),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=11
    )

ax.set_ylim(0, 17)

ax.set_xlim(1, 14)
ax.set_xlabel("Pitch [mm]")
ax.set_ylabel(r"Std [cm]")
ax.set_title(r"Std of ($z_{\rm rec} - z_{\rm true}$)")
ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig("pitch_summary.pdf", bbox_inches="tight")
