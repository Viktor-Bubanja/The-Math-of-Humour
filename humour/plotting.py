"""Shared figure style: jittered strip plots with a median bar per category."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

COLORS = {
    "misdirection": "#d1495b",
    "pun": "#00798c",
    "control": "#8d99ae",
    "novel_joke": "#e07a5f",
    "famous_joke": "#d1495b",
    "joke": "#d1495b",
    "absurdist": "#6a4c93",
}


def strip_plot(ax, groups: dict[str, list[float]], labels=None, rng=None, zero_line=True):
    """One jittered column per group, a thick bar at each group's median."""
    rng = rng or np.random.default_rng(7)
    for i, (cat, values) in enumerate(groups.items()):
        x = i + rng.uniform(-0.13, 0.13, len(values))
        ax.scatter(x, values, alpha=0.75, s=42, color=COLORS[cat], edgecolors="white", linewidth=0.5)
        ax.hlines(np.median(values), i - 0.25, i + 0.25, color=COLORS[cat], linewidth=2.5)
    if zero_line:
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels or [c.capitalize() for c in groups])
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, name: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Figure written: {path}")
    return path


def results_path(name: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    return RESULTS_DIR / name
