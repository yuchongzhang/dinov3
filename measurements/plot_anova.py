from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ANOVA variance percentages by number of Laplacian heads.")
    parser.add_argument("--inputs", required=True, nargs="+", type=Path, help="ANOVA result JSON files.")
    parser.add_argument("--output", required=True, type=Path, help="Output figure path.")
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def extract_laplacian_heads(model_name: str) -> int:
    match = re.search(r"_(\d+)L(?:_|$)", model_name)
    return int(match.group(1)) if match else 0


def load_component_percentages(path: Path) -> tuple[int, tuple[float, float, float]]:
    with path.open() as f:
        payload = json.load(f)

    model_name = payload["model_name"]
    heads = extract_laplacian_heads(model_name)
    percentages = payload.get("percent_of_total")
    if percentages is None:
        variances = payload["variances"]
        total = variances["total"]
        percentages = {
            "within_seq": 100.0 * variances["within_seq"] / total if total else 0.0,
            "within_class": 100.0 * variances["within_class"] / total if total else 0.0,
            "between_class": 100.0 * variances["between_class"] / total if total else 0.0,
        }

    return heads, (
        float(percentages["within_seq"]),
        float(percentages["within_class"]),
        float(percentages["between_class"]),
    )


def prepare_matplotlib_cache() -> None:
    cache_root = Path(f"/tmp/anova_plot_cache_{os.getuid()}")
    mpl_config_dir = Path(os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib")))
    xdg_cache_dir = Path(os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg")))
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)


def add_bar_labels(ax, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="medium",
            color="#2f3437",
        )


def main() -> int:
    args = parse_args()
    prepare_matplotlib_cache()
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "axes.labelcolor": "#2f3437",
            "axes.edgecolor": "#d0d7de",
            "font.size": 10,
            "xtick.color": "#4f565c",
            "ytick.color": "#4f565c",
        }
    )

    grouped: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for input_path in args.inputs:
        heads, values = load_component_percentages(input_path)
        grouped[heads].append(values)

    head_counts = sorted(grouped)
    if not head_counts:
        raise ValueError("No ANOVA inputs were provided.")

    means = np.array([np.array(grouped[heads], dtype=float).mean(axis=0) for heads in head_counts])
    x = np.arange(len(head_counts))
    width = 0.22
    bar_offset = 0.26
    colors = ("#4C78A8", "#F58518", "#54A24B")

    fig, ax = plt.subplots(figsize=(max(7.0, 1.35 * len(head_counts)), 5.0), facecolor="white")
    ax.set_facecolor("#fbfcfd")
    bar_style = {"edgecolor": "white", "linewidth": 1.1}
    bars_within_seq = ax.bar(x - bar_offset, means[:, 0], width, label="Within-seq", color=colors[0], **bar_style)
    bars_within_class = ax.bar(x, means[:, 1], width, label="Within-class", color=colors[1], **bar_style)
    bars_between_class = ax.bar(x + bar_offset, means[:, 2], width, label="Between-class", color=colors[2], **bar_style)

    ax.set_xlabel("Number of Laplacian heads")
    ax.set_ylabel("Percentage of total variance")
    ax.set_xticks(x)
    ax.set_xticklabels([str(heads) for heads in head_counts])
    if args.title:
        ax.set_title(args.title, pad=18)
    ax.grid(axis="y", color="#d8dee4", linewidth=0.8, alpha=0.75)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.margins(x=0.04)
    ax.set_ylim(0, max(100.0, float(means.max()) * 1.16))

    for bars in (bars_within_seq, bars_within_class, bars_between_class):
        add_bar_labels(ax, bars)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.8,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
