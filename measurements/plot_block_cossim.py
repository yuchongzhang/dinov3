from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot block-wise average cossim by model.")
    parser.add_argument("--input-dir", type=Path, default=Path("measurements") / "block_cossim_results")
    parser.add_argument("--output", type=Path, default=Path("measurements") / "block_cossim_results" / "block_cossim.png")
    parser.add_argument("--title", default="Block-wise average cossim")
    return parser.parse_args()


def extract_laplacian_heads(model_name: str) -> int:
    match = re.search(r"_(\d+)L_", model_name)
    return int(match.group(1)) if match else 0


def prepare_matplotlib_cache() -> None:
    cache_root = Path(f"/tmp/block_cossim_plot_cache_{os.getuid()}")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


def load_result(path: Path) -> tuple[int, str, list[str], list[float]]:
    with path.open() as f:
        payload = json.load(f)

    model_name = payload.get("model_name", path.stem.replace("_block_cossim", ""))
    layer_names = payload["layer_names"]
    cossim = payload.get("all_cossim")
    if cossim is None:
        cossim = [payload["layers"][name] for name in layer_names]
    return extract_laplacian_heads(model_name), model_name, layer_names, [float(value) for value in cossim]


def main() -> int:
    args = parse_args()
    results = [load_result(path) for path in sorted(args.input_dir.glob("*_block_cossim.json"))]
    if not results:
        raise FileNotFoundError(f"No block cossim JSON files found in {args.input_dir}")

    layer_names = results[0][2]
    for _, model_name, current_layers, _ in results:
        if current_layers != layer_names:
            raise ValueError(f"{model_name} has different layer names than the first result file.")

    prepare_matplotlib_cache()
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#d0d7de",
            "axes.labelcolor": "#2f3437",
            "axes.titleweight": "semibold",
            "font.size": 10,
            "legend.frameon": False,
            "xtick.color": "#4f565c",
            "ytick.color": "#4f565c",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
    ax.set_facecolor("#fbfcfd")

    x = list(range(len(layer_names)))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]
    for index, (heads, _, _, cossim) in enumerate(sorted(results)):
        label = f"{heads} Laplacian head" if heads == 1 else f"{heads} Laplacian heads"
        ax.plot(
            x,
            cossim,
            marker="o",
            markersize=5,
            linewidth=2.2,
            color=colors[index % len(colors)],
            label=label,
        )

    ax.set_title(args.title, pad=16)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Average cossim value")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=35, ha="right")
    ax.grid(axis="y", color="#d8dee4", linewidth=0.8, alpha=0.8)
    ax.grid(axis="x", color="#edf1f5", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.margins(x=0.02, y=0.12)
    ax.legend(title="Laplacian heads", ncol=2)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
