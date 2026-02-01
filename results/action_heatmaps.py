from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing.config_utils import load_config


def plot_heatmap(counts: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(counts, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("vaso bin")
    ax.set_ylabel("iv bin")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Action heatmaps")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    results_dir = cfg.results_dir
    policies = cfg.data.get("policies", [])

    for policy in policies:
        path = results_dir / f"policy_outputs_{policy}_{args.split}.npz"
        data = np.load(path, allow_pickle=True)
        bins = data["action_bin"]
        sofa = data["sofa_bucket"]
        counts = np.zeros((5, 5), dtype=int)
        for vaso_bin, iv_bin in bins:
            counts[iv_bin, vaso_bin] += 1

        plot_heatmap(counts, f"{policy} overall", results_dir / f"action_heatmap_{policy}_overall.png")
        for bucket in ["low", "medium", "high"]:
            mask = sofa == bucket
            bucket_counts = np.zeros((5, 5), dtype=int)
            for vaso_bin, iv_bin in bins[mask]:
                bucket_counts[iv_bin, vaso_bin] += 1
            plot_heatmap(
                bucket_counts,
                f"{policy} sofa {bucket}",
                results_dir / f"action_heatmap_{policy}_sofa_{bucket}.png",
            )

        df = pd.DataFrame(counts, columns=[f"vaso_{i}" for i in range(5)])
        df.insert(0, "iv_bin", list(range(5)))
        df.to_csv(results_dir / f"action_counts_{policy}.csv", index=False)


if __name__ == "__main__":
    main()
