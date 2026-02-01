from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from preprocessing.config_utils import Config, load_config


@dataclass(frozen=True)
class ActionBins:
    iv_edges: List[float]
    vaso_edges: List[float]

    def to_dict(self) -> Dict[str, List[float]]:
        return {"iv_edges": self.iv_edges, "vaso_edges": self.vaso_edges}


def compute_edges(values: np.ndarray) -> List[float]:
    non_zero = values[values > 0]
    if len(non_zero) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    q1, q2, q3, q4 = np.quantile(non_zero, [0.25, 0.5, 0.75, 1.0])
    return [float(q1), float(q2), float(q3), float(q4)]


def bin_action(value: float, edges: List[float]) -> int:
    if value <= 0:
        return 0
    if value <= edges[0]:
        return 1
    if value <= edges[1]:
        return 2
    if value <= edges[2]:
        return 3
    return 4


def to_action_id_25(iv_bin: int, vaso_bin: int) -> int:
    return iv_bin * 5 + vaso_bin


def from_action_id_25(action_id: int) -> Tuple[int, int]:
    iv_bin = action_id // 5
    vaso_bin = action_id % 5
    return iv_bin, vaso_bin


def load_action_bins(path: Path) -> ActionBins:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return ActionBins(iv_edges=data["iv_edges"], vaso_edges=data["vaso_edges"])


def save_action_bins(bins: ActionBins, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "action_bins.json"
    csv_path = results_dir / "action_bins.csv"
    tex_path = results_dir / "action_bins.tex"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(bins.to_dict(), f, indent=2)

    rows = [
        {"action": "iv_input", "q1": bins.iv_edges[0], "q2": bins.iv_edges[1], "q3": bins.iv_edges[2], "q4": bins.iv_edges[3]},
        {"action": "vaso_input", "q1": bins.vaso_edges[0], "q2": bins.vaso_edges[1], "q3": bins.vaso_edges[2], "q4": bins.vaso_edges[3]},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    tex = df.to_latex(index=False, float_format="%.4f")
    tex_path.write_text(tex, encoding="utf-8")


def compute_action_bins(cfg: Config) -> ActionBins:
    train = pd.read_csv(cfg.paths["train_csv"])
    iv = train[cfg.columns.get("iv_input", "iv_input")].fillna(0).to_numpy(dtype=float)
    vaso = train[cfg.columns.get("vaso_input", "vaso_input")].fillna(0).to_numpy(dtype=float)
    return ActionBins(iv_edges=compute_edges(iv), vaso_edges=compute_edges(vaso))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute action bin edges")
    parser.add_argument("--config", required=True, help="Path to final_config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    bins = compute_action_bins(cfg)
    save_action_bins(bins, cfg.results_dir)
    print(f"Saved action bins to {cfg.results_dir}")


if __name__ == "__main__":
    main()
