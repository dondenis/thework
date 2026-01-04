"""Summarize mortality overall and by SOFA bins for the main RL datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np


DEFAULT_SPLITS = (
    "rl_train_data_final_cont_noterm.csv",
    "rl_val_data_final_cont_noterm.csv",
    "rl_test_data_final_cont_noterm.csv",
)


def sofa_bins(sofa_values: np.ndarray) -> np.ndarray:
    sofa = sofa_values.astype(float)
    if sofa.max() <= 1.0:
        sofa = sofa * 24.0
    bins = np.empty(len(sofa), dtype=object)
    bins[sofa < 5] = "low"
    bins[(sofa >= 5) & (sofa <= 15)] = "medium"
    bins[sofa > 15] = "high"
    return bins


def mortality_summary(df: pd.DataFrame) -> Dict[str, Any]:
    mortality = df["mortality_90d"].astype(float)
    bins = sofa_bins(df["SOFA"].to_numpy())
    summary: Dict[str, Any] = {
        "rows": int(len(df)),
        "overall_mortality": float(mortality.mean()),
    }
    for label, key in [("low", "low"), ("medium", "mid"), ("high", "high")]:
        mask = bins == label
        summary[f"mortality_{key}"] = float(mortality[mask].mean()) if np.any(mask) else float("nan")
        summary[f"rows_{key}"] = int(mask.sum())
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore mortality by SOFA group for dataset splits")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--splits", nargs="*", default=list(DEFAULT_SPLITS))
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries: Dict[str, Any] = {}
    for split in args.splits:
        split_path = args.data_dir / split
        df = pd.read_csv(split_path)
        summaries[split] = mortality_summary(df)

    print("Mortality summaries:")
    for split_name, summary in summaries.items():
        print(f"- {split_name}:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(summaries, indent=2))
        print(f"Saved JSON summary to {args.output_json}")


if __name__ == "__main__":
    main()
