"""Summarize survivability endpoints overall and by SOFA bins for RL datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np


DEFAULT_SPLITS = (
    "rl_train_data_final_cont.csv",
    "rl_val_data_final_cont.csv",
    "rl_test_data_final_cont.csv",
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
    in_hosp = df["died_in_hosp"].astype(float)
    bins = sofa_bins(df["SOFA"].to_numpy())

    terminal = df.groupby("icustayid", sort=False).tail(1).copy()
    terminal_reward = pd.to_numeric(terminal["reward"], errors="coerce")
    terminal_death = (terminal_reward < 0).astype(float)

    summary: Dict[str, Any] = {
        "rows": int(len(df)),
        "overall_in_hosp_mortality": float(in_hosp.mean()),
        "stays": int(terminal["icustayid"].nunique()),
        "overall_terminal_reward_mortality": float(terminal_death.mean()),
        "terminal_reward_min": float(np.nanmin(terminal_reward.to_numpy())),
        "terminal_reward_max": float(np.nanmax(terminal_reward.to_numpy())),
    }

    for label, key in [("low", "low"), ("medium", "mid"), ("high", "high")]:
        mask = bins == label
        summary[f"in_hosp_mortality_{key}"] = float(in_hosp[mask].mean()) if np.any(mask) else float("nan")
        summary[f"rows_{key}"] = int(mask.sum())

    if "died_in_hosp" in terminal.columns:
        terminal_died = pd.to_numeric(terminal["died_in_hosp"], errors="coerce").fillna(0).clip(0, 1)
        summary["terminal_vs_in_hosp_agreement"] = float((terminal_death == terminal_died).mean())

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore in-hospital and terminal-reward mortality by SOFA group")
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

    print("Survivability endpoint summaries:")
    for split_name, summary in summaries.items():
        print(f"- {split_name}:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(summaries, indent=2))
        print(f"Saved JSON summary to {args.output_json}")


if __name__ == "__main__":
    main()
