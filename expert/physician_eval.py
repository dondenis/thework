"""Evaluate physician (behavior) policy on the test split."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from ope import compute_direct_method, compute_phwdr, compute_phwis
from ope.phwdr import build_episodes as build_phwdr_episodes
from ope.phwis import build_episodes as build_phwis_episodes

NUM_ACTIONS = 25


def repo_root() -> Path:
    return ROOT


def action_ids(df: pd.DataFrame) -> np.ndarray:
    iv = df["iv_input"].fillna(0).astype(int)
    vaso = df["vaso_input"].fillna(0).astype(int)
    return (iv * 5 + vaso).to_numpy()


def episode_start_indices(df: pd.DataFrame) -> np.ndarray:
    icustay = df["icustayid"].to_numpy()
    starts = [0]
    for idx in range(1, len(df)):
        if icustay[idx] != icustay[idx - 1]:
            starts.append(idx)
    return np.asarray(starts, dtype=int)


def sofa_bins(sofa_values: np.ndarray) -> np.ndarray:
    sofa = sofa_values.astype(float)
    if sofa.max() <= 1.0:
        sofa = sofa * 24.0
    bins = np.empty(len(sofa), dtype=object)
    bins[sofa < 5] = "low"
    bins[(sofa >= 5) & (sofa <= 15)] = "medium"
    bins[sofa > 15] = "high"
    return bins


def mortality_summary(df: pd.DataFrame) -> Dict[str, float]:
    mortality = df["mortality_90d"].astype(float)
    bins = sofa_bins(df["SOFA"].to_numpy())

    summary = {
        "overall_mortality": float(mortality.mean()),
    }
    for label in ["low", "medium", "high"]:
        mask = bins == label
        if not np.any(mask):
            summary[f"mortality_{label}"] = float("nan")
            continue
        summary[f"mortality_{label}"] = float(mortality[mask].mean())
    return summary


def physician_action_counts(df: pd.DataFrame) -> Dict[str, Any]:
    actions = action_ids(df)
    counts = np.bincount(actions, minlength=NUM_ACTIONS)[1:]
    bins = sofa_bins(df["SOFA"].to_numpy())
    by_sofa = {}
    for label, key in [("low", "low"), ("medium", "mid"), ("high", "high")]:
        mask = bins == label
        if not np.any(mask):
            by_sofa[key] = [0] * (NUM_ACTIONS - 1)
        else:
            bin_counts = np.bincount(actions[mask], minlength=NUM_ACTIONS)[1:]
            by_sofa[key] = bin_counts.astype(int).tolist()
    return {
        "physician_action_counts_24": counts.astype(int).tolist(),
        "physician_action_counts_24_by_sofa": by_sofa,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate physician policy")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root() / "data",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="rl_test_data_final_cont_noterm.csv",
    )
    parser.add_argument(
        "--policy-json",
        type=Path,
        default=repo_root() / "outputs" / "physician" / "physician_policy.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "outputs" / "physician",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: To run on Colab (T4 GPU), mount Drive and set --data-dir/--output-dir
    # to your Drive paths (e.g., /content/drive/MyDrive/sepsisrl/data).

    test_df = pd.read_csv(args.data_dir / args.test_csv)
    actions = action_ids(test_df)

    policy_meta = json.loads(args.policy_json.read_text())
    action_probs = np.array(policy_meta["action_probs"], dtype=float)

    policy_probs = np.zeros((len(test_df), NUM_ACTIONS), dtype=np.float32)
    policy_probs[np.arange(len(test_df)), actions] = 1.0
    policy_probs = np.clip(policy_probs, 1e-6, 1.0)
    policy_probs = policy_probs / policy_probs.sum(axis=1, keepdims=True)

    q_values = np.zeros_like(policy_probs)

    phwis = compute_phwis(build_phwis_episodes(test_df, policy_probs))
    phwdr = compute_phwdr(build_phwdr_episodes(test_df, policy_probs, q_values))
    am = compute_direct_method(policy_probs, q_values, episode_start_indices(test_df))

    metrics = {
        "phwis": phwis,
        "phwdr": phwdr,
        "am": am,
    }
    metrics.update(mortality_summary(test_df))
    metrics.update(physician_action_counts(test_df))

    metrics_path = args.output_dir / "physician_eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    traj_path = args.output_dir / "physician_trajectory_actions.csv"
    with traj_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["icustayid", "bloc", "physician_action"])
        for idx, row in test_df.iterrows():
            writer.writerow([row["icustayid"], row["bloc"], int(actions[idx])])

    print("Physician evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved trajectory actions to {traj_path}")


if __name__ == "__main__":
    main()
