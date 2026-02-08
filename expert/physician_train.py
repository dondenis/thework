"""Prepare physician policy artifacts from the training split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

NUM_ACTIONS = 25


def repo_root() -> Path:
    return ROOT


def action_ids(df: pd.DataFrame) -> np.ndarray:
    iv = df["iv_input"].fillna(0).astype(int)
    vaso = df["vaso_input"].fillna(0).astype(int)
    return (iv * 5 + vaso).to_numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train physician (behavior) policy")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root() / "data",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default="rl_train_data_final_cont.csv",
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

    train_df = pd.read_csv(args.data_dir / args.train_csv)
    actions = action_ids(train_df)
    counts = np.bincount(actions, minlength=NUM_ACTIONS).astype(float)
    probs = (counts + 1e-6) / (counts.sum() + NUM_ACTIONS * 1e-6)

    metadata = {
        "train_csv": args.train_csv,
        "num_actions": NUM_ACTIONS,
        "action_counts": counts.tolist(),
        "action_probs": probs.tolist(),
    }

    (args.output_dir / "physician_policy.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print("Saved physician policy artifacts.")


if __name__ == "__main__":
    main()
