"""Evaluate a trained CQL policy and compute OPE + mortality diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import tensorflow as tf

from preprocessing.sofa_utils import sofa_bins

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from ope import compute_direct_method, compute_phwdr, compute_phwis
from ope.phwdr import build_episodes as build_phwdr_episodes
from ope.phwis import build_episodes as build_phwis_episodes
from preprocessing.config_utils import load_config

DEFAULT_GAMMA = 0.99
DEFAULT_TEMPERATURE = 1.0
NUM_ACTIONS = 25


def repo_root() -> Path:
    return ROOT


def load_state_features(data_dir: Path) -> list[str]:
    with (data_dir / "state_features.txt").open() as handle:
        return handle.read().split()


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / np.sum(exp, axis=1, keepdims=True)


class DuelingQNetwork(tf.keras.Model):
    def __init__(self, input_dim: int, num_actions: int) -> None:
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(128, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization(center=True, scale=True)
        self.act1 = tf.keras.layers.LeakyReLU(alpha=0.01)

        self.fc2 = tf.keras.layers.Dense(128, use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization(center=True, scale=True)
        self.act2 = tf.keras.layers.LeakyReLU(alpha=0.01)

        kinit = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
        self.adv_head = tf.keras.layers.Dense(num_actions, kernel_initializer=kinit)
        self.val_head = tf.keras.layers.Dense(1, kernel_initializer=kinit)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.fc1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)

        stream_a, stream_v = tf.split(x, num_or_size_splits=2, axis=-1)
        adv = self.adv_head(stream_a)
        val = self.val_head(stream_v)
        return val + (adv - tf.reduce_mean(adv, axis=1, keepdims=True))


def compute_policy_outputs(
    model: tf.keras.Model,
    states: np.ndarray,
    temperature: float,
    batch_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_values = []
    for start in range(0, len(states), batch_size):
        batch = states[start : start + batch_size]
        q_values.append(model(batch).numpy())
    q_values = np.vstack(q_values)
    policy_probs = softmax(q_values, temperature)
    actions = np.argmax(q_values, axis=1)
    return q_values, policy_probs, actions


def action_ids(df: pd.DataFrame) -> np.ndarray:
    iv = df["iv_input"].fillna(0).astype(int)
    vaso = df["vaso_input"].fillna(0).astype(int)
    return (iv * 5 + vaso).to_numpy()


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


def episode_start_indices(df: pd.DataFrame) -> np.ndarray:
    icustay = df["icustayid"].to_numpy()
    starts = [0]
    for idx in range(1, len(df)):
        if icustay[idx] != icustay[idx - 1]:
            starts.append(idx)
    return np.asarray(starts, dtype=int)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CQL policy")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--train_or_load", action="store_true")
    parser.add_argument("--eval_split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root() / "data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="rl_test_data_final_cont_noterm.csv",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=repo_root() / "outputs" / "cql" / "model.weights.h5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "outputs" / "cql",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config:
        cfg = load_config(args.config)
        split_key = f"{args.eval_split}_csv"
        split_path = Path(cfg.paths[split_key])
        args.data_dir = split_path.parent
        args.test_csv = split_path.name

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: To run on Colab (T4 GPU), mount Drive and set --data-dir/--output-dir
    # to your Drive paths (e.g., /content/drive/MyDrive/sepsisrl/data).

    state_features = load_state_features(args.data_dir)
    test_df = pd.read_csv(args.data_dir / args.test_csv)
    states = test_df[state_features].to_numpy(dtype=np.float32)

    model = DuelingQNetwork(states.shape[1], NUM_ACTIONS)
    _ = model(tf.zeros((1, states.shape[1]), dtype=tf.float32))
    model.load_weights(args.model_dir)
    q_values, policy_probs, greedy_actions = compute_policy_outputs(
        model, states, args.temperature
    )

    policy_path = args.output_dir / "policy_eval_test.npz"
    np.savez(
        policy_path,
        q_values=q_values,
        policy_probs=policy_probs,
        greedy_actions=greedy_actions,
    )

    phwis_episodes = build_phwis_episodes(test_df, policy_probs)
    phwdr_episodes = build_phwdr_episodes(test_df, policy_probs, q_values)

    metrics = {
        "phwis": compute_phwis(phwis_episodes, gamma=args.gamma),
        "phwdr": compute_phwdr(phwdr_episodes, gamma=args.gamma),
        "am": compute_direct_method(
            policy_probs, q_values, episode_start_indices(test_df)
        ),
    }
    metrics.update(mortality_summary(test_df))
    metrics.update(physician_action_counts(test_df))

    metrics_path = args.output_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"Saved policy outputs to {policy_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
