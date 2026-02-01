"""Evaluate the model-based PPO+BNN expert (MB-only)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf

from preprocessing.sofa_utils import sofa_bins

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from ope import compute_direct_method, compute_phwdr, compute_phwis
from ope.phwdr import build_episodes as build_phwdr_episodes
from ope.phwis import build_episodes as build_phwis_episodes

NUM_ACTIONS = 25
DEFAULT_GAMMA = 0.99
DEFAULT_TAU = 0.25


class DynamicsModel(tf.keras.Model):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation="relu")
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.fc2 = tf.keras.layers.Dense(128, activation="relu")
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.out = tf.keras.layers.Dense(output_dim)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.fc1(x)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.dropout2(x, training=training)
        return self.out(x)


class ValueNet(tf.keras.Model):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation="relu")
        self.fc2 = tf.keras.layers.Dense(128, activation="relu")
        self.value = tf.keras.layers.Dense(1)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        return tf.squeeze(self.value(x), axis=1)


def repo_root() -> Path:
    return ROOT


def load_state_features(data_dir: Path) -> list[str]:
    return (data_dir / "state_features.txt").read_text().split()


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

def one_hot_actions(actions: np.ndarray) -> np.ndarray:
    return np.eye(NUM_ACTIONS, dtype=np.float32)[actions]


def build_transitions(
    df: pd.DataFrame, state_features: list[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = df[state_features].fillna(0).to_numpy(dtype=np.float32)
    actions = action_ids(df).astype(np.int32)
    rewards = df["reward"].fillna(0).to_numpy(dtype=np.float32)
    next_states = np.zeros_like(states)
    dones = np.zeros(len(df), dtype=np.float32)

    icustay = df["icustayid"].to_numpy()
    for idx in range(len(df)):
        if idx + 1 < len(df) and icustay[idx] == icustay[idx + 1]:
            next_states[idx] = states[idx + 1]
            dones[idx] = 0.0
        else:
            next_states[idx] = 0.0
            dones[idx] = 1.0

    return states, actions, rewards, next_states, dones


def compute_q_mb(
    dynamics_models: list[DynamicsModel],
    value_net: ValueNet,
    states: np.ndarray,
    gamma: float,
) -> np.ndarray:
    action_one_hot = np.eye(NUM_ACTIONS, dtype=np.float32)
    q_ensemble = []

    for model in dynamics_models:
        all_q = []
        for action_idx in range(NUM_ACTIONS):
            action_vec = np.repeat(action_one_hot[action_idx][None, :], states.shape[0], axis=0)
            dyn_input = np.hstack([states, action_vec])
            delta_reward = model(dyn_input, training=False).numpy()
            delta_reward = np.nan_to_num(delta_reward, nan=0.0, posinf=0.0, neginf=0.0)
            next_state = states + delta_reward[:, :-1]
            reward = delta_reward[:, -1]
            next_value = value_net(next_state).numpy()
            all_q.append(np.nan_to_num(reward + gamma * next_value, nan=0.0, posinf=0.0, neginf=0.0))
        q_ensemble.append(np.stack(all_q, axis=1))

    q_ensemble = np.stack(q_ensemble, axis=0)
    return np.min(q_ensemble, axis=0)


def episode_start_indices(df: pd.DataFrame) -> np.ndarray:
    icustay = df["icustayid"].to_numpy()
    starts = [0]
    for idx in range(1, len(df)):
        if icustay[idx] != icustay[idx - 1]:
            starts.append(idx)
    return np.asarray(starts, dtype=int)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MB-only PPO+BNN expert")
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
        "--output-dir",
        type=Path,
        default=repo_root() / "outputs" / "mb",
    )
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: To run on Colab (T4 GPU), mount Drive and set --data-dir/--output-dir
    # to your Drive paths (e.g., /content/drive/MyDrive/sepsisrl/data).

    state_features = load_state_features(args.data_dir)
    test_df = pd.read_csv(args.data_dir / args.test_csv)

    test_states, test_actions, _, _, _ = build_transitions(test_df, state_features)

    dynamics_dir = args.output_dir / "dynamics"
    dyn_models = []
    dyn_input_dim = test_states.shape[1] + NUM_ACTIONS
    dyn_output_dim = test_states.shape[1] + 1
    for weight_file in sorted(dynamics_dir.glob("model_*.weights.h5")):
        model = DynamicsModel(dyn_input_dim, dyn_output_dim)
        _ = model(tf.zeros((1, dyn_input_dim)))
        model.load_weights(weight_file)
        dyn_models.append(model)

    value_net = ValueNet(test_states.shape[1])
    _ = value_net(test_states[:1])
    value_net.load_weights(args.output_dir / "ppo_value.weights.h5")

    q_mb = compute_q_mb(dyn_models, value_net, test_states, gamma=args.gamma)
    q_mb = np.nan_to_num(q_mb, nan=0.0, posinf=0.0, neginf=0.0)

    policy_probs = tf.nn.softmax(q_mb / args.tau).numpy()
    policy_probs = np.clip(policy_probs, 1e-6, 1.0)
    policy_probs = policy_probs / policy_probs.sum(axis=1, keepdims=True)

    phwis = compute_phwis(build_phwis_episodes(test_df, policy_probs))
    phwdr = compute_phwdr(build_phwdr_episodes(test_df, policy_probs, q_mb))
    am = compute_direct_method(policy_probs, q_mb, episode_start_indices(test_df))

    metrics = {
        "phwis": phwis,
        "phwdr": phwdr,
        "am": am,
    }
    metrics.update(physician_action_counts(test_df))

    metrics_path = args.output_dir / "mb_eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    traj_path = args.output_dir / "mb_trajectory_actions.csv"
    with traj_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["icustayid", "bloc", "physician_action", "mb_action"])
        mb_action = np.argmax(q_mb, axis=1)
        for idx, row in test_df.iterrows():
            writer.writerow(
                [
                    row["icustayid"],
                    row["bloc"],
                    int(test_actions[idx]),
                    int(mb_action[idx]),
                ]
            )

    print("MB evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved trajectory actions to {traj_path}")


if __name__ == "__main__":
    main()
