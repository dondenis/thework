"""Evaluate hybrid (CQL + model-based PPO/BNN) policy with gating."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from ope import compute_direct_method, compute_phwdr, compute_phwis
from ope.phwdr import build_episodes as build_phwdr_episodes
from ope.phwis import build_episodes as build_phwis_episodes

NUM_ACTIONS = 25
DEFAULT_GAMMA = 0.99
DEFAULT_TAU = 0.25
DEFAULT_DELTA = 0.0

FEATURES_FOR_GATE = [
    "age",
    "elixhauser",
    "SOFA",
    "FiO2_1",
    "BUN",
    "GCS",
    "Albumin",
]


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


class PolicyNet(tf.keras.Model):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation="relu")
        self.fc2 = tf.keras.layers.Dense(128, activation="relu")
        self.logits = tf.keras.layers.Dense(NUM_ACTIONS)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        return self.logits(x)


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


def repo_root() -> Path:
    return ROOT


def load_state_features(data_dir: Path) -> list[str]:
    return (data_dir / "state_features.txt").read_text().split()


def action_ids(df: pd.DataFrame) -> np.ndarray:
    iv = df["iv_input"].fillna(0).astype(int)
    vaso = df["vaso_input"].fillna(0).astype(int)
    return (iv * 5 + vaso).to_numpy()


def one_hot_actions(actions: np.ndarray) -> np.ndarray:
    return np.eye(NUM_ACTIONS, dtype=np.float32)[actions]


def build_transitions(
    df: pd.DataFrame, state_features: list[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = df[state_features].fillna(0).to_numpy(dtype=np.float32)
    actions = action_ids(df).astype(np.int32)
    rewards = df["reward"].to_numpy(dtype=np.float32)
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


def dynamics_residuals(
    dynamics_models: list[DynamicsModel],
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
) -> Tuple[float, float]:
    action_one_hot = one_hot_actions(actions)
    inputs = np.hstack([states, action_one_hot])

    preds = []
    for model in dynamics_models:
        delta_reward = model(inputs, training=False).numpy()
        delta_reward = np.nan_to_num(delta_reward, nan=0.0, posinf=0.0, neginf=0.0)
        preds.append(states + delta_reward[:, :-1])

    preds = np.stack(preds, axis=0)
    mean_pred = preds.mean(axis=0)
    mse = float(np.mean((mean_pred - next_states) ** 2))

    dispersion = np.std(np.linalg.norm(preds - mean_pred[None, :, :], axis=2))
    return mse, float(dispersion)


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


def gate_features(df: pd.DataFrame, traj_lengths: np.ndarray, max_neighbor_dist: np.ndarray) -> np.ndarray:
    base = df[FEATURES_FOR_GATE].fillna(0).to_numpy(dtype=np.float32)
    return np.column_stack([base, traj_lengths, max_neighbor_dist])


def compute_traj_lengths(df: pd.DataFrame) -> np.ndarray:
    counts = df.groupby("icustayid").size().to_dict()
    return df["icustayid"].map(counts).to_numpy(dtype=np.float32)


def max_neighbor_distance(states: np.ndarray, ref_states: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(ref_states) - 1)
    distances = []
    for state in states:
        dist = np.linalg.norm(ref_states - state, axis=1)
        topk = np.partition(dist, k)[:k]
        distances.append(np.max(topk))
    return np.array(distances, dtype=np.float32)


def clinician_action_mask(
    states: np.ndarray,
    train_states: np.ndarray,
    train_actions: np.ndarray,
    k: int,
    min_freq: float = 0.01,
) -> Tuple[np.ndarray, float]:
    k = min(k, len(train_states) - 1)
    masks = []
    masked_rates = []
    for state in states:
        dist = np.linalg.norm(train_states - state, axis=1)
        topk_idx = np.argpartition(dist, k)[:k]
        neighbor_actions = train_actions[topk_idx]
        counts = np.bincount(neighbor_actions, minlength=NUM_ACTIONS)
        freqs = counts / counts.sum()
        mask = freqs < min_freq
        masks.append(mask)
        masked_rates.append(np.mean(mask))
    return np.array(masks, dtype=bool), float(np.mean(masked_rates))


def apply_action_mask(policy_probs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask, 0.0, policy_probs)
    row_sums = masked.sum(axis=1, keepdims=True)
    zero_rows = row_sums == 0
    row_sums[zero_rows] = 1.0
    normalized = masked / row_sums
    if np.any(zero_rows):
        normalized[zero_rows[:, 0]] = 1.0 / masked.shape[1]
    return normalized


def episode_start_indices(df: pd.DataFrame) -> np.ndarray:
    icustay = df["icustayid"].to_numpy()
    starts = [0]
    for idx in range(1, len(df)):
        if icustay[idx] != icustay[idx - 1]:
            starts.append(idx)
    return np.asarray(starts, dtype=int)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hybrid gating policy")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root() / "data",
    )
    parser.add_argument("--train-csv", type=str, default="rl_train_data_final_cont_noterm.csv")
    parser.add_argument("--test-csv", type=str, default="rl_test_data_final_cont_noterm.csv")
    parser.add_argument(
        "--cql-weights",
        type=Path,
        default=repo_root() / "outputs" / "cql" / "model.weights.h5",
    )
    parser.add_argument(
        "--hybrid-dir",
        type=Path,
        default=repo_root() / "outputs" / "hybrid",
    )
    parser.add_argument("--neighbor-k", type=int, default=10)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--mask-actions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.hybrid_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: To run on Colab (T4 GPU), mount Drive and set --data-dir/--hybrid-dir
    # to your Drive paths (e.g., /content/drive/MyDrive/sepsisrl/data).

    state_features = load_state_features(args.data_dir)
    train_df = pd.read_csv(args.data_dir / args.train_csv)
    test_df = pd.read_csv(args.data_dir / args.test_csv)

    train_states = train_df[state_features].to_numpy(dtype=np.float32)
    train_actions = action_ids(train_df)

    test_states, test_actions, _, test_next, _ = build_transitions(
        test_df, state_features
    )

    cql_model = DuelingQNetwork(test_states.shape[1], NUM_ACTIONS)
    _ = cql_model(test_states[:1])
    cql_model.load_weights(args.cql_weights)
    q_cql = np.nan_to_num(cql_model(test_states).numpy(), nan=0.0, posinf=0.0, neginf=0.0)

    dynamics_dir = args.hybrid_dir / "dynamics"
    dyn_models = []
    dyn_input_dim = test_states.shape[1] + NUM_ACTIONS
    dyn_output_dim = test_states.shape[1] + 1
    for weight_file in sorted(dynamics_dir.glob("model_*.weights.h5")):
        model = DynamicsModel(dyn_input_dim, dyn_output_dim)
        _ = model(tf.zeros((1, dyn_input_dim)))
        model.load_weights(weight_file)
        dyn_models.append(model)

    policy = PolicyNet(test_states.shape[1])
    value_net = ValueNet(test_states.shape[1])
    _ = policy(test_states[:1])
    _ = value_net(test_states[:1])
    policy.load_weights(args.hybrid_dir / "ppo_policy.weights.h5")
    value_net.load_weights(args.hybrid_dir / "ppo_value.weights.h5")

    q_mb = compute_q_mb(dyn_models, value_net, test_states, gamma=args.gamma)
    q_mb = np.nan_to_num(q_mb, nan=0.0, posinf=0.0, neginf=0.0)
    q_mb = np.minimum(q_mb, q_cql + args.delta)

    gating = json.loads((args.hybrid_dir / "gating.json").read_text())
    traj_lengths = compute_traj_lengths(test_df)
    max_dist = max_neighbor_distance(test_states, train_states, args.neighbor_k)
    gate_x = gate_features(test_df, traj_lengths, max_dist)
    gate_x = (gate_x - np.array(gating["mean"])) / np.array(gating["std"])

    gate_logits = gate_x @ np.array(gating["coef"]) + gating["intercept"]
    gate_probs = 1.0 / (1.0 + np.exp(-gate_logits))
    p_mb = np.where(gate_probs >= gating["threshold"], gate_probs, 0.0)

    q_hyb = (1.0 - p_mb[:, None]) * q_cql + p_mb[:, None] * q_mb
    q_hyb = np.nan_to_num(q_hyb, nan=0.0, posinf=0.0, neginf=0.0)
    policy_probs = tf.nn.softmax(q_hyb / args.tau).numpy()
    policy_probs = np.clip(policy_probs, 1e-6, 1.0)
    policy_probs = policy_probs / policy_probs.sum(axis=1, keepdims=True)

    mask_rate = 0.0
    mask = np.zeros_like(policy_probs, dtype=bool)
    if args.mask_actions:
        mask, mask_rate = clinician_action_mask(
            test_states, train_states, train_actions, k=args.neighbor_k
        )
        policy_probs = apply_action_mask(policy_probs, mask)

    phwis = compute_phwis(build_phwis_episodes(test_df, policy_probs))
    phwdr = compute_phwdr(build_phwdr_episodes(test_df, policy_probs, q_hyb))
    am = compute_direct_method(policy_probs, q_hyb, episode_start_indices(test_df))

    residual_mse, uncertainty = dynamics_residuals(
        dyn_models, test_states, test_actions, test_next
    )

    q_diff = q_hyb - q_cql
    diagnostics = {
        "qhyb_minus_cql_mean": float(np.mean(q_diff)),
        "qhyb_minus_cql_std": float(np.std(q_diff)),
        "mask_rate": mask_rate,
        "residual_mse": residual_mse,
        "uncertainty": uncertainty,
        "phwis": phwis,
        "phwdr": phwdr,
        "am": am,
    }

    metrics_path = args.hybrid_dir / "hybrid_eval_metrics.json"
    metrics_path.write_text(json.dumps(diagnostics, indent=2))

    traj_path = args.hybrid_dir / "hybrid_trajectory_actions.csv"
    with traj_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "icustayid",
                "bloc",
                "physician_action",
                "cql_action",
                "mb_action",
                "hybrid_action",
                "p_mb",
                "masked_fraction",
            ]
        )
        cql_action = np.argmax(q_cql, axis=1)
        mb_action = np.argmax(q_mb, axis=1)
        hybrid_action = np.argmax(policy_probs, axis=1)
        for idx, row in test_df.iterrows():
            masked_fraction = 0.0
            if args.mask_actions:
                masked_fraction = float(np.mean(mask[idx]))
            writer.writerow(
                [
                    row["icustayid"],
                    row["bloc"],
                    int(test_actions[idx]),
                    int(cql_action[idx]),
                    int(mb_action[idx]),
                    int(hybrid_action[idx]),
                    float(p_mb[idx]),
                    masked_fraction,
                ]
            )

    print("Hybrid evaluation metrics:")
    for key, value in diagnostics.items():
        print(f"  {key}: {value}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved trajectory actions to {traj_path}")


if __name__ == "__main__":
    main()
