"""Train a model-based PPO head, then learn a gating model to blend with CQL."""
from __future__ import annotations

# Colab quickstart (optional)
# from google.colab import drive
# drive.mount("/content/drive")
# %cd /content/drive/MyDrive/sepsisrl
# !pip install -r requirements.txt
# python expert/hybrid_train.py --config final_config.yaml --train_or_load --eval_split val
#
# --- CLI entrypoint for reproducible reporting ---
import sys
from pathlib import Path

if "--config" in sys.argv:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from expert.policy_runner import run_policy_cli

    run_policy_cli("hybrid")
    raise SystemExit(0)

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from ope.phwdr import build_episodes as build_phwdr_episodes
from ope.phwis import build_episodes as build_phwis_episodes
from ope import compute_phwdr, compute_phwis

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


@dataclass
class GateModel:
    coef: np.ndarray
    intercept: float
    mean: np.ndarray
    std: np.ndarray
    threshold: float


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


def one_hot_actions(actions: np.ndarray) -> np.ndarray:
    return np.eye(NUM_ACTIONS, dtype=np.float32)[actions]


def train_dynamics_ensemble(
    inputs: np.ndarray,
    targets: np.ndarray,
    ensemble_size: int,
    epochs: int,
    batch_size: int,
) -> List[DynamicsModel]:
    models = []
    for idx in range(ensemble_size):
        model = DynamicsModel(inputs.shape[1], targets.shape[1])
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        model.fit(
            inputs,
            targets,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            shuffle=True,
        )
        models.append(model)
    return models


def ppo_rollout(
    policy: PolicyNet,
    value_net: ValueNet,
    dynamics_models: List[DynamicsModel],
    states: np.ndarray,
    gamma: float,
    rollout_horizon: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    batch_states = []
    batch_actions = []
    batch_logprobs = []
    batch_returns = []
    batch_advantages = []

    for start_idx in range(states.shape[0]):
        state = states[start_idx]
        rewards = []
        values = []
        logprobs = []
        actions = []
        step_states = []

        for _ in range(rollout_horizon):
            logits = policy(state[None, :]).numpy()[0]
            probs = tf.nn.softmax(logits).numpy()
            if np.any(np.isnan(probs)):
                probs = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS)
            action = rng.choice(NUM_ACTIONS, p=probs)
            logprob = float(np.log(probs[action] + 1e-8))

            model = dynamics_models[rng.integers(0, len(dynamics_models))]
            action_one_hot = one_hot_actions(np.array([action]))
            dyn_input = np.hstack([state[None, :], action_one_hot])
            delta_reward = model(dyn_input, training=False).numpy()[0]
            delta_reward = np.nan_to_num(delta_reward, nan=0.0, posinf=0.0, neginf=0.0)
            next_state = state + delta_reward[:-1]
            reward = float(delta_reward[-1])

            step_states.append(state)
            actions.append(action)
            rewards.append(reward)
            logprobs.append(logprob)
            values.append(float(value_net(state[None, :]).numpy()[0]))

            state = next_state

        returns = []
        discounted = 0.0
        for reward in reversed(rewards):
            discounted = reward + gamma * discounted
            returns.insert(0, discounted)

        advantages = np.array(returns) - np.array(values)
        advantages = np.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        batch_states.extend(step_states)
        batch_actions.extend(actions)
        batch_logprobs.extend(logprobs)
        batch_returns.extend(returns)
        batch_advantages.extend(advantages)

    return {
        "states": np.array(batch_states, dtype=np.float32),
        "actions": np.array(batch_actions, dtype=np.int32),
        "logprobs": np.array(batch_logprobs, dtype=np.float32),
        "returns": np.array(batch_returns, dtype=np.float32),
        "advantages": np.array(batch_advantages, dtype=np.float32),
    }


def ppo_update(
    policy: PolicyNet,
    value_net: ValueNet,
    batch: Dict[str, np.ndarray],
    clip_ratio: float,
    epochs: int,
) -> float:
    optimizer = tf.keras.optimizers.Adam(3e-4)

    for _ in range(epochs):
        with tf.GradientTape() as tape:
            logits = policy(batch["states"])
            log_probs = tf.nn.log_softmax(logits)
            action_mask = tf.one_hot(batch["actions"], NUM_ACTIONS)
            new_logprobs = tf.reduce_sum(log_probs * action_mask, axis=1)
            ratio = tf.exp(new_logprobs - batch["logprobs"])
            clipped = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
            advantages = tf.where(
                tf.math.is_finite(batch["advantages"]),
                batch["advantages"],
                tf.zeros_like(batch["advantages"]),
            )
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped * advantages))

            values = value_net(batch["states"])
            returns = tf.where(
                tf.math.is_finite(batch["returns"]),
                batch["returns"],
                tf.zeros_like(batch["returns"]),
            )
            value_loss = tf.reduce_mean(tf.square(returns - values))
            loss = policy_loss + 0.5 * value_loss

        grads = tape.gradient(loss, policy.trainable_variables + value_net.trainable_variables)
        optimizer.apply_gradients(
            zip(grads, policy.trainable_variables + value_net.trainable_variables)
        )

    return float(loss.numpy())


def compute_q_mb(
    dynamics_models: List[DynamicsModel],
    policy_value: ValueNet,
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
            next_value = policy_value(next_state).numpy()
            all_q.append(np.nan_to_num(reward + gamma * next_value, nan=0.0, posinf=0.0, neginf=0.0))
        q_ensemble.append(np.stack(all_q, axis=1))

    q_ensemble = np.stack(q_ensemble, axis=0)
    return np.min(q_ensemble, axis=0)


def compute_short_horizon_phwdr(
    df: pd.DataFrame,
    policy_probs: np.ndarray,
    q_values: np.ndarray,
    horizon: int,
    gamma: float,
) -> float:
    episodes = build_phwdr_episodes(df, policy_probs, q_values)
    truncated = []
    for ep in episodes:
        truncated.append(
            type(ep)(
                actions=ep.actions[:horizon],
                rewards=ep.rewards[:horizon],
                eval_probs=ep.eval_probs[:horizon],
                beh_probs=ep.beh_probs[:horizon],
                q_values=ep.q_values[:horizon],
            )
        )
    return compute_phwdr(truncated, gamma=gamma)


def compute_episode_phwdr_scores(
    df: pd.DataFrame,
    policy_probs: np.ndarray,
    q_values: np.ndarray,
    horizon: int,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    episodes = build_phwdr_episodes(df, policy_probs, q_values)
    scores = []
    lengths = []
    for ep in episodes:
        truncated = [
            type(ep)(
                actions=ep.actions[:horizon],
                rewards=ep.rewards[:horizon],
                eval_probs=ep.eval_probs[:horizon],
                beh_probs=ep.beh_probs[:horizon],
                q_values=ep.q_values[:horizon],
            )
        ]
        scores.append(compute_phwdr(truncated, gamma=gamma))
        lengths.append(len(ep.actions))
    return np.array(scores, dtype=np.float32), np.array(lengths, dtype=np.int32)


def dynamics_residuals(
    dynamics_models: List[DynamicsModel],
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

def gate_features(
    df: pd.DataFrame,
    traj_lengths: np.ndarray,
    max_neighbor_dist: np.ndarray,
) -> np.ndarray:
    base = df[FEATURES_FOR_GATE].fillna(0).to_numpy(dtype=np.float32)
    features = np.column_stack([base, traj_lengths, max_neighbor_dist])
    return features


def compute_traj_lengths(df: pd.DataFrame) -> np.ndarray:
    counts = df.groupby("icustayid").size().to_dict()
    return df["icustayid"].map(counts).to_numpy(dtype=np.float32)


def max_neighbor_distance(states: np.ndarray, ref_states: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(ref_states) - 1)
    distances = []
    for state in states:
        diffs = ref_states - state
        dist = np.linalg.norm(diffs, axis=1)
        topk = np.partition(dist, k)[:k]
        distances.append(np.max(topk))
    return np.array(distances, dtype=np.float32)


def train_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    steps: int = 2000,
    lr: float = 1e-2,
) -> Tuple[np.ndarray, float]:
    weights = np.zeros(features.shape[1], dtype=np.float32)
    bias = 0.0

    for _ in range(steps):
        logits = features @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-logits))
        grad_w = features.T @ (probs - labels) / len(labels)
        grad_b = float(np.mean(probs - labels))
        weights -= lr * grad_w
        bias -= lr * grad_b

    return weights, bias


def calibrate_threshold(probs: np.ndarray, target_rate: float) -> float:
    thresholds = np.linspace(0.0, 1.0, 101)
    best = 0.5
    best_diff = 1e9
    for thr in thresholds:
        rate = np.mean(probs >= thr)
        diff = abs(rate - target_rate)
        if diff < best_diff:
            best_diff = diff
            best = thr
    return float(best)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hybrid gating model")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root() / "data",
    )
    parser.add_argument("--train-csv", type=str, default="rl_train_data_final_cont_noterm.csv")
    parser.add_argument("--val-csv", type=str, default="rl_val_data_final_cont_noterm.csv")
    parser.add_argument(
        "--cql-weights",
        type=Path,
        default=repo_root() / "outputs" / "cql" / "model.weights.h5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "outputs" / "hybrid",
    )
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--bnn-epochs", type=int, default=5)
    parser.add_argument("--ppo-steps", type=int, default=4000)
    parser.add_argument("--ppo-rollout-horizon", type=int, default=5)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--short-horizon", type=int, default=5)
    parser.add_argument("--target-mb-rate", type=float, default=0.5)
    parser.add_argument("--neighbor-k", type=int, default=10)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: To run on Colab (T4 GPU), mount Drive and set --data-dir/--output-dir
    # to your Drive paths (e.g., /content/drive/MyDrive/sepsisrl/data).

    state_features = load_state_features(args.data_dir)
    train_df = pd.read_csv(args.data_dir / args.train_csv)
    val_df = pd.read_csv(args.data_dir / args.val_csv)

    train_states, train_actions, train_rewards, train_next, train_dones = build_transitions(
        train_df, state_features
    )
    val_states, val_actions, _, val_next, _ = build_transitions(val_df, state_features)

    dyn_inputs = np.hstack([train_states, one_hot_actions(train_actions)])
    dyn_targets = np.hstack([
        train_next - train_states,
        train_rewards[:, None],
    ])

    dynamics_models = train_dynamics_ensemble(
        dyn_inputs,
        dyn_targets,
        ensemble_size=args.ensemble_size,
        epochs=args.bnn_epochs,
        batch_size=256,
    )

    policy = PolicyNet(train_states.shape[1])
    value_net = ValueNet(train_states.shape[1])
    _ = policy(train_states[:1])
    _ = value_net(train_states[:1])

    eval_history: List[Dict[str, float]] = []
    best_checkpoint = None
    best_median = -np.inf
    no_improve = 0
    last_diag = None

    for step in range(0, args.ppo_steps, args.ppo_rollout_horizon):
        sample_idx = np.random.choice(len(train_states), size=32, replace=False)
        batch = ppo_rollout(
            policy,
            value_net,
            dynamics_models,
            train_states[sample_idx],
            gamma=args.gamma,
            rollout_horizon=args.ppo_rollout_horizon,
        )
        loss = ppo_update(
            policy,
            value_net,
            batch,
            clip_ratio=args.ppo_clip,
            epochs=args.ppo_epochs,
        )

        if step % 2000 == 0:
            q_mb = compute_q_mb(dynamics_models, value_net, val_states, gamma=args.gamma)
            q_mb = np.nan_to_num(q_mb, nan=0.0, posinf=0.0, neginf=0.0)
            policy_probs_mb = tf.nn.softmax(q_mb / args.tau).numpy()
            policy_probs_mb = np.clip(policy_probs_mb, 1e-6, 1.0)
            policy_probs_mb = policy_probs_mb / policy_probs_mb.sum(axis=1, keepdims=True)

            cql_model = DuelingQNetwork(val_states.shape[1], NUM_ACTIONS)
            _ = cql_model(val_states[:1])
            cql_model.load_weights(args.cql_weights)
            q_cql = np.nan_to_num(cql_model(val_states).numpy(), nan=0.0, posinf=0.0, neginf=0.0)
            policy_probs_cql = tf.nn.softmax(q_cql / args.tau).numpy()
            policy_probs_cql = np.clip(policy_probs_cql, 1e-6, 1.0)
            policy_probs_cql = policy_probs_cql / policy_probs_cql.sum(axis=1, keepdims=True)

            phwdr_mb = compute_short_horizon_phwdr(
                val_df, policy_probs_mb, q_mb, args.short_horizon, args.gamma
            )
            phwdr_cql = compute_short_horizon_phwdr(
                val_df, policy_probs_cql, q_cql, args.short_horizon, args.gamma
            )
            phwis_mb = compute_phwis(build_phwis_episodes(val_df, policy_probs_mb))
            residual_mse, uncertainty = dynamics_residuals(
                dynamics_models, val_states, val_actions, val_next
            )

            eval_history.append(
                {
                    "step": step,
                    "loss": loss,
                    "phwdr_mb": phwdr_mb,
                    "phwdr_cql": phwdr_cql,
                    "phwis_mb": phwis_mb,
                    "residual_mse": residual_mse,
                    "uncertainty": uncertainty,
                }
            )

            window = [m["phwdr_mb"] for m in eval_history if step - m["step"] <= 5000]
            rolling_median = float(np.median(window)) if window else phwdr_mb
            improved = rolling_median > best_median
            tie_break = (
                best_checkpoint is not None
                and np.isclose(rolling_median, best_median)
                and phwis_mb > best_checkpoint["phwis_mb"]
            )
            if improved or tie_break:
                best_median = rolling_median
                no_improve = 0
                best_checkpoint = {
                    "phwdr_mb": rolling_median,
                    "phwis_mb": phwis_mb,
                    "policy_weights": policy.get_weights(),
                    "value_weights": value_net.get_weights(),
                }
            else:
                no_improve += 1

            stable = False
            if last_diag is not None:
                stable = (
                    abs(last_diag["residual_mse"] - residual_mse) < 1e-4
                    and abs(last_diag["uncertainty"] - uncertainty) < 1e-4
                )
            last_diag = {"residual_mse": residual_mse, "uncertainty": uncertainty}

            print(f"[ppo] step {step} loss={loss:.4f} phwdr_mb={phwdr_mb:.4f}")

            if no_improve >= 3 and stable:
                print("[ppo] early stopping: PHWDR median stalled and diagnostics stable.")
                break

    if best_checkpoint is not None:
        policy.set_weights(best_checkpoint["policy_weights"])
        value_net.set_weights(best_checkpoint["value_weights"])

    q_mb = compute_q_mb(dynamics_models, value_net, val_states, gamma=args.gamma)
    q_mb = np.nan_to_num(q_mb, nan=0.0, posinf=0.0, neginf=0.0)
    cql_model = DuelingQNetwork(val_states.shape[1], NUM_ACTIONS)
    _ = cql_model(val_states[:1])
    cql_model.load_weights(args.cql_weights)
    q_cql = np.nan_to_num(cql_model(val_states).numpy(), nan=0.0, posinf=0.0, neginf=0.0)

    policy_probs_mb = tf.nn.softmax(q_mb / args.tau).numpy()
    policy_probs_mb = np.clip(policy_probs_mb, 1e-6, 1.0)
    policy_probs_mb = policy_probs_mb / policy_probs_mb.sum(axis=1, keepdims=True)
    policy_probs_cql = tf.nn.softmax(q_cql / args.tau).numpy()
    policy_probs_cql = np.clip(policy_probs_cql, 1e-6, 1.0)
    policy_probs_cql = policy_probs_cql / policy_probs_cql.sum(axis=1, keepdims=True)

    phwdr_mb = compute_short_horizon_phwdr(
        val_df, policy_probs_mb, q_mb, args.short_horizon, args.gamma
    )
    phwdr_cql = compute_short_horizon_phwdr(
        val_df, policy_probs_cql, q_cql, args.short_horizon, args.gamma
    )
    mb_scores, ep_lengths = compute_episode_phwdr_scores(
        val_df, policy_probs_mb, q_mb, args.short_horizon, args.gamma
    )
    cql_scores, _ = compute_episode_phwdr_scores(
        val_df, policy_probs_cql, q_cql, args.short_horizon, args.gamma
    )
    # We assign each state in an episode the episode-level short-horizon PHWDR winner.
    labels = []
    for mb_score, cql_score, length in zip(mb_scores, cql_scores, ep_lengths):
        label = 1.0 if mb_score >= cql_score else 0.0
        labels.extend([label] * int(length))
    labels = np.array(labels, dtype=np.float32)

    traj_lengths = compute_traj_lengths(val_df)
    max_dist = max_neighbor_distance(val_states, train_states, args.neighbor_k)
    gate_x = gate_features(val_df, traj_lengths, max_dist)

    mean = gate_x.mean(axis=0)
    std = gate_x.std(axis=0) + 1e-6
    gate_x = (gate_x - mean) / std

    coef, intercept = train_logistic_regression(gate_x, labels)
    probs = 1.0 / (1.0 + np.exp(-(gate_x @ coef + intercept)))
    threshold = calibrate_threshold(probs, args.target_mb_rate)

    gate_model = GateModel(
        coef=coef,
        intercept=float(intercept),
        mean=mean,
        std=std,
        threshold=threshold,
    )

    dynamics_dir = args.output_dir / "dynamics"
    dynamics_dir.mkdir(exist_ok=True)
    for idx, model in enumerate(dynamics_models):
        model.save_weights(dynamics_dir / f"model_{idx}.weights.h5")

    policy.save_weights(args.output_dir / "ppo_policy.weights.h5")
    value_net.save_weights(args.output_dir / "ppo_value.weights.h5")

    gate_path = args.output_dir / "gating.json"
    gate_path.write_text(
        json.dumps(
            {
                "coef": gate_model.coef.tolist(),
                "intercept": gate_model.intercept,
                "mean": gate_model.mean.tolist(),
                "std": gate_model.std.tolist(),
                "threshold": gate_model.threshold,
                "features": FEATURES_FOR_GATE + ["traj_len", "max_neighbor_dist"],
            },
            indent=2,
        )
    )

    (args.output_dir / "train_history.json").write_text(
        json.dumps(eval_history, indent=2)
    )

    print("Saved dynamics, PPO policy/value, and gating model.")


if __name__ == "__main__":
    main()
