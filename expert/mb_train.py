"""Train a model-based PPO+BNN expert (MB-only) on the preprocessed dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

NUM_ACTIONS = 25
DEFAULT_GAMMA = 0.99


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


def train_dynamics_ensemble(
    inputs: np.ndarray,
    targets: np.ndarray,
    ensemble_size: int,
    epochs: int,
    batch_size: int,
) -> List[DynamicsModel]:
    models = []
    for _ in range(ensemble_size):
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
) -> dict[str, np.ndarray]:
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
    batch: dict[str, np.ndarray],
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MB-only PPO+BNN expert")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root() / "data",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default="rl_train_data_final_cont_noterm.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "outputs" / "mb",
    )
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--bnn-epochs", type=int, default=5)
    parser.add_argument("--ppo-steps", type=int, default=4000)
    parser.add_argument("--ppo-rollout-horizon", type=int, default=5)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: To run on Colab (T4 GPU), mount Drive and set --data-dir/--output-dir
    # to your Drive paths (e.g., /content/drive/MyDrive/sepsisrl/data).

    state_features = load_state_features(args.data_dir)
    train_df = pd.read_csv(args.data_dir / args.train_csv)

    train_states, train_actions, train_rewards, train_next, _ = build_transitions(
        train_df, state_features
    )

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
            print(f"[ppo] step {step} loss={loss:.4f}")

    dynamics_dir = args.output_dir / "dynamics"
    dynamics_dir.mkdir(exist_ok=True)
    for idx, model in enumerate(dynamics_models):
        model.save_weights(dynamics_dir / f"model_{idx}.weights.h5")

    policy.save_weights(args.output_dir / "ppo_policy.weights.h5")
    value_net.save_weights(args.output_dir / "ppo_value.weights.h5")

    metadata = {
        "state_features": state_features,
        "train_csv": args.train_csv,
        "ensemble_size": args.ensemble_size,
        "bnn_epochs": args.bnn_epochs,
        "ppo_steps": args.ppo_steps,
        "ppo_rollout_horizon": args.ppo_rollout_horizon,
        "ppo_epochs": args.ppo_epochs,
        "ppo_clip": args.ppo_clip,
        "gamma": args.gamma,
    }
    (args.output_dir / "train_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("Saved MB dynamics and PPO policy/value weights.")


if __name__ == "__main__":
    main()
