"""Train a Dueling Double DQN (DDDQN) policy on the preprocessed dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

NUM_ACTIONS = 25
DEFAULT_GAMMA = 0.99
DEFAULT_BATCH_SIZE = 256
DEFAULT_STEPS = 20000
DEFAULT_TAU = 0.005
DEFAULT_LR = 1e-4


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_state_features(data_dir: Path) -> list[str]:
    with (data_dir / "state_features.txt").open() as handle:
        return handle.read().split()


def action_ids(df: pd.DataFrame) -> np.ndarray:
    iv = df["iv_input"].fillna(0).astype(int)
    vaso = df["vaso_input"].fillna(0).astype(int)
    return (iv * 5 + vaso).to_numpy()


def build_transitions(
    df: pd.DataFrame, state_features: list[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required_cols = state_features + ["reward", "iv_input", "vaso_input", "icustayid"]
    missing_mask = df[required_cols].isna().any(axis=1)
    dropped = int(missing_mask.sum())
    if dropped > 0:
        print(f"[debug] Dropping {dropped} rows with missing required fields.")
    df = df.loc[~missing_mask].reset_index(drop=True)

    states = df[state_features].to_numpy(dtype=np.float32)
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

    if not np.isfinite(states).all():
        raise ValueError("Non-finite values detected in states after preprocessing.")
    if not np.isfinite(rewards).all():
        raise ValueError("Non-finite values detected in rewards after preprocessing.")
    if actions.min() < 0 or actions.max() >= NUM_ACTIONS:
        raise ValueError("Actions out of range after preprocessing.")

    return states, actions, rewards, next_states, dones


class DuelingDDQN(tf.keras.Model):
    def __init__(self, input_dim: int, num_actions: int) -> None:
        super().__init__()
        self.shared_1 = tf.keras.layers.Dense(128)
        self.shared_2 = tf.keras.layers.Dense(128)
        self.value = tf.keras.layers.Dense(1)
        self.advantage = tf.keras.layers.Dense(num_actions)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = tf.nn.leaky_relu(self.shared_1(inputs))
        x = tf.nn.leaky_relu(self.shared_2(x))
        value = self.value(x)
        advantage = self.advantage(x)
        return value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))


def soft_update(target: tf.keras.Model, source: tf.keras.Model, tau: float) -> None:
    for target_var, source_var in zip(target.variables, source.variables):
        target_var.assign(tau * source_var + (1.0 - tau) * target_var)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDDQN policy")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root() / "data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default="rl_train_data_final_cont_noterm.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "outputs" / "dddqn",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: To run on Colab (T4 GPU), mount Drive and set --data-dir/--output-dir
    # to your Drive paths (e.g., /content/drive/MyDrive/sepsisrl/data).

    state_features = load_state_features(args.data_dir)
    train_df = pd.read_csv(args.data_dir / args.train_csv)

    states, actions, rewards, next_states, dones = build_transitions(
        train_df, state_features
    )

    main_model = DuelingDDQN(states.shape[1], NUM_ACTIONS)
    target_model = DuelingDDQN(states.shape[1], NUM_ACTIONS)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    mse = tf.keras.losses.MeanSquaredError()

    @tf.function
    def train_step(
        batch_states: tf.Tensor,
        batch_actions: tf.Tensor,
        batch_rewards: tf.Tensor,
        batch_next_states: tf.Tensor,
        batch_dones: tf.Tensor,
    ) -> tf.Tensor:
        with tf.GradientTape() as tape:
            q_next_main = main_model(batch_next_states)
            next_actions = tf.argmax(q_next_main, axis=1)
            q_next_target = target_model(batch_next_states)
            q_next = tf.gather(q_next_target, next_actions, batch_dims=1)
            targets = batch_rewards + args.gamma * (1.0 - batch_dones) * q_next

            q_values = main_model(batch_states)
            q_selected = tf.gather(q_values, batch_actions, batch_dims=1)
            loss = mse(targets, q_selected)

        gradients = tape.gradient(loss, main_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, main_model.trainable_variables))
        return loss

    target_model.set_weights(main_model.get_weights())

    for step in range(1, args.steps + 1):
        idx = np.random.choice(len(states), size=args.batch_size, replace=False)
        batch_states = tf.convert_to_tensor(states[idx])
        batch_actions = tf.convert_to_tensor(actions[idx])
        batch_rewards = tf.convert_to_tensor(rewards[idx])
        batch_next_states = tf.convert_to_tensor(next_states[idx])
        batch_dones = tf.convert_to_tensor(dones[idx])

        loss = train_step(
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_dones,
        )
        soft_update(target_model, main_model, args.tau)

        if step % 1000 == 0:
            print(f"Step {step}: loss={loss.numpy():.6f}")

    model_path = args.output_dir / "model.weights.h5"
    main_model.save_weights(model_path)

    metadata = {
        "state_features": state_features,
        "train_csv": args.train_csv,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "tau": args.tau,
        "lr": args.lr,
    }
    (args.output_dir / "train_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved model weights to {model_path}")


if __name__ == "__main__":
    main()
