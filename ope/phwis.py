# %% [markdown]
# # Per-Horizon Weighted Importance Sampling (PHWIS)
#
# This script estimates policy value using **Per-Horizon Weighted Importance Sampling**.
# It expects:
# - A preprocessed dataset from `data/` (with `icustayid`, `iv_input`, `vaso_input`, `reward`).
# - A saved evaluation policy file (`.npz`) produced by the DDQN evaluation script.
#
# The estimator uses global behavior action frequencies as \(\pi_b(a)\) (with smoothing),
# and per-state evaluation probabilities from the saved policy.
#
# **Output**: a single PHWIS estimate (float).

# %%
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# %%
NUM_ACTIONS = 25
DEFAULT_GAMMA = 0.99
SMOOTHING = 1e-6


@dataclass
class EpisodeData:
    actions: np.ndarray
    rewards: np.ndarray
    eval_probs: np.ndarray
    beh_probs: np.ndarray


# %%
# ## Data helpers

def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "reward" not in df.columns:
        raise ValueError("Expected a `reward` column in the dataset.")
    return df


def action_ids(df: pd.DataFrame) -> np.ndarray:
    iv = df["iv_input"].fillna(0).astype(int)
    vaso = df["vaso_input"].fillna(0).astype(int)
    return (iv * 5 + vaso).to_numpy()


def behavior_policy(actions: np.ndarray, num_actions: int = NUM_ACTIONS) -> np.ndarray:
    counts = np.bincount(actions, minlength=num_actions).astype(float)
    probs = (counts + SMOOTHING) / (counts.sum() + num_actions * SMOOTHING)
    return probs


def build_episodes(
    df: pd.DataFrame,
    eval_policy_probs: np.ndarray,
) -> List[EpisodeData]:
    actions = action_ids(df)
    beh_probs = behavior_policy(actions)

    episodes: List[EpisodeData] = []
    start = 0
    icustay = df["icustayid"].to_numpy()
    for idx in range(1, len(df) + 1):
        if idx == len(df) or icustay[idx] != icustay[idx - 1]:
            end = idx
            ep_actions = actions[start:end]
            ep_rewards = df["reward"].fillna(0).iloc[start:end].to_numpy(dtype=float)
            ep_eval_probs = eval_policy_probs[start:end]
            ep_beh_probs = beh_probs[ep_actions]
            episodes.append(
                EpisodeData(
                    actions=ep_actions,
                    rewards=ep_rewards,
                    eval_probs=ep_eval_probs,
                    beh_probs=ep_beh_probs,
                )
            )
            start = end
    return episodes


# %%
# ## PHWIS estimator

def compute_phwis(episodes: List[EpisodeData], gamma: float = DEFAULT_GAMMA) -> float:
    max_len = max(len(ep.actions) for ep in episodes)
    estimate = 0.0

    for t in range(max_len):
        rhos = []
        rewards_t = []
        for ep in episodes:
            if t >= len(ep.actions):
                continue
            ratios = ep.eval_probs[: t + 1, ep.actions[: t + 1]] / ep.beh_probs[: t + 1]
            rho_t = float(np.prod(ratios))
            rhos.append(rho_t)
            rewards_t.append(ep.rewards[t])

        if not rhos:
            continue

        rhos = np.asarray(rhos, dtype=float)
        rhos = np.nan_to_num(rhos, nan=0.0, posinf=0.0, neginf=0.0)
        denom = np.sum(rhos)
        if denom <= 0 or np.isnan(denom):
            weights = np.full_like(rhos, 1.0 / len(rhos))
        else:
            weights = rhos / denom
        estimate += (gamma**t) * float(np.sum(weights * np.asarray(rewards_t)))

    return float(estimate)


# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PHWIS estimator")
    parser.add_argument("--data", type=Path, required=True, help="Path to test CSV")
    parser.add_argument(
        "--policy", type=Path, required=True, help="Path to evaluation policy .npz"
    )
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.data)
    policy = np.load(args.policy)
    eval_policy_probs = policy["policy_probs"]
    episodes = build_episodes(df, eval_policy_probs)

    estimate = compute_phwis(episodes, gamma=args.gamma)
    print(f"PHWIS estimate: {estimate:.6f}")


if __name__ == "__main__":
    main()
