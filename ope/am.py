# %% [markdown]
# # Approximate Model (AM) / Direct Method
#
# This script computes a **Direct Method** estimate using the DDQN Q-values:
# it evaluates \(V(s) = \sum_a \pi_e(a|s) Q(s,a)\) and averages over episode starts.
#
# **Inputs**
# - Test dataset CSV with `icustayid`, `iv_input`, `vaso_input`.
# - Policy `.npz` with `policy_probs` and `q_values`.

# %%
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# %%
DEFAULT_GAMMA = 0.99


# %%
# ## Helpers

def load_dataset(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def episode_start_indices(df: pd.DataFrame) -> np.ndarray:
    icustay = df["icustayid"].to_numpy()
    starts = [0]
    for idx in range(1, len(df)):
        if icustay[idx] != icustay[idx - 1]:
            starts.append(idx)
    return np.asarray(starts, dtype=int)


def compute_direct_method(
    policy_probs: np.ndarray, q_values: np.ndarray, start_indices: np.ndarray
) -> float:
    v_values = np.sum(policy_probs * q_values, axis=1)
    return float(np.mean(v_values[start_indices]))


# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Approximate Model (Direct Method)")
    parser.add_argument("--data", type=Path, required=True, help="Path to test CSV")
    parser.add_argument(
        "--policy", type=Path, required=True, help="Path to evaluation policy .npz"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.data)
    policy = np.load(args.policy)
    policy_probs = policy["policy_probs"]
    q_values = policy["q_values"]

    starts = episode_start_indices(df)
    estimate = compute_direct_method(policy_probs, q_values, starts)
    print(f"AM (Direct Method) estimate: {estimate:.6f}")


if __name__ == "__main__":
    main()
