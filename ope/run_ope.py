from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from preprocessing.action_bins import bin_action, load_action_bins
from preprocessing.config_utils import load_config


NUM_ACTIONS = 25


@dataclass
class Episode:
    actions: np.ndarray
    rewards: np.ndarray
    pi: np.ndarray
    beh_probs: np.ndarray
    q_values: np.ndarray


def behavior_policy(actions: np.ndarray) -> np.ndarray:
    counts = np.bincount(actions, minlength=NUM_ACTIONS).astype(float)
    counts = counts + 1e-6
    return counts / counts.sum()


def compute_action_ids(df: pd.DataFrame, bins, columns: Dict[str, str]) -> np.ndarray:
    vaso_col = columns.get("vaso_input", "vaso_input")
    iv_col = columns.get("iv_input", "iv_input")
    vaso = df[vaso_col].fillna(0).to_numpy(dtype=float)
    iv = df[iv_col].fillna(0).to_numpy(dtype=float)
    vaso_bins = np.array([bin_action(v, bins.vaso_edges) for v in vaso], dtype=int)
    iv_bins = np.array([bin_action(v, bins.iv_edges) for v in iv], dtype=int)
    return iv_bins * 5 + vaso_bins


def build_episodes(
    df: pd.DataFrame,
    pi: np.ndarray,
    q_values: np.ndarray,
    beh_probs: np.ndarray,
    columns: Dict[str, str],
) -> List[Episode]:
    actions = compute_action_ids(df, bins, columns)
    episodes: List[Episode] = []
    start = 0
    stays = df[columns.get("icustayid", "icustayid")].to_numpy()
    for idx in range(1, len(df) + 1):
        if idx == len(df) or stays[idx] != stays[idx - 1]:
            end = idx
            ep_actions = actions[start:end]
            reward_col = columns.get("reward", "reward")
            ep_rewards = df[reward_col].fillna(0).iloc[start:end].to_numpy(dtype=float)
            ep_pi = pi[start:end]
            ep_q = q_values[start:end]
            ep_beh = beh_probs[ep_actions]
            episodes.append(Episode(ep_actions, ep_rewards, ep_pi, ep_beh, ep_q))
            start = end
    return episodes


def compute_phwis(episodes: List[Episode], gamma: float) -> float:
    max_len = max(len(ep.actions) for ep in episodes)
    estimate = 0.0
    for t in range(max_len):
        rhos = []
        rewards_t = []
        for ep in episodes:
            if t >= len(ep.actions):
                continue
            ratios = ep.pi[: t + 1, ep.actions[: t + 1]] / ep.beh_probs[: t + 1]
            rho_t = float(np.prod(ratios))
            rhos.append(rho_t)
            rewards_t.append(ep.rewards[t])
        if not rhos:
            continue
        rhos = np.nan_to_num(np.asarray(rhos), nan=0.0, posinf=0.0, neginf=0.0)
        denom = np.sum(rhos)
        weights = rhos / denom if denom > 0 else np.full_like(rhos, 1.0 / len(rhos))
        estimate += (gamma ** t) * float(np.sum(weights * np.asarray(rewards_t)))
    return float(estimate)


def compute_phwdr(episodes: List[Episode], gamma: float) -> float:
    max_len = max(len(ep.actions) for ep in episodes)
    estimate = 0.0
    for t in range(max_len):
        rhos = []
        deltas = []
        for ep in episodes:
            if t >= len(ep.actions):
                continue
            ratios = ep.pi[: t + 1, ep.actions[: t + 1]] / ep.beh_probs[: t + 1]
            rho_t = float(np.prod(ratios))
            rhos.append(rho_t)
            v_next = 0.0
            if t + 1 < len(ep.actions):
                v_next = float(np.sum(ep.pi[t + 1] * ep.q_values[t + 1]))
            q_sa = float(ep.q_values[t, ep.actions[t]])
            delta = ep.rewards[t] + gamma * v_next - q_sa
            deltas.append(delta)
        if not rhos:
            continue
        rhos = np.nan_to_num(np.asarray(rhos), nan=0.0, posinf=0.0, neginf=0.0)
        denom = np.sum(rhos)
        weights = rhos / denom if denom > 0 else np.full_like(rhos, 1.0 / len(rhos))
        estimate += (gamma ** t) * float(np.sum(weights * np.asarray(deltas)))
    v0 = np.mean([
        float(np.sum(ep.pi[0] * ep.q_values[0])) for ep in episodes if len(ep.actions) > 0
    ])
    return float(v0 + estimate)


def compute_am(df: pd.DataFrame, pi: np.ndarray, q_values: np.ndarray, columns: Dict[str, str]) -> float:
    starts = [0]
    stays = df[columns.get("icustayid", "icustayid")].to_numpy()
    for idx in range(1, len(df)):
        if stays[idx] != stays[idx - 1]:
            starts.append(idx)
    v_values = np.sum(pi * q_values, axis=1)
    return float(np.mean(v_values[starts]))


def bootstrap(
    episodes: List[Episode],
    metric_fn,
    gamma: float,
    n_boot: int,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(0)
    estimates = []
    for _ in range(n_boot):
        sample = rng.choice(episodes, size=len(episodes), replace=True)
        estimates.append(metric_fn(list(sample), gamma))
    estimates = np.asarray(estimates)
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5)), float(np.mean(estimates))


def run_ope(config_path: str, split: str) -> Dict[str, bool]:
    cfg = load_config(config_path)
    results_dir = cfg.results_dir
    bins_path = results_dir / "action_bins.json"
    if not bins_path.exists():
        raise FileNotFoundError("action_bins.json not found; run preprocessing/action_bins.py")
    global bins
    bins = load_action_bins(bins_path)

    df = pd.read_csv(cfg.paths[f"{split}_csv"])
    policies = cfg.data.get("policies", [])
    outputs = {}
    row_index = None

    for policy in policies:
        path = results_dir / f"policy_outputs_{policy}_{split}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing policy outputs: {path}")
        data = np.load(path, allow_pickle=True)
        outputs[policy] = data
        if row_index is None:
            row_index = data["row_index"]
        elif not np.array_equal(row_index, data["row_index"]):
            raise ValueError("Row index mismatch across policies")
        pi = data["pi"]
        if not np.allclose(pi.sum(axis=1), 1.0, atol=1e-3):
            raise ValueError(f"Policy {policy} probabilities do not sum to 1")

    beh_actions = compute_action_ids(df, bins, cfg.columns)
    beh_probs = behavior_policy(beh_actions)
    gamma = float(cfg.data.get("hyperparams", {}).get("cql", {}).get("gamma", 0.99))
    n_boot = int(cfg.data.get("reporting", {}).get("bootstrap_samples", 200))

    overall_rows = []
    by_sofa_rows = []
    has_nan = False

    for policy, data in outputs.items():
        pi = data["pi"]
        q_values = data["q_values"] if "q_values" in data else np.zeros_like(pi)
        episodes = build_episodes(df, pi, q_values, beh_probs, cfg.columns)
        phwis_ci = bootstrap(episodes, compute_phwis, gamma, n_boot)
        phwdr_ci = bootstrap(episodes, compute_phwdr, gamma, n_boot)
        am_value = compute_am(df, pi, q_values, cfg.columns)

        overall_rows.append(
            {
                "policy": policy,
                "phwis": phwis_ci[2],
                "phwis_ci_low": phwis_ci[0],
                "phwis_ci_high": phwis_ci[1],
                "phwdr": phwdr_ci[2],
                "phwdr_ci_low": phwdr_ci[0],
                "phwdr_ci_high": phwdr_ci[1],
                "am": am_value,
            }
        )

        sofa_bucket = data["sofa_bucket"]
        for bucket in ["low", "medium", "high"]:
            mask = sofa_bucket == bucket
            if not np.any(mask):
                continue
            bucket_df = df.loc[mask]
            bucket_pi = pi[mask]
            bucket_q = q_values[mask]
            bucket_actions = compute_action_ids(bucket_df, bins, cfg.columns)
            bucket_beh_probs = behavior_policy(bucket_actions)
            bucket_eps = build_episodes(bucket_df, bucket_pi, bucket_q, bucket_beh_probs, cfg.columns)
            phwis = compute_phwis(bucket_eps, gamma)
            phwdr = compute_phwdr(bucket_eps, gamma)
            am_value = compute_am(bucket_df, bucket_pi, bucket_q, cfg.columns)
            by_sofa_rows.append(
                {
                    "policy": policy,
                    "sofa_bucket": bucket,
                    "phwis": phwis,
                    "phwdr": phwdr,
                    "am": am_value,
                }
            )

    overall_df = pd.DataFrame(overall_rows)
    by_sofa_df = pd.DataFrame(by_sofa_rows)
    if overall_df.isna().any().any() or by_sofa_df.isna().any().any():
        has_nan = True

    overall_df.to_csv(results_dir / "ope_metrics_overall.csv", index=False)
    (results_dir / "ope_metrics_overall.json").write_text(overall_df.to_json(orient="records", indent=2), encoding="utf-8")
    by_sofa_df.to_csv(results_dir / "ope_metrics_by_sofa.csv", index=False)
    (results_dir / "ope_metrics_by_sofa.json").write_text(by_sofa_df.to_json(orient="records", indent=2), encoding="utf-8")

    return {"has_nan": has_nan}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OPE metrics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ope(args.config, args.split)


if __name__ == "__main__":
    main()
