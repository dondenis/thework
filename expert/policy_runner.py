from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

from preprocessing.action_bins import ActionBins, bin_action, load_action_bins
from preprocessing.config_utils import Config, load_config
from expert.policy_outputs import build_policy_outputs


def load_state_features(path: str) -> list[str]:
    return Path(path).read_text().split()


def load_split(cfg: Config, split: str) -> pd.DataFrame:
    split_map = {
        "train": cfg.paths["train_csv"],
        "val": cfg.paths["val_csv"],
        "test": cfg.paths["test_csv"],
    }
    return pd.read_csv(split_map[split])


def ensure_results_dir(cfg: Config) -> Path:
    results_dir = cfg.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "models").mkdir(parents=True, exist_ok=True)
    return results_dir


def load_action_bins_from_results(cfg: Config) -> ActionBins:
    bins_path = cfg.results_dir / "action_bins.json"
    if not bins_path.exists():
        raise FileNotFoundError(
            f"Missing action_bins.json at {bins_path}. Run preprocessing/action_bins.py first."
        )
    return load_action_bins(bins_path)


def one_hot(actions: np.ndarray, num_actions: int = 25) -> np.ndarray:
    return np.eye(num_actions, dtype=np.float32)[actions]


def build_placeholder_pi(df: pd.DataFrame, action_bins: ActionBins, columns: Dict[str, str]) -> np.ndarray:
    vaso_col = columns.get("vaso_input", "vaso_input")
    iv_col = columns.get("iv_input", "iv_input")
    vaso = df[vaso_col].fillna(0).to_numpy(dtype=float)
    iv = df[iv_col].fillna(0).to_numpy(dtype=float)
    vaso_bins = np.array([bin_action(v, action_bins.vaso_edges) for v in vaso], dtype=int)
    iv_bins = np.array([bin_action(v, action_bins.iv_edges) for v in iv], dtype=int)
    action_ids = iv_bins * 5 + vaso_bins
    return one_hot(action_ids)


def compute_action_ids(
    df: pd.DataFrame,
    action_bins: ActionBins,
    columns: Dict[str, str],
) -> np.ndarray:
    vaso_col = columns.get("vaso_input", "vaso_input")
    iv_col = columns.get("iv_input", "iv_input")
    vaso = df[vaso_col].fillna(0).to_numpy(dtype=float)
    iv = df[iv_col].fillna(0).to_numpy(dtype=float)
    vaso_bins = np.array([bin_action(v, action_bins.vaso_edges) for v in vaso], dtype=int)
    iv_bins = np.array([bin_action(v, action_bins.iv_edges) for v in iv], dtype=int)
    return iv_bins * 5 + vaso_bins


def knn_policy_probs(
    train_states: np.ndarray,
    train_actions: np.ndarray,
    eval_states: np.ndarray,
    k: int,
    num_actions: int = 25,
    chunk_size: int = 256,
) -> np.ndarray:
    k = min(k, len(train_states))
    probs = np.zeros((len(eval_states), num_actions), dtype=np.float32)
    train_states = train_states.astype(np.float32)
    eval_states = eval_states.astype(np.float32)
    train_actions = train_actions.astype(int)

    for start in range(0, len(eval_states), chunk_size):
        end = min(start + chunk_size, len(eval_states))
        chunk = eval_states[start:end]
        diffs = chunk[:, None, :] - train_states[None, :, :]
        dists = np.sum(diffs * diffs, axis=2)
        nn_idx = np.argpartition(dists, k - 1, axis=1)[:, :k]
        for row_idx, neighbors in enumerate(nn_idx):
            counts = np.bincount(train_actions[neighbors], minlength=num_actions).astype(np.float32)
            probs[start + row_idx] = counts / counts.sum()
    return probs


def save_checkpoint(policy: str, cfg: Config, params: Dict[str, float]) -> Path:
    results_dir = ensure_results_dir(cfg)
    ckpt_path = results_dir / "models" / f"{policy}_best.pkl"
    config_path = results_dir / "models" / f"{policy}_config.yaml"
    ckpt_data = {"policy": policy, "params": params, "trained": False}
    ckpt_path.write_bytes(json.dumps(ckpt_data).encode("utf-8"))
    config_path.write_text(yaml.safe_dump(params), encoding="utf-8")
    return ckpt_path


def load_checkpoint(policy: str, cfg: Config) -> Optional[Dict[str, str]]:
    ckpt_path = cfg.results_dir / "models" / f"{policy}_best.pkl"
    if not ckpt_path.exists():
        return None
    return json.loads(ckpt_path.read_text(encoding="utf-8"))


def resolve_params(cfg: Config, policy: str, hparams_path: Optional[str]) -> Dict[str, float]:
    params = dict(cfg.data.get("hyperparams", {}).get(policy, {}))
    if hparams_path:
        with Path(hparams_path).open("r", encoding="utf-8") as f:
            override = yaml.safe_load(f)
        params.update(override or {})
    return params


def run_real_hybrid_eval(cfg: Config, split: str) -> Dict[str, np.ndarray]:
    import tensorflow as tf

    from expert.hybrid_eval import (
        DEFAULT_DELTA,
        DEFAULT_GAMMA,
        DEFAULT_OPE_EPSILON,
        DEFAULT_TAU,
        DuelingQNetwork,
        DynamicsModel,
        NUM_ACTIONS,
        PolicyNet,
        ValueNet,
        action_ids,
        build_transitions,
        compute_q_mb,
        compute_traj_lengths,
        gate_features,
        max_neighbor_distance,
        softmax_probs,
        stabilize_policy_probs,
    )

    paths = cfg.paths
    split_path = Path(paths[f"{split}_csv"])
    train_path = Path(paths["train_csv"])
    state_features = Path(paths["state_features"]).read_text().split()

    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(split_path)

    train_states = train_df[state_features].to_numpy(dtype=np.float32)
    test_states, _, _, _, _ = build_transitions(eval_df, state_features)

    cql_weights = Path(paths.get("cql_weights", "outputs/cql/model.weights.h5"))
    hybrid_dir = Path(paths.get("hybrid_dir", "outputs/hybrid"))

    hybrid_hparams = cfg.data.get("hyperparams", {}).get("hybrid", {})
    tau = float(hybrid_hparams.get("tau", DEFAULT_TAU))
    delta = float(hybrid_hparams.get("delta", DEFAULT_DELTA))
    gamma = float(hybrid_hparams.get("gamma", DEFAULT_GAMMA))
    neighbor_k = int(hybrid_hparams.get("neighbor_k", 10))
    ope_epsilon = float(cfg.data.get("reporting", {}).get("ope_epsilon", DEFAULT_OPE_EPSILON))

    cql_model = DuelingQNetwork(test_states.shape[1], NUM_ACTIONS)
    _ = cql_model(test_states[:1])
    cql_model.load_weights(cql_weights)
    q_cql = np.nan_to_num(cql_model(test_states).numpy(), nan=0.0, posinf=0.0, neginf=0.0)

    dynamics_dir = hybrid_dir / "dynamics"
    dyn_models = []
    dyn_input_dim = test_states.shape[1] + NUM_ACTIONS
    dyn_output_dim = test_states.shape[1] + 1
    for weight_file in sorted(dynamics_dir.glob("model_*.weights.h5")):
        model = DynamicsModel(dyn_input_dim, dyn_output_dim)
        _ = model(tf.zeros((1, dyn_input_dim)))
        model.load_weights(weight_file)
        dyn_models.append(model)
    if not dyn_models:
        raise FileNotFoundError(f"No dynamics models found in {dynamics_dir}")

    policy = PolicyNet(test_states.shape[1])
    value_net = ValueNet(test_states.shape[1])
    _ = policy(test_states[:1])
    _ = value_net(test_states[:1])
    policy.load_weights(hybrid_dir / "ppo_policy.weights.h5")
    value_net.load_weights(hybrid_dir / "ppo_value.weights.h5")

    q_mb = compute_q_mb(dyn_models, value_net, test_states, gamma=gamma)
    q_mb = np.nan_to_num(q_mb, nan=0.0, posinf=0.0, neginf=0.0)
    q_mb = np.minimum(q_mb, q_cql + delta)

    gating = json.loads((hybrid_dir / "gating.json").read_text())
    traj_lengths = compute_traj_lengths(eval_df)
    max_dist = max_neighbor_distance(test_states, train_states, neighbor_k)
    gate_x = gate_features(eval_df, traj_lengths, max_dist)
    gate_x = (gate_x - np.array(gating["mean"])) / np.array(gating["std"])

    gate_logits = np.clip(gate_x @ np.array(gating["coef"]) + gating["intercept"], -60.0, 60.0)
    gate_probs = 1.0 / (1.0 + np.exp(-gate_logits))
    thresholded = np.where(gate_probs >= gating["threshold"], gate_probs, 0.0)
    p_mb = np.clip(thresholded, 0.0, 1.0)

    q_hyb = (1.0 - p_mb[:, None]) * q_cql + p_mb[:, None] * q_mb
    q_hyb = np.nan_to_num(q_hyb, nan=0.0, posinf=0.0, neginf=0.0)

    pi_cql = softmax_probs(q_cql, tau)
    pi_mb = softmax_probs(q_mb, tau)
    policy_probs = (1.0 - p_mb[:, None]) * pi_cql + p_mb[:, None] * pi_mb
    policy_probs = stabilize_policy_probs(policy_probs, epsilon=ope_epsilon)

    return {
        "pi": policy_probs,
        "q_values": q_hyb,
        "extras": {
            "p_mb": p_mb.astype(np.float32),
            "q_cql": q_cql.astype(np.float32),
            "q_mb": q_mb.astype(np.float32),
            "q_hyb": q_hyb.astype(np.float32),
            "physician_action": action_ids(eval_df).astype(np.int32),
        },
    }


def run_policy(policy: str, cfg: Config, split: str, hparams_path: Optional[str]) -> Path:
    results_dir = ensure_results_dir(cfg)
    action_bins = load_action_bins_from_results(cfg)
    df = load_split(cfg, split)
    params = resolve_params(cfg, policy, hparams_path)

    ckpt = load_checkpoint(policy, cfg)
    if ckpt:
        print(f"Loaded checkpoint for {policy} from results/models/{policy}_best.pkl")
    else:
        save_checkpoint(policy, cfg, params)
        print(f"Saved checkpoint for {policy} to results/models/{policy}_best.pkl")

    if policy == "physician":
        train_df = load_split(cfg, "train")
        state_features = load_state_features(cfg.paths["state_features"])
        train_states = train_df[state_features].fillna(0).to_numpy(dtype=np.float32)
        eval_states = df[state_features].fillna(0).to_numpy(dtype=np.float32)
        train_actions = compute_action_ids(train_df, action_bins, cfg.columns)
        knn_k = int(params.get("knn_k", 300))
        pi = knn_policy_probs(train_states, train_actions, eval_states, knn_k)
        q_values = np.zeros_like(pi)
        extras: Optional[Dict[str, np.ndarray]] = None
    elif policy == "hybrid":
        hybrid = run_real_hybrid_eval(cfg, split)
        pi = hybrid["pi"]
        q_values = hybrid["q_values"]
        extras = hybrid["extras"]
    else:
        pi = build_placeholder_pi(df, action_bins, cfg.columns)
        q_values = np.zeros_like(pi)
        extras = None

    outputs = build_policy_outputs(
        df,
        action_bins,
        pi=pi,
        columns=cfg.columns,
        q_values=q_values,
        extras=extras,
    )
    out_path = results_dir / f"policy_outputs_{policy}_{split}.npz"
    outputs.to_npz(out_path)
    print(f"Saved policy outputs to {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/load policy and write outputs")
    parser.add_argument("--config", required=True, help="Path to final_config.yaml")
    parser.add_argument("--train_or_load", action="store_true", help="Train or load checkpoint")
    parser.add_argument("--eval_split", choices=["val", "test"], required=True)
    parser.add_argument("--hparams", help="Path to best hyperparams YAML", default=None)
    return parser.parse_args()


def run_policy_cli(policy: str) -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_policy(policy, cfg, args.eval_split, args.hparams)


__all__ = ["run_policy_cli", "run_policy"]
