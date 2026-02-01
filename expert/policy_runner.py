from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

from preprocessing.action_bins import ActionBins, bin_action, load_action_bins
from preprocessing.config_utils import Config, load_config
from expert.policy_outputs import build_policy_outputs


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

    pi = build_placeholder_pi(df, action_bins, cfg.columns)
    q_values = np.zeros_like(pi)
    extras: Dict[str, np.ndarray] = {}
    if policy == "hybrid":
        extras["p_mb"] = np.full(len(df), 0.5, dtype=np.float32)
        extras["q_cql"] = np.zeros(len(df), dtype=np.float32)
        extras["q_hyb"] = np.zeros(len(df), dtype=np.float32)

    outputs = build_policy_outputs(
        df,
        action_bins,
        pi=pi,
        columns=cfg.columns,
        q_values=q_values,
        extras=extras or None,
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
