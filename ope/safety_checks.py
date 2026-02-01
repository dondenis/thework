from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from preprocessing.action_bins import load_action_bins
from preprocessing.config_utils import load_config


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(np.where(p > 0, p * np.log(p / m), 0.0))
    kl_qm = np.sum(np.where(q > 0, q * np.log(q / m), 0.0))
    return float(0.5 * (kl_pm + kl_qm))


def compute_support_mask(actions: np.ndarray, min_freq: float = 0.01) -> np.ndarray:
    actions = actions.astype(int)
    counts = np.bincount(actions, minlength=25)
    freqs = counts / counts.sum()
    return freqs < min_freq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safety and plausibility checks")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    results_dir = cfg.results_dir
    policies = cfg.data.get("policies", [])
    bins = load_action_bins(results_dir / "action_bins.json")

    outputs: Dict[str, np.lib.npyio.NpzFile] = {}
    for policy in policies:
        path = results_dir / f"policy_outputs_{policy}_{args.split}.npz"
        outputs[policy] = np.load(path, allow_pickle=True)

    physician = outputs.get("physician")
    if physician is None:
        raise FileNotFoundError("Physician policy outputs required for JSD")

    rows = []
    for policy, data in outputs.items():
        action_bin = data["action_bin"]
        extreme = np.mean((action_bin[:, 0] == 4) | (action_bin[:, 1] == 4))
        action_ids = data["action_id_25"].astype(int)
        support_mask = compute_support_mask(physician["action_id_25"])
        masked_rate = float(np.mean(support_mask[action_ids]))

        policy_dist = np.bincount(action_ids, minlength=25).astype(float)
        phys_dist = np.bincount(physician["action_id_25"].astype(int), minlength=25).astype(float)
        jsd_val = jsd(policy_dist, phys_dist)

        row = {
            "policy": policy,
            "extreme_dose_rate": float(extreme),
            "masked_action_rate": masked_rate,
            "jsd_vs_physician": jsd_val,
        }

        if policy == "hybrid":
            p_mb = data["p_mb"] if "p_mb" in data else np.zeros(len(action_ids))
            sofa = data["sofa_bucket"]
            row["median_p_mb_all"] = float(np.median(p_mb))
            row["median_p_mb_high_sofa"] = float(np.median(p_mb[sofa == "high"])) if np.any(sofa == "high") else float("nan")
            q_cql = data["q_cql"] if "q_cql" in data else np.zeros(len(action_ids))
            q_hyb = data["q_hyb"] if "q_hyb" in data else np.zeros(len(action_ids))
            delta = float(cfg.data.get("hyperparams", {}).get("hybrid", {}).get("delta", 0.1))
            row["pr_q_hyb_gt_q_cql_plus_delta"] = float(np.mean((q_hyb - q_cql) > delta))

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "safety_plausibility.csv", index=False)
    (results_dir / "safety_plausibility.json").write_text(df.to_json(orient="records", indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
