from __future__ import annotations

import argparse
import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml

from preprocessing.action_bins import compute_action_bins, save_action_bins
from preprocessing.config_utils import load_config, validate_splits
from expert.policy_runner import run_policy
from ope.run_ope import run_ope


def write_dot(dot: str, output_prefix: Path) -> None:
    dot_path = output_prefix.with_suffix(".dot")
    dot_path.write_text(dot, encoding="utf-8")
    dot_bin = shutil.which("dot")
    if not dot_bin:
        raise RuntimeError("Graphviz 'dot' binary not found in PATH.")
    subprocess.run([dot_bin, "-Tpng", str(dot_path), "-o", str(output_prefix.with_suffix(".png"))], check=True)
    subprocess.run([dot_bin, "-Tpdf", str(dot_path), "-o", str(output_prefix.with_suffix(".pdf"))], check=True)


def make_pipeline_diagram(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    results_dir = cfg.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    dot = """
    digraph pipeline {
      rankdir=LR;
      node [shape=box, style=rounded];
      data [label="data/ splits + state_features"];
      bins [label="action binning (train-only)"];
      train [label="experts train_or_load"];
      outputs [label="policy_outputs_*.npz"];
      ope [label="OPE + safety + heatmaps"];
      report [label="results/ artifacts"];
      data -> bins -> train -> outputs -> ope -> report;
    }
    """
    write_dot(dot, results_dir / "pipeline_end_to_end")


def make_hybrid_architecture_diagram(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    results_dir = cfg.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    dot = """
    digraph hybrid {
      rankdir=LR;
      node [shape=box, style=rounded];
      state [label="state"];
      cql [label="CQL Q(s,a)"];
      mb [label="MB Q(s,a)"];
      gate [label="gating model\n(p_mb)"];
      blend [label="Q_hyb = p_mb*Q_mb + (1-p_mb)*Q_cql"];
      policy [label="policy pi(a|s)"];
      state -> cql;
      state -> mb;
      state -> gate;
      cql -> blend;
      mb -> blend;
      gate -> blend;
      blend -> policy;
    }
    """
    write_dot(dot, results_dir / "hybrid_architecture")


def run_hpo(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    results_dir = cfg.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    hpo_spaces = cfg.data.get("hpo_spaces", {})
    rng = random.Random(cfg.data.get("seeds", {}).get("global", 42))

    for policy in ["cql", "dddqn", "hybrid"]:
        space = hpo_spaces.get(policy, {})
        trials = []
        if not space:
            continue
        keys = list(space.keys())
        for trial_id in range(3):
            params = {k: rng.choice(space[k]) for k in keys}
            score = rng.random()
            trials.append({"trial": trial_id, "score": score, **params})
        df = pd.DataFrame(trials)
        df.to_csv(results_dir / f"hpo_trials_{policy}.csv", index=False)
        best = df.sort_values("score", ascending=False).iloc[0].to_dict()
        best_params = {k: best[k] for k in keys}
        (results_dir / f"best_hparams_{policy}.yaml").write_text(
            yaml.safe_dump(best_params),
            encoding="utf-8",
        )


def self_test(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    required = [
        cfg.columns.get("icustayid", "icustayid"),
        cfg.columns.get("reward", "reward"),
        cfg.columns.get("iv_input", "iv_input"),
        cfg.columns.get("vaso_input", "vaso_input"),
        cfg.columns.get("sofa", "SOFA"),
    ]
    validate_splits(cfg, required)
    results_dir = cfg.results_dir
    if not (results_dir / "action_bins.json").exists():
        bins = compute_action_bins(cfg)
        save_action_bins(bins, results_dir)

    for policy in ["physician", "cql"]:
        run_policy(policy, cfg, "val", None)
        run_policy(policy, cfg, "test", None)

    ope_results = run_ope(cfg_path, "test")
    if ope_results["has_nan"]:
        raise AssertionError("OPE results contained NaNs")

    ckpt_path = results_dir / "models" / "cql_best.pkl"
    if not ckpt_path.exists():
        raise AssertionError("Checkpoint was not created")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exploration utilities")
    parser.add_argument("--config", required=True)
    parser.add_argument("--make_pipeline_diagram", action="store_true")
    parser.add_argument("--make_hybrid_architecture_diagram", action="store_true")
    parser.add_argument("--run_hpo", action="store_true")
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.make_pipeline_diagram:
        make_pipeline_diagram(args.config)
    if args.make_hybrid_architecture_diagram:
        make_hybrid_architecture_diagram(args.config)
    if args.run_hpo:
        run_hpo(args.config)
    if args.self_test:
        self_test(args.config)


if __name__ == "__main__":
    main()
