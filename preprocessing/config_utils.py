from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import yaml


@dataclass(frozen=True)
class Config:
    data: Dict[str, Any]

    @property
    def paths(self) -> Dict[str, Any]:
        return self.data.get("paths", {})

    @property
    def columns(self) -> Dict[str, str]:
        return self.data.get("columns", {})

    @property
    def results_dir(self) -> Path:
        return Path(self.paths.get("results_dir", "results"))


def load_config(path: str | Path) -> Config:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(data=data)


def load_splits(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = cfg.paths
    train = pd.read_csv(paths["train_csv"])
    val = pd.read_csv(paths["val_csv"])
    test = pd.read_csv(paths["test_csv"])
    return train, val, test


def validate_splits(cfg: Config, required_cols: list[str]) -> None:
    train, val, test = load_splits(cfg)
    missing = [col for col in required_cols if col not in train.columns]
    if missing:
        raise ValueError(f"Missing required columns in train split: {missing}")
    for name, df in [("val", val), ("test", test)]:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {name} split: {missing}")

    icu_col = cfg.columns.get("icustayid", "icustayid")
    train_ids = set(train[icu_col].unique())
    val_ids = set(val[icu_col].unique())
    test_ids = set(test[icu_col].unique())
    overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    if overlap:
        raise ValueError(f"Found overlapping icustayid across splits: {len(overlap)}")

    print(
        "Split sizes:"
        f" train={len(train)} (stays={len(train_ids)}),"
        f" val={len(val)} (stays={len(val_ids)}),"
        f" test={len(test)} (stays={len(test_ids)})"
    )


__all__ = ["Config", "load_config", "load_splits", "validate_splits"]
