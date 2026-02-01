from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from preprocessing.action_bins import ActionBins, bin_action, to_action_id_25
from preprocessing.sofa_utils import sofa_bins


@dataclass(frozen=True)
class PolicyOutputs:
    row_index: np.ndarray
    icustayid: np.ndarray
    sofa: np.ndarray
    sofa_bucket: np.ndarray
    action_cont: np.ndarray
    action_bin: np.ndarray
    action_id_25: np.ndarray
    pi: np.ndarray
    q_values: Optional[np.ndarray] = None
    extras: Optional[Dict[str, np.ndarray]] = None

    def to_npz(self, path: Path) -> None:
        payload = {
            "row_index": self.row_index,
            "icustayid": self.icustayid,
            "sofa": self.sofa,
            "sofa_bucket": self.sofa_bucket,
            "action_cont": self.action_cont,
            "action_bin": self.action_bin,
            "action_id_25": self.action_id_25,
            "pi": self.pi,
        }
        if self.q_values is not None:
            payload["q_values"] = self.q_values
        if self.extras:
            payload.update(self.extras)
        np.savez(path, **payload)


def build_policy_outputs(
    df: pd.DataFrame,
    action_bins: ActionBins,
    pi: np.ndarray,
    columns: Optional[Dict[str, str]] = None,
    q_values: Optional[np.ndarray] = None,
    extras: Optional[Dict[str, np.ndarray]] = None,
) -> PolicyOutputs:
    columns = columns or {}
    icu_col = columns.get("icustayid", "icustayid")
    sofa_col = columns.get("sofa", "SOFA")
    vaso_col = columns.get("vaso_input", "vaso_input")
    iv_col = columns.get("iv_input", "iv_input")

    row_index = df.index.to_numpy()
    icustayid = df[icu_col].to_numpy()
    sofa = df[sofa_col].to_numpy()
    sofa_bucket = sofa_bins(sofa)
    vaso = df[vaso_col].fillna(0).to_numpy(dtype=float)
    iv = df[iv_col].fillna(0).to_numpy(dtype=float)
    action_cont = np.column_stack([vaso, iv])

    vaso_bins = np.array([bin_action(v, action_bins.vaso_edges) for v in vaso], dtype=int)
    iv_bins = np.array([bin_action(v, action_bins.iv_edges) for v in iv], dtype=int)
    action_bin = np.column_stack([vaso_bins, iv_bins])
    action_id = np.array([to_action_id_25(iv_b, v_b) for iv_b, v_b in zip(iv_bins, vaso_bins)], dtype=int)

    return PolicyOutputs(
        row_index=row_index,
        icustayid=icustayid,
        sofa=sofa,
        sofa_bucket=sofa_bucket,
        action_cont=action_cont,
        action_bin=action_bin,
        action_id_25=action_id,
        pi=pi,
        q_values=q_values,
        extras=extras,
    )
