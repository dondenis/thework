from __future__ import annotations

import numpy as np


def sofa_bins(sofa_values: np.ndarray) -> np.ndarray:
    sofa = sofa_values.astype(float)
    if sofa.max() <= 1.0:
        sofa = sofa * 24.0
    bins = np.empty(len(sofa), dtype=object)
    bins[sofa < 5] = "low"
    bins[(sofa >= 5) & (sofa <= 15)] = "medium"
    bins[sofa > 15] = "high"
    return bins


__all__ = ["sofa_bins"]
