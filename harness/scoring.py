"""Trusted scoring and the accept/reject rule."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import mean, variance

import numpy as np


def score(preds: dict[str, np.ndarray], data_dir: Path) -> dict[str, float]:
    """Accuracy and macro-F1 on both held-out splits. Runs in the parent process only."""
    from sklearn.metrics import f1_score

    out = {}
    for split in ("val", "test"):
        y = np.load(data_dir / "private" / f"{split}_y.npy")
        p = preds[split]
        out[f"{split}_acc"] = float((p == y).mean())
        out[f"{split}_f1"] = float(f1_score(y, p, average="macro"))
    return out


def screen(candidate_first_seed: float, incumbent: list[float]) -> bool:
    """Cheap first gate: one seed must beat the incumbent's mean to earn more seeds."""
    return candidate_first_seed > mean(incumbent)


def decide(candidate: list[float], incumbent: list[float], min_delta: float) -> tuple[bool, str]:
    """Accept only if the mean gain exceeds both a floor and the seed noise.

    The noise term is the standard error of the difference of the two means.
    With three seeds this is a coarse estimate; it exists to stop the loop from
    ratcheting upward on lucky seeds, not to claim statistical significance.
    """
    diff = mean(candidate) - mean(incumbent)
    n_c, n_i = len(candidate), len(incumbent)
    se = math.sqrt(
        (variance(candidate) / n_c if n_c > 1 else 0.0)
        + (variance(incumbent) / n_i if n_i > 1 else 0.0)
    )
    threshold = max(min_delta, se)
    verdict = diff > threshold
    return verdict, f"mean gain {diff:+.5f} vs threshold {threshold:.5f} (seed noise {se:.5f})"
