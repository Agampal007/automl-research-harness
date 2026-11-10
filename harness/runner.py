"""Trusted entry point executed inside the candidate subprocess.

    python -m harness.runner fit   <solution.py> <public_dir> <out_dir> <seed> <budget_s>
    python -m harness.runner probe <probe.py>    <public_dir> <out_dir>

`fit` imports the candidate, calls its `fit(...)` on the training arrays only,
then applies the returned predictor to the validation and test features. The
candidate never receives held-out features during training and never receives
held-out labels at all; scoring happens in the parent process.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np


def _load(path: str):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_fit(solution: str, public: Path, out: Path, seed: int, budget_s: float) -> None:
    with np.load(public / "train.npz") as z:
        X_train, y_train = z["X"], z["y"]
    _seed_everything(seed)
    module = _load(solution)

    curve_file = open(out / "curve.jsonl", "w", buffering=1)
    start = time.time()

    def log(step, **metrics) -> None:
        row = {"step": step, "t": round(time.time() - start, 2)}
        row.update({k: float(v) for k, v in metrics.items()})
        curve_file.write(json.dumps(row) + "\n")

    predict = module.fit(
        X_train, y_train, seed=seed, device=_device(), time_budget_s=budget_s, log=log
    )
    fit_seconds = time.time() - start
    preds = {}
    for split in ("val", "test"):
        X = np.load(public / f"{split}_X.npy")
        preds[split] = np.asarray(predict(X))
    np.savez(out / "preds.npz", **preds)
    (out / "runner.json").write_text(json.dumps({"fit_seconds": round(fit_seconds, 2)}))


def run_probe(script: str, public: Path) -> None:
    with np.load(public / "train.npz") as z:
        X_train, y_train = z["X"], z["y"]
    _seed_everything(0)
    _load(script).probe(X_train, y_train)


def main(argv: list[str]) -> None:
    mode = argv[0]
    if mode == "fit":
        run_fit(argv[1], Path(argv[2]), Path(argv[3]), int(argv[4]), float(argv[5]))
    elif mode == "probe":
        run_probe(argv[1], Path(argv[2]))
    else:
        raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
