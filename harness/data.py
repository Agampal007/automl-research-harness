"""Dataset preparation and profiling.

A dataset is a directory with this layout:

    meta.json            name, n_classes, feature_names, description
    public/train.npz     X (float32), y (int64)   - visible to candidate code
    public/val_X.npy     validation features      - only the trusted runner reads these
    public/test_X.npy    test features
    private/val_y.npy    validation labels        - only the scorer reads these
    private/test_y.npy   test labels              - never shown to the researcher

Candidate code only ever receives the training arrays. Validation and test
labels live in a separate directory that is never passed to the subprocess.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SPLIT_SEED = 0


def write_dataset(
    out: Path,
    X: np.ndarray,
    y: np.ndarray,
    *,
    name: str,
    feature_names: list[str],
    description: str,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> None:
    """Stratified train/val/test split with a fixed seed, written in harness layout."""
    from sklearn.model_selection import train_test_split

    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    X_tr, X_hold, y_tr, y_hold = train_test_split(
        X, y, test_size=val_frac + test_frac, stratify=y, random_state=SPLIT_SEED
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_hold,
        y_hold,
        test_size=test_frac / (val_frac + test_frac),
        stratify=y_hold,
        random_state=SPLIT_SEED,
    )
    (out / "public").mkdir(parents=True, exist_ok=True)
    (out / "private").mkdir(parents=True, exist_ok=True)
    np.savez(out / "public" / "train.npz", X=X_tr, y=y_tr)
    np.save(out / "public" / "val_X.npy", X_val)
    np.save(out / "public" / "test_X.npy", X_te)
    np.save(out / "private" / "val_y.npy", y_val)
    np.save(out / "private" / "test_y.npy", y_te)
    meta = {
        "name": name,
        "description": description,
        "n_classes": int(y.max()) + 1,
        "n_features": int(X.shape[1]),
        "feature_names": feature_names,
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_te)),
        "split_seed": SPLIT_SEED,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


def prepare_covertype(out: Path) -> None:
    """Forest Covertype (UCI): 581k rows, 54 features, 7 classes."""
    from sklearn.datasets import fetch_covtype

    bunch = fetch_covtype()
    names = (
        [
            "Elevation", "Aspect", "Slope",
            "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon",
            "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points",
        ]
        + [f"Wilderness_Area_{i}" for i in range(4)]
        + [f"Soil_Type_{i}" for i in range(40)]
    )
    write_dataset(
        out,
        bunch.data,
        bunch.target - 1,
        name="covertype",
        feature_names=names,
        description=(
            "Forest Covertype: predict the forest cover type (7 classes) of a 30x30 m "
            "cell from cartographic variables. 10 continuous features, a 4-way one-hot "
            "wilderness area, and a 40-way one-hot soil type."
        ),
    )


def prepare_synthetic(out: Path, n: int = 6000, seed: int = 0) -> None:
    """Small non-linear toy problem used by the test suite."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8)).astype(np.float32)
    score = X[:, 0] * X[:, 1] + np.sin(2 * X[:, 2]) + 0.5 * X[:, 3]
    y = np.digitize(score, np.quantile(score, [1 / 3, 2 / 3]))
    write_dataset(
        out, X, y, name="synthetic", feature_names=[f"f{i}" for i in range(8)],
        description="Synthetic 3-class problem with an interaction and a periodic term.",
    )


def load_meta(data_dir: Path) -> dict:
    return json.loads((data_dir / "meta.json").read_text())


def profile(data_dir: Path) -> str:
    """Plain-text profile of the training split, shown to the researcher."""
    meta = load_meta(data_dir)
    with np.load(data_dir / "public" / "train.npz") as z:
        X, y = z["X"], z["y"]
    lines = [
        f"Dataset: {meta['name']} - {meta['description']}",
        f"Rows: train={meta['n_train']}, val={meta['n_val']}, test={meta['n_test']} "
        f"(stratified split, fixed seed). Features: {X.shape[1]}, classes: {meta['n_classes']}.",
        "",
        "Class balance (train): "
        + ", ".join(f"{c}: {p:.3%}" for c, p in enumerate(np.bincount(y) / len(y))),
        "",
    ]
    binary, dense = [], []
    for j, name in enumerate(meta["feature_names"]):
        col = X[:, j]
        if np.isin(col, (0.0, 1.0)).all():
            binary.append((j, name, float(col.mean())))
        else:
            dense.append((j, name, col))
    if dense:
        lines.append("Non-binary features (index name: mean, std, min, median, max, skew):")
        for j, name, col in dense:
            mu, sd = float(col.mean()), float(col.std())
            skew = float(((col - mu) ** 3).mean() / (sd**3 + 1e-12))
            lines.append(
                f"  [{j}] {name}: {mu:.4g}, {sd:.4g}, {col.min():.4g}, "
                f"{np.median(col):.4g}, {col.max():.4g}, {skew:.2f}"
            )
    if binary:
        lines.append("")
        lines.append("Binary 0/1 features (index name: fraction of rows set):")
        lines.append("  " + "; ".join(f"[{j}] {name}: {frac:.4f}" for j, name, frac in binary))
    return "\n".join(lines)
