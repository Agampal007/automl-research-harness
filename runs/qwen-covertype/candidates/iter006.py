"""Increase max_leaf_nodes from 31 to 63.

Single change: more leaves per boosting round. Iteration 5 showed that
halving leaves (31→15) cost ~3 points of accuracy, indicating the model
is capacity-limited. Doubling leaves should let each round capture more
complex boundaries (elevation × soil-type interactions) even though the
total iteration count drops proportionally within the 60 s wall-clock
budget.
"""

import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def _make_model(max_iter, seed):
    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=0.3,
        max_leaf_nodes=63,          # <-- changed from 31
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=seed,
    )


def fit(X_train, y_train, *, seed, device, time_budget_s, log):
    start = time.time()
    X = np.ascontiguousarray(X_train, dtype=np.float32)
    y = np.ascontiguousarray(y_train, dtype=np.int64)
    n = len(X)

    # --- Measure per-iteration cost so we stay inside the wall-clock budget. ---
    t0 = time.time()
    _make_model(5, seed).fit(X, y)
    t1 = time.time()
    est1 = (t1 - t0) / 5.0
    if est1 * 25 < time_budget_s * 0.2:
        _make_model(30, seed).fit(X, y)
        t2 = time.time()
        per_iter = max((t2 - t1) / 25.0, est1)
    else:
        t2 = t1
        per_iter = est1
    print(f"timing: first fit {t1 - t0:.2f}s (incl. binning), per-iter {per_iter * 1e3:.0f} ms")

    budget_left = time_budget_s * 0.85 - (t2 - start)
    max_iter = int(min(500, max(50, budget_left / per_iter)))
    print(f"planned max_iter={max_iter}")

    # --- Fit the final model on all training rows. ---
    model = _make_model(max_iter, seed)
    model.fit(X, y)
    print(f"fit done in {time.time() - start:.1f}s, n_iter_={model.n_iter_}")

    # --- Holdout slice of the train set, for logging only. ---
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = min(20000, n // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    holdout_acc = float(np.mean(model.predict(X[val_idx]) == y[val_idx]))
    train_acc = float(np.mean(model.predict(X[tr_idx]) == y[tr_idx]))

    log(1, train_loss=float("nan"), holdout_acc=holdout_acc)
    print(f"train-subset holdout acc: {holdout_acc:.5f} (train acc {train_acc:.5f})")

    def predict(X_new):
        X_new = np.ascontiguousarray(X_new, dtype=np.float32)
        out = []
        for i in range(0, len(X_new), 65536):
            out.append(model.predict(X_new[i:i + 65536]))
        return np.concatenate(out)

    return predict
