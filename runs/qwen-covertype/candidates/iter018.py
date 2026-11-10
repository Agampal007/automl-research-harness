"""Ensemble of 2 HistGradientBoosting models (1023 leaves each).

Hypothesis: averaging predicted probabilities from two independently-trained
models reduces variance and beats a single larger model, given the ~3%
train/val gap suggests overfitting that ensembling can mitigate.
"""

import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def _make_model(max_iter, seed):
    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=0.3,
        max_leaf_nodes=1023,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=seed,
    )


def fit(X_train, y_train, *, seed, device, time_budget_s, log):
    start = time.time()
    X = np.ascontiguousarray(X_train, dtype=np.float32)
    y = np.ascontiguousarray(y_train, dtype=np.int64)
    n = len(X)

    # --- Measure per-iteration cost with 1023 leaves. ---
    t0 = time.time()
    _make_model(5, seed).fit(X, y)
    t1 = time.time()
    est1 = (t1 - t0) / 5.0
    if est1 * 20 < time_budget_s * 0.15:
        _make_model(25, seed).fit(X, y)
        t2 = time.time()
        per_iter = max((t2 - t1) / 20.0, est1)
    else:
        t2 = t1
        per_iter = est1
    print(f"timing: first fit {t1 - t0:.2f}s (incl. binning), per-iter {per_iter * 1e3:.0f} ms")

    # Budget: split 85% of time_budget between 2 models, leaving 15% for overhead.
    budget_per_model = (time_budget_s * 0.85 - (t2 - start)) / 2.0
    max_iter = int(min(200, max(10, budget_per_model / per_iter)))
    print(f"planned max_iter per model={max_iter}, budget per model={budget_per_model:.1f}s")

    # --- Fit 2 models with different seeds. ---
    models = []
    for i in range(2):
        model_seed = seed + i * 1000
        model = _make_model(max_iter, model_seed)
        model.fit(X, y)
        elapsed = time.time() - start
        print(f"model {i} fit done in {elapsed:.1f}s, n_iter_={model.n_iter_}")
        models.append(model)

    # --- Holdout slice for logging. ---
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = min(20000, n // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    # Ensemble prediction on holdout
    proba_val = np.zeros((len(val_idx), 7), dtype=np.float64)
    for m in models:
        proba_val += m.predict_proba(X[val_idx])
    proba_val /= len(models)
    holdout_acc = float(np.mean(np.argmax(proba_val, axis=1) == y[val_idx]))

    proba_tr = np.zeros((len(tr_idx), 7), dtype=np.float64)
    for m in models:
        proba_tr += m.predict_proba(X[tr_idx])
    proba_tr /= len(models)
    train_acc = float(np.mean(np.argmax(proba_tr, axis=1) == y[tr_idx]))

    log(1, train_loss=float("nan"), holdout_acc=holdout_acc)
    print(f"ensemble holdout acc: {holdout_acc:.5f} (train acc {train_acc:.5f})")

    def predict(X_new):
        X_new = np.ascontiguousarray(X_new, dtype=np.float32)
        n_new = len(X_new)
        proba = np.zeros((n_new, 7), dtype=np.float64)
        for m in models:
            for i in range(0, n_new, 65536):
                chunk = X_new[i:i + 65536]
                proba[i:i + 65536] += m.predict_proba(chunk)
        proba /= len(models)
        return np.argmax(proba, axis=1)

    return predict
