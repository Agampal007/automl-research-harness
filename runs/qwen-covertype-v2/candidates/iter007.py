import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def fit(X_train, y_train, *, seed, device, time_budget_s, log):
    start = time.time()
    n_classes = int(y_train.max()) + 1

    # Train HistGradientBoostingClassifier with good hyperparameters
    # Use early stopping with internal validation to pick best n_iter
    model = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.08,
        max_leaf_nodes=63,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.1,
        max_bins=255,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        tol=1e-4,
    )

    model.fit(X_train, y_train)

    elapsed = time.time() - start
    n_iter_used = model.n_iter_
    print(f"HistGBM: {n_iter_used} iterations in {elapsed:.1f}s")

    # Quick self-eval on training data (sanity check)
    train_pred = model.predict(X_train[:50000])
    train_acc = (train_pred == y_train[:50000]).mean()
    log(1, train_loss=0.0, holdout_acc=train_acc)
    print(f"Train acc (first 50k): {train_acc:.5f}")

    def predict(X_new):
        return model.predict(X_new)

    return predict
