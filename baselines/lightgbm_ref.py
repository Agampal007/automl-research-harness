"""Reference: LightGBM, the usual strong tabular baseline.

It has no time-budget hook, so it runs to its iteration cap; the baselines table
reports how long that took next to the score.

min_sum_hessian_in_leaf matters here: with the default (1e-3) the multiclass
loss diverges on this dataset after ~50 rounds because leaves for the rarest
class (0.5% of rows) get near-zero hessians and enormous values.
"""

from lightgbm import LGBMClassifier


def fit(X_train, y_train, *, seed, device, time_budget_s, log):
    model = LGBMClassifier(
        n_estimators=500, learning_rate=0.1, num_leaves=255,
        min_sum_hessian_in_leaf=1.0, random_state=seed, verbose=-1,
    )
    model.fit(X_train, y_train)
    return model.predict
