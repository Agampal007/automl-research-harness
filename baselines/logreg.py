"""Reference floor: multinomial logistic regression on standardised features."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit(X_train, y_train, *, seed, device, time_budget_s, log):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, random_state=seed))
    model.fit(X_train, y_train)
    return model.predict
