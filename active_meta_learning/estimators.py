import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge

from active_meta_learning.data_utils import form_datasets_from_tasks

# Inner algorithms
class GDLeastSquares(BaseEstimator, RegressorMixin):
    def __init__(self, learning_rate, adaptation_steps):
        self.learning_rate = learning_rate
        self.adaptation_steps = adaptation_steps

    def fit(self, X, y, w_0=None):
        n, d = X.shape
        if w_0 is None:
            self.w_hat_ = np.zeros(d)
        else:
            self.w_hat_ = w_0
        for _ in range(self.adaptation_steps):
            self.w_hat_ -= self.learning_rate * 2 * X.T @ (X @ self.w_hat_ - y)
        return self

    def predict(self, X):
        return X @ self.w_hat_


class RidgeRegression(BaseEstimator, RegressorMixin):
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
        self.w_hat_ = Ridge(alpha=self.alpha, fit_intercept=False).fit(X, y).coef_
        return self

    def predict(self, X):
        return X @ self.w_hat_


class BiasedRidgeRegression(BaseEstimator, RegressorMixin):
    # Note: to stay consistent with RidgeRegPrototypeEstimator
    # and GDLeastSquares which does *not* take n into account, we
    # use \alpha = n\lambda as regulariser
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y, w_0):
        # Fitting this is equivalent to solving
        # a new RR problem through change of variable
        # and then changing back
        y_ = y - X @ w_0
        w_tilde = Ridge(alpha=self.alpha, fit_intercept=False).fit(X, y_).coef_
        self.w_hat_ = w_tilde + w_0
        return self

    def predict(self, X):
        return X @ self.w_hat_


# Prototype algorithms
class RidgeRegPrototypeEstimator:
    def __init__(self, alpha):
        self.alpha = alpha

    def transform(self, tasks):
        datasets = form_datasets_from_tasks(tasks)
        weights = np.array(
            [self._get_weights(dataset).squeeze() for dataset in datasets]
        )
        return weights

    def _get_weights(self, dataset):
        X, y = dataset[:, :-1], dataset[:, -1:]
        return Ridge(alpha=self.alpha, fit_intercept=False).fit(X, y).coef_

    def set_params(self, **params):
        self.alpha = params["alpha"]

    def get_params(self):
        return {"alpha": self.alpha}


class TrueWeightPrototypeEstimator:
    def __init__(self):
        pass

    def transform(self, tasks):
        weights = np.array([task["w"].squeeze() for task in tasks])
        return weights

    def set_params(self, **params):
        pass

    def get_params(self):
        return {}
