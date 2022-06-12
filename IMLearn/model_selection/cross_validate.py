from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_score = 0
    validation_score = 0
    folds = np.remainder(np.arange(len(X)), cv)
    for k in range(cv):
        train_k_x = X[folds != k]
        train_k_y = y[folds != k]
        validate_k_x = X[folds == k]
        validate_k_y = y[folds == k]

        estimator.fit(train_k_x, train_k_y)
        train_k_pred = estimator.predict(train_k_x)
        validate_k_pred = estimator.predict(validate_k_x)

        train_score += scoring(train_k_y, train_k_pred)
        validation_score += scoring(validate_k_y, validate_k_pred)

    return train_score / cv, validation_score / cv
