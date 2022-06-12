from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    response = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)

    X = np.linspace(-1.2, 2, n_samples)
    y = response(X)

    eps = np.random.normal(scale=noise, size=n_samples)
    y_noise = y + eps

    train_set, train_response, test_set, test_response = split_train_test(
        pd.DataFrame(X), pd.Series(y_noise), (2 / 3))

    train_set, train_response, test_set, test_response = train_set.to_numpy().reshape(
        (-1)), train_response.to_numpy().reshape(
        (-1)), test_set.to_numpy().reshape(
        (-1)), test_response.to_numpy().reshape((-1))

    fig = go.Figure(
        [go.Scatter(name="true model", x=X, y=y,
                    mode="lines"),
         go.Scatter(name="train set", x=train_set,
                    y=train_response,
                    mode="markers", marker=dict(color="blue", opacity=.7)),
         go.Scatter(name="test set", x=test_set,
                    y=test_response,
                    mode="markers", marker=dict(color="red", opacity=.7))
         ]
    )
    fig.update_layout(title="Generated Data")
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    K = np.arange(11)
    avg_training = []
    avg_validate = []
    for k in K:
        pf = PolynomialFitting(k)
        train_score, validation_score = cross_validate(pf,
                                                       train_set,
                                                       train_response,
                                                       mean_square_error)
        avg_training.append(train_score)
        avg_validate.append(validation_score)

    fig = go.Figure(
        [go.Scatter(name="train error", x=K, y=avg_training,
                    mode="lines"),
         go.Scatter(name="validate error", x=K,
                    y=avg_validate,
                    mode="lines")],
        go.Layout(
            title="Train and Validations errors as function of polynomial degree",
            xaxis={"title": "polynomial degree"},
            yaxis={"title": "error"})
    )

    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = np.argmin(avg_validate)
    pf_ = PolynomialFitting(K[k_star])
    pf_.fit(train_set, train_response)
    score = round(pf_.loss(test_set, test_response), 2)
    print("The best K is: " + str(K[k_star]))
    print("The Validation error: " + str(
        round(avg_validate[int(k_star)], 2)) + " The test error: " + str(
        score))


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_x, train_y = X[:n_samples], y[:n_samples]
    test_x, test_y = X[n_samples:], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    min_evaluation = 0.001
    max_evaluation = 5
    lamdas = np.linspace(min_evaluation, max_evaluation, n_evaluations).astype(
        float)
    avg_training_ridge = []
    avg_validate_ridge = []
    avg_training_lasso = []
    avg_validate_lasso = []

    for lam in lamdas:
        ridge = RidgeRegression(lam)
        train_score_ridge, validation_score_ridge = cross_validate(ridge,
                                                                   train_x,
                                                                   train_y,
                                                                   mean_square_error)
        avg_training_ridge.append(train_score_ridge)
        avg_validate_ridge.append(validation_score_ridge)

        lasso = Lasso(lam)
        train_score_lasso, validation_score_lasso = cross_validate(lasso,
                                                                   train_x,
                                                                   train_y,
                                                                   mean_square_error)
        avg_training_lasso.append(train_score_lasso)
        avg_validate_lasso.append(validation_score_lasso)

    fig = go.Figure(
        [go.Scatter(name="Ridge - train error", x=lamdas, y=avg_training_ridge,
                    mode="lines"),
         go.Scatter(name="Ridge - validate error", x=lamdas,
                    y=avg_validate_ridge,
                    mode="lines"),
         go.Scatter(name="Lasso - train error", x=lamdas, y=avg_training_lasso,
                    mode="lines"),
         go.Scatter(name="Lasso - validate error", x=lamdas,
                    y=avg_validate_lasso,
                    mode="lines")
         ],
        go.Layout(
            title="Train and Validations errors as function of lamda",
            xaxis={"title": "lamda"},
            yaxis={"title": "error"})
    )
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    lam_star_ridge = np.argmin(avg_validate_ridge)
    lam_star_lasso = np.argmin(avg_validate_lasso)

    print("The best regularization parameter for ridge is: " + str(
        lamdas[lam_star_ridge]))
    print("The best regularization parameter for lasso is: " + str(
        lamdas[lam_star_lasso]))

    ridge_ = RidgeRegression(lamdas[lam_star_ridge])
    lasso_ = Lasso(alpha=lamdas[lam_star_lasso])
    lr_ = LinearRegression()

    ridge_.fit(train_x, train_y)
    lasso_.fit(train_x, train_y)
    lr_.fit(train_x, train_y)

    score_ridge = round(ridge_.loss(test_x, test_y), 2)
    score_lasso = round(mean_square_error(test_y, lasso_.predict(test_x)), 2)
    score_LS = round(lr_.loss(test_x, test_y), 2)

    print("The test error for ridge is: " + str(
        score_ridge))
    print("The test error for lasso is: " + str(
        score_lasso))
    print("The test error for LS is: " + str(
        score_LS))


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)

    select_regularization_parameter()
