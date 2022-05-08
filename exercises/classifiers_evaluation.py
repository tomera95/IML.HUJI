import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X_data, y_data = \
            load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def perceptron_callback(fit: Perceptron, x: np.ndarray, y: int):
            losses.append(fit.loss(X_data, y_data))

        Perceptron(callback=perceptron_callback).fit(X_data, y_data)
        losses_amount = np.linspace(1, len(losses), len(losses)).astype(int)
        # Plot figure of loss as function of fitting iteration
        fig = go.Figure([go.Scatter(x=losses_amount, y=losses,
                                    mode="lines")],
                        layout=go.Layout(
                            title="Perceptron algorithm's training loss "
                                  "values for " + n + " data",
                            xaxis_title="training iterations",
                            yaxis_title="loss value",
                        ))
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X_data, y_data = \
            load_dataset("../datasets/" + f)

        # Fit models and predict over training set
        lda = LDA()
        gnb = GaussianNaiveBayes()
        y_pred_lda = lda.fit(X_data, y_data).predict(X_data)
        y_pred_gnb = gnb.fit(X_data, y_data).predict(X_data)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        lda_accuracy = accuracy(y_data, y_pred_lda)
        gnb_accuracy = accuracy(y_data, y_pred_gnb)

        lda_title = "LDA classifier: " + str(lda_accuracy)
        gnb_title = "Gaussian Naive Bayes classifier: " + str(gnb_accuracy)

        fig = make_subplots(1, 2, vertical_spacing=1,
                            subplot_titles=[gnb_title, lda_title])
        fig.update_layout(title={'text': f[:9] + " dataset",
                                 'x': 0.5,
                                 'xanchor': 'center'
                                 }, title_font_size=20, showlegend=False)

        # Add traces for data-points setting symbols and colors
        models = [gnb, lda]
        predicts = [y_pred_gnb, y_pred_lda]
        symbols = np.array(["circle", "square", "triangle-up"])

        for i, model in enumerate(models):
            fig.add_trace(
                go.Scatter(x=X_data[:, 0], y=X_data[:, 1], mode="markers",
                           marker=dict(color=predicts[i],
                                       symbol=symbols[y_data], size=10),
                           line=dict(color="black", width=1))
                , row=1, col=i + 1)

            # Add `X` dots specifying fitted Gaussians' means
            fig.add_trace(
                go.Scatter(x=model.mu_[:, 0], y=model.mu_[:, 1],
                           mode="markers",
                           marker=dict(color="black", symbol="x", size=15)),
                row=1, col=i + 1)

        # Add ellipses depicting the covariances of the fitted Gaussians
        # for GNB
        for i in range(len(gnb.mu_)):
            fig.add_trace(get_ellipse(gnb.mu_[i, :], gnb.cov_[i]),
                          row=1, col=1)

        # for LDA
        for i in range(len(lda.classes_)):
            fig.add_trace(get_ellipse(lda.mu_[i, :], lda.cov_),
                          row=1, col=2)

        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()

