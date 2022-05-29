import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ab = AdaBoost(DecisionStump, n_learners)
    ab.fit(train_X, train_y)
    training_loss, test_loss = [], []
    for i in range(1, n_learners):
        training_loss.append(ab.partial_loss(train_X, train_y, i))
        test_loss.append(ab.partial_loss(test_X, test_y, i))
    n_learners_arr = np.arange(1, n_learners)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(name="training loss", x=n_learners_arr, y=training_loss,
                   mode="lines"))
    fig.add_trace(go.Scatter(name="test loss", x=n_learners_arr, y=test_loss,
                             mode="lines"))
    fig.update_layout(title="Adaboost losses",
                      xaxis_title="number of fitted learners",
                      yaxis_title="loss value")
    fig.show()

    # Question 2: Plotting decision surfaces
    def decision_surface(predict, xrange, yrange, num_of_learners, density=120,
                         dotted=False,
                         colorscale=custom, showscale=True):
        xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange,
                                                                    density)
        xx, yy = np.meshgrid(xrange, yrange)
        pred = predict(np.c_[xx.ravel(), yy.ravel()], num_of_learners)

        if dotted:
            return go.Scatter(x=xx.ravel(), y=yy.ravel(), opacity=1,
                              mode="markers", marker=dict(color=pred, size=1,
                                                          colorscale=colorscale,
                                                          reversescale=False),
                              hoverinfo="skip", showlegend=False)
        return go.Contour(x=xrange, y=yrange, z=pred.reshape(xx.shape),
                          colorscale=colorscale, reversescale=False,
                          opacity=.7, connectgaps=True, hoverinfo="skip",
                          showlegend=False, showscale=showscale)

    def decision_boundaries_fig(T, ab, decision_surface, lims, symbols, test_X,
                                test_y, title, rows, cols, size):
        fig = make_subplots(rows=rows, cols=cols,
                            subplot_titles=[rf"$\textbf{{{t}}}$" for t in
                                            T],
                            horizontal_spacing=0.01, vertical_spacing=.03)
        for i, t in enumerate(T):
            fig.add_traces(
                [decision_surface(ab.partial_predict, lims[0], lims[1], t,
                                  showscale=False),
                 go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                            showlegend=False,
                            marker=dict(color=test_y.astype(int), size=size,
                                        symbol=symbols[test_y.astype(int)],
                                        colorscale=[custom[0],
                                                    custom[-1]],
                                        line=dict(color="black",
                                                  width=1)))],
                rows=(i // rows) + 1, cols=(i % cols) + 1)
        fig.update_layout(
            title=rf"$\textbf{{{title}}}$",
            margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)
        fig.show()

    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])
    symbols = np.array(["circle", "x"])
    Q2_title = "Decision Boundaries Of Adaboost For different number of " \
               "learners "
    decision_boundaries_fig(T, ab, decision_surface, lims, symbols, test_X,
                            test_y, Q2_title, 2, 2, None)

    # Question 3: Decision surface of best performing ensemble
    T_best = np.min(np.argmin(test_loss)) + 1
    from IMLearn.metrics import accuracy
    ensemble_accuracy = accuracy(test_y,
                                 ab.partial_predict(test_X, int(T_best)))
    Q3_title = "Ensemble Size: " + str(T_best) + " Ensemble Accuracy: " + str(
        ensemble_accuracy)
    decision_boundaries_fig([int(T_best)], ab, decision_surface, lims, symbols,
                            test_X,
                            test_y, Q3_title, 1, 1, None)

    # Question 4: Decision surface with weighted samples
    size = ab.D_
    size = (size / np.max(size)) * 5
    Q4_title = "Decision surface with weighted samples"
    decision_boundaries_fig([max(T)], ab, decision_surface, lims, symbols,
                            test_X,
                            test_y, Q4_title, 1, 1, size)


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
