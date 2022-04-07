from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data = pd.read_csv(filename)

    # deleting the irrelevant rows
    data = data.loc[data["price"] > 0]  # where price is 0
    data.dropna().drop_duplicates()  # where values are null

    # deleting the irrelevant columns
    data.drop(labels=["id", "date", "lat", "long"], axis=1, inplace=True)

    # setting categorical values to fit the model
    data = pd.concat([data.drop("zipcode", axis=1),
                      pd.get_dummies(data["zipcode"])], axis=1)

    # making groups from the renovated years and then make dummies.
    low_values = [1934, 1951, 1971, 1990, 2000, 2010]
    high_values = [1950, 1970, 1989, 1999, 2009, 2015]
    new_vals = [50, 70, 80, 90, 100, 110]
    for i in range(len(low_values)):
        data["yr_renovated"].mask((low_values[i] <= data["yr_renovated"]) &
                                  (data["yr_renovated"] <= high_values[i]),
                                  new_vals[i], inplace=True)

    data = pd.concat([data.drop("yr_renovated", axis=1),
                      pd.get_dummies(data["yr_renovated"])], axis=1)
    price_df = data.pop("price")
    return data, price_df


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # to store correlations
    correlations = dict()
    for feature in X:
        correlation = ((np.cov(X[feature], y)[0, 1]) /
                       (np.std(X[feature]) * np.std(y)))
        correlations[feature] = correlation

    #  plotting 2 features
    features = ["sqft_living", "condition"]
    for feature_to_plot in features:
        title = "Pearson correlation between " +\
            str(X[feature_to_plot].name) + " and " + str(y.name) + " is " +\
            str(correlations[feature_to_plot])
        fig = go.Figure([go.Scatter(x=X[feature_to_plot], y=y,
                        mode='markers')],
                        layout=go.Layout(
                            title=title,
                            xaxis_title=feature_to_plot,
                            yaxis_title=y.name
                        ))
        fig.write_image(output_path + "/" + str(feature_to_plot) + ".png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")

    # # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, "C:/Users/tomer/Desktop/graphs/1")

    # # # Question 3 - Split samples into training- and testing sets.
    train_set, train_response, test_set, test_response = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    first_p = 0.1
    last_p = 1
    space = 91
    repeat = 10

    lr = LinearRegression()
    loss_arr = []
    mean_of_loss = np.zeros(space)
    std_of_loss = np.zeros(space)

    p_arr = np.linspace(first_p, last_p, space)
    j = 0
    for p in p_arr:
        for i in range(repeat):
            train_set_p = train_set.sample(frac=p)
            train_response_p = train_response.reindex_like(train_set_p)
            lr.fit(train_set_p.to_numpy(), train_response_p.to_numpy())
            loss_arr.append(lr.loss(test_set.to_numpy(),
                                    test_response.to_numpy()))
        mean_of_loss[j] = np.mean(loss_arr)
        std_of_loss[j] = np.std(loss_arr)
        loss_arr.clear()
        j += 1

    fig = go.Figure(
        [go.Scatter(x=p_arr, y=mean_of_loss - 2 * std_of_loss, fill=None,
                    mode="lines", line=dict(color="lightgrey"),
                    showlegend=False),
         go.Scatter(x=p_arr, y=mean_of_loss + 2 * std_of_loss,
                    fill='tonexty',
                    mode="lines", line=dict(color="lightgrey"),
                    showlegend=False),
         go.Scatter(x=p_arr, y=mean_of_loss, mode="markers+lines",
                    marker=dict(color="black", size=1),
                    showlegend=False)],
        layout=go.Layout(
            title=r"$\text{Mean loss as function of increasing percentage "
                  "of train dataset}$",
            height=300))
    fig.update_xaxes(ticksuffix="%", title_text="Percentage of "
                                                "train dataset [%]")
    fig.update_yaxes(title_text="Average loss")
    fig.write_image("C:/Users/tomer/Desktop/graphs/loss2.png")

