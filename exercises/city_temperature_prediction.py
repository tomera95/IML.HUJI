import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=["Date"])

    # deleting the irrelevant rows
    data.dropna().drop_duplicates()  # where values are null
    temps = data["Temp"].to_numpy()
    data = data.loc[data["Temp"] > temps.min()]

    #  adding day of year column
    day_of_year = []
    for i in range(len(data)):
        day_of_year.append(pd.to_datetime(data["Date"].to_numpy()[i])
                           .timetuple().tm_yday)
    data["DayOfYear"] = day_of_year
    return data


if __name__ == '__main__':

    def plot_bar(x, y, label_x, label_y, title, path):
        fig = px.bar(x=x, y=y,
                     title=title,
                     labels={"x": label_x,
                             "y": label_y})
        fig.update_layout(
            xaxis=dict(
                tickmode='linear',
                tick0=1,
                dtick=1))
        fig.write_image(path)

    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    # first figure
    data_il = data[data["Country"] == "Israel"]
    temp_il = data_il["Temp"]
    day_of_year_il = data_il["DayOfYear"]
    years_il = data_il["Year"].astype("category")

    fig = px.scatter(data_il, x=day_of_year_il, y=temp_il, color=years_il,
                     title="Daily temperature in Israel", labels={
                        "DayOfYear": "Day of year",
                        "Temp": "Temperature",
                        "color": "Years"})
    fig.write_image("C:/Users/tomer/Desktop/graphs/temptoyearil.png")

    # second figure
    data_months_group = data_il.groupby("Month").std()
    temp_stds = data_months_group["Temp"]

    plot_bar(data_months_group.index, temp_stds, "Month",
             "Standard deviation of temperatures",
             "Standard deviation of temperatures in each month",
             "C:/Users/tomer/Desktop/graphs/barplot.png")

    # Question 3 - Exploring differences between countries
    data_group_mean = data.groupby(by=["Month", "Country"]).mean()
    temp_avg = data_group_mean["Temp"]
    data_group_std = data.groupby(by=["Month", "Country"]).std()
    temp_std = data_group_std["Temp"]
    fig = px.line(x=data_group_mean.reset_index()["Month"], y=temp_avg,
                  error_y=temp_std,
                  color=data_group_mean.reset_index()["Country"],
                  title="Average monthly temperature",
                  labels={"x": "Month",
                          "y": "Average temperatures",
                          "color": "Countries"})
    fig.write_image("C:/Users/tomer/Desktop/graphs/average_temp_country.png")

    # # Question 4 - Fitting model for different values of `k`
    losses = []
    y = data_il["Temp"]
    X = data_il["DayOfYear"]
    ks = np.linspace(1, 10, 10).astype(int)

    train_set, train_response, test_set, test_response = \
        split_train_test(X, y)
    for k in ks:
        pf = PolynomialFitting(k)
        pf.fit(train_set.to_numpy(), train_response.to_numpy())
        loss = round(pf.loss(test_set.to_numpy(), test_response.to_numpy()), 2)
        losses.append(loss)
        print("k: ", k, "loss: ", loss)

    plot_bar(ks, losses, "Test error for each value of K", "K", "Test error",
             "C:/Users/tomer/Desktop/graphs/Kbarplot.png")

    # # Question 5 - Evaluating fitted model on different countries
    nations = ["Jordan", "South Africa", "The Netherlands"]
    loss_by_nation = []
    best_k = 5
    pf = PolynomialFitting(best_k)
    pf.fit(X, y)
    for nation in nations:
        data_nation = data[data["Country"] == nation]
        loss_by_nation.append(pf.loss(data_nation["DayOfYear"],
                                      data_nation["Temp"]))

    plot_bar(nations, loss_by_nation, "nations", "Test error",
             "Test error for each nation with K=5",
             "C:/Users/tomer/Desktop/graphs/nations_bar_plot.png")
