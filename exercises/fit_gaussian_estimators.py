from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"
NUM_OF_SAMPLES = 1000


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    univariate_gaussian = UnivariateGaussian(False)
    mu = 10
    sigma = 1
    X = np.random.normal(mu, sigma, NUM_OF_SAMPLES)
    univariate_gaussian.fit(X)
    expectation, variance = univariate_gaussian.mu_, univariate_gaussian.var_
    print((expectation, variance))

    # Question 2 - Empirically showing sample mean is consistent
    min_samples = 10
    space = 100
    ms = np.linspace(min_samples, NUM_OF_SAMPLES, space).astype(np.int)
    estimated_mean_vs_mu = []
    for m in ms:
        iter_X = X[1:m]
        estimated_mean_vs_mu.append(abs(np.mean(iter_X) - mu))

    go.Figure([go.Scatter(x=ms, y=estimated_mean_vs_mu, mode='markers')],
              layout=go.Layout(
                  title="Absolute distance between the estimated-"
                        " and true value of the expectation,"
                        "as a function of the sample size.",
                  xaxis_title="Number of samples",
                  yaxis_title="Absolute distance between the estimated- "
                              "and true value of the expectation",
              )).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = univariate_gaussian.pdf(X)
    go.Figure([go.Scatter(x=X, y=pdfs, mode='markers')],
              layout=go.Layout(
                  title="PDF as function of sample value",
                  xaxis_title="Sample value",
                  yaxis_title="PDF value"
              )).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    multivariate_gaussian = MultivariateGaussian()
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(np.transpose(mu), sigma, NUM_OF_SAMPLES)
    multivariate_gaussian.fit(X)
    expectation, covariance = multivariate_gaussian.mu_, \
                              multivariate_gaussian.cov_
    print(expectation)
    print(covariance)

    # Question 5 - Likelihood evaluation
    space = 200
    log_arr = np.zeros([space, space])
    f1 = np.linspace(-10, 10, space)
    f3 = np.linspace(-10, 10, space)

    for i in range(len(f1)):
        for j in range(len(f3)):
            mu_new = np.array([f1[i], 0, f3[j], 0])
            log_arr[i][j] = \
                multivariate_gaussian.log_likelihood(mu_new, sigma, X)

    go.Figure(go.Heatmap(x=f3, y=f1, z=log_arr),
              layout=go.Layout(title="Heatmap",
                               xaxis_title="f3",
                               yaxis_title="f1"
                               )).show()

    # Question 6 - Maximum likelihood
    max_vals = np.unravel_index(np.argmax(log_arr, axis=None),
                                log_arr.shape)
    print("feature 1 maximum: ", f1[max_vals[0]])
    print("feature 3 maximum: ", f3[max_vals[1]])
    print("maximum log-likelihood value: ", np.max(log_arr))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
