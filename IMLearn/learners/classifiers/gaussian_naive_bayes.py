from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

from numpy.linalg import det, inv


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m = len(X)
        self.classes_, n_k = np.unique(y, return_counts=True)
        self.pi_ = n_k / m

        n_classes = len(self.classes_)
        n_features = len(X[0])
        self.mu_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))
        self.cov_ = np.zeros((n_classes, n_features, n_features))
        for k in self.classes_:
            self.mu_[k] = np.mean(X[y == k], axis=0)
            self.vars_[k] = np.var(X[y == k], axis=0, ddof=1)
            self.cov_[k] = np.diag(self.vars_[k])


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        responses = np.zeros(len(X))
        likelihood_arr = self.likelihood(X)
        for i in range(len(likelihood_arr)):
            responses[i] = self.classes_[np.argmax(likelihood_arr[i])]
        return responses

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")
        n_samples, n_features = len(X), len(X[0])
        likelihood_arr = np.zeros((n_samples, self.classes_.size))
        cov_inv = inv(self.cov_)

        for k in range(len(self.classes_)):
            likelihood = self.pi_[k] * np.sqrt(
                1 / ((2 * np.pi) ** n_features * det(self.cov_[k]))) \
                         * np.exp(-1 / 2 * np.diag(
                (X - self.mu_[k].T) @ cov_inv[k] @ (
                        X - self.mu_[k].T).T))
            likelihood_arr[:, k] = likelihood

        return likelihood_arr

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
