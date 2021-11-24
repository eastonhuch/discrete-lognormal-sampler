import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal


class Parameter():
    def __init__(self, values) -> None:
        self.values = values
        self.last_idx = 0

    def get_last_value(self):
        return self.values[self.last_idx]

    def set_next_value(self, next_value):
        self.value[self.last_idx+1] = next_value.copy()
        self.last_idx += 1


class GibbsSampler():
    def __init__(self, alpha_rho=None, beta_rho=None):
        none_rho_params = [p for p in [alpha_rho, beta_rho] if p is None]
        if none_rho_params == 1:
            raise ValueError("alpha_rho, beta_rho must both be specified or both be None")
        self.alpha_rho = alpha_rho
        self.beta_rho = beta_rho
        self.use_correlated_model = alpha_rho is not None

    def fit(self, data, n_iter=100):

        # Prepare data/parameters
        X = data.X
        y = data.y
        n, p = X.shape
        if n != y.size:
            raise ValueError("y.size must equal the number of rows in X")
        self.create_parameters(n, p, n_iter)

        # Model-fitting loop
        for i in range(1, n_iter+1):
            self.sample_z()
            self.sample_beta()
            self.sample_alpha()
            if self.use_correlated_model:
                self.sample_rho()

    def create_parameters(self, n, p, n_iter):
        # Hyperparameters
        self.mu_beta = np.zeros(p)
        self.sigma_beta = np.diag([0]*2 + [1]*(p-2))
        self.sigma_beta[0, 0] = 0
        self.sigma_beta[1, 1] = 0

        self.mu_alpha = self.mu_beta.copy()
        self.sigma_alpha = self.sigma_beta.copy()

        # Estimated parameters
        self.beta = Parameter(np.zeros([n_iter, p]))
        self.mu = Parameter(np.zeros([n_iter, n]))
        self.alpha = Parameter(np.zeros([n_iter, p]))
        self.sigma = Parameter(np.zeros([n_iter, n]))
        self.z
        if self.use_correlated_model:
            rho_prior_mean = self.alpha_rho / self.beta_rho
            self.rho = Parameter(np.full(n_iter, rho_prior_mean))

    def sample_z(self):
        pass

    def sample_beta(self):
        pass

    def sample_alpha(self):
        pass

    def sample_rho(self):
        pass