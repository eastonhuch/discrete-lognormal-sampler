import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm, multivariate_normal


class Parameter():
    def __init__(self, values) -> None:
        self.values = values
        self.last_idx = 0

    def get_last_value(self):
        return self.values[self.last_idx].copy()

    def set_next_value(self, next_value):
        self.values[self.last_idx+1] = next_value.copy()
        self.increment_idx()

    def increment_idx(self):
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
        self.x = data.x
        self.y = data.y
        n, p = self.x.shape
        self.n = n
        self.p = p
        if n != self.y.size:
            raise ValueError("data.y.size must equal the number of rows in data.x")
        self.n_iter = n_iter
        self.create_parameters()

        # Model-fitting loop
        for i in range(1, n_iter+1):
            self.sample_z()
            self.sample_beta()
            break
            # self.sample_alpha()
            # if self.use_correlated_model:
            #     self.sample_rho()

    def create_parameters(self):
        n = self.n
        p = self.p
        n_iter = self.n_iter

        # Hyperparameters
        self.mu_beta = np.zeros(p)
        self.sigma_beta_inv = np.diag([0]*2 + [1]*(p-2))
        self.beta_prior_weight = self.sigma_beta_inv @ self.mu_beta
        self.mu_alpha = self.mu_beta.copy()
        self.sigma_alpha_inv = self.sigma_beta_inv.copy()

        # Estimated parameters
        self.beta = Parameter(np.zeros([n_iter, p]))
        self.mu = Parameter(np.zeros([n_iter, n]))
        self.alpha = Parameter(np.zeros([n_iter, p]))
        self.sigma = Parameter(np.ones([n_iter, n]))
        self.z = Parameter(np.zeros([n_iter, n]))
        self.rho = Parameter(np.zeros(n_iter))

    def sample_z(self):
        mu = self.mu.get_last_value()
        Sigma = self.calculate_Sigma()
        Sigma_inv = self.calculate_Sigma_inv()
        z = self.z.get_last_value() # z from last iteration
        e = z - mu
        n = self.n
        for i in range(n):
            Sigma_i_ni = np.delete(Sigma[i], i)
            Sigma_inv_ni_ni = np.delete(np.delete(Sigma_inv, i, axis=0), i, axis=1)
            Sigma_inv_i_ni = np.delete(Sigma_inv[i], i)
            Sigma_ni_ni_inv = Sigma_inv_ni_ni - Sigma_inv_i_ni @ (Sigma_inv_ni_ni @ Sigma_inv_i_ni)
            Sigma_prod = Sigma_i_ni @ Sigma_ni_ni_inv

            z_i_mean = mu[i] + Sigma_prod @ np.delete(e, i)
            z_i_var = Sigma[i, i] - Sigma_prod @ Sigma_i_ni
            y_i = self.y[i]
            z[i] = truncnorm.rvs(
                np.log(y_i), np.log(y_i+1),
                loc=z_i_mean, scale=np.sqrt(z_i_var))

        self.z.set_next_value(z)

    def sample_beta(self):
        # Pull out the values I need for convenience
        x = self.x
        z = self.z.get_last_value()
        Sigma_inv = self.calculate_Sigma_inv()

        # Calculate posterior parameters and sample
        xT_Sigma_inv = x.T @ Sigma_inv
        sigma_posterior_inv = self.sigma_beta_inv + xT_Sigma_inv @ x
        sigma_posterior = np.linalg.inv(sigma_posterior_inv)
        mu_posterior = sigma_posterior @ (self.beta_prior_weight + xT_Sigma_inv @ z)
        beta_sample = multivariate_normal.rvs(mean=mu_posterior, cov=sigma_posterior)
        self.beta.set_next_value(beta_sample)

    def sample_alpha(self):
        pass

    def sample_rho(self):
        pass

    # Helper functions for calculating cov/corr matrices and their inverses
    def calculate_Sigma(self):
        R = self.calculate_R()
        sigma_2d = self.get_sigma_2d()
        return sigma_2d @ R @ sigma_2d

    def calculate_Sigma_inv(self):
        R_inv = self.calculate_R_inv()
        sigma_inv_2d = self.get_sigma_inv_2d()
        return sigma_inv_2d @ R_inv @ sigma_inv_2d

    def calculate_R(self):
        # Pull out paramter values of interest
        rho = self.rho.get_last_value()
        n = self.n

        # Create R
        ns = np.arange(n)
        ns_repeated = np.repeat(ns, n)
        ns_tiled = np.tile(ns, n)
        abs_diff = np.abs(ns_repeated - ns_tiled)
        R = rho**abs_diff
        R.resize([n, n]) # Surprisingly, this modifies in place
        return R

    def calculate_R_inv(self):
        # Pull out parameter values we need
        rho = self.rho.get_last_value()
        n = self.n

        # Calculate R_inv
        k = (1-rho**2)
        R_inv = np.eye(n)
        R_inv[1, 2] = -rho
        R_inv[2, 1] = -rho
        for i in range(1, n-1):
            R_inv[i, i] = 1 + rho**2
            R_inv[i+1, i] = -rho
            R_inv[i, i+1] = -rho
        R_inv *= k

        return R_inv

    def get_sigma_2d(self):
        return np.diag(self.sigma.get_last_value())

    def get_sigma_inv_2d(self):
        return np.diag(1/self.sigma.get_last_value())
