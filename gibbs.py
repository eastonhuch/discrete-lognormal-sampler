import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal

class GibbsSampler():
    def __init__(self, n_iter=100, alpha_rho=None, beta_rho=None):
        self.n_iter = n_iter

        none_rho_params = [p for p in [alpha_rho, beta_rho] if p is None]
        if none_rho_params == 1:
            raise ValueError("alpha_rho, beta_rho must both be specified or both be None")
        self.alpha_rho = alpha_rho
        self.beta_rho = beta_rho
        self.use_correlated_model = alpha_rho is not None


    def fit(self, data):
        X = data.X
        y = data.y
        n, p = X.shape
        if n != y.size:
            raise ValueError("y.size must equal the number of rows in X")

        self.create_parameters(n, p)


    def create_parameters(self, n, p):
        pass
        # What do I need?
        self.mu_beta = np.
        self.sigma_beta
