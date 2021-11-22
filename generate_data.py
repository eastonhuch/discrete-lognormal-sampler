import numpy as np
import pandas as pd
from scipy.stats import norm #, multivariate_normal
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from dataclasses import dataclass

np.random.seed(1) # For reproducibility

def generate_data():
    days, x_df = get_predictors()
    beta, alpha = get_coefs()
    mu, sigma = get_derived_params(x_df, beta, alpha)
    z, y = sample_data(mu, sigma)

    # Return data in dataclass
    @dataclass(frozen=True)
    class Data:
        days: pd.Series
        x_df: pd.DataFrame
        beta: np.ndarray
        alpha: np.ndarray
        mu: np.ndarray
        sigma: np.ndarray
        z: np.ndarray
        y: np.ndarray

    data = Data(
        days=days,
        x_df=x_df,
        beta=beta,
        alpha=alpha,
        mu=mu,
        sigma=sigma,
        z=z,
        y=y)

    return data


def get_predictors():
    start_date = dt.strptime("2017-01-01", "%Y-%m-%d").date()
    num_years = 3
    max_date = start_date + relativedelta(years=num_years, days=-1)

    days = [start_date]
    while days[-1] < max_date:
        days.append(
            days[-1] + relativedelta(days=1))


    # Put date in data frame
    df = pd.DataFrame({"day": days})

    # Simple transformations
    df = df.assign(
        intercept=1.,
        day_of_week=df.day.apply(lambda d: d.weekday()),
        days_since_start=df.day.apply(lambda d: (d - start_date).days),
        day_of_year=df.day.apply(lambda d: d.timetuple().tm_yday),
    )

    # Small modifications to transformations
    df = df.assign(
        days_since_start=(df.days_since_start - df.days_since_start.mean()) /
            df.days_since_start.std(), # Rescaling
        year_radians=df.day_of_year*2*np.pi/365,
    )

    # Day of week
    day_of_week = pd.get_dummies(df.day_of_week, prefix="day_of_week")

    # Long-term trends
    trends = (df
        .assign(days_since_start_squared=df.days_since_start**2)
        .loc[:, ["days_since_start", "days_since_start_squared"]]
    )

    # Seasonality
    seasonality = df.assign(
        seasonality_cos=np.cos(df.year_radians),
        seasonality_sin=np.sin(df.year_radians)).loc[
        :, ["seasonality_cos", "seasonality_sin"]
    ]

    # Create design matrix
    df_list = [
        df.loc[:, ["intercept"]],
        day_of_week,
        trends,
        seasonality,
    ]

    x_df = pd.concat(df_list, axis=1)

    return df.day, x_df


def get_coefs():
    # Set beta
    beta_intercept = [5.]
    beta_day_of_week = [-0.3, 0.03, 0.06, 0.1, 0.09, -0.04, -0.23]
    beta_trends = [0.4, -0.17]
    beta_seasonality = [0.2, -0.1]
    beta = np.array(
        beta_intercept +
            beta_day_of_week +
            beta_trends +
            beta_seasonality)

    # Set alpha
    alpha_intercept = [-2.]
    alpha_day_of_week = [-0.3, 0.03, 0.06, 0.1, 0.09, -0.04, -0.23]
    alpha_trends = [-0.2, -0.03]
    alpha_seasonality = [0.16, -0.05]
    alpha = np.array(
        alpha_intercept +
            alpha_day_of_week +
            alpha_trends +
            alpha_seasonality)

    return beta, alpha


def get_derived_params(x_df, beta, alpha):
    x = x_df.values
    mu = x @ beta
    exp_mu = np.exp(mu)
    sigma = np.exp(x @ alpha)
    return mu, sigma


def sample_data(mu, sigma):
    z = norm.rvs(loc=mu, scale=sigma)
    y = np.floor(np.exp(z))
    return z, y