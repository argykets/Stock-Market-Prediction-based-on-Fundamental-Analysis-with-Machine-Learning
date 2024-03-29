import numpy as np
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import scipy.sparse as sp
import scipy

import cvxpy as cp

import os
import glob

from pypfopt.risk_models import semicovariance
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt import risk_models
from pypfopt.risk_models import risk_matrix
from pypfopt import objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation


def portfolio_optimization(backtesting_data, keep_top_k_stocks, validation, weights):
    unique_tickers = backtesting_data['ticker'].unique().tolist()
    unique_dates = np.sort(backtesting_data['date'].unique())
    tickers_to_delete = []
    for ticker in unique_tickers:
        inside = backtesting_data.loc[backtesting_data['ticker'] == ticker]
        if len(inside['date']) != (2 if validation else 6):    # !!! change to 6 for test set and 2 for validation set
            tickers_to_delete.append(ticker)
    for ticker in tickers_to_delete:
        backtesting_data.drop(backtesting_data[backtesting_data['ticker'] == ticker].index, inplace=True)

    unique_tickers = backtesting_data['ticker'].unique().tolist()
    for ticker in unique_tickers:
        backtesting_data['date'].loc[backtesting_data['ticker'] == ticker] = (['2019-12-31', '2020-03-31'] if validation
                                                                              else ['2020-06-30', '2020-09-30', '2020-12-31', '2021-03-31', '2021-06-30', '2021-09-30'])
            #['2020-06-30', '2020-09-30', '2020-12-31', '2021-03-31', '2021-06-30', '2021-09-30']
    # ['2019-12-31', '2020-03-31']
    expected_returns = pd.DataFrame()
    for ticker in unique_tickers:
        inside = backtesting_data['expected_returns'].loc[backtesting_data['ticker'] == ticker].values
        expected_returns[ticker] = inside

    #expected_returns.index = ['2020-06-30', '2020-09-30', '2020-12-31', '2021-03-31', '2021-06-30', '2021-09-30']

    expected_returns = expected_returns.iloc[:, :keep_top_k_stocks]
    unique_tickers = expected_returns.columns.tolist()

    port_returns = []
    port_volatility = []
    port_weights = []

    individual_rets = expected_returns.mean()

    num_portfolios = weights.shape[0]
    for port in range(num_portfolios):
        port_weights.append(weights[port])
        returns = np.dot(weights[port], individual_rets)
        port_returns.append(returns)

        var_matrix = expected_returns.cov()
        var = var_matrix.mul(weights[port], axis=0).mul(weights[port], axis=1).sum().sum()
        sd = np.sqrt(var)
        port_volatility.append(sd)

    index_of_min_volatility = np.argmin(port_volatility)

    return port_weights[index_of_min_volatility], unique_tickers


def calc_covariance(unique_tickers):
    df = pd.DataFrame()
    for ticker in unique_tickers:
        inside_df = pd.read_csv(f'Price data/{ticker}.csv')
        inside_df.drop(columns='Date', inplace=True)
        preprocessed_df = preprocessing_historical_prices(inside_df)
        df[ticker] = preprocessed_df
    return risk_models.risk_matrix(df, returns_data=True)


def preprocessing_historical_prices(inside_df):
    inside_df = inside_df.pct_change()
    inside_df.dropna(inplace=True)
    return inside_df


def calc_portfolio_performance(optimal_weights, unique_tickers, validation):
    optimal_weights = pd.DataFrame(optimal_weights, index=unique_tickers)
    df = pd.DataFrame()
    for ticker in unique_tickers:
        inside_df = pd.read_csv(f'Price data/{ticker}.csv')
        # filter dates
        #inside_df = inside_df.loc[inside_df['Date'].isin(['2020-06-30', '2020-09-30', '2020-12-31', '2021-03-31', '2021-06-30', '2021-09-30'])]
        # VALIDATION FILTERING
        if validation:
            inside_df = inside_df.loc[inside_df['Date'].isin(['2019-12-31', '2020-03-31'])]
        else:
            inside_df = inside_df.loc[inside_df['Date'].isin(
                ['2020-06-30', '2020-09-30', '2020-12-31', '2021-03-31', '2021-06-30', '2021-09-30'])]
        inside_df.drop(columns='Date', inplace=True)
        df[ticker] = inside_df

    # find columns that contain nan values
    tickers_to_delete = df.columns[df.isna().any()].tolist()

    df.dropna(inplace=True, axis=1)
    df.reset_index(drop=True, inplace=True)

    # delete weights according to tickers to delete
    optimal_weights = optimal_weights.drop(index=tickers_to_delete)
    optimal_weights = optimal_weights.values
    optimal_weights = optimal_weights.reshape(len(optimal_weights))
    #total_price_returns = (df.iloc[-1, :] - df.iloc[0, :]) / df.iloc[0, :]
    total_price_returns = df.pct_change()
    total_price_returns.dropna(inplace=True, axis=0)
    # compute weighted returns
    portfolio_value = 10000
    portfolio_returns = np.zeros(total_price_returns.shape[0])
    for i in range(portfolio_returns.shape[0]):
        portfolio_returns[i] = sum(optimal_weights * total_price_returns.iloc[i, :])

    if validation:
        portfolio_returns = pd.DataFrame(portfolio_returns, index=['2020-03-31'])
    else:
        portfolio_returns = pd.DataFrame(portfolio_returns, index=['2020-09-30', '2020-12-31', '2021-03-31', '2021-06-30', '2021-09-30'])
    # index=['2020-09-30', '2020-12-31', '2021-03-31', '2021-06-30', '2021-09-30']
    # ['2020-03-31']
    print(portfolio_returns)
    cumulative_return = (portfolio_returns + 1).cumprod() - 1
    cumulative_return.dropna(inplace=True)
    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std()
    volatility = portfolio_returns.std()
    return portfolio_returns, cumulative_return, sharpe_ratio, volatility


