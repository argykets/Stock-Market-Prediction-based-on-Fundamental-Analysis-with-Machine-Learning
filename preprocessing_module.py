import numpy as np
import sklearn
import tensorflow
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import pandas as pd

from tensorflow.keras.losses import Huber, CosineSimilarity, MeanSquaredLogarithmicError


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import portfolio_optimization_module


def preprocessing(data):
    # Split to training and test set
    training_data_with_dates = data.loc[data['date'] <= '2019-05-01']
    validation_data_with_dates = data.loc[(data['date'] > '2019-10-01') & (data['date'] <= '2020-05-01')]
    test_data_with_dates = data.loc[data['date'] > '2020-05-01']

    # Drop irrelevant columns: {Date, Ticker}
    train_data = training_data_with_dates.drop(columns=['ticker', 'date'])
    val_data = validation_data_with_dates.drop(columns=['ticker', 'date'])
    test_data = test_data_with_dates.drop(columns=['ticker', 'date'])

    # Select target value --- simple returns
    y_train = train_data['log_returns'].values
    X_train = train_data.drop(columns=['log_returns', 'close', 'returns'])

    # Get feature names
    columns = X_train.columns.values.tolist()
    X_train = X_train.values

    y_test = test_data['log_returns'].values
    X_test = test_data.drop(columns=['log_returns', 'close', 'returns']).values

    y_val = val_data['log_returns'].values
    X_val = val_data.drop(columns=['log_returns', 'close', 'returns']).values

    y_train = np.asarray(y_train).astype('float32').reshape((-1, 1))
    y_test = np.asarray(y_test).astype('float32').reshape((-1, 1))


    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, columns, test_data_with_dates, X_val, y_val, validation_data_with_dates


def grid_construction():
    grid = {'batch_size': [32, 64, 128],
            'loss': ['mse', 'mae', 'huber'],
            'epochs': [20, 40, 60, 80, 100],
            'learning_rate': [0.001, 0.01, 0.1]}

    return grid


def evaluation(y_val, y_hat_val):
    mse = mean_squared_error(y_val, y_hat_val)
    mae = mean_absolute_error(y_val, y_hat_val)
    r2 = r2_score(y_val, y_hat_val)
    # convert to float32
    y_val = y_val.astype('float32')
    y_hat_val = y_hat_val.astype('float32')
    h = Huber()
    huber = h(y_val, y_hat_val).numpy()

    return mse, mae, huber, r2


def financialEvaluation(data_with_dates, y_pred, validation):
    backtesting_data = data_with_dates[['date', 'ticker']]
    backtesting_data.reset_index(inplace=True, drop=True)
    backtesting_data['expected_returns'] = y_pred

    # portfolio optimization
    print(backtesting_data.shape[0])
    keep_top_k_stocks = 50  # 20 or 50 or 421
    if validation:
        num_portfolios = 10000
        np.random.seed(42)
        weights = np.random.rand(num_portfolios, keep_top_k_stocks)
        r = weights.sum(axis=1)
        weights_matrix = np.zeros((num_portfolios, keep_top_k_stocks))
        for i in range(num_portfolios):
            weights_matrix[i, :] = weights[i, :] / r[i]
    else:
        num_portfolios = 100000
        np.random.seed(42)
        weights = np.random.rand(num_portfolios, keep_top_k_stocks)
        r = weights.sum(axis=1)
        weights_matrix = np.zeros((num_portfolios, keep_top_k_stocks))
        for i in range(num_portfolios):
            weights_matrix[i, :] = weights[i, :] / r[i]


    optimal_weights, unique_tickers = portfolio_optimization_module.portfolio_optimization(backtesting_data,
                                                                                           keep_top_k_stocks, validation, weights_matrix)

    # calculate portfolio performance
    portfolio_ret, cum_ret, sharpe_ratio, volatility = portfolio_optimization_module.calc_portfolio_performance(optimal_weights, unique_tickers, validation)
    return portfolio_ret, cum_ret, sharpe_ratio, volatility, unique_tickers
