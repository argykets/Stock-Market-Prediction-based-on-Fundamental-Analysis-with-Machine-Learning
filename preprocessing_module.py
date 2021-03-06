import numpy as np
import sklearn
import tensorflow
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import pandas as pd

from tensorflow.keras.losses import Huber, CosineSimilarity, MeanSquaredLogarithmicError


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import portfolio_optimization_module

def preprocessing(data):
    # Fill NaN with zeros and reset index
    data.fillna(0, inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Polynomial features for close price
    poly = PolynomialFeatures(4)
    numerical_features = data['Close'].values
    numerical_features = numerical_features.reshape(-1, 1)
    polynomial_features = pd.DataFrame(poly.fit_transform(numerical_features))
    polynomial_features = polynomial_features.rename(
        columns={0: "poly1", 1: "poly2", 2: "poly3", 3: "poly4", 4: "poly5"})

    # Drop poly1 and poly2 features
    polynomial_features.drop(columns=['poly1', 'poly2'], inplace=True)

    data = pd.concat([data, polynomial_features], axis=1, ignore_index=False)

    # Drop duplicate rows
    data.drop_duplicates(inplace=True)
    data = data[data['date'] != 0]

    # Split to training and test set
    training_data_with_dates = data.loc[data['date'] <= '2019-05-01']
    validation_data_with_dates = data.loc[(data['date'] > '2019-10-01') & (data['date'] <= '2020-05-01')]
    test_data_with_dates = data.loc[data['date'] > '2020-05-01']

    # Remove nan values
    train_data_with_dates = training_data_with_dates.fillna(0)
    validation_data_with_dates = validation_data_with_dates.fillna(0)
    test_data_with_dates = test_data_with_dates.fillna(0)

    # Remove None values and replace them with zeros
    train_data_with_dates = train_data_with_dates.replace(to_replace='None', value=0)
    validation_data_with_dates = validation_data_with_dates.replace(to_replace='None', value=0)
    test_data_with_dates = test_data_with_dates.replace(to_replace='None', value=0)

    # Drop irrelevant columns: {Date, Ticker}
    train_data = train_data_with_dates.drop(columns=['ticker', 'date'])
    val_data = validation_data_with_dates.drop(columns=['ticker', 'date'])
    test_data = test_data_with_dates.drop(columns=['ticker', 'date'])

    # Select target value --- simple returns
    y_train = train_data['log returns'].values
    y_train_class = train_data['log returns trend'].values
    X_train = train_data.drop(columns=['simple returns', 'log returns', 'log returns trend'])

    # Get feature names
    columns = X_train.columns.values.tolist()
    X_train = X_train.values

    y_test = test_data['log returns'].values
    y_test_class = test_data['log returns trend'].values
    X_test = test_data.drop(columns=['simple returns', 'log returns', 'log returns trend']).values

    y_val = val_data['log returns'].values
    y_val_class = val_data['log returns trend'].values
    X_val = val_data.drop(columns=['simple returns', 'log returns', 'log returns trend']).values

    y_train = np.asarray(y_train).astype('float32').reshape((-1, 1))
    y_test = np.asarray(y_test).astype('float32').reshape((-1, 1))
    y_train_class = np.asarray(y_train_class).astype('float32').reshape((-1, 1))
    y_test_class = np.asarray(y_test_class).astype('float32').reshape((-1, 1))

    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, y_train_class, y_test_class, columns, test_data_with_dates, X_val, y_val,\
           y_val_class, validation_data_with_dates


def grid_construction():
    grid = {'batch_size': [60, 80, 100],
            'epochs': [10, 20, 30],
            'loss': ['mse', 'mae', Huber(), CosineSimilarity(), MeanSquaredLogarithmicError()]}

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
    cs = CosineSimilarity()
    cosine_similarity = cs(y_val, y_hat_val).numpy()
    msle = MeanSquaredLogarithmicError()
    logarithmic_error = msle(y_val, y_hat_val)

    return mse, mae, r2, huber, cosine_similarity, logarithmic_error


def financialEvaluation(data_with_dates, y_pred):
    backtesting_data = data_with_dates[['date', 'ticker']]
    backtesting_data.reset_index(inplace=True, drop=True)
    backtesting_data['expected_returns'] = y_pred

    # portfolio optimization
    print(backtesting_data.shape[0])
    keep_top_k_stocks = 50  # 20 or 50 or 421
    optimal_weights, unique_tickers = portfolio_optimization_module.portfolio_optimization(backtesting_data,
                                                                                           keep_top_k_stocks)
    # calculate portfolio performance
    return portfolio_optimization_module.calc_portfolio_performance(optimal_weights, unique_tickers)
