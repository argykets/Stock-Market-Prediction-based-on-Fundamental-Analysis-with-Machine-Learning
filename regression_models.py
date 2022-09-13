import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, activations
from tensorflow.keras.optimizers import Nadam, Adam

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import xgboost


def NN(learning_rate, loss):
    # Basic neural network for log returns forecasting
    model = models.Sequential()
    model.add(layers.Dense(64, input_dim=34, activation='relu', kernel_initializer='he_normal'))
    model.add(layers.Dense(64, activation='relu', kernel_initializer='he_normal'))
    model.add(layers.Dense(1, activation='linear'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics='mse')

    return model


# Random forest regressor
def RFRegressor(X_train, y_train, X_test, y_test):
    # Define dummy regressor model
    regressor = RandomForestRegressor(random_state=42)

    # return optimal hyperparameters
    best_parameters = {'min_samples_split': 2, 'n_estimators': 1000}

    # Declare new regressor with optimal hyperparameters
    regressor = RandomForestRegressor(random_state=42, n_estimators=best_parameters['n_estimators'],
                                      min_samples_split=best_parameters['min_samples_split'])

    # Training
    regressor.fit(X_train, y_train)

    # Prediction
    y_pred = regressor.predict(X_test)

    # MAE
    mae = mean_absolute_error(y_test, y_pred)

    # RMSE
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    return y_pred, mae, rmse


# Support vector machine regressor
def SVRegressor(X_train, y_train, X_test, y_test):
    # Define dummy regressor model
    regressor = SVR()

    # return optimal hyperparameters
    best_parameters = {'C': 1, 'kernel': 'rbf'}

    # Declare new regressor with optimal hyperparameters
    regressor = SVR(kernel=best_parameters['kernel'], C=best_parameters['C'])

    # Training
    regressor.fit(X_train, y_train)

    # Prediction
    y_pred = regressor.predict(X_test)

    # MAE
    mae = mean_absolute_error(y_test, y_pred)

    # RMSE
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    return y_pred, mae, rmse


def xgboostRegressor(X_train, y_train, X_test, y_test):
    # Define dummy regressor model
    regressor = xgboost.XGBRegressor(random_state=42)

    # return optimal hyperparameters
    best_parameters = {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}

    # Declare new regressor with optimal hyperparameters
    regressor = xgboost.XGBRegressor(random_state=42, n_estimators=best_parameters['n_estimators'],
                                     max_depth=best_parameters['max_depth'],
                                     learning_rate=best_parameters['learning_rate'])

    # Training
    regressor.fit(X_train, y_train)

    # Feature importance
    importances = regressor.feature_importances_

    # Prediction
    y_pred = regressor.predict(X_test)

    # MAE
    mae = mean_absolute_error(y_test, y_pred)

    # MSE
    mse = mean_squared_error(y_test, y_pred)

    return y_pred, mae, mse, regressor, importances
