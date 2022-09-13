# Import libraries
import json
import pandas as pd
import math
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, activations
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.losses import Huber, CosineSimilarity, MeanSquaredLogarithmicError

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures


from matplotlib import pyplot as plt
import yfinance as yf

import gridSearchOptimization
import portfolio_optimization_module
import featureEngineering_module
import preprocessing_module
import regression_models

if __name__ == '__main__':
    # Specify tickers
    tickers = open('tickers.txt', 'r')
    tickers = tickers.read().split(",")

    # feature engineering phase
    data = featureEngineering_module.featureEngineering(tickers, use_macros=True)
    print(data.shape)

    # transform to percentage representation
    columns_to_transform_to_percentage = data.columns.tolist()
    unwanted = {'Close', 'date', 'ticker', 'log returns'}
    columns_for_pct_change = [ele for ele in columns_to_transform_to_percentage if ele not in unwanted]
    transform_to_percentage = True
    if transform_to_percentage:
        for column in columns_for_pct_change:
            data[column] = data[column].pct_change()

    # preprocessing phase
    X_train, y_train, X_test, y_test, columns,\
    test_data_with_dates, X_val, y_val, validation_data_with_dates = preprocessing_module.preprocessing(data)

    # run xgboost regressor model
    reg_pred, mae, mse, xgboostRegressor, importance = regression_models.xgboostRegressor(X_train, y_train, X_test, y_test)

    # create importances dataframe
    importances = pd.DataFrame(importance, columns=['importances'])
    importances['feature'] = columns
    importances['type of feature'] = ['fundamental', 'fundamental', 'fundamental', 'fundamental', 'fundamental',
                                      'fundamental', 'fundamental', 'fundamental', 'fundamental', 'fundamental',
                                      'fundamental', 'fundamental', 'fundamental', 'fundamental', 'fundamental',
                                      'fundamental', 'fundamental', 'fundamental', 'fundamental', 'technical',
                                      'macro', 'macro', 'macro', 'macro', 'macro', 'macro', 'macro', 'macro', 'macro',
                                      'macro', 'macro', 'fundamental', 'fundamental', 'fundamental']

    # sort importances in ascending order
    importances.sort_values(by='importances', inplace=True)

    # plot feature importances
    for i in range(len(importances)):
        if importances['type of feature'].iloc[i] == 'macro':
            plt.bar(importances['feature'].iloc[i], importances['importances'].iloc[i], color='b')
        elif importances['type of feature'].iloc[i] == 'technical':
            plt.bar(importances['feature'].iloc[i], importances['importances'].iloc[i], color='r')
        elif importances['type of feature'].iloc[i] == 'fundamental':
            plt.bar(importances['feature'].iloc[i], importances['importances'].iloc[i], color='g')
        plt.xticks(rotation=90)
        plt.tight_layout()

    # report XGBoost regressor metrics
    print('XGBoost regression metrics')
    print('MAE: ', mae)
    print('MSE: ', mse)

    # perform grid search on validation set
    #optimization_results = gridSearchOptimization.gridSearch(X_train, y_train, X_val, y_val, validation_data_with_dates)

    optimization_results = pd.read_csv('optimization_results.csv', index_col=0)

    # store predictions on a list
    predictions = []

    # store results in lists
    cum_list = []
    sharpe_list = []
    volatility_list = []
    returns_list = []

    # train and test model #1 (min mse)
    # reshape y_val
    y_val = y_val.reshape(y_val.shape[0], 1)
    model = regression_models.NN(learning_rate=0.01, loss=Huber())
    history = model.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0),
                        batch_size=32, epochs=100, validation_data=(X_val, y_val), verbose=True)

    # plot training and validation losses
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Huber Loss')
    plt.title('Training and validation loss for model #3')

    # make predictions on test set and evaluate
    y_pred = model.predict(X_test)
    predictions.append(y_pred)
    mse, mae, r2, huber, cs, msle = preprocessing_module.evaluation(y_test, y_pred)
    print('MSE =', np.round(mse, 5))
    print('MAE =', np.round(mae, 5))
    print('Huber =', np.round(huber, 5))

    # financial evaluation
    portfolio_ret, cum_ret, sharpe_ratio, volatility = preprocessing_module.financialEvaluation(test_data_with_dates, y_pred)
    print('Cumulative return =', cum_ret.iloc[-1].values)
    print('Sharpe ratio', sharpe_ratio.values)
    print('Volatility', volatility.values)
    print('Detailed portfolio returns', portfolio_ret)

    # store results in list
    cum_list.append(cum_ret.iloc[-1].values)
    sharpe_list.append(sharpe_ratio.values)
    volatility_list.append(volatility.values)
    returns_list.append(cum_ret)

    # define starting fund
    portfolio_value = 10000
    returns = returns_list[0]

    # load s&p 500 index data
    gspc = pd.read_csv('Price data/gspc.csv', index_col=0)
    gspc = gspc.pct_change()
    gspc.dropna(inplace=True)
    gspc = (gspc + 1).cumprod() - 1

    # plot P&L plot for best model and S&P 500 index
    plt.plot(np.array(gspc.index), gspc.values, color='r', label='S&P 500')
    plt.plot(np.array(returns.index), returns.values, color='b', label='Our Portfolio')
    plt.legend()
    plt.title('Profit and Loss of our Portfolio and S&P 500')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.show()





