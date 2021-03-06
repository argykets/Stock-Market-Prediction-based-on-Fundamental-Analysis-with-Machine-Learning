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

import xgboost

from scipy.stats import pearsonr, spearmanr
from matplotlib import pyplot as plt
import yfinance as yf

import portfolio_optimization_module

if __name__ == '__main__':
    # Specify tickers
    tickers = open('tickers.txt', 'r')
    tickers = tickers.read().split(",")

    # feature engineering phase
    import featureEngineering_module
    data = featureEngineering_module.featureEngineering(tickers)
    print(data.shape)

    # preprocessing phase
    import preprocessing_module
    X_train, y_train, X_test, y_test, y_train_class, y_test_class, columns,\
    test_data_with_dates, X_val, y_val, y_val_class, validation_data_with_dates = preprocessing_module.preprocessing(data)

    # run xgboost regressor model
    import regression_models
    reg_pred, mae, mse, r2, xgboostRegressor, importances = regression_models.xgboostRegressor(X_train, y_train, X_test, y_test)

    # sort importances
    importances, columns = zip(*sorted(zip(importances, columns)))

    # plot features importances
    plt.plot(columns, importances)
    plt.xticks(rotation=90)
    plt.tight_layout()

    # report regression metrics
    print('Regression metrics')
    print('MAE: ', mae)
    print('MSE: ', mse)
    print('R2: ', r2)

    # Construct new dataset based on xgboost regression results
    X_train, X_test, X_val = featureEngineering_module.hybrid_dataset_construction(xgboostRegressor, X_train, X_test, X_val)
    columns = columns + ('xgboost_predictions',)

    # keep fundamentals and xgboost predictions
    X_train = X_train[:, np.r_[0:13, -1]]
    X_val = X_val[:, np.r_[0:13, -1]]
    del columns[13:-1]

    X_test = X_test[:, np.r_[0:13, -1]]

    '''
    # perform grid search on validation set
    # define the grid search parameters
    grid = preprocessing_module.grid_construction()
    hyperparameter_list = []
    mse_list = []
    mae_list = []
    r2_list = []
    huber_list = []
    cosine_similarity_list = []
    msle_list = []
    portfolio_value_list = []
    weights_list = []
    port_volatility = []
    y_hat_val_list = []
    for batch_size in grid['batch_size']:
        for epoch in grid['epochs']:
            for loss in grid['loss']:
                model = regression_models.NN(learning_rate=0.01, loss=loss)
                model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch)
                y_hat_val = model.predict(X_val)
                y_hat_val_list.append(y_hat_val)
                mse, mae, r2, huber, cs, msle = preprocessing_module.evaluation(y_val, y_hat_val)
                hyperparameter_list.append(f'batch_size:{batch_size}, epochs:{epoch}, loss:{loss}')
                mse_list.append(mse)
                mae_list.append(mae)
                r2_list.append(r2)
                huber_list.append(huber)
                cosine_similarity_list.append(cs)
                msle_list.append(msle)

                # add date and ticker to predictions
                backtesting_data = validation_data_with_dates[['date', 'ticker']]
                backtesting_data.reset_index(inplace=True, drop=True)
                backtesting_data['expected_returns'] = y_hat_val

                # portfolio optimization
                keep_top_k_stocks = 10
                optimal_weights, unique_tickers = portfolio_optimization_module.portfolio_optimization(backtesting_data,
                                                                                                       keep_top_k_stocks)
                weights_list.append(optimal_weights)
                # calculate portfolio performance
                portfolio_ret, cum_ret, sharpe_ratio, volatility = preprocessing_module.financialEvaluation(
                    validation_data_with_dates, y_hat_val)

                portfolio_value_list.append(cum_ret.iloc[-1].values)


    
    optimization_results = pd.DataFrame([hyperparameter_list,
                                         mse_list, mae_list, r2_list, huber_list, cosine_similarity_list, msle_list, portfolio_value_list],
                                        index=['hyperparameters', 'mse', 'mae', 'r2', 'huber' , 'cosine similarity', 'msle', 'cum return'])
    optimization_results = optimization_results.T

    optimization_results.index = optimization_results['hyperparameters']
    optimization_results.drop(columns='hyperparameters', inplace=True)

    optimization_results = optimization_results.astype(float).round(4)

    optimization_results.to_csv('optimization_results.csv')
    '''
    optimization_results = pd.read_csv('optimization_results.csv', index_col=0)

    # select model that minimizes mse
    # AFTER EXPERIMENTATION: batch_size:80, epochs:10, loss:mse
    print(optimization_results['mse'].idxmin())

    # select model that minimizes mae
    # AFTER EXPERIMENTATION: batch_size:80, epochs:10, loss:mse
    print(optimization_results['mae'].idxmin())

    # select model tha maximizes R-squared
    # AFTER EXPERIMENTATION: batch_size:80, epochs:10, loss:mse
    print(optimization_results['r2'].idxmax())

    # select model tha minimizes huber
    # AFTER EXPERIMENTATION: batch_size:80, epochs:10, loss:mse
    print(optimization_results['huber'].idxmin())

    # select model that maximizes portfolio profitability
    # AFTER EXPERIMENTATION: batch_size:100, epochs:20, learning_rate:mae
    print(optimization_results['cum return'].idxmax())

    # store predictions on a list
    predictions = []

    # train and test model #1 (min mse)
    print('MIN MSE')
    # reshape y_val
    y_val = y_val.reshape(y_val.shape[0], 1)
    model = regression_models.NN(learning_rate=0.01, loss='mse')
    history = model.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0),
                        batch_size=80, epochs=10, validation_data=(X_val, y_val), verbose=False)

    fig, ax = plt.subplots(5, sharex=True)
    fig.set_figheight(10)
    fig.subplots_adjust(hspace=1)
    fig.set_figwidth(6)
    #fig.tight_layout()
    for a in ax.flat:
        a.set(xlabel='Epochs', ylabel='Loss')


    ax[0].plot(history.history['loss'], color='b', label='train loss')
    ax[0].plot(history.history['val_loss'], color='r', label='val_loss')
    ax[0].legend()
    ax[0].set_title('Training loss for MIN MSE model')
    # make predictions on test set and evaluate
    y_pred = model.predict(X_test)
    predictions.append(y_pred)
    mse, mae, r2, huber, cs, msle = preprocessing_module.evaluation(y_test, y_pred)
    print('MSE =', np.round(mse, 5))
    print('MAE =', np.round(mae, 5))
    print('R2 =', np.round(r2, 5))
    print('Huber =', np.round(huber, 5))
    print('Cosine Similarity =', np.round(cs, 5))
    print('MSLE =', np.round(msle, 5))
    print('Pearson Correlation =', np.round(pearsonr(y_pred[:, 0], y_test[:, 0]), 5))

    # financial evaluation
    portfolio_ret, cum_ret, sharpe_ratio, volatility = preprocessing_module.financialEvaluation(test_data_with_dates, y_pred)
    print('Cumulative return =', cum_ret.iloc[-1].values)
    print('Sharpe ratio', sharpe_ratio.values)
    print('Volatility', volatility.values)
    print('Detailed portfolio returns', portfolio_ret)

    # train and test model #2 (min mae)
    print('MIN MAE')
    # reshape y_val
    #y_val = y_val.reshape(y_val.shape[0], 1)
    model = regression_models.NN(learning_rate=0.01, loss='mse')
    history = model.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0),
                        batch_size=80, epochs=10, validation_data=(X_val, y_val), verbose=False)

    ax[1].plot(history.history['loss'], color='b', label='train loss')
    ax[1].plot(history.history['val_loss'], color='r', label='val_loss')
    ax[1].set_title('Training loss for MIN MAE model')

    # make predictions on test set and evaluate
    y_pred = model.predict(X_test)
    predictions.append(y_pred)
    mse, mae, r2, huber, cs, msle = preprocessing_module.evaluation(y_test, y_pred)
    print('MSE =', np.round(mse, 5))
    print('MAE =', np.round(mae, 5))
    print('R2 =', np.round(r2, 5))
    print('Huber =', np.round(huber, 5))
    print('Cosine Similarity =', np.round(cs, 5))
    print('MSLE =', np.round(msle, 5))
    print('Pearson Correlation =', np.round(pearsonr(y_pred[:, 0], y_test[:, 0]), 5))

    # financial evaluation
    portfolio_ret, cum_ret, sharpe_ratio, volatility = preprocessing_module.financialEvaluation(test_data_with_dates, y_pred)
    print('Cumulative return =', cum_ret.iloc[-1].values)
    print('Sharpe ratio', sharpe_ratio.values)
    print('Volatility', volatility.values)
    print('Detailed portfolio returns', portfolio_ret)

    # train and test model #3 (max R-squared)
    print('MAX R-SQUARED')
    # reshape y_val
    model = regression_models.NN(learning_rate=0.01, loss='mse')
    history = model.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0),
                        batch_size=80, epochs=10, validation_data=(X_val, y_val), verbose=False)

    ax[2].plot(history.history['loss'], color='b', label='train loss')
    ax[2].plot(history.history['val_loss'], color='r', label='val_loss')
    ax[2].set_title('Training loss for MAX R-Squared model')

    # make predictions on test set and evaluate
    y_pred = model.predict(X_test)
    predictions.append(y_pred)
    predictions.append(y_pred)
    mse, mae, r2, huber, cs, msle = preprocessing_module.evaluation(y_test, y_pred)
    print('MSE =', np.round(mse, 5))
    print('MAE =', np.round(mae, 5))
    print('R2 =', np.round(r2, 5))
    print('Huber =', np.round(huber, 5))
    print('Cosine Similarity =', np.round(cs, 5))
    print('MSLE =', np.round(msle, 5))
    print('Pearson Correlation =', np.round(pearsonr(y_pred[:, 0], y_test[:, 0]), 5))

    # financial evaluation
    portfolio_ret, cum_ret, sharpe_ratio, volatility = preprocessing_module.financialEvaluation(test_data_with_dates,
                                                                                                y_pred)
    print('Cumulative return =', cum_ret.iloc[-1].values)
    print('Sharpe ratio', sharpe_ratio.values)
    print('Volatility', volatility.values)
    print('Detailed portfolio returns', portfolio_ret)

    # train and test model #4 (min Huber)
    print('MIN HUBER')
    # reshape y_val
    model = regression_models.NN(learning_rate=0.01, loss='mse')
    history = model.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0),
                        batch_size=80, epochs=10, validation_data=(X_val, y_val), verbose=False)

    ax[3].plot(history.history['loss'], color='b', label='train loss')
    ax[3].plot(history.history['val_loss'], color='r', label='val_loss')
    ax[3].set_title('Training loss for MIN Huber model')

    # make predictions on test set and evaluate
    y_pred = model.predict(X_test)
    predictions.append(y_pred)
    predictions.append(y_pred)
    mse, mae, r2, huber, cs, msle = preprocessing_module.evaluation(y_test, y_pred)
    print('MSE =', np.round(mse, 5))
    print('MAE =', np.round(mae, 5))
    print('R2 =', np.round(r2, 5))
    print('Huber =', np.round(huber, 5))
    print('Cosine Similarity =', np.round(cs, 5))
    print('MSLE =', np.round(msle, 5))
    print('Pearson Correlation =', np.round(pearsonr(y_pred[:, 0], y_test[:, 0]), 5))

    # financial evaluation
    portfolio_ret, cum_ret, sharpe_ratio, volatility = preprocessing_module.financialEvaluation(test_data_with_dates,
                                                                                                y_pred)
    print('Cumulative return =', cum_ret.iloc[-1].values)
    print('Sharpe ratio', sharpe_ratio.values)
    print('Volatility', volatility.values)
    print('Detailed portfolio returns', portfolio_ret)
    '''
    # train and test model #5 (max Cosine Similarity)
    print('MAX COSINE SIMILARITY')
    # reshape y_val
    model = regression_models.NN(learning_rate=0.01, loss=CosineSimilarity())
    history = model.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0),
                        batch_size=60, epochs=30, validation_data=(X_val, y_val), verbose=False)

    plt.figure(5)
    plt.plot(history.history['loss'], color='b', label='train loss')
    plt.plot(history.history['val_loss'], color='r', label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('Huber loss')
    plt.legend()
    plt.title('Training loss for MAX Cosine Similarity model')

    # make predictions on test set and evaluate
    y_pred = model.predict(X_test)
    predictions.append(y_pred)
    predictions.append(y_pred)
    mse, mae, r2, huber, cs, msle = preprocessing_module.evaluation(y_test, y_pred)
    print('MSE =', np.round(mse, 5))
    print('MAE =', np.round(mae, 5))
    print('R2 =', np.round(r2, 5))
    print('Huber =', np.round(huber, 5))
    print('Cosine Similarity =', np.round(cs, 5))
    print('MSLE =', np.round(msle, 5))

    # financial evaluation
    portfolio_ret, cum_ret, sharpe_ratio, volatility = preprocessing_module.financialEvaluation(test_data_with_dates,
                                                                                                y_pred)
    print('Cumulative return =', cum_ret.iloc[-1].values)
    print('Sharpe ratio', sharpe_ratio.values)
    print('Volatility', volatility.values)
    print('Detailed portfolio returns', portfolio_ret)
    '''

    '''
    # train and test model #6 (min Mean Squared Logarithmic Error)
    print('MIN SQUARED LOGARITHMIC ERROR')
    # reshape y_val
    model = regression_models.NN(learning_rate=0.01, loss=CosineSimilarity())
    history = model.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0),
                        batch_size=60, epochs=10, validation_data=(X_val, y_val), verbose=False)

    plt.figure(6)
    plt.plot(history.history['loss'], color='b', label='train loss')
    plt.plot(history.history['val_loss'], color='r', label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('Huber loss')
    plt.legend()
    plt.title('Training loss for MIN MSLE model')

    # make predictions on test set and evaluate
    y_pred = model.predict(X_test)
    predictions.append(y_pred)
    predictions.append(y_pred)
    mse, mae, r2, huber, cs, msle = preprocessing_module.evaluation(y_test, y_pred)
    print('MSE =', np.round(mse, 5))
    print('MAE =', np.round(mae, 5))
    print('R2 =', np.round(r2, 5))
    print('Huber =', np.round(huber, 5))
    print('Cosine Similarity =', np.round(cs, 5))
    print('MSLE =', np.round(msle, 5))

    # financial evaluation
    portfolio_ret, cum_ret, sharpe_ratio, volatility = preprocessing_module.financialEvaluation(test_data_with_dates,
                                                                                                y_pred)
    print('Cumulative return =', cum_ret.iloc[-1].values)
    print('Sharpe ratio', sharpe_ratio.values)
    print('Volatility', volatility.values)
    print('Detailed portfolio returns', portfolio_ret)
    '''
    # train and test model #7 (max portfolio return)
    print('MAX PORTFOLIO RETURN')
    # reshape y_val
    model = regression_models.NN(learning_rate=0.01, loss='mae')
    history = model.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0),
                        batch_size=100, epochs=10, validation_data=(X_val, y_val), verbose=False)

    ax[4].plot(history.history['loss'], color='b', label='train loss')
    ax[4].plot(history.history['val_loss'], color='r', label='val_loss')
    ax[4].set_title('Training loss for MAX PROFIT model')

    # make predictions on test set and evaluate
    y_pred = model.predict(X_test)
    predictions.append(y_pred)
    mse, mae, r2, huber, cs, msle = preprocessing_module.evaluation(y_test, y_pred)
    print('MSE =', np.round(mse, 5))
    print('MAE =', np.round(mae, 5))
    print('R2 =', np.round(r2, 5))
    print('Huber =', np.round(huber, 5))
    print('Cosine Similarity =', np.round(cs, 5))
    print('MSLE =', np.round(msle, 5))
    print('Pearson Correlation =', np.round(pearsonr(y_pred[:, 0], y_test[:, 0]), 5))

    # financial evaluation
    portfolio_ret, cum_ret, sharpe_ratio, volatility = preprocessing_module.financialEvaluation(test_data_with_dates, y_pred)
    print('Cumulative return =', cum_ret.iloc[-1].values)
    print('Sharpe ratio', sharpe_ratio.values)
    print('Volatility', volatility.values)
    print('Detailed portfolio returns', portfolio_ret)

