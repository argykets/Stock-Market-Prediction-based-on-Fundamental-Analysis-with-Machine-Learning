# Import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt

import feature_engineering
import preprocessing_module
import regression_models
import gridSearchOptimization
import portfolio_optimization_module
import preprocessing.featureImportance as fi

if __name__ == '__main__':
    transform_to_percentage = True
    # Specify tickers
    tickers = open('tickers.txt', 'r')
    tickers = tickers.read().split(",")

    # feature engineering phase
    all_stocks_dataset = feature_engineering.feature_engineering(tickers, use_macros=True, compute_differences=True)
    print(all_stocks_dataset.shape)

    # preprocessing phase
    X_train, y_train, X_test, y_test, columns, test_data_with_dates, X_val, y_val, validation_data_with_dates = \
        preprocessing_module.preprocessing(all_stocks_dataset)

    # run xgboost regressor model
    reg_pred, mae, mse, xgboostRegressor, importance = regression_models.xgboostRegressor(X_train, y_train, X_test, y_test)

    # concatenate XGBoost results to all_stocks_dataset
    X_train = np.c_[X_train, xgboostRegressor.predict(X_train)]
    X_val = np.c_[X_val, xgboostRegressor.predict(X_val)]
    X_test = np.c_[X_test, xgboostRegressor.predict(X_test)]

    # create importances df
    importances = fi.create_feature_importances_dataframe(importance, columns)

    # plot importances
    fi.plot_feature_importances(importances)

    # report XGBoost regressor metrics
    print('XGBoost regression metrics')
    print('MAE: ', mae)
    print('MSE: ', mse)

    # perform grid search on validation set
    #optimization_results = gridSearchOptimization.gridSearch(X_train, y_train, X_val, y_val, validation_data_with_dates)

    #optimization_results = pd.read_csv('optimization_results.csv', index_col=0)

    # store predictions on a list
    predictions = []

    # store results in lists
    cum_list = []
    sharpe_list = []
    volatility_list = []
    returns_list = []

    loss_list = ['mse', 'mae', 'huber']
    for loss in loss_list:
        # train and test model
        # reshape y_val
        tf.random.set_seed(42)
        y_val = y_val.reshape(y_val.shape[0], 1)
        model = regression_models.NN(learning_rate=0.01, loss=loss)
        history = model.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0),
                        batch_size=128, epochs=40, validation_data=(X_val, y_val), verbose=True)

        # plot training and validation losses
        plt.figure()
        plt.plot(history.history['loss'], label='Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(f'{loss.upper()} Loss')
        plt.title(f'Training and validation loss for {loss.upper()} loss function')
        plt.show()

        # make predictions on test set and evaluate
        y_pred = model.predict(X_test)
        predictions.append(y_pred)
        mse, mae, huber, r2 = preprocessing_module.evaluation(y_test, y_pred)
        print('MSE =', np.round(mse, 5))
        print('MAE =', np.round(mae, 5))
        print('Huber =', np.round(huber, 5))
        print('R2 =', np.round(r2, 5))

        # financial evaluation
        portfolio_ret, cum_ret, sharpe_ratio, volatility, unique_tickers = preprocessing_module.financialEvaluation(test_data_with_dates, y_pred, validation=False)
        print('Cumulative return =', cum_ret.iloc[-1].values)
        print('Sharpe ratio', sharpe_ratio.values)
        print('Volatility', volatility.values)
        print('Detailed portfolio returns', portfolio_ret)

        # store results in list
        cum_list.append(cum_ret.iloc[-1].values)
        sharpe_list.append(sharpe_ratio.values)
        volatility_list.append(volatility.values)
        returns_list.append(cum_ret)


    # load s&p 500 index data
    gspc = pd.read_csv('Price data/gspc.csv', index_col=0)
    gspc = gspc.pct_change()
    gspc = gspc.iloc[3:]
    gspc.dropna(inplace=True)
    gspc = (gspc + 1).cumprod() - 1

    # EWP top 50 evaluation
    keep_top_k_stocks = 50
    optimal_weights = np.tile(1/keep_top_k_stocks, keep_top_k_stocks)
    ew_portfolio_ret, ew_cum_ret, ew_sharpe_ratio, ew_volatility = portfolio_optimization_module.calc_portfolio_performance(optimal_weights, unique_tickers, validation=False)

    # plot P&L plot for best model and S&P 500 index
    plt.plot(np.array(returns_list[0].index), returns_list[0].values, color='b', label='MSE')
    plt.plot(np.array(returns_list[1].index), returns_list[1].values, color='g', label='MAE')
    plt.plot(np.array(returns_list[2].index), returns_list[2].values, color='y', label='Huber')
    plt.plot(np.array(ew_cum_ret.index), ew_cum_ret.values, color='m', label='Top 50 EWP')
    plt.plot(np.array(gspc.index), gspc.values, color='r', label='S&P 500')
    plt.legend()
    plt.title('Profit and loss of proposed portfolios and S&P 500')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.savefig('P_and_L_plot.jpeg', dpi=300)





