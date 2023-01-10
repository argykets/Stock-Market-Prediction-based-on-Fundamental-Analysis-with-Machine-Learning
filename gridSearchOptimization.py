import pandas as pd

import tensorflow as tf
import preprocessing_module
import regression_models
import portfolio_optimization_module


def gridSearch(X_train, y_train, X_val, y_val, validation_data_with_dates):
    grid = preprocessing_module.grid_construction()
    hyperparameter_list = []
    mse_list = []
    mae_list = []
    huber_list = []
    r2_list = []
    portfolio_value_list = []
    sharpe_ratio_list = []
    volatility_list = []
    y_hat_val_list = []
    for loss in grid['loss']:
        for epoch in grid['epochs']:
            for batch_size in grid['batch_size']:
                for lr in grid['learning_rate']:
                    tf.random.set_seed(42)
                    model = regression_models.NN(learning_rate=lr, loss=loss)
                    model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch)
                    y_hat_val = model.predict(X_val)
                    y_hat_val_list.append(y_hat_val)
                    mse, mae, huber, r2 = preprocessing_module.evaluation(y_val, y_hat_val)
                    hyperparameter_list.append(f'loss:{loss}, epochs:{epoch} batch_size:{batch_size}, learning_rate:{lr}')
                    mse_list.append(mse)
                    mae_list.append(mae)
                    huber_list.append(huber)
                    r2_list.append(r2)

                    # add date and ticker to predictions
                    backtesting_data = validation_data_with_dates[['date', 'ticker']]
                    backtesting_data.reset_index(inplace=True, drop=True)
                    backtesting_data['expected_returns'] = y_hat_val

                    # portfolio optimization
                    portfolio_ret, cum_ret, sharpe_ratio, volatility = preprocessing_module.financialEvaluation(
                        validation_data_with_dates, y_hat_val, validation=True)
                    print(sharpe_ratio)
                    sharpe_ratio_list.append(sharpe_ratio)
                    volatility_list.append(volatility)
                    portfolio_value_list.append(cum_ret)
                    continue

    optimization_results = pd.DataFrame([hyperparameter_list,
                                         mse_list, mae_list, huber_list,
                                         r2_list, portfolio_value_list, sharpe_ratio_list, volatility_list],
                                        index=['hyperparameters', 'mse', 'mae', 'huber',
                                               'r2', 'cum return', 'sharpe ratio', 'volatility'])
    optimization_results = optimization_results.T

    optimization_results.index = optimization_results['hyperparameters']
    optimization_results.drop(columns='hyperparameters', inplace=True)

    optimization_results = optimization_results.astype(float).round(4)

    optimization_results.to_csv('optimization_results.csv')

    return optimization_results
