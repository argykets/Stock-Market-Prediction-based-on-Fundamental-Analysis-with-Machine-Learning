import pandas as pd

import preprocessing_module
import regression_models
import portfolio_optimization_module


def gridSearch(X_train, y_train, X_val, y_val, validation_data_with_dates):
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
        for loss in grid['loss']:
            model = regression_models.NN(learning_rate=0.01, loss=loss)
            model.fit(X_train, y_train, batch_size=batch_size, epochs=100)
            y_hat_val = model.predict(X_val)
            y_hat_val_list.append(y_hat_val)
            mse, mae, r2, huber, cs, msle = preprocessing_module.evaluation(y_val, y_hat_val)
            hyperparameter_list.append(f'batch_size:{batch_size}, loss:{loss}')
            mse_list.append(mse)
            mae_list.append(mae)
            r2_list.append(r2)
            huber_list.append(huber)
            cosine_similarity_list.append(cs)
            msle_list.append(msle)

            # add date and ticker to predictions
            backtesting_data = validation_data_with_dates[['date', 'ticker']]
            # backtesting_data.reset_index(inplace=True, drop=True)
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
                                         mse_list, mae_list, r2_list, huber_list, cosine_similarity_list, msle_list,
                                         portfolio_value_list],
                                        index=['hyperparameters', 'mse', 'mae', 'r2', 'huber', 'cosine similarity',
                                               'msle', 'cum return'])
    optimization_results = optimization_results.T

    optimization_results.index = optimization_results['hyperparameters']
    optimization_results.drop(columns='hyperparameters', inplace=True)

    optimization_results = optimization_results.astype(float).round(4)

    optimization_results.to_csv('optimization_results.csv')

    return optimization_results