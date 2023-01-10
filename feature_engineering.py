import warnings
import json
import pandas as pd
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)


def construct_fundamental(income_statement, balance_sheet, cash_flow, ticker):
    # initialize fundamental_data dataframe
    fundamental_data = pd.DataFrame()
    # extract info from income statement
    # date, net income
    net_income = []
    revenue = []
    dates = []
    cost_of_goods_sold = []
    expenses = []
    interests = []
    for quarter in income_statement['quarterlyReports']:
        net_income.append(quarter['netIncome'])
        dates.append(quarter['fiscalDateEnding'])
        revenue.append(quarter['totalRevenue'])
        cost_of_goods_sold.append(quarter['costofGoodsAndServicesSold'])
        expenses.append(quarter['operatingExpenses'])
        interests.append(quarter['interestIncome'])

    fundamental_data['date'] = dates
    fundamental_data['ticker'] = ticker
    fundamental_data['net_income'] = net_income
    fundamental_data['revenue'] = revenue
    fundamental_data['cost_of_goods_sold'] = cost_of_goods_sold
    fundamental_data['interests'] = interests

    # extract info from balance sheet
    # total assets, common shares outstanding
    total_assets = []
    common_shares_outstanding = []
    total_liabilities = []
    current_liabilities = []
    current_debt = []
    current_assets = []
    for quarter in balance_sheet['quarterlyReports']:
        common_shares_outstanding.append(quarter['commonStockSharesOutstanding'])
        total_assets.append(quarter['totalAssets'])
        total_liabilities.append(quarter['totalLiabilities'])
        current_liabilities.append(quarter['totalCurrentLiabilities'])
        current_debt.append(quarter['currentDebt'])
        current_assets.append(quarter['totalCurrentAssets'])

    fundamental_data['total_assets'] = total_assets
    fundamental_data['total_liabilities'] = total_liabilities
    fundamental_data['common_shares_outstanding'] = common_shares_outstanding
    fundamental_data['current_liabilities'] = current_liabilities
    fundamental_data['current_assets'] = current_assets
    fundamental_data['current_debt'] = current_debt

    # extract info from cash flow
    # investments, preferred dividends
    investments = []
    preferred_dividends = []
    for quarter in cash_flow['quarterlyReports']:
        investments.append(quarter['cashflowFromInvestment'])
        preferred_dividends.append(quarter['dividendPayout'])

    fundamental_data['investments'] = investments
    fundamental_data['preferred_dividends'] = preferred_dividends

    # compute fundamental ratios
    # Replace 'None' values with 0
    fundamental_data.replace(to_replace='None', value=0, inplace=True)
    # specify dtypes
    convert_dict = {'date': str,
                    'ticker': str,
                    'net_income': int,
                    'revenue': int,
                    'total_assets': int,
                    'total_liabilities': int,
                    'common_shares_outstanding': int,
                    'current_liabilities': int,
                    'current_assets': int,
                    'current_debt': int,
                    'investments': int,
                    'preferred_dividends': int,
                    'interests': int,
                    'cost_of_goods_sold': int
                    }
    fundamental_data = fundamental_data.astype(convert_dict)

    # Compute return on assets
    fundamental_data['return_on_assets'] = fundamental_data['net_income'] / fundamental_data['total_assets']

    # Compute Debt Ratio
    fundamental_data['debt_ratio'] = fundamental_data['total_liabilities'] / fundamental_data['total_assets']

    # Compute Current Ratio
    fundamental_data['current_ratio'] = fundamental_data['current_debt'] / fundamental_data['current_assets']

    # Compute Gross Profit
    fundamental_data['gross_profit'] = fundamental_data['revenue'] - fundamental_data['cost_of_goods_sold']

    # Compute Book Value - Equity
    fundamental_data['book_value'] = fundamental_data['total_assets'] - fundamental_data['total_liabilities']

    # Compute EPS
    fundamental_data['EPS'] = (fundamental_data['net_income'] -
                               fundamental_data['preferred_dividends']) / fundamental_data['common_shares_outstanding']

    return fundamental_data


def construct_technical(ticker):
    technical_data = pd.read_csv(f'data/prices/{ticker}_quarters.csv')
    technical_data['ticker'] = ticker
    technical_data.rename(columns={'Close': 'close', 'Volume': 'volume'}, inplace=True)

    # Remove Open, Low, Adj Close, Volume for prices dataframe
    technical_data.drop(columns=['Open', 'High', 'Low', 'Adj Close'], inplace=True)
    # Reindex prices so that old values are on top
    technical_data.sort_values(by='date', inplace=True, ascending=False)

    # Compute log returns
    technical_data['log_returns'] = np.log(technical_data['close']) - np.log(technical_data['close'].shift(-1))
    technical_data['returns'] = (technical_data['close'] - technical_data['close'].shift(-1)) /\
                                technical_data['close'].shift(-1)

    # Compute log returns trend
    # prices['log returns trend'] = prices['log returns'].apply(lambda x: 1 if x > 0 else 0)

    return technical_data


def feature_engineering(tickers, use_macros, compute_differences):
    all_stocks_dataset = pd.DataFrame()
    for ticker in tickers:
        # Import financial statements
        with open(f'data/{ticker}_income_statement.json') as json_file:
            income_statement = json.load(json_file)
        with open(f'data/{ticker}_balance_sheet.json') as json_file:
            balance_sheet = json.load(json_file)
        with open(f'data/{ticker}_cash_flow.json') as json_file:
            cash_flow = json.load(json_file)

        # Construct fundamental
        fundamental_data = construct_fundamental(income_statement, balance_sheet, cash_flow, ticker)
        # Construct technical
        technical_data = construct_technical(ticker)
        # Merge fundamental and technical data
        fundamental_data.sort_values(by=['date'], inplace=True)
        dataset = fundamental_data.merge(technical_data, how='outer', on=['date', 'ticker'])

        # Merge macros if needed
        if use_macros:
            # Merge macroeconomic variables
            macroeconomics = pd.read_csv('macro.csv')
            macroeconomics = macroeconomics.loc[
                (macroeconomics['date'] >= '2016-12-31') & (macroeconomics['date'] <= '2021-10-01')]
            macroeconomics.drop(columns='date', inplace=True)
            macroeconomics.reset_index(inplace=True, drop=True)
            macroeconomics['consumer_sentiment'] = pd.to_numeric(macroeconomics['CONSUMER_SENTIMENT'])
            macroeconomics.drop(columns=['CONSUMER_SENTIMENT'], inplace=True)
            dataset = dataset.join(macroeconomics)

        dataset.dropna(axis=0, inplace=True)

        # Compute additional ratios
        dataset['pe_ratio'] = dataset['close'] / dataset['EPS']
        dataset['pb_ratio'] = dataset['close'] / (dataset['book_value'] / dataset['common_shares_outstanding'])
        dataset['net_margin'] = 100 * dataset['net_income'] / dataset['revenue']

        if compute_differences:
            unwanted = {'close', 'date', 'ticker', 'log_returns'}
            columns_for_pct_change = [ele for ele in dataset if ele not in unwanted]
            for column in columns_for_pct_change:
                dataset[column] = dataset[column].pct_change()
            dataset.replace([np.inf, -np.inf], 0, inplace=True)
            dataset.fillna(0, inplace=True)

        all_stocks_dataset = all_stocks_dataset.append(dataset)

    all_stocks_dataset.replace([np.inf, -np.inf], 0, inplace=True)
    return all_stocks_dataset


def hybrid_dataset_construction(regressor, X_train, X_test, X_val):
    Y_hat_train = regressor.predict(X_train)
    Y_hat_train = np.reshape(Y_hat_train, (Y_hat_train.shape[0], 1))
    X_train = np.append(X_train, Y_hat_train, axis=1)

    Y_hat_test = regressor.predict(X_test)
    Y_hat_test = np.reshape(Y_hat_test, (Y_hat_test.shape[0], 1))
    X_test = np.append(X_test, Y_hat_test, axis=1)

    Y_hat_val = regressor.predict(X_val)
    Y_hat_val = np.reshape(Y_hat_val, (Y_hat_val.shape[0], 1))
    X_val = np.append(X_val, Y_hat_val, axis=1)
    return X_train, X_test, X_val
