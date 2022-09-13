import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
import pandas as pd
import numpy as np


def featureEngineering(tickers, use_macros):
    # Import fundamental and price data and construct features
    dataset = pd.DataFrame()
    for ticker in tickers:
        # Import fundamental data
        with open(f'data/{ticker}_income_statement.json') as json_file:
            income_statement = json.load(json_file)
        with open(f'data/{ticker}_balance_sheet.json') as json_file:
            balance_sheet = json.load(json_file)
        with open(f'data/{ticker}_cash_flow.json') as json_file:
            cash_flow = json.load(json_file)

        # Import price data
        prices = pd.read_csv(f'data/prices/{ticker}_quarters.csv')
        prices['ticker'] = ticker

        # FEATURE 1 - Return on Assets(ROA) = Net Income / Total Assets
        # -------------------------------------------------------------
        quarterlyIS = income_statement['quarterlyReports']
        netIncome = []
        dates = []
        for quarter in quarterlyIS:
            netIncome.append(int(quarter['netIncome']))
            dates.append(quarter['fiscalDateEnding'])

        data = pd.DataFrame(netIncome, columns=['netIncome'])

        # Select total assets from balance sheet
        quarterlyBS = balance_sheet['quarterlyReports']
        totalAssets = []
        commonSharesOutstanding = []
        for quarter in quarterlyBS:
            if quarter['commonStockSharesOutstanding'] == 'None':
                commonSharesOutstanding.append(0)
            else:
                commonSharesOutstanding.append(quarter['commonStockSharesOutstanding'])

            if quarter['totalAssets'] == 'None':
                totalAssets.append(0)
            else:
                totalAssets.append(int(quarter['totalAssets']))

        data['totalAssets'] = totalAssets
        data['commonSharesOutstanding'] = np.array(commonSharesOutstanding, dtype='float64')


        # Compute return on assets
        data['ReturnOnAssets'] = data['netIncome'] / data['totalAssets']

        # FEATURE 2 - Debt Ratio = Total Liabilities / Total Assets
        # -------------------------------------------------------------
        totalLiabilities = []
        for quarter in quarterlyBS:
            if quarter['totalLiabilities'] == 'None':
                totalLiabilities.append(0)
            else:
                totalLiabilities.append(int(quarter['totalLiabilities']))

        data['totalLiabilities'] = totalLiabilities

        # Compute Debt Ratio
        data['DebtRatio'] = data['totalLiabilities'] / data['totalAssets']

        # FEATURE 3 - Current Ratio = Current Debt / Current Assets
        # -------------------------------------------------------------
        currentDebt = []
        currentLiabilities = []
        for quarter in quarterlyBS:
            if quarter['totalCurrentLiabilities'] == 'None':
                currentLiabilities.append(0)
            else:
                currentLiabilities.append(quarter['totalCurrentLiabilities'])

            if quarter['currentDebt'] == 'None':
                currentDebt.append(0)
            else:
                currentDebt.append(quarter['currentDebt'])

        data['totalCurrentLiabilities'] = np.array(currentLiabilities, dtype='float64')
        currentAssets = []
        for quarter in quarterlyBS:
            if quarter['totalCurrentAssets'] == 'None':
                currentAssets.append(0)
            else:
                currentAssets.append(int(quarter['totalCurrentAssets']))

        data['currentDebt'] = np.array(currentDebt, dtype='float64')
        data['currentAssets'] = np.array(currentAssets, dtype='float64')

        # Compute Current Ratio
        data['currentRatio'] = data['currentDebt'] / data['currentAssets']

        # FEATURE 4 - Gross Margin = Revenue - Cost of Goods sold / Revenue
        # -------------------------------------------------------------
        revenue = []
        for quarter in quarterlyIS:
            if quarter['totalRevenue'] == 'None':
                revenue.append(0)
            else:
                revenue.append(int(quarter['totalRevenue']))

        costOfGoodsSold = []
        for quarter in quarterlyIS:
            if quarter['costofGoodsAndServicesSold'] == 'None':
                costOfGoodsSold.append(0)
            else:
                costOfGoodsSold.append(int(quarter['costofGoodsAndServicesSold']))

        data['revenue'] = revenue
        data['costOfGoodsSold'] = costOfGoodsSold

        # Expenses
        expenses = []
        for quarter in quarterlyIS:
            if quarter['operatingExpenses'] == 'None':
                expenses.append(0)
            else:
                expenses.append(quarter['operatingExpenses'])
        data['expenses'] = np.array(expenses, dtype='float64')

        # Interests
        interests = []
        for quarter in quarterlyIS:
            if quarter['interestIncome'] == 'None':
                interests.append(0)
            else:
                interests.append(quarter['interestIncome'])
        data['interests'] = np.array(interests, dtype='float64')
        data['interests'].replace('None', 0, inplace=True)

        # Gross Profit
        data['gross_profit'] = data['revenue'] - data['costOfGoodsSold']

        # Investments
        quarterlyCF = cash_flow['quarterlyReports']
        investments = []
        preferred_dividends = []
        for quarter in quarterlyCF:
            if quarter['cashflowFromInvestment'] == 'None':
                investments.append(0)
            else:
                investments.append(quarter['cashflowFromInvestment'])

            if quarter['dividendPayout'] == 'None':
                preferred_dividends.append(0)
            else:
                preferred_dividends.append(quarter['dividendPayout'])

        data['investments'] = np.array(investments, dtype='float64')
        data['preferred_dividends'] = np.array(preferred_dividends, dtype='float64')
        # Liabilities
        data['totalLiabilities'] = totalLiabilities
        # Equity - Book Value
        data['book_value'] = data['totalAssets'] - data['totalLiabilities']
        # Ticker
        data['ticker'] = ticker
        # Date
        data['date'] = dates

        # cash flow fundamental data
        data['EPS'] = (data['netIncome'] - data['preferred_dividends']) / data['commonSharesOutstanding']


        # Remove Open, Low, Adj Close, Volume for prices dataframe
        prices = prices.drop(columns=['Open', 'High', 'Low', 'Adj Close'])
        # prices = prices.drop(columns=['Adj Close'])
        # Reindex prices so that old values are on top
        prices.sort_values(by='date', inplace=True, ascending=False)

        # Compute log returns
        prices['log returns'] = np.log(prices['Close']) - np.log(prices['Close'].shift(-1))

        # Compute log returns trend
        #prices['log returns trend'] = prices['log returns'].apply(lambda x: 1 if x > 0 else 0)

        # Merge Fundamental and price data
        FD = data
        FD.sort_values(by=['date'], inplace=True)
        FD = FD.merge(prices, how='outer', on=['date', 'ticker'])

        if use_macros:
            # Merge macroeconomic variables
            macroeconomics = pd.read_csv('macro.csv')
            macroeconomics = macroeconomics.loc[
                (macroeconomics['date'] >= '2016-12-31') & (macroeconomics['date'] <= '2021-10-01')]
            macroeconomics.drop(columns='date', inplace=True)
            macroeconomics.reset_index(inplace=True, drop=True)
            FD = FD.join(macroeconomics)
            FD['CONSUMER_SENTIMENT'] = np.array(FD['CONSUMER_SENTIMENT'], dtype='float64')

        FD.dropna(axis=0, inplace=True)

        FD['P/E ratio'] = FD['Close'] / FD['EPS']
        FD['P/B ratio'] = FD['Close'] / (FD['book_value'] / FD['commonSharesOutstanding'])
        FD['Net Margin'] = 100 * FD['netIncome'] / FD['revenue']

        dataset = dataset.append(FD)

    dataset.replace([np.inf, -np.inf], 0, inplace=True)
    return dataset


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


