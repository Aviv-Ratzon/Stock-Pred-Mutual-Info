import pandas as pd
import yfinance as yf
import os.path


def download_data():
    dir = 'data/stocks/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Load config of stock groups
    StockNames = pd.read_csv("config/StockNames.csv")
    StockSymbols = pd.read_csv("config/StockSymbols.csv")

    # Identify stock symbols for download
    column_values = StockSymbols.values.ravel()
    symbols = pd.unique(column_values)
    symbols = [x for x in symbols if str(x) != 'nan']

    column_values = StockNames.values.ravel()
    names = pd.unique(column_values)
    names = [x for x in names if str(x) != 'nan']

    for sym, name in zip(symbols, names):
        path = dir + sym + '.csv'
        # Only download non-existing data
        if not os.path.isfile(path):
            print('Downloading ' + name + ' stock data.......')
            # Download entire stock history
            ticker = yf.Ticker(sym)
            df = ticker.history(period="max")
            if df.empty:
                raise RuntimeError('Failed to to download stock, check symbol [' + sym + ']')
            # Remove unwanted columns, clear 0 values of volume and sort by date
            df = df.drop(['Dividends', 'Stock Splits'], axis=1)
            df['Volume'].replace(to_replace=0, method='ffill', inplace=True)
            df.sort_values('Date', inplace=True)

            # save
            df.to_csv(path, index=True)
            print('Downloaded ' + name + ' stock data with ' + str(df.shape[0]) + ' entries')
