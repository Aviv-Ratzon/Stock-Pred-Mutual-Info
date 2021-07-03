from train import *
import pandas as pd
import numpy as np
import torch
import config
from datetime import datetime, timedelta


def get_group_data(group):
    data_dir = config.DATA_CONFIG['data_dir']

    # Load config of stock groups
    StockNames = pd.read_csv(config.DATA_CONFIG['stock_names_file'])
    StockSymbols = pd.read_csv(config.DATA_CONFIG['stock_symbols_file'])

    # Extract non-nan names and symbols
    names = [x for x in StockNames[group] if str(x) != 'nan']
    symbols = [x for x in StockSymbols[group] if str(x) != 'nan']

    # Calculate date limits for data alignment
    min_date = max([datetime.strptime(pd.read_csv(data_dir + symbol + '.csv')['Date'].min(), '%Y-%m-%d') for name, symbol in zip(names, symbols)])
    max_date = min([datetime.strptime(pd.read_csv(data_dir + symbol + '.csv')['Date'].max(), '%Y-%m-%d') for name, symbol in zip(names, symbols)])

    # List dates that exist for all datasets
    df_dates = [pd.read_csv(data_dir + symbol + '.csv')['Date'].values for name, symbol in zip(names, symbols)]
    delta = max_date - min_date  # as timedelta
    valid_dates = []
    for i in range(delta.days + 1):
        day = (min_date + timedelta(days=i)).strftime('%Y-%m-%d')
        valid = np.logical_and.reduce([day in dates for dates in df_dates])
        if valid:
            valid_dates.append(day)

    data = {}
    for name, symbol in zip(names, symbols):
        df = pd.read_csv(data_dir + symbol + '.csv')
        df.set_index('Date', inplace=True)
        df = df[df.index.isin(valid_dates)]
        df = df.dropna()
        # Align dates to other stocks in group
        df = df[df.index.isin(valid_dates)]

        # Create arithmetic returns column - use rolling mean to verify that model can learn an easier task
        df['Open'] = df['Open'].pct_change()#.rolling(10).mean()
        df['High'] = df['High'].pct_change()#.rolling(10).mean()
        df['Low'] = df['Low'].pct_change()#.rolling(10).mean()
        df['Close'] = df['Close'].pct_change()#.rolling(10).mean()
        df['Volume'] = df['Volume'].pct_change()#.rolling(10).mean()

        # Zero out nans and infs
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.replace(np.nan, 0, inplace=True)

        ###############################################################################
        '''Create indexes to split dataset'''

        times = sorted(df.index.values)
        last_10pct = sorted(df.index.values)[-int(0.1 * len(times))]  # Last 10% of series

        ###############################################################################
        '''Normalize price columns'''
        #
        min_return = min(df[(df.index < last_10pct)][['Open', 'High', 'Low', 'Close']].min(axis=0))
        max_return = max(df[(df.index < last_10pct)][['Open', 'High', 'Low', 'Close']].max(axis=0))

        # Min-max normalize price columns (0-1 range)
        df['Open'] = (df['Open'] - min_return) / (max_return - min_return)
        df['High'] = (df['High'] - min_return) / (max_return - min_return)
        df['Low'] = (df['Low'] - min_return) / (max_return - min_return)
        df['Close'] = (df['Close'] - min_return) / (max_return - min_return)

        ###############################################################################
        '''Normalize volume column'''

        min_volume = df[(df.index < last_10pct)]['Volume'].min(axis=0)
        max_volume = df[(df.index < last_10pct)]['Volume'].max(axis=0)

        # Min-max normalize volume columns (0-1 range)
        df['Volume'] = (df['Volume'] - min_volume) / (max_volume - min_volume)

        ###############################################################################
        '''Create training, validation and test split'''

        df_train = df[(df.index < last_10pct)]  # Training data are 80% of total data
        df_test = df[(df.index >= last_10pct)]

        # Remove date column
        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)

        # Convert pandas columns into arrays
        train_data = torch.tensor(df_train.values).float().to(config.TRAIN_CONFIG['device'])
        test_data = torch.tensor(df_test.values).float().to(config.TRAIN_CONFIG['device'])

        df_train.head()
        data[name] = [train_data, test_data]
    print('Training data shape: {}'.format(train_data.shape))
    print('Test data shape: {}'.format(test_data.shape))

    return data


def batchify(data):
    bsz = config.TRAIN_CONFIG['bsz']
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1, data.shape[-1]).t().contiguous()
    return data.to(config.TRAIN_CONFIG['device'])


def get_batch(source, i, ):
    seq_len = config.TRAIN_CONFIG['seq_len']
    data = source[i:i + seq_len]
    target = source[i+seq_len+1, config.DATA_CONFIG['predict_column']]
    return data, target


def evaluate(eval_model, data_source, criterion):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    seq_len = config.TRAIN_CONFIG['seq_len']
    outputs = []
    with torch.no_grad():
        for i in range(0, data_source.size(0) - seq_len - 1):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            total_loss += criterion(output, targets).item()
            outputs.append(output)
    return total_loss / (data_source.size(0) - seq_len - 1), torch.tensor(outputs)
