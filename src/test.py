from utils import *
from config import config

group_names = pd.read_csv("config/StockNames.csv").columns
for group_name in group_names:
    data_dir = config.DATA_CONFIG['data_dir']

    # Load config of stock groups
    StockNames = pd.read_csv(config.DATA_CONFIG['stock_names_file'])
    StockSymbols = pd.read_csv(config.DATA_CONFIG['stock_symbols_file'])

    names = [x for x in StockNames[group_name] if str(x) != 'nan']
    symbols = [x for x in StockSymbols[group_name] if str(x) != 'nan']

    min_date = max([datetime.strptime(pd.read_csv(data_dir + symbol + '.csv')['Date'].min(), '%Y-%m-%d') for name, symbol in zip(names, symbols)])
    max_date = min([datetime.strptime(pd.read_csv(data_dir + symbol + '.csv')['Date'].max(), '%Y-%m-%d') for name, symbol in zip(names, symbols)])

    # List dates that exist for all datasets
    mini = np.min([pd.read_csv(data_dir + symbol + '.csv').shape[0] for name, symbol in zip(names, symbols)])
    ind = np.argmin([pd.read_csv(data_dir + symbol + '.csv').shape[0] for name, symbol in zip(names, symbols)])
    if mini < 2000:
        print(names[ind])
    print(group_name, ': ', names)