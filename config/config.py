import torch

# Options for running project
RUN_CONFIG = {
    'run train': True,
    'retrain': False,
    # options are ['All'] or list of names e.g ['Banks', 'Unrelated 4', 'Unrelated 5']
    'train stocks': ['All'],
    # options are ['All'] or list of names
    'compare stocks': ['All']
}

# Hyper parameters for TransformerEncoder
MODEL_CONFIG = {
    'n_hid': 256,
    'dropout': 0.1,
    'n_layers': 2,
}

# Hyper parameters for training
TRAIN_CONFIG = {
    'lr': 0.1,
    'num_epochs': 15,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'seq_len': 128,
    'bsz': 32,
    'gamma': 0.9,
    'log_interval': 1000,
    'chkpt_path': './checkpoints/',
    'results_path': './data/results/'
}

# Options for data source
DATA_CONFIG = {
    'data_dir': 'data/stocks/',
    'stock_names_file': 'config/StockNames.csv',
    'stock_symbols_file': 'config/StockSymbols.csv',
    'n_dim': 5,
    'predict_column': 3
}

