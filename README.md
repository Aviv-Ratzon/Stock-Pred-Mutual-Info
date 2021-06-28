# Stock-Pred-Mutual-Info
PyTorch implementation of Least-Squares DQN with extras (DuelingDQN, Boosted FQI)

![results](https://github.com/Aviv-Ratzon/Shit-Class_Stock-Pred/blob/main/images/Loss_Change_By_Group.png)

- [Stock-Pred-Mutual-Info](#Stock-Pred-Mutual-Info)
  * [Background](#background)
  * [Usage](#usage)
  * [Prerequisites](#prerequisites)
  * [Files in the repository](#files-in-the-repository)
  * [Parameters](#parameters)
  * [References](#references)

## Background
This is a POC tool ment to check the mutual information between a stock and a group of other stocks. 


## Usage

1. Fill in group names and stock names in `/config/StockNames.csv`
1. Fill in group names and stock symbols in `/config/StockSymbols.csv` (in accordance with `StockNames.csv`)
1. Edit `config/config.py`, mainly `RUN_CONFIG`:
    - `'RUN_CONFIG':'run train'` whether to train models
    - `'train stocks':'train stocks'` which stock groups to train
    - `'compare stocks':'compare stocks'` which stock groups to compare
1. run main.py


## Prerequisites
|Library         | Version |
|----------------------|----|
|`Python`|  `3.8`|
|`torch`|  `1.7.0`|
|`pandas`|  `1.1.4`|
|`matplotlib`|  `3.3.4`|
|`numpy`|  `1.19.4`|
|`yahoo-finance`|  `1.6`|


## Files in the repository

|File name         | Purpsoe |
|----------------------|------|
|`config.py`| configuration of which models to run hyperparameters |
|`download_data.py`| function for downloading the stocks listed in StockSymbols.csv |
|`Main.py`| main function, run this file after configuration|
|`model.py`| contains TransformerEncoder model and Time2Vec model|
|`run_compare.py`| function that loads results(losses and predicitons) and plots comparison |
|`run_train.py`| function that runs training procedure for the configured groups and saves models and results|
|`train.py`| functions that handle training of a model|
|`utils.py`| utility functions|

# Parameters
- MODEL_CONFIG
    - `n_hid`: Number of hidden units for the attention layers
    - `dropout`: Dropout of transformer model
    - `n_layers`: Number of multi-head attention layers
- TRAIN_CONFIG
    - `lr`: Learning rate
    - `num_epochs`: number of epochs
    - `seq_len`: Number of consequtive days that the model is trained on to predict one day ahead
    - `bsz`: Batch size (number of sequences that are trained before calling `backward`
    - `gamma`: lr decrease at each epoch
    - `log_interval`: how many sequences to run before saving train loss to list and printing progress
    - `chkpt_path`: path for saving chekpoints
    - `results_path`: path for saving results
- TRAIN_CONFIG
    - `data_dir`: location of data directory where stocks are saved
    - `stock_names_file`: location of file containing stock names
    - `stock_symbols_file`: location of file containing stock symbols
    - `n_dim`: dimensionality of data (5 for OHLCV)
    - `predict_column`: The dimension of input to predict
    - `log_interval`: how many sequences to run before saving train loss to list and printing progress

## References
[1] Kazemi, Seyed Mehran, et al. "Time2vec: Learning a vector representation of time." arXiv preprint arXiv:1907.05321 (2019).



