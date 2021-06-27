# Stock-Pred-Mutual-Info
PyTorch implementation of Least-Squares DQN with extras (DuelingDQN, Boosted FQI)

- [Stock-Pred-Mutual-Info](#Stock-Pred-Mutual-Info)
  * [Background](#background)
  * [Prerequisites](#prerequisites)
  * [Files in the repository](#files-in-the-repository)
  * [Usage](#usage)
  * [Parameters](#parameters)
  * [References](#references)

## Background
This is a POC tool ment to check the mutual information between a stock and a group of other stocks. 



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


## Usage

1. Fill in group names and stock names in /config/StockNames.csv
2. Fill in group names and stock symbols in /config/StockSymbols.csv (in accordance with StockNames.csv)
3. Edit config/config.py, mainly RUN_CONFIG:


# Parameters
* kernel_size = size of the line segement kernel (usually 1/30 of the height/width of the original image)
* stroke_width = thickness of the strokes in the Stroke Map (0, 1, 2)
* num_of_directions = stroke directions in the Stroke Map (used for the kernels)
* smooth_kernel = how the image is smoothed (Gaussian Kernel - "gauss", Median Filter - "median")
* gradient_method = how the gradients for the Stroke Map are calculated (0 - forward gradient, 1 - Sobel)
* rgb = True if the original image has 3 channels, False if grayscale
* w_group = 3 possible weight groups (0, 1, 2) for the histogram distribution, according to the paper (brighter to darker)
* pencil_texture_path = path to the Pencil Texture Map to use (4 options in "./pencils", you can add your own)
* stroke_darkness = 1 is the same, up is darker.
* tone_darkness = as above


## References
[1] Kazemi, Seyed Mehran, et al. "Time2vec: Learning a vector representation of time." arXiv preprint arXiv:1907.05321 (2019).



