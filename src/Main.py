import config
from run_train import run_train
from run_compare import run_compare
from download_data import download_data

# Download missing data
download_data()

# Train models if needed
if config.RUN_CONFIG['run train']:
    run_train()

# Compare and display results of single inputs and mutual inputs
run_compare()
