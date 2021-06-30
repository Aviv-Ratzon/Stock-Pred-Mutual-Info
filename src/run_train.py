from train import *
from utils import *
import config
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import os


def run_train():
    # Determine which groups to train on
    if config.RUN_CONFIG['train stocks'] == ['All']:
        group_names = pd.read_csv("config/StockNames.csv").columns
    else:
        group_names = config.RUN_CONFIG['train stocks']

    for group_name in group_names:
        # Make relevant directories
        path_ckpt = config.TRAIN_CONFIG['chkpt_path'] + group_name + '/'
        path_results = config.TRAIN_CONFIG['results_path'] + group_name + '/'
        if not os.path.exists(path_ckpt):
            os.makedirs(path_ckpt)
        if not os.path.exists(path_results):
            os.makedirs(path_results)

        # Process group data
        stock_datas = get_group_data(group_name)

        for stock_name, stock_data in stock_datas.items():
            train_data, test_data = stock_data

            # Train single model
            filename = path_ckpt + stock_name + '_single_ckpt.pth'
            if not os.path.isfile(filename) or config.RUN_CONFIG['retrain']:
                print('Training ' + stock_name + ' single model.............')
                model_single, test_losses_single, train_losses_single = train(train_data, test_data)
                print('==> Saving model ...')
                state = {'net': model_single.state_dict()}
                torch.save(state, filename)

                # Evaluate single model
                train_loss_single, train_preds_single = evaluate(model_single, train_data, nn.MSELoss())
                test_loss_single, test_preds_single = evaluate(model_single, test_data, nn.MSELoss())
                # Save data of losses and prediction for both train and test
                print('Test loss for ' + stock_name + ' single model = {:.5f}'.format(test_loss_single))
                print('Saving results for ' + stock_name + ' single.............')
                torch.save([train_data, train_loss_single, train_preds_single,
                            test_data, test_loss_single, test_preds_single], path_results + stock_name + '_single_results.pth')


            # Train mutual model
            filename = path_ckpt + stock_name + '_mutual_ckpt.pth'
            if not os.path.isfile(filename) or config.RUN_CONFIG['retrain']:
                # Concatenate other stocks closing prices to current stock
                train_data_mutual = torch.cat([train_data] + [v[0][:,[config.DATA_CONFIG['predict_column']]] for k, v in stock_datas.items() if k != stock_name], axis=1)
                test_data_mutual = torch.cat([test_data] + [v[1][:,[config.DATA_CONFIG['predict_column']]] for k, v in stock_datas.items() if k != stock_name], axis=1)

                print('Training ' + stock_name + ' mutual model.............')
                model_mutual, test_losses_mutual, train_losses_mutual = train(train_data_mutual, test_data_mutual)
                print('==> Saving model ...')
                state = {'net': model_mutual.state_dict()}
                torch.save(state, filename)

                # Evaluate mutual model
                train_loss_mutual, train_preds_mutual = evaluate(model_mutual, train_data_mutual, nn.MSELoss())
                test_loss_mutual, test_preds_mutual = evaluate(model_mutual, test_data_mutual, nn.MSELoss())

                # Save data of losses and prediction for both train and test
                print('Test loss for ' + stock_name + ' mutual model = {:.5f}'.format(test_loss_mutual))
                # Save data of losses and prediction for both train and test of single and mutual models
                print('Saving results for ' + stock_name + ' mutual.............')
                torch.save([train_data_mutual, train_loss_mutual, train_preds_mutual,
                            test_data_mutual, test_loss_mutual, test_preds_mutual], path_results + stock_name + '_mutual_results.pth')
