from utils import *
from config import config
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def run_compare():
    single_loss_train = []
    mututal_loss_train = []
    single_loss_test = []
    mututal_loss_test = []
    names = []
    names_stocks = []

    if config.RUN_CONFIG['compare stocks'] == ['All']:
        group_names = pd.read_csv("config/StockNames.csv").columns
    else:
        group_names = config.RUN_CONFIG['compare stocks']

    StockNames = pd.read_csv(config.DATA_CONFIG['stock_names_file'])
    for group_name in group_names:
        path_results = './data/results/' + group_name + '/'
        for stock_name in StockNames[group_name][StockNames[group_name].notna()]:
            names_stocks.append(stock_name)
            names.append(group_name)
            # Load results of single and mutual models for the current stock
            try:
                train_data, train_loss_single, train_preds_single, test_data, test_loss_single, test_preds_single = torch.load(path_results + stock_name + '_single_results.pth')
                train_data_mutual, train_loss_mutual, train_preds_mutual, test_data_mutual, test_loss_mutual, test_preds_mutual = torch.load(path_results + stock_name + '_mutual_results.pth')
            except:
                raise RuntimeError('Missing ' + group_name + ' - ' + stock_name + ' model, reconfig and run train')
            
            # Append losses to lists
            single_loss_train.append(train_loss_single)
            single_loss_test.append(test_loss_single)
            mututal_loss_train.append(train_loss_mutual)
            mututal_loss_test.append(test_loss_mutual)

    names = np.array(names)
    inds = np.where(names[:-1] != names[1:])[0]

    # Calculate % change from single model to mutual model for train set and display
    loss_changes_train = np.array([100*(1-l1/l2) for l1,l2 in zip(single_loss_train,mututal_loss_train)])
    plt.figure()
    plt.bar(np.arange(len(loss_changes_train)), loss_changes_train)
    plt.legend()
    plt.axhline(0, c='r')
    [plt.axvline(ind+0.5, c='gray', alpha=0.25, linestyle='--') for ind in inds]
    if 'Unrelated 1' in names:
        plt.axvline(np.where(names=='Unrelated 1')[0][0]-0.5, c='orange', alpha=0.5, linestyle='--')
    plt.title('Train Loss Change')
    plt.xticks(np.arange(len(loss_changes_train)), names, rotation=90)
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    for i, v in enumerate(loss_changes_train):
        plt.text(i, v+0.1*np.sign(v), names_stocks[i], color='k', fontweight='bold', fontsize=8, horizontalalignment='center', alpha=0.7, rotation='vertical')

    # Calculate % change from single model to mutual model for test set and display
    loss_changes_test = np.array([100*(1-l1/l2) for l1,l2 in zip(single_loss_test,mututal_loss_test)])
    plt.figure()
    plt.bar(np.arange(len(loss_changes_test)), loss_changes_test)
    plt.legend()
    plt.axhline(0, c='r')
    [plt.axvline(ind+0.5, c='gray', alpha=0.25, linestyle='--') for ind in inds]
    if 'Unrelated 1' in names:
        plt.axvline(np.where(names=='Unrelated 1')[0][0]-0.5, c='orange', alpha=0.5, linestyle='--')
    plt.title('Test Loss Change')
    plt.xticks(np.arange(len(loss_changes_test)), names, rotation=90)
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    for i, v in enumerate(loss_changes_test):
        plt.text(i, v+0.1*np.sign(v), names_stocks[i], color='k', fontweight='bold', fontsize=8, horizontalalignment='center', alpha=0.7, rotation='vertical')

    # Calculate mean % change from single model to mutual model for test set and display for each group
    avg_changes_train = [np.mean(loss_changes_train[np.where(names==name)[0]]) for name in np.unique(names)]
    avg_changes_test = [np.mean(loss_changes_test[np.where(names==name)[0]]) for name in np.unique(names)]
    plt.figure()
    plt.bar(np.arange(len(avg_changes_train))-0.125, avg_changes_train, label='train', width=0.25)
    plt.bar(np.arange(len(avg_changes_train))+0.125, avg_changes_test, label='test', width=0.25)
    plt.legend()
    plt.axhline(0, c='r')
    if 'Unrelated 1' in names:
        plt.axvline(np.where(np.unique(names) =='Unrelated 1')[0][0]-0.5, c='orange', alpha=0.5, linestyle='--')
    plt.title('Average Loss Change by Group', fontsize=15)
    plt.xticks(np.arange(len(avg_changes_train)), np.unique(names), rotation=45, fontsize=15)
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())

    plt.show()
