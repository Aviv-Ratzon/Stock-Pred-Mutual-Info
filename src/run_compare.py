from utils import *
import config
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')


seq_len = config.TRAIN_CONFIG['seq_len']
predict_column = config.DATA_CONFIG['predict_column']

def run_compare():
    single_loss_train_vec = []
    single_loss_test_vec = []
    mututal_loss_train_vec = []
    mututal_loss_test_vec = []
    target_test_vec = []
    single_pred_test_vec = []
    mutual_pred_test_vec = []
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
                train_data, single_loss_train, single_pred_train, test_data, single_loss_test, single_pred_test = torch.load(path_results + stock_name + '_single_results.pth')
                train_data, mututal_loss_train, mututal_pred_train, test_data, mututal_loss_test, mutual_pred_test = torch.load(path_results + stock_name + '_mutual_results.pth')
            except:
                raise RuntimeError('Missing ' + group_name + ' - ' + stock_name + ' model, reconfig and run train')

            # Append losses to lists
            single_loss_train_vec.append(single_loss_train)
            single_loss_test_vec.append(single_loss_test)
            mututal_loss_train_vec.append(mututal_loss_train)
            mututal_loss_test_vec.append(mututal_loss_test)
            target_test_vec.append(test_data[seq_len:, predict_column])
            single_pred_test_vec.append(single_pred_test)
            mutual_pred_test_vec.append(mutual_pred_test)

    names = np.array(names)
    inds = np.where(names[:-1] != names[1:])[0]

    # Calculate % change from single model to mutual model for train set and display
    loss_changes_train = np.array([100*(1-l1/l2) for l1,l2 in zip(single_loss_train_vec,mututal_loss_train_vec)])
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
    loss_changes_test = np.array([100*(1-l1/l2) for l1,l2 in zip(single_loss_test_vec,mututal_loss_test_vec)])
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


    ind_best = loss_changes_test.argmin()
    ind_worst = loss_changes_test.argmax()

    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.plot(target_test_vec[ind_best], label='target')
    ax1.plot(single_pred_test_vec[ind_best], label='single model pred')
    ax1.plot(mutual_pred_test_vec[ind_best], label='mutual model pred')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('pct change')
    ax1.set_title('Most Improved Model')
    ax1.legend()

    ax2.plot(target_test_vec[ind_worst], label='target')
    ax2.plot(single_pred_test_vec[ind_worst], label='single model pred')
    ax2.plot(mutual_pred_test_vec[ind_worst], label='mutual model pred')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('pct change')
    ax2.set_title('Least Improved Model')
    ax2.legend()


    plt.show()
