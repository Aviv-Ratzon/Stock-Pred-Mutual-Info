from utils import *
import config
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
plt.style.use('seaborn')


seq_len = config.TRAIN_CONFIG['seq_len']
predict_column = config.DATA_CONFIG['predict_column']
if not os.path.exists('images/'):
    os.makedirs('images/')

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
    loss_changes_train = np.array([100*(1-l1/l2) for l1,l2 in zip(single_loss_train_vec,mututal_loss_train_vec)])
    loss_changes_test = np.array([100*(1-l1/l2) for l1,l2 in zip(single_loss_test_vec,mututal_loss_test_vec)])
    avg_changes_train = [np.mean(loss_changes_train[np.where(names==name)[0]]) for name in np.unique(names)]
    avg_changes_test = [np.mean(loss_changes_test[np.where(names==name)[0]]) for name in np.unique(names)]
    if 'Unrelated 1' in names:
        ind_split = np.where(names=='Unrelated 1')[0][0]
        # Calculate % change from single model to mutual model for train set and display
        mean_changes_train =[np.mean(loss_changes_train[:ind_split]), np.mean(loss_changes_train[ind_split:])]
        mean_changes_test = [np.mean(loss_changes_test[:ind_split]), np.mean(loss_changes_test[ind_split:])]
        std_changes_train = [np.std(loss_changes_train[:ind_split]), np.std(loss_changes_train[ind_split:])]
        std_changes_test = [np.var(loss_changes_test[:ind_split]), np.std(loss_changes_test[ind_split:])]
        fig = plt.figure(figsize=(12,7))
        plt.bar([0.25, 1.25], mean_changes_train, width=0.25, label='train', capsize=2)
        plt.bar([0.75, 1.75], mean_changes_test, width=0.25, label='test')
        plt.errorbar([0.25, 1.25], mean_changes_train, yerr=std_changes_train, fmt='o', c='k', label='std')
        plt.errorbar([0.75, 1.75], mean_changes_test, yerr=std_changes_test, fmt='o', c='k')
        plt.text(0.5, 0.5, 'N='+str(ind_split), color='k', fontweight='bold', fontsize=12, horizontalalignment='center')
        plt.text(1.5, 0.5, 'N='+str(len(names)-ind_split), color='k', fontweight='bold', fontsize=12, horizontalalignment='center')
        plt.legend(fontsize=14)
        plt.axhline(0, c='gray')
        plt.axvline(1, c='orange', alpha=0.5, linestyle='--')
        plt.title('Loss Change Across Related and Unrelated Groups', fontsize=20)
        plt.xticks([0.5, 1.5], ['Related', 'Unrelated'], fontsize=14)
        plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
        plt.ylabel('% loss change', fontsize=14)
        plt.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.2)
        fig.savefig('images/LossChangeStats.png')

    inds_ticks = np.append(np.insert(inds, 0, 0), len(names))
    xticks = inds_ticks[:-1] + np.diff(inds_ticks)/2
    # Calculate % change from single model to mutual model for train set and display
    fig = plt.figure(figsize=(12,7))
    plt.bar(np.arange(len(loss_changes_train)), loss_changes_train)
    plt.legend(fontsize=14)
    plt.axhline(0, c='gray')
    [plt.axvline(ind+0.5, c='gray', alpha=0.25, linestyle='--') for ind in inds]
    if 'Unrelated 1' in names:
        plt.axvline(np.where(names=='Unrelated 1')[0][0]-0.5, c='orange', alpha=0.5, linestyle='--')
    plt.title('Train Loss Change', fontsize=20)
    # plt.xticks(np.arange(len(loss_changes_train)), names, rotation=90, fontsize=14)
    plt.xticks(xticks, names[inds_ticks[:-1]+1], rotation=45, fontsize=14)
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    plt.ylabel('% loss change', fontsize=14)
    for i, v in enumerate(loss_changes_train):
        plt.text(i, v+0.1*np.sign(v), names_stocks[i], color='k', fontweight='bold', fontsize=8, horizontalalignment='center', alpha=0.7, rotation='vertical')
    plt.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.2)
    fig.savefig('images/TrainLossChange.png')

    # Calculate % change from single model to mutual model for test set and display
    fig = plt.figure(figsize=(12,7))
    plt.bar(np.arange(len(loss_changes_test)), loss_changes_test)
    plt.legend(fontsize=14)
    plt.axhline(0, c='gray')
    [plt.axvline(ind+0.5, c='gray', alpha=0.25, linestyle='--') for ind in inds]
    if 'Unrelated 1' in names:
        plt.axvline(np.where(names=='Unrelated 1')[0][0]-0.5, c='orange', alpha=0.5, linestyle='--')
    plt.title('Test Loss Change', fontsize=20)
    # plt.xticks(np.arange(len(loss_changes_train)), names, rotation=90, fontsize=14)
    plt.xticks(xticks, names[inds_ticks[:-1]+1], rotation=45, fontsize=14)
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    plt.ylabel('% loss change', fontsize=14)
    for i, v in enumerate(loss_changes_test):
        plt.text(i, v+0.1*np.sign(v), names_stocks[i], color='k', fontweight='bold', fontsize=8, horizontalalignment='center', alpha=0.7, rotation='vertical')
    plt.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.2)
    fig.savefig('images/TestLossChange.png')

    # Calculate mean % change from single model to mutual model for test set and display for each group
    fig = plt.figure(figsize=(12,7))
    plt.axhline(0, c='gray')
    plt.bar(np.arange(len(avg_changes_train))-0.125, avg_changes_train, label='train', width=0.25)
    plt.bar(np.arange(len(avg_changes_train))+0.125, avg_changes_test, label='test', width=0.25)
    plt.legend(fontsize=14)
    if 'Unrelated 1' in names:
        plt.axvline(np.where(np.unique(names) =='Unrelated 1')[0][0]-0.5, c='orange', alpha=0.5, linestyle='--')
    plt.title('Average Loss Change by Group', fontsize=20)
    plt.xticks(np.arange(len(avg_changes_train)), np.unique(names), rotation=45, fontsize=14)
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    plt.ylabel('% loss change', fontsize=14)
    plt.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.2)
    fig.savefig('images/LossChangeByGroup.png')


    # ind_best = loss_changes_test.argmin()
    # ind_worst = loss_changes_test.argmax()
    #
    # fig, (ax1, ax2) = plt.subplots(2,1)
    # ax1.plot(target_test_vec[ind_best], label='target')
    # ax1.plot(single_pred_test_vec[ind_best], label='single model pred')
    # ax1.plot(mutual_pred_test_vec[ind_best], label='mutual model pred')
    # ax1.set_xlabel('Date')
    # ax1.set_ylabel('pct change')
    # ax1.set_title('Most Improved Model')
    # ax1.legend()
    #
    # ax2.plot(target_test_vec[ind_worst], label='target')
    # ax2.plot(single_pred_test_vec[ind_worst], label='single model pred')
    # ax2.plot(mutual_pred_test_vec[ind_worst], label='mutual model pred')
    # ax2.set_xlabel('Date')
    # ax2.set_ylabel('pct change')
    # ax2.set_title('Least Improved Model')
    # ax2.legend()
    #
    #
    # plt.show()