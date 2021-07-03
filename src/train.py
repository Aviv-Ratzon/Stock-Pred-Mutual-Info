import time
import torch
from model import *
import config
from utils import *


def train(train_data, test_data):
    n_hid = config.MODEL_CONFIG['n_hid']
    dropout = config.MODEL_CONFIG['dropout']
    n_layers = config.MODEL_CONFIG['n_layers']
    device = config.TRAIN_CONFIG['device']
    lr = config.TRAIN_CONFIG['lr']
    num_epochs = config.TRAIN_CONFIG['num_epochs']
    gamma = config.TRAIN_CONFIG['gamma']

    model = TransformerModel(train_data.shape[-1], train_data.shape[-1]+2, n_hid, n_layers, dropout).to(device)
    print('Number of params = {:d}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    optim = 'SGD'
    criterion = nn.MSELoss()
    optimizer = getattr(torch.optim, optim)(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=gamma)

    test_losses = []
    train_losses = []
    for epoch in range(num_epochs):
        train_loss = train_epoch(train_data, model, criterion, optimizer, scheduler, epoch)
        train_losses.extend(train_loss)
        scheduler.step()
        test_losses.append(evaluate(model, test_data, criterion)[0])
        print('| finished epoch {:d}/{:d} | Train loss {:f}'.format(
            epoch+1, num_epochs, test_losses[-1]))
    return model, test_losses, train_losses


def train_epoch(train_data, model, criterion, optimizer, scheduler, epoch):
    model.train()  # Turn on the train mode
    seq_len = config.TRAIN_CONFIG['seq_len']
    bsz = config.TRAIN_CONFIG['bsz']
    log_interval = config.TRAIN_CONFIG['log_interval']
    total_loss = 0.
    start_time = time.time()
    loss = 0
    losses = []
    for batch, i in enumerate(range(0, train_data.size(0) - seq_len - 1)):
        data, targets = get_batch(train_data, i)

        output = model(data)
        loss += criterion(output, targets)

        # Run bsz sequences before backward step to get batch effect
        if batch % bsz == 0 or batch == train_data.size(0) - seq_len - 2:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            loss = 0

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} sequences | '
                  'lr {:f} | ms/interval {:.3f} | '
                  'train loss {:f}'.format(
                epoch+1, batch, len(train_data), scheduler.get_last_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss))
            losses.append(cur_loss)
            total_loss = 0
            start_time = time.time()
    return losses



