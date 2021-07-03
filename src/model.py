import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import config


class Time2Vector(nn.Module):
    def __init__(self):
        super(Time2Vector, self).__init__()
        seq_len = config.TRAIN_CONFIG['seq_len']

        '''Initialize weights and biases with shape (batch, seq_len)'''
        self.weights_linear = nn.parameter.Parameter(torch.rand(seq_len))
        self.bias_linear = nn.parameter.Parameter(torch.rand(seq_len))

        self.weights_periodic = nn.parameter.Parameter(torch.rand(seq_len))
        self.bias_periodic = nn.parameter.Parameter(torch.rand(seq_len))

    def forward(self, x):
        '''Calculate linear and periodic time features'''
        x = torch.mean(x[:, :4], dim=-1)
        time_linear = self.weights_linear * x + self.bias_linear  # Linear time feature
        time_linear = time_linear.unsqueeze(dim=-1)  # Add dimension (batch, seq_len, 1)

        time_periodic = torch.sin(torch.dot(x, self.weights_periodic) + self.bias_periodic)
        time_periodic = time_periodic.unsqueeze(dim=-1)  # Add dimension (batch, seq_len, 1)
        return torch.cat([time_linear, time_periodic], dim=-1)  # shape = (batch, seq_len, 2)


class TransformerModel(nn.Module):
    def __init__(self, ninp, nhead, nhid, nlayers, dropout):
        super(TransformerModel, self).__init__()
        self.temp_encoder = Time2Vector()
        encoder_layers = TransformerEncoderLayer(ninp+2, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.decoder = nn.Linear(config.TRAIN_CONFIG['seq_len'], 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = torch.cat([src, self.temp_encoder(src)], dim=-1)
        output = self.transformer_encoder(src.unsqueeze(1))
        output = self.decoder(output.mean(dim=-1).T)
        return output
