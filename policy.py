import time
import constants
import torch
import torch.nn as nn
import torch.nn.functional as F


_device = constants.DEVICE

_input_dim = constants.INPUT_DIM
_output_dim = constants.OUTPUT_DIM
_hidden_dim = constants.HIDDEN_DIM
_layers = constants.GRU_LAYERS
_dropout = constants.DROP_OUT


class PolicyNetwork(nn.Module):

    def __init__(self):
        super(PolicyNetwork, self).__init__()
        
        self.gru = nn.GRU(_input_dim, _hidden_dim, _layers, batch_first=True, dropout=_dropout)
        self.lin = nn.Linear(_hidden_dim, _output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.probs = []
        self.rewards = []
        
    def forward(self, x, h):
        out, h = self.gru(x, h)

        out = out[:,-1]
        out = self.lin(out)
        out = self.relu(out)
        out = self.softmax(out)

        return out, h
    
    def init_hidden(self, batch_size):

        return torch.zeros(_layers, batch_size, _hidden_dim).to(_device)
