import time
import tokens
import torch
import torch.nn as nn
import torch.nn.functional as F
import constants


input_dim = tokens.count()
output_dim = tokens.count()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class PolicyNetwork(nn.Module):

    def __init__(self):
        super(PolicyNetwork, self).__init__()
        
        self.gru = nn.GRU(input_dim,
            constants.GENERATOR_HIDDEN_DIM,
            constants.GENERATOR_LAYERS,
            batch_first=True,
            dropout=constants.GENERATOR_DROPOUT)

        self.lin = nn.Linear(constants.GENERATOR_HIDDEN_DIM, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

        self.probs = []
        self.rewards = []
        
    def forward(self, x, h):
        
        out, h = self.gru(x, h)

        out = out[:,-1]
        out = self.lin(out)
        out = self.relu(out)
        out = self.softmax(out)

        return out, h
    
    def init_hidden(self):

        return torch.zeros(constants.GENERATOR_LAYERS, 
            constants.ADVERSARIAL_BATCHSIZE, 
            constants.GENERATOR_HIDDEN_DIM).to(device)
