import time
import tokens
import torch
import torch.nn as nn
import torch.nn.functional as F


input_dim = tokens.count()
output_dim = tokens.count()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class PolicyNetwork(nn.Module):

    def __init__(self, hidden_dim, layers, dropout):
        super(PolicyNetwork, self).__init__()
        
        self.gru = nn.GRU(input_dim,hidden_dim,layers,batch_first=True,dropout=dropout)

        self.lin = nn.Linear(hidden_dim, output_dim)
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
    
    def init_hidden(self, layers, batch_size, hidden_dim):

        return torch.zeros(layers, batch_size, hidden_dim).to(device)
