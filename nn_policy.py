import time
import tokens
import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg


class Policy(nn.Module):

    def __init__(self, oracle=False):
        super(Policy, self).__init__()

        self.input_dim = tokens.count()
        self.output_dim = tokens.count()
        self.hidden_dim = cfg.g_cfg.hidden_dim
        self.dropout = cfg.g_cfg.dropout
        self.layers = cfg.g_cfg.layers
        
        self.gru = nn.GRU(self.input_dim,self.hidden_dim,self.layers,batch_first=True,dropout=self.dropout)
        self.lin = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1) # in forward (batch_size, num_features)

        if oracle:
            for param in self.parameters():
                torch.nn.init.normal_(param, 0, 1)

        self.probs = []
        self.rewards = []

        self.running_reward = 0.0
        
    def forward(self, x, h):
        
        out, h = self.gru(x, h)

        out = out[:,-1]
        out = self.lin(out)
        out = self.relu(out)
        out = self.softmax(out)

        return out, h
    
    def initial(self):

        batch = torch.zeros(cfg.app_cfg.batchsize, 1, self.input_dim)
        hidden = torch.zeros(self.layers, cfg.app_cfg.batchsize, self.hidden_dim)
        
        batch.to(cfg.app_cfg.device)
        hidden.to(cfg.app_cfg.device)
        batch.requires_grad = False

        return batch, hidden

    def save(self, file):

        torch.save(self.state_dict(), file)

    def load(self, file):

        self.load_state_dict(torch.load(file))

    def set_parameters_to(self, policy):

        self.load_state_dict(policy.state_dict())


class Oracle(Policy):

    def __init__(self):
        super(Oracle, self).__init__(oracle=True)

        self.eval()
        self.running_loss = 0.0
