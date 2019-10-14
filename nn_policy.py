import time
import tokens
import torch
import torch.nn as nn
import torch.nn.functional as F
import constants


class Policy(nn.Module):

    def __init__(self, oracle=False):
        super(Policy, self).__init__()

        self.input_dim = tokens.count()
        self.output_dim = tokens.count()
        self.hidden_dim = constants.GENERATOR_HIDDEN_DIM
        self.dropout = constants.GENERATOR_DROPOUT
        self.layers = constants.GENERATOR_LAYERS
        
        self.gru = nn.GRU(input_dim,self.hidden_dim,self.layers,batch_first=True,self.dropout)
        self.lin = nn.Linear(self.hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

        if oracle:
            for param in self.parameters():
                torch.nn.init.normal(p, 0, 1)

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

        batch = torch.zeros(constants.ADVERSARIAL_BATCHSIZE,1,self.input_dim)
        hidden = torch.zeros(self.layers,constants.ADVERSARIAL_BATCHSIZE,self.hidden_dim)
        
        if torch.cuda.is_available():
            batch.to('cuda')
            hidden.to('cuda')
        else:
            batch.to('cpu')
            hidden.to('cpu')

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
        super(self).__init__(oracle=True)

        self.eval()
        self.running_loss = 0.0
