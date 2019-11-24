from torch import nn
from config import config, paths
from helper import store

import tokens
import torch
import distribution


class Policy(nn.Module):
    """
    The policy net consists of two recurrent layers using GRUs, a fully connected layer and a softmax function so that
    the output can be seen as distribution over the possible token choices. The output will be a tensor of size
    (batch size, 1, one hot length) with each index of the one hot dimension representing a token choice.
    """

    def __init__(self):
        super(Policy, self).__init__()

        self.input_dim = tokens.count()
        self.output_dim = tokens.count()
        self.hidden_dim = config.g_hidden_dim
        self.dropout = config.g_dropout
        self.layers = config.g_layers

        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.layers, batch_first=True, dropout=self.dropout)
        self.lin = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # in forward (batch_size, num_features)
        self.optimizer = None

        if config.g_bias:
            bias = distribution.load(paths.bias_term)

            assert bias is not None
            assert len(bias) == self.output_dim

            self.bias = torch.tensor(bias).float()
            self.lin.bias = torch.nn.Parameter(self.bias, requires_grad=True)

    def forward(self, x, h):
        out, h = self.gru(x, h)

        out = out[:, -1]
        out = self.lin(out)
        out = self.relu(out)
        out = self.softmax(out)

        return out, h

    def initial(self):
        batch = torch.zeros(config.batch_size, 1, self.input_dim, device=config.device)
        hidden = torch.zeros(self.layers, config.batch_size, self.hidden_dim, device=config.device)

        return batch, hidden

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file):
        self.load_state_dict(torch.load(file, map_location=torch.device(config.device)))

    def set_parameters_to(self, policy):
        self.load_state_dict(policy.state_dict())


def step(policy, batch, hidden, save_prob=False):
    """
    This function performs a single step on the given policy net give a batch of unfinished subsequences.

    :param policy: The policy net which guides the decision making process.
    :param batch: The batch of input sequences size (batch size, sequence length, onehot length).
    :param hidden: The hidden state of the policy net.
    :param save_prob: If true, the probabilities for the chosen action will be saved for the policy net. Should be true
        if it is a step in the policy net training and false if it is a rollout or sample step.
    :return: Returns batch, hidden with the new encoding tensors for the chosen actions.
    """

    # avoid feeding whole sequences redundantly
    state = batch[:, -1, :][:, None, :]
    policies, hidden = policy(state, hidden)

    # sample next actions
    distributions = torch.distributions.Categorical(policies)
    actions = distributions.sample()

    # save log probabilities for gradient computation
    if save_prob:
        store.get('List: Mean Policies Per Single Step').append(torch.mean(policies, dim=0))
        store.get('List: Mean Entropies Per Single Step').append(torch.mean(distributions.entropy(), dim=0))
        store.get('List: Sampled Actions Per Single Step').append(actions)
        store.get('List: Log Probabilites Per Actions Of Single Step').append(distributions.log_prob(actions))

    # concat onehot tokens with the batch of sequences
    encodings = torch.tensor([tokens.onehot(id) for id in actions], device=config.device)
    encodings = encodings[:, None, :].float()
    batch = torch.cat((batch, encodings), dim=1)

    # if batch still has the empty start token remove it
    if torch.sum(batch[:, 0, :]) == 0:
        batch = batch[:, 1:, :]

    return batch, hidden


def rollout(policy, batch, hidden):
    """
    This function finished a sequence for a given subsequence without saving probabilities or gradients.

    :param policy: The policy guiding the decision making process.
    :param batch: The batch of subsequences to finish.
    :param hidden: The current hidden state of the net.
    :return: Returns a batch of finished sequencens, tensor of size (batchsize, sequence length, onehot length).
    """

    with torch.no_grad():
        while batch.shape[1] < config.sequence_length:

            batch, hidden = step(policy, batch, hidden)

    return batch


def sample(policy, num_batches):
    """
    This function samples finished sequences for a given policy.

    :param policy: The policy guiding the decision making process.
    :param num_batches: The amount of batches to generate.
    :return: Returns a python list of tensors of size (batch size, sequence length, onehot length).
    """

    batch = torch.empty((0, config.sequence_length, tokens.count()), device=config.device)

    with torch.no_grad():
        for _ in range(num_batches):
            out, hidden = policy.initial()
            out, hidden = step(policy, out, hidden)
            out = rollout(policy, out, hidden)

            batch = torch.cat([batch, out], dim=0)

    return batch
