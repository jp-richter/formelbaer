from torch import nn

import tokens
import torch
import os
import config as config


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()

        self.input_dim = tokens.count()
        self.output_dim = tokens.count()
        self.hidden_dim = config.generator.hidden_dim
        self.dropout = config.generator.dropout
        self.layers = config.generator.layers

        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.layers, batch_first=True, dropout=self.dropout)
        self.lin = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # in forward (batch_size, num_features)

        self.probs = []
        self.rewards = []
        self.running_reward = 0.0
        self.optimizer = None

    def forward(self, x, h):
        out, h = self.gru(x, h)

        out = out[:, -1]
        out = self.lin(out)
        out = self.relu(out)
        out = self.softmax(out)

        return out, h

    def initial(self):
        batch = torch.zeros(config.general.batch_size, 1, self.input_dim, device=config.general.device)
        hidden = torch.zeros(self.layers, config.general.batch_size, self.hidden_dim, device=config.general.device)

        return batch, hidden

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file):
        self.load_state_dict(torch.load(file))

    def set_parameters_to(self, policy):
        self.load_state_dict(policy.state_dict())


class Oracle(Policy):

    def __init__(self):
        super(Oracle, self).__init__()

        self.eval()
        self.running_score = 0.0
        self.criterion = None

        if not os.path.exists(config.paths.oracle):
            [torch.nn.init.normal_(param, 0, 1) for param in self.parameters()]
            self.save(config.paths.oracle)

        else:
            self.load(config.paths.oracle)


def step(nn_policy, batch, hidden, save_prob=False):
    """
    This function performs a single step on the given policy net give a batch of unfinished subsequences.

    :param nn_policy: The policy net which guides the decision making process.
    :param batch: The batch of input sequences size (batch size, sequence length, onehot length).
    :param hidden: The hidden state of the policy net.
    :param save_prob: If true, the probabilities for the chosen action will be saved for the policy net. Should be true
        if it is a step in the policy net training and false if it is a rollout or sample step.
    :return: Returns batch, hidden with the new encoding tensors for the chosen actions.
    """

    # avoid feeding whole sequences redundantly
    state = batch[:, -1, :][:, None, :]
    policies, hidden = nn_policy(state, hidden)

    # sample next actions
    policies = torch.distributions.Categorical(policies)
    actions = policies.sample()

    # save log probabilities for gradient computation
    if save_prob:
        log_probs = policies.log_prob(actions)
        nn_policy.probs.append(log_probs)

    # concat onehot tokens with the batch of sequences
    encodings = torch.tensor([tokens.onehot(id) for id in actions], device=config.general.device)
    encodings = encodings[:, None, :].float()
    batch = torch.cat((batch, encodings), dim=1)

    # if batch still has the empty start token remove it
    if torch.sum(batch[:, 0, :]) == 0:
        batch = batch[:, 1:, :]

    return batch, hidden


def rollout(nn_policy, batch, hidden):
    """
    This function finished a sequence for a given subsequence without saving probabilities or gradients.

    :param nn_policy: The policy guiding the decision making process.
    :param batch: The batch of subsequences to finish.
    :param hidden: The current hidden state of the net.
    :return: Returns a batch of finished sequencens, tensor of size (batchsize, sequence length, onehot length).
    """

    with torch.no_grad():
        while batch.shape[1] < config.general.sequence_length:
            batch, hidden = step(nn_policy, batch, hidden)

    return batch


def sample(nn_policy, num_batches):
    """
    This function samples finished sequences for a given policy.

    :param nn_policy: The policy guiding the decision making process.
    :param num_batches: The amount of batches to generate.
    :return: Returns a python list of tensors of size (batch size, sequence length, onehot length).
    """

    batches = []

    with torch.no_grad():
        for _ in range(num_batches):
            batch, hidden = nn_policy.initial()
            batch, hidden = step(nn_policy, batch, hidden)
            batch = rollout(nn_policy, batch, hidden)
            batches.append(batch)

    return batches


def policy_gradient_update(nn_policy):
    """
    This function adjusts the parameters of the give policy net with the policy gradient method. The respective returns
    and probabilities should be saved in the policy.rewards and policy.probs attributes and will be deleted after the
    update.

    :param nn_policy: The net which parameters should be updated.
    """

    total = 0
    loss = []
    returns = []

    nn_policy.optimizer.zero_grad()

    # compute state action values for each step
    for reward in nn_policy.rewards[::-1]:
        total = reward + config.generator.gamma * total
        returns.insert(0, total)

    assert len(nn_policy.probs) == len(returns)

    # weight state action values by log probability of action
    for log_prob, reward in zip(nn_policy.probs, returns):
        loss.append(-log_prob * reward)

    # sum rewards over all steps for each sample
    loss = torch.stack(loss)
    loss = torch.sum(loss, dim=1)

    # average rewards over batch
    batch_size = loss.shape[0]
    loss = torch.sum(loss) / batch_size

    loss.backward()
    nn_policy.optimizer.step()

    nn_policy.running_reward += -1 * loss.item()

    del nn_policy.rewards[:]
    del nn_policy.probs[:]
