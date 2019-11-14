from torch import nn

import tokens
import torch
import os
import distribution
import config as config


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
        self.hidden_dim = config.generator.hidden_dim
        self.dropout = config.generator.dropout
        self.layers = config.generator.layers

        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.layers, batch_first=True, dropout=self.dropout)
        self.lin = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # in forward (batch_size, num_features)

        self.probs = []
        self.rewards = []
        self.entropies = []

        self.running_loss = 0.0
        self.loss_divisor = 0
        self.running_reward = 0.0
        self.reward_divisor = 0
        self.running_prediction = 0.0
        self.prediction_divisor = 0

        self.optimizer = None

        if config.generator.bias:
            bias = distribution.load(config.paths.distribution_bias)
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
        batch = torch.zeros(config.general.batch_size, 1, self.input_dim, device=config.general.device)
        hidden = torch.zeros(self.layers, config.general.batch_size, self.hidden_dim, device=config.general.device)

        return batch, hidden

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file):
        self.load_state_dict(torch.load(file, map_location=torch.device(config.general.device)))

    def set_parameters_to(self, policy):
        self.load_state_dict(policy.state_dict())


class Oracle(Policy):
    """
    This class represents an oracle. An oracle is a fake real distribution of data which can be very useful to make
    statements about performance of a policy. If a policy gets trained to model an oracle the oracle distribution is
    well known in contrast to real data. In theory an unlimited amount of real samples can be generated and measuring
    the exact difference of distributions is possible. Obviously the oracle does not produce any semantically
    meaningful data. The weights get initialized following a normal distribution to garantuee variance in the parameters
    to avoid creating similar policy and oracle models.
    """

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
        entropy = policies.entropy()
        nn_policy.probs.append(log_probs)
        nn_policy.entropies.append(entropy)

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

    batch = torch.empty((0, config.general.sequence_length, len(tokens.possibilities())), device=config.general.device)

    with torch.no_grad():
        for _ in range(num_batches):
            out, hidden = nn_policy.initial()
            out, hidden = step(nn_policy, out, hidden)
            out = rollout(nn_policy, out, hidden)

            batch = torch.cat([batch, out], dim=0)

    return batch


def policy_gradient_update(nn_policy):
    """
    This function adjusts the parameters of the give policy net with the REINFORCE algorithm.

    :param nn_policy: The net which parameters should be updated.
    """

    nn_policy.optimizer.zero_grad()

    # assumption: policy stores lists with tensors of size (batchsize) of length (steps until update)
    assert len(nn_policy.rewards) == len(nn_policy.probs)
    assert all(tensor.size() == (config.general.batch_size,) for tensor in nn_policy.probs)
    assert all(tensor.size() == (config.general.batch_size,) for tensor in nn_policy.rewards)

    # weight state action values by log probability of action
    total = torch.zeros(config.general.batch_size, device=config.general.device)
    reward = torch.zeros(config.general.batch_size, device=config.general.device)
    reward_without_log = torch.zeros(config.general.batch_size, device=config.general.device)

    for log, rew in zip(nn_policy.probs, nn_policy.rewards):
        reward_without_log = reward_without_log + rew
        total = total + (rew - config.generator.baseline)
        reward = reward + (log * total)

    # actual task is to maximize this value
    reward_without_log = torch.sum(reward_without_log)
    reward_without_log = reward_without_log / config.general.batch_size

    # average log prob * reward over batchsize
    reward = torch.sum(reward)
    reward = reward / config.general.batch_size

    # final prediction / equals reward if update every step
    prediction = nn_policy.rewards[-1]
    prediction = torch.sum(prediction) / config.general.batch_size

    # negate for gradient descent and substract entropy
    entropy = torch.stack(nn_policy.entropies, dim=1).to(config.general.device)
    entropy = torch.sum(entropy, dim=1)
    entropy = torch.sum(entropy) / config.general.batch_size
    entropy = entropy * 0.005

    loss = - (reward + entropy)
    loss.backward()
    nn_policy.optimizer.step()

    nn_policy.running_loss += loss.item()
    nn_policy.loss_divisor += 1
    nn_policy.running_reward += reward_without_log.item()
    nn_policy.reward_divisor += 1
    nn_policy.running_prediction += prediction.item()
    nn_policy.prediction_divisor += 1

    del nn_policy.rewards[:]
    del nn_policy.probs[:]
    del nn_policy.entropies[:]
