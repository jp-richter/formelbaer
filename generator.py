from numpy.random import choice
from numpy import empty, finfo, float32
from policy import PolicyNetwork

import tokens
import torch
import constants


_policy = PolicyNetwork().to(constants.DEVICE)
_optimizer = torch.optim.Adam(_policy.parameters(), lr=constants.LEARN_RATE)
_eps = finfo(float32).eps.item()

_default_batch_size = constants.BATCH_SIZE
_default_batch = torch.zeros([_default_batch_size,1,tokens.count()])

_rollout = PolicyNetwork().to(constants.DEVICE).eval()
_rollout.load_state_dict(_policy.state_dict())


def step(batch=None, h=None):

    batch = _default_batch if batch is None else batch
    h = _policy.init_hidden(batch.shape[0]) if h is None else h

    _policy.train()

    batch.requires_grad = False
    batch.to(constants.DEVICE)

    policies, actions, batch, h = decision(_policy, batch, h)
    _policy.probs.append(policies.log_prob(actions))

    return batch, h


def rollout(batch=None, h=None):

    batch = _default_batch if batch is None else batch
    h = _policy.init_hidden(batch.shape[0]) if h is None else h

    for _ in range(constants.SEQ_LENGTH - batch.shape[1]):
        _, _, batch, h = decision(_rollout, batch, h)

    return batch


def decision(net, batch, h):

    state = batch[:,-1,:][:,None,:]
    policies, h = net(state, h)
    policies = torch.distributions.Categorical(policies)
    actions = policies.sample() 

    encodings = torch.Tensor(list(map(tokens.onehot, actions)))
    encodings = encodings[:,None,:]
    batch = torch.cat((batch,encodings),dim=1)

    return policies, actions, batch, h


def feedback(reward):

    _policy.rewards.append(reward)


def update_policy():

    total = 0
    policy_loss = []
    returns = []

    for r in _policy.rewards[::-1]:
        total = r + constants.GAMMA * total
        returns.insert(0, total)

    returns = returns[0]
    returns = (returns - returns.mean()) / (returns.std() + _eps)

    for log_prob, total in zip(_policy.probs, returns):
        policy_loss.append(-log_prob * total)

    _optimizer.zero_grad()

    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()

    _optimizer.step()

    del _policy.rewards[:]
    del _policy.probs[:]


def update_rollout():

    _rollout.load_state_dict(_policy.state_dict())
