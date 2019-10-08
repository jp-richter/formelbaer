from numpy.random import choice
from numpy import empty, finfo, float32
from nn_policy import PolicyNetwork, learnrate, baseline, gamma

import tokens
import torch


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
policy_net = PolicyNetwork().to(device)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=learnrate)
eps = finfo(float32).eps.item()

rollout_net = PolicyNetwork().to(device).eval()
rollout_net.load_state_dict(policy_net.state_dict())


def step(batch_size, batch=None, h=None):

    batch = torch.zeros(batch_size,1,tokens.count()) if batch is None else batch
    h = policy_net.init_hidden(batch.shape[0]) if h is None else h

    policy_net.train()

    batch.requires_grad = False
    batch.to(device)

    policies, actions, batch, h = decision(policy_net, batch, h)
    policy_net.probs.append(policies.log_prob(actions))

    return batch, h


def rollout(batch_size, sequence_length, batch=None, h=None):

    with torch.no_grad():

        batch = torch.zeros([batch_size,1,tokens.count()]) if batch is None else batch
        h = rollout_net.init_hidden(batch.shape[0]) if h is None else h

        for _ in range(sequence_length - batch.shape[1]):
            _, _, batch, h = decision(rollout_net, batch, h)

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

    policy_net.rewards.append(reward)


def updatepolicy():

    total = 0
    policy_loss = []
    returns = []

    for r in policy_net.rewards[::-1]:
        total = r + gamma * total
        returns.insert(0, total)

    returns = returns[0]
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for log_prob, total in zip(policy_net.probs, returns):
        policy_loss.append(-log_prob * total)

    optimizer.zero_grad()

    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()

    optimizer.step()

    del policy_net.rewards[:]
    del policy_net.probs[:]


def updaterollout():

    rollout_net.load_state_dict(policy_net.state_dict())
