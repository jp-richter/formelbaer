from numpy.random import choice
from numpy import empty, finfo, float32
from nn_policy import PolicyNetwork

import tokens
import torch

hidden_dim = 32
layers = 2
dropout = 0.2
learnrate = 0.01
baseline = 1
gamma = 0.95

running_loss = 0.0

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

policy_net = PolicyNetwork(hidden_dim, layers, dropout).to(device)
rollout_net = PolicyNetwork(hidden_dim, layers, dropout).to(device).eval()
rollout_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.Adam(policy_net.parameters(), lr=learnrate)
eps = finfo(float32).eps.item()


def step(batch=None, hidden=None, batch_size=None):

    if batch_size is None and batch is None:
        raise ValueError('Either specify a batch size or provide a batch.')

    # generate a new batch of sequences
    if batch == None:
        optimizer.zero_grad()

        batch = torch.zeros(batch_size,1,tokens.count())
        hidden = policy_net.init_hidden(layers, batch_size, hidden_dim)

        batch.requires_grad = False
        batch.to(device)

    policy_net.train()

    policies, actions, batch, hidden = decision(policy_net, batch, hidden)
    policy_net.probs.append(policies.log_prob(actions))

    return batch, hidden


def rollout(length, batch=None, hidden=None, batch_size=None):

    if batch_size is None and batch is None:
        raise ValueError('Either specify a batch size or provide a batch.')

    with torch.no_grad():

        # generate a new batch of sequences
        if batch is None:
            batch = torch.zeros([batch_size,1,tokens.count()])
            hidden = rollout_net.init_hidden(layers,batch_size,hidden_dim)

        for _ in range(length - batch.shape[1]):
            _, _, batch, hidden = decision(rollout_net, batch, hidden)

    return batch


def decision(net, batch, hidden):

    # use hidden state to avoid feeding whole sequences redundantly
    state = batch[:,-1,:][:,None,:]
    policies, hidden = net(state, hidden)

    # sample next actions
    policies = torch.distributions.Categorical(policies)
    actions = policies.sample() 

    # concat onehot tokens with the batch of sequences
    encodings = torch.Tensor(list(map(tokens.onehot, actions)))
    encodings = encodings[:,None,:]
    batch = torch.cat((batch,encodings),dim=1)

    return policies, actions, batch, hidden


def feedback(reward):

    policy_net.rewards.append(reward)


def update_policy():
    global running_loss

    total = 0
    increment = []
    returns = []

    # compute state action values for each step
    for r in policy_net.rewards[::-1]:
        total = r + gamma * total
        returns.insert(0, total)

    # for each step standardize rewards in batch
    for i in range(len(returns)):
        returns[i] = (returns[i] - returns[i].mean()) / (returns[i].std() + eps)

    # weight state action values by log probability of action
    for log_prob, reward in zip(policy_net.probs, returns):
        increment.append(-log_prob * reward)

    # sum rewards over all steps for each sample
    increment = torch.stack(increment)
    increment = torch.sum(increment, dim=1)

    # average rewards over batch
    batch_size = increment.shape[0]
    increment = torch.sum(increment) / batch_size

    running_loss += increment.item()
    increment.backward()
    optimizer.step()

    del policy_net.rewards[:]
    del policy_net.probs[:]


def update_rollout():

    rollout_net.load_state_dict(policy_net.state_dict())
