from numpy.random import choice
from numpy import empty, finfo, float32
from nn_policy import PolicyNetwork
import constants as c

import tokens
import torch

running_reward = 0.0

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

policy_net = PolicyNetwork().to(device)
rollout_net = PolicyNetwork().to(device)
rollout_net.eval()
rollout_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.Adam(policy_net.parameters(), lr=c.GENERATOR_LEARNRATE)
eps = finfo(float32).eps.item()


def step(batch=None, hidden=None):

    if batch == None: # generate a new batch of sequences
        optimizer.zero_grad()

        batch = torch.zeros(c.ADVERSARIAL_BATCHSIZE,1,tokens.count())
        hidden = policy_net.init_hidden()

        batch.requires_grad = False
        batch.to(device)

    policy_net.train()

    policies, actions, batch, hidden = decision(policy_net, batch, hidden)
    policy_net.probs.append(policies.log_prob(actions))

    return batch, hidden


def rollout(batch=None, hidden=None):

    with torch.no_grad():

        if batch is None: # generate a new batch of sequences
            batch = torch.zeros([c.ADVERSARIAL_BATCHSIZE,1,tokens.count()])
            hidden = rollout_net.init_hidden()

        for _ in range(c.ADVERSARIAL_SEQUENCE_LENGTH - batch.shape[1]):
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
    global running_reward

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
    batchsize = increment.shape[0]
    increment = torch.sum(increment) / batchsize

    running_reward += increment.item()
    increment.backward()
    optimizer.step()

    del policy_net.rewards[:]
    del policy_net.probs[:]


def update_rollout():

    rollout_net.load_state_dict(policy_net.state_dict())


def save_parameters(folder):

    file = folder + 'generator_parameters.pt'
    torch.save(policy_net.state_dict(), file)


def load_parameters(folder):

    file = folder + 'generator_parameters.pt'
    policy_net.load_state_dict(torch.load(file))
