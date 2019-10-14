from numpy import finfo, float

import tokens
import torch

def step(policy, batch, hidden):

    # avoid feeding whole sequences redundantly
    state = batch[:,-1,:][:,None,:]
    policies, hidden = policy(state, hidden)

    # sample next actions
    policies = torch.distributions.Categorical(policies)
    actions = policies.sample() 

    # save probabilities for gradient computation
    policy.probs.append(policies.log_prob(actions))

    # concat onehot tokens with the batch of sequences
    encodings = torch.Tensor(list(map(tokens.onehot, actions)))
    encodings = encodings[:,None,:]
    batch = torch.cat((batch,encodings),dim=1)

    return batch, hidden


def rollout(policy, batch, hidden, seq_length):

    _, current_length, _ = batch.size()

    with torch.no_grad():

        for _ in range(seq_length - current_length):
            _, _, batch, hidden = step(rollout, batch, hidden)

    return batch


def sample(policy, num_batches, seq_length):

    batches = torch.empty([0,seq_length,policy.input_dim])

    with torch.no_grad():

        for _ in range(num_batches):
            batch, hidden = policy.initial()
            batch = rollout(policy, batch, hidden, seq_length)
            batches = torch.cat([batches, batch], dim=0)

    return batches


def reward(policy, rewards):

    policy.rewards.append(rewards)


def update(policy, optimizer):

    total = 0
    increment = []
    returns = []

    # compute state action values for each step
    for reward in policy.rewards[::-1]:
        total = reward + gamma * total
        returns.insert(0, total)

    # for each step standardize rewards in batch
    eps = finfo(float32).eps.item()
    for i in range(len(returns)):
        returns[i] = (returns[i] - returns[i].mean()) / (returns[i].std() + eps)

    # weight state action values by log probability of action
    for log_prob, reward in zip(policy.probs, returns):
        increment.append(-log_prob * reward)

    # sum rewards over all steps for each sample
    increment = torch.stack(increment)
    increment = torch.sum(increment, dim=1)

    # average rewards over batch
    batchsize = increment.shape[0]
    increment = torch.sum(increment) / batchsize

    policy.running_reward += -1 * increment.item()
    increment.backward()
    optimizer.step()

    del policy.rewards[:]
    del policy.probs[:]
