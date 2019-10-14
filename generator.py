from numpy import finfo, float

import tokens
import torch
import const

def step(nn_policy, batch, hidden, nn_oracle, o_crit):

    # avoid feeding whole sequences redundantly
    state = batch[:,-1,:][:,None,:]
    policies, hidden = nn_policy(state, hidden)

    # in case of oracle training compute oracle loss
    if const.ORACLE:
        _, hidden = nn_oracle.inital()
        oracle_policies, _ = nn_oracle(batch, hidden)
        loss = o_crit(policies, oracle_policies)
        nn_oracle.running_loss += loss.item()

    # sample next actions
    policies = torch.distributions.Categorical(policies)
    actions = policies.sample() 

    # save probabilities for gradient computation
    nn_policy.probs.append(policies.log_prob(actions))

    # concat onehot tokens with the batch of sequences
    encodings = torch.Tensor(list(map(tokens.onehot, actions)))
    encodings = encodings[:,None,:]
    batch = torch.cat((batch,encodings),dim=1)

    return batch, hidden


def rollout(nn_policy, batch, hidden, seq_length):

    _, current_length, _ = batch.size()

    with torch.no_grad():

        for _ in range(seq_length - current_length):
            _, _, batch, hidden = step(nn_policy, batch, hidden)

    return batch


def sample(nn_policy, num_batches, seq_length):

    batches = torch.empty([0,seq_length,nn_policy.input_dim])

    with torch.no_grad():

        for _ in range(num_batches):
            batch, hidden = nn_policy.initial()
            batch = rollout(nn_policy, batch, hidden, seq_length)
            batches = torch.cat([batches, batch], dim=0)

    return batches


def reward(nn_policy, rewards):

    nn_policy.rewards.append(rewards)


def update(nn_policy, optimizer):

    total = 0
    increment = []
    returns = []

    # compute state action values for each step
    for reward in nn_policy.rewards[::-1]:
        total = reward + gamma * total
        returns.insert(0, total)

    # for each step standardize rewards in batch
    eps = finfo(float32).eps.item()
    for i in range(len(returns)):
        returns[i] = (returns[i] - returns[i].mean()) / (returns[i].std() + eps)

    # weight state action values by log probability of action
    for log_prob, reward in zip(nn_policy.probs, returns):
        increment.append(-log_prob * reward)

    # sum rewards over all steps for each sample
    increment = torch.stack(increment)
    increment = torch.sum(increment, dim=1)

    # average rewards over batch
    batchsize = increment.shape[0]
    increment = torch.sum(increment) / batchsize

    nn_policy.running_reward += -1 * increment.item()
    increment.backward()
    optimizer.step()

    del nn_policy.rewards[:]
    del nn_policy.probs[:]
