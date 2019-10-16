from numpy import finfo, float32

import tokens
import torch
import math
import config as cfg


def step(nn_policy, batch, hidden, nn_oracle, o_crit, save_prob=False):

    # avoid feeding whole sequences redundantly
    state = batch[:,-1,:][:,None,:]
    policies, hidden = nn_policy(state, hidden)

    # in case of oracle training compute oracle loss
    if cfg.app_cfg.oracle and nn_oracle is not None:
        _, hidden = nn_oracle.initial()
        oracle_policies, _ = nn_oracle(batch, hidden)

        # kl divergence module expects input prob logs
        log_policies = torch.log(policies)

        loss = o_crit(log_policies, oracle_policies)
        nn_oracle.running_loss += loss.item()

    # sample next actions
    policies = torch.distributions.Categorical(policies)
    actions = policies.sample() 

    # save log probabilities for gradient computation
    if save_prob:
        log_probs = policies.log_prob(actions) 
        nn_policy.probs.append(log_probs)

    # concat onehot tokens with the batch of sequences
    encodings = torch.Tensor(list(map(tokens.onehot, actions)))
    encodings = encodings[:,None,:]
    batch = torch.cat((batch,encodings),dim=1)

    # if batch still has the empty start token remove it
    if torch.sum(batch[:,0,:]) == 0:
        batch = batch[:,1:,:]

    return batch, hidden


def rollout(nn_policy, batch, hidden):

    with torch.no_grad():

        while batch.shape[1] < cfg.app_cfg.seq_length:
            batch, hidden = step(nn_policy, batch, hidden, None, None)

    return batch


def sample(nn_policy, num_batches):

    batches = torch.empty([0,cfg.app_cfg.seq_length,nn_policy.input_dim])

    with torch.no_grad():

        for _ in range(num_batches):
            batch, hidden = nn_policy.initial()
            batch, hidden = step(nn_policy, batch, hidden, None, None)
            batch = rollout(nn_policy, batch, hidden)
            batches = torch.cat([batches, batch], dim=0)

    return batches


def reward(nn_policy, rewards):

    nn_policy.rewards.append(rewards)


def update(nn_policy, optimizer):

    total = 0
    loss = []
    returns = []

    # compute state action values for each step
    for reward in nn_policy.rewards[::-1]:
        total = reward + cfg.g_cfg.gamma * total
        returns.insert(0, total)

    # for each step standardize rewards in batch
    eps = finfo(float32).eps.item()
    for i in range(len(returns)):
        returns[i] = (returns[i] - returns[i].mean()) / (returns[i].std() + eps)

    assert len(nn_policy.probs) == len(returns)

    # weight state action values by log probability of action
    for log_prob, reward in zip(nn_policy.probs, returns):
        loss.append(-log_prob * reward)

    # sum rewards over all steps for each sample
    loss = torch.stack(loss)
    loss = torch.sum(loss, dim=1)

    # average rewards over batch
    batchsize = loss.shape[0]
    loss = torch.sum(loss) / batchsize

    loss.backward()
    optimizer.step()

    # max reward * prob ~> max reward * log prob ~> min -(reward * log prob)
    # the loss is positive (even if we multiplied with - log_prob) because the log prob
    # itsself is negative. instead of starting negative and max reward we start positive
    # and minimize the negative reward. for clarity we still keep track of the negative
    # reward we're going to max

    nn_policy.running_reward += -1 * loss.item()

    del nn_policy.rewards[:]
    del nn_policy.probs[:]
