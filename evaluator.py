import sys
import os
import torch
import numpy
import tokens
import config
import generator
import matplotlib.pyplot as plt


def chart_single(x, y):
    figure, axis = plt.subplots()
    axis.bar(x, y, width=0.5)
    return figure


def chart_multiple(x, y, legend):
    figure, axis = plt.subplots()

    for y, label in zip(y, legend):
        axis.bar(x, y, alpha=0.5, label=label)

    plt.legend(loc='best')
    return figure


def policy_step(nn_policy, batch, hidden):
    state = batch[:, -1, :][:, None, :]
    policies, hidden = nn_policy(state, hidden)

    distributions = torch.distributions.Categorical(policies)
    actions = distributions.sample()

    encodings = torch.tensor([tokens.onehot(id) for id in actions])
    encodings = encodings[:, None, :].float()
    batch = torch.cat((batch, encodings), dim=1)

    return policies, batch, hidden


def policy_average(filepath):
    nn_policy = generator.Policy()
    nn_policy.load(filepath)
    batch, hidden = nn_policy.initial()
    results = torch.empty((0, batch.shape[0], batch.shape[2]))

    for _ in range(config.general.sequence_length):
        policies, batch, hidden = policy_step(nn_policy, batch, hidden)
        policies = policies.unsqueeze(dim=0)
        results = torch.cat((results, policies), dim=0)

    results = torch.mean(results, dim=0)  # average over steps
    results = torch.mean(results, dim=0)  # average over batch
    results = results.tolist()

    return results


def plot(filepath=config.paths.policies):
    policy_paths = []  # (path)
    policies = []  # (average policy)
    stepsize = 10

    iter = os.scandir(filepath)
    for entry in iter:
        if entry.name.endswith('.pt'):
            policy_paths.append(filepath + '/' + entry.name)

    names = [tokens.get(id).name for id in tokens.possibilities()]
    for i, path in enumerate(policy_paths):
        if i % stepsize == 0:
            policy = policy_average(path)
            with open('{}.txt'.format(path[:-2]), 'w', encoding="utf-8") as file:
                string = ''
                for name, value in zip(names, policy):
                    string += '{}: {}\n'.format(name, value)
                file.write(string)
            policies.append(policy)

    x = numpy.array(tokens.possibilities())  # token ids for all tokens
    y = numpy.array([p for p in policies])

    # save single plots
    for i, policy in enumerate(y):
        figure = chart_single(x, policy)
        epoch = i * stepsize
        figure.savefig('{}/{}.png'.format(filepath, epoch))

    # save multiplots every tenth policy
    legend = [str(epoch * stepsize) for epoch in range(len(policies))]
    figure = chart_multiple(x, y, legend)
    figure.savefig('{}/all_distributions.png'.format(filepath))


if __name__ == '__main__':
    assert len(sys.argv) == 2
    _, filepath = sys.argv
    plot(filepath)
