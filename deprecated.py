import sys
import os
import torch
import numpy
import tokens
import config
import generator
import matplotlib.pyplot as plt


class PolicyEvaluator:

    def _plot(self, x, y, legend):
        figure, axis = plt.subplots()

        for y, label in zip(y, legend):
            axis.bar(x, y, alpha=0.5, label=label)

        plt.legend(loc='best')
        return figure

    def _policy_step(self, nn_policy, batch, hidden):
        state = batch[:, -1, :][:, None, :]
        policies, hidden = nn_policy(state, hidden)

        distributions = torch.distributions.Categorical(policies)
        actions = distributions.sample()

        encodings = torch.tensor([tokens.onehot(id) for id in actions])
        encodings = encodings[:, None, :].float()
        batch = torch.cat((batch, encodings), dim=1)

        return policies, batch, hidden

    def _policy_average(self, filepath):
        nn_policy = generator.Policy()
        nn_policy.load(filepath)
        batch, hidden = nn_policy.initial()
        results = torch.empty((0, batch.shape[0], batch.shape[2]))

        for _ in range(config.general.sequence_length):
            policies, batch, hidden = self._policy_step(nn_policy, batch, hidden)
            policies = policies.unsqueeze(dim=0)
            results = torch.cat((results, policies), dim=0)

        results = torch.mean(results, dim=0)  # average over steps
        results = torch.mean(results, dim=0)  # average over batch
        results = results.tolist()

        return results

    def evaluate(self, filepath):
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
                policy = self._policy_average(path)
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
            figure = self._plot(x, policy, [''])
            epoch = i * stepsize
            figure.savefig('{}/{}.png'.format(filepath, epoch))

        # save multiplots every tenth policy
        legend = [str(epoch * stepsize) for epoch in range(len(policies))]
        figure = self._plot(x, y, legend)
        figure.savefig('{}/all_distributions.png'.format(filepath))


class LogEvaluator():

    # example: values = [(x,y)] with x = np.array(..), ...
    #          legend = ['reward', 'loss', ..]

    def _plot(self, values, xlabel, ylabel, legend, title, fontsize, path):
        figure, axis = plt.subplots()
        lines = ['b', 'r', 'y']

        for (x, y), line, label in zip(values, lines, legend):
            axis.plot(x, y, line, label=label, linewidth=0.3)

        plt.title(title, fontsize=fontsize)
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)

        leg = plt.legend()
        for line in leg.get_lines():
            line.set_linewidth(1)
        for text in leg.get_texts():
            text.set_fontsize('x-large')

        axis.grid()
        figure.savefig(path)

    def _parse(self, filepath, target):
        with open(filepath, 'r') as file:
            string = file.read()

        targets = {
            'greward': r'Generator\sReward\sas\sSequence:\s.*',
            'gloss': r'Generator\sLoss\sas\sSequence:\s.*',
            'gprediction': r'Generator\sPrediction\sas\sSequence:\s.*',
            'dloss': r'Discriminator\sLoss\sas\sSequence:\s.*'
        }

        targets_substrings = {
            'greward': lambda s: s[30:],
            'gloss': lambda s: s[28:],
            'gprediction': lambda s: s[34:],
            'dloss': lambda s: s[32:]
        }

        pattern = re.compile(targets[target])
        result = []

        for match in re.finditer(pattern, string):
            result.append(match.group())

        assert len(result) == 1
        result = targets_substrings[target](result[0])
        ls = result.split(',')
        ls = [float(n) for n in ls]

        return ls

    def evaluate(self, filepath):
        targets = ['greward', 'gloss', 'gprediction']
        targets_labels = {
            'greward': 'Generator Reward',
            'gloss': 'Generator Loss',
            'gprediction': 'Generator Prediction'
        }
        results = []

        # single plots
        for t in targets:
            numbers = self._parse(filepath, t)
            x = numpy.arange(0, len(numbers), 1)
            y = numpy.array(numbers)

            # save single plot
            self._plot([(x, y)], 'Step', targets_labels[t], [''], '', 12, '{}/{}_plot.png'.format(filepath[:-11], t))

            if t == 'gprediction':
                pass

            if t == 'gloss':
                # y = (y - y.mean()) / (y.std() + numpy.finfo(numpy.float32).eps.item())
                y = (y - y.min()) / (y.max() - y.min())

            if t == 'greward':
                # y = (y - y.mean()) / (y.std() + numpy.finfo(numpy.float32).eps.item())
                y = (y - y.min()) / (y.max() - y.min())

            results.append((x, y))

        # plot all on same surface
        legend = ['Generator Reward', 'Generator Loss', 'Discriminator Prediction']
        self._plot(results, 'Step', '', legend, '', 12, '{}/gen_plot.png'.format(filepath[:-11]))

        numbers = self._parse(filepath, 'dloss')
        x = numpy.arange(0, len(numbers), 1)
        y = numpy.array(numbers)
        self._plot([(x, y)], 'Epoch', 'Discriminator Loss', [''], '', 12, '{}/dis_plot.png'.format(filepath[:-11]))


if __name__ == '__main__':
    assert len(sys.argv) == 2
    _, filepath = sys.argv

    try:
        eval = PolicyEvaluator()
        eval.evaluate(filepath)
    except:
        pass

    try:
        eval = LogEvaluator()
        eval.evaluate(filepath)
    except:
        pass
