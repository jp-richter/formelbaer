import datetime
import json
import traceback
import numpy
import io
import logging
import torchsummary
import matplotlib.pyplot

from torch.utils.tensorboard import SummaryWriter
from contextlib import redirect_stdout


def plot(path: str, values: list, title: str, labels: list, dtype: str):
    figure, axis = matplotlib.pyplot.subplots()

    types = {
        'bar': axis.bar,
        'plot': axis.plot,
        'hist': axis.hist
    }

    colors = ['b', 'r', 'y']

    for y, label, color in zip(values, labels, colors):
        x = numpy.arange(0, len(y), 1)
        function = types[dtype]
        function(x, y, color, alpha=0.5 if len(values)>1 else 1, label=label)

    matplotlib.pyplot.legend(loc='best')
    matplotlib.pyplot.title(title)
    axis.grid()

    figure.save_fig(path)


class ExperimentInfo():

    def __init__(self, folder: str, name: str):
        self.folder = folder
        self.name = name
        self.date = str(datetime.datetime.now()).replace(':', '-').replace(' ', '-')[:-7]
        self.dict = {}

    def __str__(self) -> str:
        string = 'Date: {}\n'.format(self.date) if self.date is not None else ''

        for key, value in self.dict.items():
            string += '{}: {}\n'.format(key, value)

        return string

    def __getitem__(self, item: str):
        return self.dict[item]

    def __setitem__(self, key: str, value):
        self.dict[key] = value

    def __delitem__(self, key: str):
        del self.dict[key]

    def __iter__(self):
        return iter(self.dict)

    def __len__(self):
        return len(self.dict)

    def items(self):
        return self.dict.items()

    def values(self):
        return self.dict.values()

    def keys(self):
        return self.dict.keys()

    def load(self):
        with open('{}/{}_info.json'.format(self.folder, self.name), "r") as file:
            dict = json.load(file)
            self.__dict__ = dict

    def save(self):
        parameters = self.__dict__
        with open('{}/{}_info.json'.format(self.folder, self.name), "w") as file:
            json.dump(parameters, file, indent=4)

    def plot(self, keys: list, title: str, labels: list, dtype: str, normalized: bool = False):
        if not len(keys) > 0:
            print('List of keys to plot is empty.')
            traceback.print_last(2)
            return

        if not len(keys) == len(labels):
            print('Amount of keys differs from amount of labels.')
            traceback.print_last(2)
            return

        dtypes = ['bar', 'plot']

        if not dtype in dtypes:
            print('Type {} is invalid, expected one of {}.'.format(dtype, dtypes))
            traceback.print_last(2)
            return

        ls = []

        for key in keys:
            try:
                values = numpy.array(self.dict[key])
            except KeyError:
                print('Key {} does not exist.'.format(key))
                traceback.print_last(2)
                continue
            except TypeError:
                print('Value of {} is of type {}, expected type list.'.format(key, type(self.dict[key])))
                traceback.print_last(2)
                continue

            if normalized:
                values = (values - values.min()) / (values.max() - values.min())

            ls.append(values)

        path = '{}/{}.png'.format(self.folder, title)
        plot(path, ls, title, labels, dtype)


class Logger():

    def __init__(self, path: str):
        self.path = path
        self.logs = {}
        self.levels = {
            'debug': 10,
            'info': 20,
            'warning': 30,
            'error': 40,
            'critical': 50
        }

    def setup(self, id: str):
        log = logging.getLogger(id)
        log.setLevel(logging.INFO)

        handler = logging.FileHandler('{}/{}.log'.format(self.path, id), mode='w')
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        log.addHandler(handler)
        self.logs[id] = log

    def write(self, id: str, level: str, message: str):
        self.logs[id].log(self.levels[level], message)

    def set_level(self, id: str, level: str):
        self.logs[id] = self.levels[level]


class Board(SummaryWriter):

    def __init__(self, folder: str, name: str):
        super().__init__(folder)
        self.info = ExperimentInfo(folder, name)

    def setup(self, models: list, hyperparameters: dict, notes: str = ''):
        def summary(model):
            channel = io.StringIO()
            with redirect_stdout(channel):
                torchsummary.summary(model)
            return channel.getvalue()

        for model in models:
            sum = summary(model)
            self.add_text('Models', sum)
            self.info['models'] = sum

        for parameter, value in hyperparameters.items():
            self.add_text('Hyperparameters', '{}: {}'.format(parameter, value))
        self.info['hyperparameters'] = hyperparameters

        self.add_text('Notes', notes)
        self.info['notes'] = notes



# all hyperparameters of a run
configuration = {}

# state_dicts, the parameters of the policiy nets for each training step
generator_parameters = []

# tensors(batchsize x actions), the policies for each generation step
generator_policies = []

# tensors(batchsize), the actions sampled from the policies for each generation step
generator_sampled_actions = []

# tensors(batchsize), rewards recieved for the sampled actions for each generation step
generator_rewards = []

# floats, the average prediction of the discriminator for synthetic samples for each generator training step
generator_predictions = []

# floats, losses for each training step averaged over the batches
generator_losses = []

# floats, the entropies of policies for each training step averaged over batch and steps
generator_entropies = []

# floats, the losses for each training step averaged over the batches
discriminator_losses = []

board = Board('/Users/jan/Desktop/test')
board.add_text('ein tag', 'ein text')
