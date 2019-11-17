from typing import Union

import datetime
import json
import os
import traceback
import numpy
import logging
import torchvision
import requests
import torch
import matplotlib.pyplot
import sys


class _TracePrints:

    def __init__(self):
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write("Writing %r\n" % s)
        traceback.print_stack(file=self.stdout)


class Tracer:

    def start(self):
        sys.stdout = _TracePrints()

    def stop(self):
        sys.stdout = sys.__stdout__


class HiddenPrints:
    """Use with contextmanager: with info.Hiddenprints(): .."""

    def __enter__(self):
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


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
        function(x, y, color, alpha=0.5 if len(values) > 1 else 1, label=label)

    matplotlib.pyplot.legend(loc='best')
    matplotlib.pyplot.title(title)
    axis.grid()

    figure.savefig(path)


class ExperimentInfo:

    HYPERPARAMETER = 1
    NOTES = 2
    TEXT_DICT = 3
    SCALAR_DICT = 4
    IMAGE_DICT = 5
    DISTRIBUTION_DICT = 6
    HISTOGRAM_DICT = 7

    def __init__(self, folder: str, hyperparameter: dict={}, notes: str=''):
        self.date = str(datetime.datetime.now()).replace(':', '-').replace(' ', '-')[:-7]
        self.folder = folder

        self.hyperparameter = hyperparameter
        self.notes = notes
        self.text_dict = {}
        self.scalar_dict = {}
        self.image_dict = {}
        self.distribution_dict = {}
        self.histogram_dict = {}

        self.dictionaries = {
            self.HYPERPARAMETER: self.hyperparameter,
            self.NOTES: self.notes,
            self.TEXT_DICT: self.text_dict,
            self.SCALAR_DICT: self.scalar_dict,
            self.IMAGE_DICT: self.image_dict,
            self.DISTRIBUTION_DICT: self.distribution_dict,
            self.HISTOGRAM_DICT: self.histogram_dict
        }

        self.files = {
            self.HYPERPARAMETER: 'hyperparameter',
            self.NOTES: 'notes',
            self.TEXT_DICT: 'text_info',
            self.SCALAR_DICT: 'scalar_info',
            self.IMAGE_DICT: 'image_info',
            self.DISTRIBUTION_DICT: 'distribution_info',
            self.HISTOGRAM_DICT: 'histogram_info'
        }

    def __dict__(self) -> dict:
        return {
            **self.text_dict,
            **self.scalar_dict,
            **self.image_dict,
            **self.distribution_dict,
            **self.histogram_dict
        }

    def __str__(self) -> str:
        string = 'Date: {}\n'.format(self.date) if self.date is not None else ''

        try:
            string += 'Hyperparameter:\n {} \n\n'.format(self.hyperparameter)
            string += 'Notes:\n {} \n\n'.format(self.notes)
        except:
            pass

        for d in dict(self):
            try:
                string += '{} \n\n'.format(d)
            except:
                pass

        return string

    def __iter__(self):
        return iter(dict(self))

    def __len__(self):
        return len(dict(self))

    def _add(self, dict, tag, value, step):
        if tag in dict.keys():
            dict[tag].append((step, value))
        else:
            dict[tag] = [(step, value)]

    def add_text(self, tag: str, value: str, step: int):
        self._add(self.text_dict, tag, value, step)

    def add_scalar(self, tag: str, value: Union[int, float], step: int):
        self._add(self.scalar_dict, tag, value, step)

    def add_distribution(self, tag: str, value: list, step: int):
        self._add(self.distribution_dict, tag, value, step)

    def add_sample(self, tag: str, value: list, step: int):
        self._add(self.histogram_dict, tag, value, step)

    def add_image(self, tag: str, value: torch.Tensor, step: int):
        self._add(self.image_dict, tag, value, step)

    def load(self):
        try:
            with open('{}/{}.json'.format(self.folder, self.files[self.HYPERPARAMETER]), "r") as file:
                string = json.loads(file)
                self.hyperparameter = string
        except:
            pass

        try:
            with open('{}/{}.json'.format(self.folder, self.files[self.NOTES]), "r") as file:
                string = json.loads(file)
                self.notes = string
        except:
            pass

        try:
            with open('{}/{}.json'.format(self.folder, self.files[self.TEXT_DICT]), "r") as file:
                dict = json.load(file)
                self.text_dict = dict
        except:
            pass

        try:
            with open('{}/{}.json'.format(self.folder, self.files[self.SCALAR_DICT]), "r") as file:
                dict = json.load(file)
                self.scalar_dict = dict
        except:
            pass

        try:
            with open('{}/{}.json'.format(self.folder, self.files[self.DISTRIBUTION_DICT]), "r") as file:
                dict = json.load(file)
                self.distribution_dict = dict
        except:
            pass

        try:
            with open('{}/{}.json'.format(self.folder, self.files[self.HISTOGRAM_DICT]), "r") as file:
                dict = json.load(file)
                self.histogram_dict = dict
        except:
            pass

    def save(self):
        def _json(target_id, dictionary):
            with open('{}/{}.json'.format(self.folder, self.files[target_id]), "w") as file:
                json.dump(dictionary, file, indent=4)

        def _image(target_id, dictionary):
            transform = torchvision.transforms.ToPILImage()
            for tag, ls in dictionary.items():
                for i, tensor, step in enumerate(ls):
                    try:
                        if not os.path.exists('{}/images-{}'.format(self.folder, tag)):
                            os.makedirs('{}/images-{}'.format(self.folder, tag))
                        image = transform(tensor)
                        image.save('{}/images-{}/step_{}_{}.png'.format(self.folder, tag, step, i))
                    except:
                        pass

        def _plot(target_id, dictionary):
            for tag, ls in dictionary.items():
                y = numpy.array([v for (s,v) in ls])
                path = '{}/{}.png'.format(self.folder, tag)
                plot(path, [y], '', [tag], 'plot')

        def _histogram(target_id, dictionary):
            for tag, ls in dictionary.items():
                for i, sample in enumerate([v for (s,v) in ls]):
                    y = numpy.array(sample)
                    if not os.path.exists('{}/histograms-{}'.format(self.folder, tag)):
                        os.makedirs('{}/histograms-{}'.format(self.folder, tag))
                    path = '{}/histograms-{}/{}.png'.format(self.folder, tag, i)
                    plot(path, [y], tag, [tag], 'bar')

        def _disctribution(target_id, dictionary):
            for tag, ls in dictionary.items():
                for i, sample in enumerate([v for (s,v) in ls]):
                    y = numpy.array(sample)
                    if not os.path.exists('{}/histograms-{}'.format(self.folder, tag)):
                        os.makedirs('{}/histograms-{}'.format(self.folder, tag))
                    path = '{}/histograms-{}/{}.png'.format(self.folder, tag, i)
                    plot(path, [y], tag, [tag], 'hist')

        protocols = {
            self.HYPERPARAMETER: (_json,),
            self.NOTES: (_json,),
            self.TEXT_DICT: (_json,),
            self.SCALAR_DICT: (_json, _plot),
            self.IMAGE_DICT: (_image,),
            self.DISTRIBUTION_DICT: (_json, _disctribution),
            self.HISTOGRAM_DICT: (_json, _histogram)
        }

        for target_id, dictionary in self.dictionaries.items():
            protocol = protocols[target_id]
            for function in protocol:
                function(target_id, dictionary)

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
                values = numpy.array(dict(self)[key])
            except KeyError:
                print('Key {} does not exist.'.format(key))
                traceback.print_last(2)
                continue
            except TypeError:
                print('Value of {} is of type {}, expected type list.'.format(key, type(dict(self)[key])))
                traceback.print_last(2)
                continue

            if normalized:
                values = (values - values.min()) / (values.max() - values.min())

            ls.append(values)

        path = '{}/{}.png'.format(self.folder, title)
        plot(path, ls, title, labels, dtype)


class Logger:

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


class TelegramService:

    MESSAGE = 'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&parse_mode=Markdown&text={' \
              'message} '
    STATUS = 'https://api.telegram.org/bot{bot_token}/getUpdates'

    def __init__(self):
        self.bot_token = '1064137368:AAFy7T0T-DR5Sob2aA0kfSRoFob9HxI_RrY'
        self.chat_id = '560229425'

    def send(self, message: str):
        send_text = self.MESSAGE.format(bot_token=self.bot_token, chat_id=self.chat_id, message=message)
        response = requests.get(send_text)
        return response.json()
