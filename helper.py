from typing import Any, Iterable, KeysView

import datetime
import os
import traceback
import logging
import requests
import sys
import pickle


class _TracePrints:

    def __init__(self):
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write("Writing %r\n" % s)
        traceback.print_stack(file=self.stdout)


class TracePrints:
    """Use with contextmanager: with info.TracePrints(): .."""

    def __enter__(self):
        sys.stdout = _TracePrints()

    def __exit__(self, exc_type, exc_val, exc_tb):
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


class _DataStore:

    NONE = 0
    PLOTTABLE = 1
    PLOTTABLE_ELEMENTS = 2

    def __init__(self):
        self.date = None
        self.folder = None
        self.hyperparameter = None
        self.notes = None
        self._data = {}
        self._attributes = {}
        self.paths = {}

    def __dict__(self) -> dict:
        return self._data

    def __str__(self) -> str:
        string = 'Date: {}\n'.format(self.date) if self.date is not None else ''
        string += 'Hyperparameter:\n {} \n\n'.format(self.hyperparameter)
        string += 'Notes:\n {} \n\n'.format(self.notes)
        return string

    def __iter__(self) -> Iterable:
        for key, value in self._data.items():
            yield (key, value)

    def __len__(self) -> int:
        return len(self._data)

    def setup(self, folder: str = None, hyperparameter: dict = None, notes: str = None):
        global store

        if not folder or not hyperparameter or not notes:
            raise ValueError('Please provide initialization arguments to setup the store object.')

        self.date = str(datetime.datetime.now()).replace(':', '-').replace(' ', '-')[:-7]
        self.folder = folder
        self.hyperparameter = hyperparameter
        self.notes = notes

        if not os.path.exists(folder):
            os.makedirs(folder)

        self.paths = {
            '{}/date.pickle'.format(self.folder): self.date,
            '{}/hyperparameters.pickle'.format(self.folder): self.hyperparameter,
            '{}/notes.pickle'.format(self.folder): self.notes,
            '{}/data.pickle'.format(self.folder): self._data
        }

        _STORE = self

    def set(self, tag: str, value: Any, attributes: list = None, if_exists: bool = True):
        if tag in self._data.keys() and not if_exists:
            return

        # TODO add type and assert items added are of type and make method get type

        self._data[tag] = value

        if attributes is not None:
            self._attributes[tag] = attributes

    def get(self, tag: str, raise_error: bool = False) -> Any:
        if tag not in self._data.keys():
            if not raise_error:
                return None
            else:
                raise KeyError('Tag {} not found.'.format(tag))

        return self._data[tag]

    def rmget(self, tag: str, raise_error: bool = False) -> Any:
        if tag not in self._data.keys():
            if not raise_error:
                return None
            else:
                raise KeyError('Tag {} not found.'.format(tag))

        temp = self._data[tag]
        del self._data[tag]
        return temp

    def attributes(self, tag):
        return self._attributes[tag] if tag in self._attributes.keys() else []

    def rm(self, tag: str, raise_exception: bool = False):
        if tag not in self._data.keys() and raise_exception:
            raise KeyError('Tag {} not found.'.format(tag))

        elif tag in self._data.keys():
            del self._data[tag]

    def get_tags(self) -> KeysView:
        return self._data.keys()

    def load(self):
        for path in self.paths.keys():
            try:
                with open(path, "rb") as file:
                    self.paths[path] = pickle.load(file)
            except Exception as e:
                print(traceback.format_exc())
                print('{}: {}'.format(e, path))

    def save(self):
        for path, value in self.paths.items():
            try:
                with open(path, "wb") as file:
                    pickle.dump(value, file)
            except Exception as e:
                print(traceback.format_exc())
                print('{}: {}'.format(e, path))


store = _DataStore()


def get_logger(path: str, name: str):
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)

    handler = logging.FileHandler('{}/{}.log'.format(path, name), mode='w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    log.addHandler(handler)
    return log


class TelegramService:
    MESSAGE = 'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&parse_mode=Markdown&text={' \
              'message} '
    STATUS = 'https://api.telegram.org/bot{bot_token}/getUpdates'

    def __init__(self):
        self.bot_token = ''  # insert api token
        self.chat_id = '560229425'

    def send(self, message: str):
        send_text = self.MESSAGE.format(bot_token=self.bot_token, chat_id=self.chat_id, message=message)
        response = requests.get(send_text)
        return response.json()
