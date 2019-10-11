from tokens import *
from numpy import random
import tree
import tokens
import torch
import pathlib
import constants
import generator
import converter

print('hello')

samples = generator.rollout(5, batch_size=10)
converter.convert(samples, constants.DIRECTORY_GENERATED_DATA)
