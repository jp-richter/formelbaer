import torch
from os import path, makedirs
from pathlib import Path

#
# APPLICATION PARAMETERS
#

HOME = str(Path.home()) + '/formelbaer'
ARXIV = HOME + '/arxiv'
GENERATED = HOME + '/generated'

if not path.exists(HOME):
	makedirs(HOME)

if not path.exists(GENERATED):
    makedirs(GENERATED)

if not path.exists(ARXIV):
    raise ValueError('Please save the training set of arxiv .png or .pt files in ' \
    	+ ARXIV + ' or change the ARXIV directory in constants.py accordingly.')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DISCRIMINATOR_STEPS = 2
GENERATOR_STEPS = 2
ITERATIONS = 1

SEQUENCE_LENGTH = 6
MONTECARLO = 2

#
# GRU HYPER PARAMETERS
#

import tokens
GRU_INPUT_DIM = tokens.count()
GRU_OUTPUT_DIM = tokens.count()

GRU_BATCH_SIZE = 32
GRU_HIDDEN_DIM = 32
GRU_LAYERS = 2

GRU_DROP_OUT = 0.2
GRU_LEARN_RATE = 0.01

GRU_BASELINE = 1
GRU_GAMMA = 0.95

#
# CNN HYPER PARAMETERS
#

CNN_LEARN_RATE = 0.01

