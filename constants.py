#
# APPLICATION PARAMETERS
#

from pathlib import Path
HOME = str(Path.home()) + '/formelbaer'

from os import path, makedirs
if not path.exists(HOME):
    makedirs(HOME)

import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DSTEPS = 2
GSTEPS = 2
ITERATIONS = 1

SEQ_LENGTH = 6
MONTECARLO = 2

#
# GRU HYPER PARAMETERS
#

import tokens
INPUT_DIM = tokens.count()
OUTPUT_DIM = tokens.count()

BATCH_SIZE = 32
HIDDEN_DIM = 32
GRU_LAYERS = 2

DROP_OUT = 0.2
LEARN_RATE = 0.01

BASELINE = 1
GAMMA = 0.95

#
# CNN HYPER PARAMETERS
#
