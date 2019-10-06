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

