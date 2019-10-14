import os
import pathlib
import multiprocessing
import logging

#
# APPLICATION 
#

# '/rdata/schill/arxiv_processed/all/pngs'
# '/rdata/schill/equationlearning'

if os.path.exists('/rdata'):
    DIRECTORY_APPLICATION = '/ramdisk/formelbaer_data/'

else:
    DIRECTORY_APPLICATION = str(pathlib.Path.home()) + '/formelbaer'

DIRECTORY_GENERATED_DATA = DIRECTORY_APPLICATION + '/generated'
DIRECTORY_ARXIV_DATA = DIRECTORY_APPLICATION + '/arxiv'
FILE_LOG = c.DIRECTORY_APPLICATION + '/results.log'

ADVERSARIAL_ITERATIONS = 100
ADVERSARIAL_DISCRIMINATOR_STEPS = 2 # (*2) due to computational cost reasons
ADVERSARIAL_GENERATOR_STEPS = 1
ADVERSARIAL_SEQUENCE_LENGTH = 16
ADVERSARIAL_MONTECARLO_TRIALS = 16 # seqGan
ADVERSARIAL_BATCHSIZE = multiprocessing.cpu_count()*4

ORACLE = False

LABEL_SYNTH = 1
LABEL_ARXIV = 0

#
# GENERATOR
#

GENERATOR_HIDDEN_DIM
GENERATOR_LAYERS = 2
GENERATOR_DROPOUT = 0.2
GENERATOR_LEARNRATE = 0.01
GENERATOR_BASELINE = 1
GENERATOR_GAMMA = 0.98

#
# DISCRIMINATOR
#

DISCRIMINATOR_DROPOUT = 0.2
DISCRIMINATOR_LEARNRATE = 0.01
