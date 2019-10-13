import os
import pathlib
import multiprocessing
import logging

#
# APPLICATION 
#

# DIRECTORY_SFB_CLUSTER_ARXIV_DATA = '/rdata/schill/arxiv_processed/all/pngs'

DIRECTORY_APPLICATION = str(pathlib.Path.home()) + '/ramdisk/formelbaer'
DIRECTORY_GENERATED_DATA = str(pathlib.Path.home()) + '/ramdisk/formelbaer/generated'
DIRECTORY_ARXIV_DATA = str(pathlib.Path.home()) + '/ramdisk/formelbaer/arxiv'
DIRECTORY_SFB_CLUSTER_ARXIV_DATA = '/ramdisk/pngs'

ADVERSARIAL_ITERATIONS = 100
ADVERSARIAL_DISCRIMINATOR_STEPS = 2 # (*2) due to implementation
ADVERSARIAL_GENERATOR_STEPS = 1
ADVERSARIAL_SEQUENCE_LENGTH = 16
ADVERSARIAL_MONTECARLO_TRIALS = 16
ADVERSARIAL_BATCHSIZE = multiprocessing.cpu_count()*4

#
# GENERATOR
#

GENERATOR_HIDDEN_DIM = 32
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

#
# MAKEDIRS
#

if os.path.exists(DIRECTORY_SFB_CLUSTER_ARXIV_DATA):
	DIRECTORY_ARXIV_DATA = DIRECTORY_SFB_CLUSTER_ARXIV_DATA

if not os.path.exists(DIRECTORY_APPLICATION):
	os.makedirs(DIRECTORY_APPLICATION)

if not os.path.exists(DIRECTORY_GENERATED_DATA):
	os.makedirs(DIRECTORY_GENERATED_DATA)

if not os.path.exists(DIRECTORY_ARXIV_DATA):
	raise ValueError()

#
# SAVE SETTINGS
#

def log(folder):

	logging.basicConfig(level=logging.INFO, filename=folder + '/parameters.log')
	log = logging.getLogger(__name__)
	log.setLevel(logging.INFO)

    log.info('''Models Version 1

        Total Iterations {}
        Discriminator Steps {}
        Generator Steps {}
        Fixed Sequence Length {}
        Monte Carlo Trials {}
        Batch Size {}

        Generator Hidden Dim {}
        Generator Layers {}
        Generator Dropout {}
        Generator Learning Rate {}
        Generator Baseline {}
        Generator Gamma {}

        Discriminator Dropout {}
        Discriminator Learnrate {}'''.format(
            ADVERSARIAL_ITERATIONS, 
            ADVERSARIAL_DISCRIMINATOR_STEPS, 
            ADVERSARIAL_GENERATOR_STEPS, 
            ADVERSARIAL_SEQUENCE_LENGTH, 
            ADVERSARIAL_MONTECARLO_TRIALS, 
            ADVERSARIAL_BATCHSIZE, 
            GENERATOR_HIDDEN_DIM, 
            GENERATOR_LAYERS, 
            GENERATOR_DROPOUT, 
            GENERATOR_BASELINE,
            GENERATOR_GAMMA, 
            DISCRIMINATOR_DROPOUT, 
            DISCRIMINATOR_LEARNRATE))
