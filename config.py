import os
import pathlib
import multiprocessing
import torch

from dataclasses import dataclass


@dataclass
class Paths:

	app: str 
	synthetic_data: str
	arxiv_data: str
	oracle_data: str

	log: str
	oracle: str

	dump: str


@dataclass
class AppConfig:
	
	device: None

	iterations: int 
	d_steps: int 
	g_steps: int 
	seq_length: int 
	montecarlo_trials: int 
	batchsize: int

	oracle: bool 
	oracle_samplesize: int 

	label_synth: int 
	label_arxiv: int 


@dataclass 
class GeneratorConfig:

	hidden_dim: int 
	layers: int
	dropout: int 
	learnrate: int 
	baseline: int
	gamma: int 


@dataclass
class DiscriminatorConfig:

	dropout: int 
	learnrate: int


# '/rdata/schill/arxiv_processed/all/pngs'

# '/rdata/schill/arxiv_processed/ba_data/ml_data/train/pngs'

# '/ramdisk/arxiv'


DEFAULT_PATHS_CFG = Paths(

	app = str(pathlib.Path.home()) + '/formelbaer_data',
	synthetic_data = str(pathlib.Path.home()) + '/formelbaer_data/synthetic_samples',
	arxiv_data = str(pathlib.Path.home()) + '/formelbaer_data/arxiv_samples',
	oracle_data = str(pathlib.Path.home()) + '/formelbaer_data/oracle_samples',
	
	log = str(pathlib.Path.home()) + '/formelbaer_data/results.log',
	oracle = str(pathlib.Path.home()) + '/formelbaer_data/oracle_net.pt',

	dump = str(pathlib.Path.home()) + '/formelbaer_data/dump.log'

	)


DEFAULT_PATHS_CFG_CLUSTER = Paths(

	app = '/ramdisk/formelbaer_data',
	synthetic_data = '/ramdisk/formelbaer_data/synthetic_data',
	# arxiv_data = '/rdata/schill/arxiv_processed/ba_data/ml_data/train/pngs',
	arxiv_data = '/ramdisk/formelbaer_data/arxiv_data',
	oracle_data = '/ramdisk/formelbaer_data/oracle_data',
	
	log = '/ramdisk/formelbaer_data/results.log',
	oracle = '/ramdisk/formelbaer_data/oracle_net.pt',

	dump = '/ramdisk/formelbaer_data/dump.log'

	)


if multiprocessing.cpu_count() > 4:
	paths_cfg = DEFAULT_PATHS_CFG_CLUSTER
else:
	paths_cfg = DEFAULT_PATHS_CFG


DEFAULT_APP_CFG = AppConfig(

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),

	iterations = 150,
	d_steps = 2, # (*2) due to computational cost reasons
	g_steps = 1,
	seq_length = 16, # 16
	montecarlo_trials = 16, # 16
	# batchsize = multiprocessing.cpu_count()*4, # computational cost reasons
	batchsize = multiprocessing.cpu_count(),

	oracle = False,
	oracle_samplesize = 100,

	label_synth = 1,
	label_arxiv = 0

	)

app_cfg = DEFAULT_APP_CFG


DEFAULT_GENERATOR_CFG = GeneratorConfig(

	hidden_dim = 32,
	layers = 2,
	dropout = 0.2,
	learnrate = 0.01,
	baseline = 1,
	gamma = 0.99

	)

g_cfg = DEFAULT_GENERATOR_CFG


DEFAULT_DISCRIMINATOR_CFG = DiscriminatorConfig(

	dropout = 0.2,
	learnrate = 0.01

	)

d_cfg = DEFAULT_DISCRIMINATOR_CFG
