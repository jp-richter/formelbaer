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
# '/rdata/schill/equationlearning'


DEFAULT_PATHS_CFG = Paths(

	app = str(pathlib.Path.home()) + '/formelbaer_data',
	synthetic_data = str(pathlib.Path.home()) + '/formelbaer_data/synthetic',
	arxiv_data = str(pathlib.Path.home()) + '/formelbaer_data/arxiv',
	oracle_data = str(pathlib.Path.home()) + '/formelbaer_data/oracle',
	
	log = str(pathlib.Path.home()) + '/formelbaer_data/results.log',
	oracle = str(pathlib.Path.home()) + '/formelbaer_data/oracle.pt'

	)

paths_cfg = DEFAULT_PATHS_CFG


# default_paths = Paths(

# 	app = '/ramdisk/formelbaer_data',
# 	synthetic_data = '/ramdisk/synthetic',
# 	arxiv_data = '/ramdisk/arxiv',
# 	oracle_data = '/ramdisk/oracle',
	
# 	log = '/ramdisk/results.log',
# 	oracle = '/ramdisk/oracle.pt'

# 	)


DEFAULT_APP_CFG = AppConfig(

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),

	iterations = 2,
	d_steps = 2, # (*2) due to computational cost reasons
	g_steps = 1,
	seq_length = 2, # 16
	montecarlo_trials = 2, # 16
	batchsize = multiprocessing.cpu_count()*4, # computational cost reasons

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
