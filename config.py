import multiprocessing
import torch
import os

from pathlib import Path, PurePath
from dataclasses import dataclass


@dataclass
class Paths:
    """
    Configuration data class which contains all path variables of the script. To change paths make changes to the
    DEFAULT_PATHS_CFG instance or overwrite the cfg_paths variable with your own instance.
    """

    app: str
    synthetic_data: str
    arxiv_data: str
    log: str
    dump: str
    ray_store: str
    bias_term: str
    rdata: str
    policies: str
    results: str


@dataclass
class Config:
    """
    Configuration data class which contains various training parameters. To change training parameters make changes
    to the DEFAUT_APP_CFG instance or overwrite the cfg_app variable with your own instance.
    """

    # GENERAL

    device: torch.device
    num_real_samples: int
    num_eval_samples: int

    adversarial_steps: int
    d_epochs: int
    d_steps: int
    g_steps: int

    batch_size: int
    montecarlo_trials: int

    sequence_length: int

    label_synth: int
    label_real: int

    # GENERATOR

    g_hidden_dim: int
    g_layers: int
    g_dropout: float
    g_learnrate: float
    g_baseline: float
    g_gamma: float
    g_bias: bool

    # DISCRIMINATOR

    d_dropout: float
    d_learnrate: float


# '/rdata/schill/arxiv_processed/all/pngs'
# '/rdata/schill/arxiv_processed/ba_data/ml_data/train/pngs'

_home = '/ramdisk' if os.path.exists('/ramdisk') else str(Path.home())

DEFAULT_PATHS = Paths(

    app=_home + '/formelbaer-data',
    synthetic_data=_home + '/formelbaer-data/synthetic-data',
    arxiv_data=_home + '/formelbaer-data/arxiv-data',
    log=_home + '/formelbaer-data/results.log',
    dump=_home + '/formelbaer-data/dump.txt',
    ray_store=_home + '/formelbaer-data/ray-plasma-store',
    bias_term=str(PurePath(Path(__file__).resolve().parent, 'bias_term.txt')),
    rdata='/rdata/richter2/experiments',
    policies=_home + '/formelbaer-data/policies',
    results='/rdata/richter2/experiments' if os.path.exists('/rdata/richter2/experiments') else _home + '/formelbaer-data/experiments'
)

paths = DEFAULT_PATHS

DEFAULT_CONFIG = Config(

    # GENERAL

    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),

    num_real_samples=2000,  # 5000
    num_eval_samples=100,  # 100

    adversarial_steps=150,  # 150
    d_epochs=1,
    d_steps=1,
    g_steps=10,  # 10

    batch_size=multiprocessing.cpu_count() * 4,  # computational cost reasons
    montecarlo_trials=10,  # 10

    sequence_length=20,  # 20

    label_synth=1,  # discriminator outputs P(x ~ synthetic)
    label_real=0,

    # GENERATOR

    g_hidden_dim=32,
    g_layers=2,
    g_dropout=0.2,
    g_learnrate=0.02,
    g_baseline=0.001,
    g_gamma=1,
    g_bias=True,

    # DISCRIMINATOR

    d_dropout=0.2,
    d_learnrate=0.001
)

config = DEFAULT_CONFIG
