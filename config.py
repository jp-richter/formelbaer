import multiprocessing
import torch

from pathlib import Path
from dataclasses import dataclass


@dataclass
class Paths:
    """Configuration data class which contains all path variables of the script. To change paths make changes to the
    DEFAULT_PATHS_CFG instance or overwrite the cfg_paths variable with your own instance."""

    app: str
    synthetic_data: str
    arxiv_data: str
    oracle_data: str

    log: str
    oracle: str

    dump: str


@dataclass
class AppConfig:
    """Configuration data class which contains various training parameters. To change training parameters make changes
    to the DEFAUT_APP_CFG instance or overwrite the cfg_app variable with your own instance."""

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
    """Configuration data class which contains various training and structural parameters specific to the generating
    neural net. To change parameters make changes to the DEFAUT_GENERATOR_CFG instance or overwrite the cfg_g variable
    with your own instance."""

    hidden_dim: int
    layers: int
    dropout: float
    learnrate: float
    baseline: int
    gamma: float


@dataclass
class DiscriminatorConfig:
    """Configuration data class which contains various training parameters specific to the discriminating neural net.
    To change training parameters make changes to the DEFAUT_DISCRIMINATOR_CFG instance or overwrite the cfg_d variable
    with your own instance."""

    dropout: float
    learnrate: float


# '/rdata/schill/arxiv_processed/all/pngs'
# '/rdata/schill/arxiv_processed/ba_data/ml_data/train/pngs'
# '/ramdisk/arxiv'

_home = str(Path.home())

DEFAULT_PATHS_CFG = Paths(

    app=_home + '/formelbaer-data',
    synthetic_data=_home + '/formelbaer-data' + '/synthetic-data',
    arxiv_data=_home + '/formelbaer-data' + '/arxiv-data',
    oracle_data=_home + '/formelbaer-data' + '/oracle-data',

    log=_home + '/formelbaer-data' + '/results.log',
    oracle=_home + '/formelbaer-data' + '/oracle-net.pt',

    dump=_home + '/formelbaer-data' + '/dump.txt'

)

paths_cfg = DEFAULT_PATHS_CFG

DEFAULT_APP_CFG = AppConfig(

    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),

    iterations=150,
    d_steps=2,  # (*2) due to computational cost reasons
    g_steps=1,
    seq_length=16,  # 16
    montecarlo_trials=16,  # 16
    batchsize=multiprocessing.cpu_count(),  # computational cost reasons

    oracle=False,
    oracle_samplesize=100,

    label_synth=1,
    label_arxiv=0

)

# overwrite this to change parameters
app_cfg = DEFAULT_APP_CFG

DEFAULT_GENERATOR_CFG = GeneratorConfig(

    hidden_dim=32,
    layers=2,
    dropout=0.2,
    learnrate=0.01,
    baseline=1,
    gamma=0.99

)

# overwrite this to change parameters
g_cfg = DEFAULT_GENERATOR_CFG

DEFAULT_DISCRIMINATOR_CFG = DiscriminatorConfig(

    dropout=0.2,
    learnrate=0.01

)

# overwrite this to change parameters
d_cfg = DEFAULT_DISCRIMINATOR_CFG
