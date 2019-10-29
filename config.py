import multiprocessing
import torch

from pathlib import Path
from dataclasses import dataclass


@dataclass
class Paths:
    """
    Configuration data class which contains all path variables of the script. To change paths make changes to the
    DEFAULT_PATHS_CFG instance or overwrite the cfg_paths variable with your own instance.
    """

    # the root directory of the script
    app: str

    # this directory stores synthetic data temporarily for conversions and evaluation by the discriminating net
    synthetic_data: str

    # positive data samples of the real distribution should be saved here
    arxiv_data: str

    # fake positive oracle data samples will be saved here
    oracle_data: str

    # the log file contains timestamps of loading times and iterations, loss and reward sequences for parsing
    log: str

    # if an oracle has been generated it will be saved here for future comparisons
    oracle: str

    dump: str

    # used by ray for shared memory plasma store
    ray: str


@dataclass
class AppConfig:
    """
    Configuration data class which contains various training parameters. To change training parameters make changes
    to the DEFAUT_APP_CFG instance or overwrite the cfg_app variable with your own instance.
    """

    device: torch.device

    # the amount of samples used as real dataset in mle pretraining and to train the discriminator in the adversarial
    # training phase. if oracle=True this amount will be generated by the oracle, else its the upper amount of samples
    # used from the arxiv dataset since training on the full set is infeasible due to computational cost
    size_real_dataset: int

    # the amount of samples used to evaluate the oracle score
    num_eval_samples: int

    # the amount of training steps with mle
    mle_epochs: int

    # the amount of training steps with kldiv
    kldiv_epochs: int

    # the amount of adversarial training steps, g steps for the generator and d steps for the discriminator
    iterations: int

    # the amount of epochs the discriminator trains on the synthetic and real data for one d step
    d_epochs: int

    # discriminator training steps per iteration, for each iteration the discriminator trains on (d steps * batch size)
    # negative and (d steps * batch size) positive samples
    d_steps: int

    # generator training steps per iteration, for each iteration the generator trains on (g steps * batch size)
    # sequences and for (g steps * batch size * sequence length) generating steps
    g_steps: int

    # the fixed length of the generated sequences
    sequence_length: int

    # the amount of rollouts to estimate rewards for unfinished sequences
    montecarlo_trials: int

    # should be equal to or a multiple of the cpu count for performance reasons
    batch_size: int

    # if true, the training target will be an fake real distribution represented by an oracle policy instead of arxiv
    oracle: bool

    # the labels should always set to synth=1 and arxiv=0, do not overwrite
    label_synth: int
    label_real: int


@dataclass
class GeneratorConfig:
    """
    Configuration data class which contains various training and structural parameters specific to the generating
    neural net. To change parameters make changes to the DEFAUT_GENERATOR_CFG instance or overwrite the cfg_g variable
    with your own instance.
    """

    hidden_dim: int
    layers: int
    dropout: float
    learnrate: float
    baseline: int
    gamma: float


@dataclass
class DiscriminatorConfig:
    """
    Configuration data class which contains various training parameters specific to the discriminating neural net.
    To change training parameters make changes to the DEFAUT_DISCRIMINATOR_CFG instance or overwrite the cfg_d variable
    with your own instance.
    """

    dropout: float
    learnrate: float


# '/rdata/schill/arxiv_processed/all/pngs'
# '/rdata/schill/arxiv_processed/ba_data/ml_data/train/pngs'

_home = '/ramdisk' if multiprocessing.cpu_count() > 4 else str(Path.home())

DEFAULT_PATHS = Paths(

    app=_home + '/formelbaer-data',
    synthetic_data=_home + '/formelbaer-data' + '/synthetic-data',
    arxiv_data=_home + '/formelbaer-data' + '/arxiv-data',
    oracle_data=_home + '/formelbaer-data' + '/oracle-data',

    log=_home + '/formelbaer-data' + '/results.log',
    oracle=_home + '/formelbaer-data' + '/oracle-net.pt',

    dump=_home + '/formelbaer-data' + '/dump.txt',

    ray=_home + '/ray-plasma-store'

)

paths = DEFAULT_PATHS

DEFAULT_GENERAL = AppConfig(

    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),

    size_real_dataset=5000,  # 10.000
    num_eval_samples=100,  # 100

    mle_epochs=2,
    kldiv_epochs=2,

    d_epochs=1,  # 3
    iterations=150,  # 150

    d_steps=1,
    g_steps=1,
    sequence_length=20,  # 16
    montecarlo_trials=10,  # 16
    batch_size=multiprocessing.cpu_count(),  # computational cost reasons

    oracle=False,

    label_synth=1,  # discriminator outputs P(x ~ synthetic)
    label_real=0

)

# overwrite this to change parameters
general = DEFAULT_GENERAL

DEFAULT_GENERATOR = GeneratorConfig(

    hidden_dim=32,
    layers=2,
    dropout=0.2,
    learnrate=0.02,
    baseline=0,
    gamma=0.99  # TODO mal gamma=1 probieren?

)

# overwrite this to change parameters
generator = DEFAULT_GENERATOR

DEFAULT_DISCRIMINATOR = DiscriminatorConfig(

    dropout=0.2,
    learnrate=0.001

)

# overwrite this to change parameters
discriminator = DEFAULT_DISCRIMINATOR
