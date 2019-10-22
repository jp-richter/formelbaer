from dataset import Dataset
from torch.utils.data import DataLoader

import config as cfg
import converter
import math
import os
import shutil
import generator
import datetime
import tokens


arxiv_data = None
oracle_data = None


def refresh():

    shutil.rmtree(cfg.paths_cfg.synthetic_data)
    os.makedirs(cfg.paths_cfg.synthetic_data)


def save_pngs(samples, directory):

    converter.convert_to_png(samples, directory)


def save_sequences(samples, directory):

    with open(directory + '/sequences.txt','w') as f:

        sequences = []
        strings = []

        for sample in samples:
            sequence = []

            for onehot in sample:
                sequence.append(tokens.id(onehot))

            sequences.append(sequence)
            strings.append(', '.join(str(s) for s in sequence))

        all_sequences = '\n'.join(str(s) for s in sequences)

        f.write(all_sequences)


def get_pos_neg_loader(synthetic_samples):

    refresh()

    save_pngs(synthetic_samples, cfg.paths_cfg.synthetic_data)
    data = Dataset(cfg.paths_cfg.synthetic_data, label=cfg.app_cfg.label_synth)

    if cfg.app_cfg.oracle:
        data.merge(oracle_data.inorder(cfg.app_cfg.batchsize))
    else:
        data.merge(arxiv_data.inorder(cfg.app_cfg.batchsize))

    return DataLoader(data, batch_size=cfg.app_cfg.batchsize, drop_last=True, shuffle=True)


def load_single_batch(synthetic_samples):

    refresh()

    save_pngs(synthetic_samples, cfg.paths_cfg.synthetic_data)
    data = Dataset(cfg.paths_cfg.synthetic_data, label=cfg.app_cfg.label_synth)
    loader = DataLoader(data, cfg.app_cfg.batchsize)

    return next(iter(loader))[0] # (samples, labels)


def get_experiment_directory():

    directory = cfg.paths_cfg.app + '/' + str(datetime.datetime.now())[-15:]
    os.makedirs(directory)

    return directory


def make_directories():

    if not os.path.exists(cfg.paths_cfg.app):
        os.makedirs(cfg.paths_cfg.app)

    if not os.path.exists(cfg.paths_cfg.synthetic_data):
        os.makedirs(cfg.paths_cfg.synthetic_data)

    if not os.path.exists(cfg.paths_cfg.oracle_data):
        os.makedirs(cfg.paths_cfg.oracle_data)

    if not os.path.exists(cfg.paths_cfg.arxiv_data) and not cfg.app_cfg.oracle:
        raise ValueError('Either train with Oracle or provide training samples.')

    if not os.path.exists(cfg.paths_cfg.dump):
        open(cfg.paths_cfg.dump, 'w+')


def load_oracle_data():
    global oracle_data

    nn_oracle = generator.Oracle()

    # generate oracle net with random weights
    if not os.path.exists(cfg.paths_cfg.oracle):
        nn_oracle.save(cfg.paths_cfg.oracle) 
    else:
        nn_oracle.load(cfg.paths_cfg.oracle)

    # store samples from oracle distribution for adversarial training
    samplesize = len([name for name in os.listdir(cfg.paths_cfg.oracle_data) 
        if os.path.isfile(os.path.join(cfg.paths_cfg.oracle_data, name))])

    missing = cfg.app_cfg.oracle_samplesize - samplesize
    batch_num = math.ceil(missing / cfg.app_cfg.batchsize)

    samples = generator.sample(nn_oracle, batch_num)
    save_pngs(samples, cfg.paths_cfg.oracle_data)

    oracle_data = Dataset(cfg.paths_cfg.oracle_data, label=cfg.app_cfg.label_arxiv)


def load_arxiv_data(log):
    global arxiv_data

    arxiv_data = Dataset(cfg.paths_cfg.arxiv_data, label=cfg.app_cfg.label_arxiv, recursive=True)

    print('{} samples loaded.'.format(len(arxiv_data)))
    log.log.info('{} samples loaded.'.format(len(arxiv_data)))

    provided = len(arxiv_data)
    needed = cfg.app_cfg.batchsize * cfg.app_cfg.d_steps * cfg.app_cfg.iterations

    message = '''Either provide more training samples or parameters:
            Batchsize {}
            Discriminator Steps {}
            Iterations {}
            Positive Samples Needed {}
            Arxiv Samples Provided {}'''.format(
                cfg.app_cfg.batchsize,
                cfg.app_cfg.d_steps,
                cfg.app_cfg.iterations,
                needed,
                provided)

    if provided < needed:
        log.log.error(message)
        raise ValueError(message)


def load_data(log):

    if cfg.app_cfg.oracle: load_oracle_data()
    else: load_arxiv_data(log)