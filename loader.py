from dataset import Dataset
from torch.utils.data import DataLoader

import torch
import config
import converter
import math
import os
import shutil
import generator
import datetime
import tokens
import ray
import log

arxiv_dataset = None
oracle_dataset = None


def clear_directory(directory) -> None:
    """
    This function deletes all files in the synthetic data directory. The synthetic data directory serves as temporary
    store for data samples meant to be evaluated by the discriminating net. This function should be called after the
    evaluation or before the next evaluation to avoid evaluating the same data again.
    """

    shutil.rmtree(directory)
    os.makedirs(directory)


def save_pngs(samples, directory) -> None:
    """
    This function saves the given batch of samples produced by the generator to the given directory in the .png format.
    It also accepts lists of batches.

    :param samples: The samples that should be converted and saved. Needs to be a torch.Tensor of size (batch size,
        sequence length, one hot encoding length) or a list of such tensor objects.
    :param directory: The path to the directory to which the .png files will be written to.
    """

    converter.convert_to_png(samples, directory)


def save_sequences(samples, directory) -> None:
    """
    This function saves the sequences of integer ids for a given batch or list of sequences to the given directory. It
    is especially useful to see nodes of the syntax trees which will get ignored by the tree generation to assert
    their grammatical correctness. Also these integer id sequences allow to construct syntax trees whereas the saved
    png files only allow visual feedback of the generators performance. The sequences will be saved in text files as
    strings seperated by ',' f.e. 31,53,2,1 with every line representing a single sequence.

    :param samples: A batch in form of a tensor of size (batch size, sequence length, onehot encoding length) or a list
        of such tensors to save.
    :param directory: The path to the target directory the sequences.txt file will be written to.
    """

    with open(directory + '/sequences.txt', 'w') as f:

        sequences = []
        strings = []

        def process(single_batch):
            for sequence in single_batch:
                sequence_ids = []

                for onehot in sequence:
                    sequence_ids.append(tokens.id(onehot))

                sequences.append(sequence_ids)
                strings.append(', '.join(str(s) for s in sequence_ids))

            all_sequences = '\n'.join(str(s) for s in sequences)
            f.write(all_sequences)

        if isinstance(samples, list):
            for batch in samples:
                process(batch)

        else:
            process(samples)


def make_directory_with_timestamp() -> str:
    """
    This function creates a directory named with the current time in the main app directory specified by the current
    configuration. This is useful to get unique directory names when saving experiment data, as the time stamp includes
    milliseconds.

    :return: The path to the directory created.
    """

    directory = config.paths.app + '/' + str(datetime.datetime.now())[-15:]
    os.makedirs(directory)

    return directory


def make_dataset(directory, nn_generator, label, num_batches) -> Dataset:

    clear_directory(directory)
    batches = generator.sample(nn_generator, num_batches)
    save_pngs(batches, directory)
    dataset = Dataset(directory, label)

    return dataset


def prepare_batch(batch) -> torch.Tensor:

    clear_directory(config.paths.synthetic_data)
    save_pngs(batch, config.paths.synthetic_data)
    dataset = Dataset(config.paths.synthetic_data, config.general.label_synth)
    loader = DataLoader(dataset, config.general.batch_size)
    images = next(iter(loader))[0]  # (images, labels)
    images = images.to(config.general.device)

    return images


def prepare_oracle_loader(num_samples, nn_generator, nn_oracle) -> DataLoader:
    pass


def prepare_arxiv_loader(num_samples, nn_generator, nn_oracle) -> DataLoader:
    dataset = Dataset()

    if not nn_generator is None:
        num_batches = math.ceil(num_samples / config.general.batch_size) // 2
        dataset = make_dataset(config.paths.synthetic_data, nn_generator, config.general.label_synth, num_batches)

        arxiv_samples = arxiv_dataset.inorder(num_samples // 2)

    else:
        arxiv_samples = arxiv_dataset.inorder(num_samples)

    dataset.append(arxiv_samples)
    data_loader = DataLoader(dataset, config.general.batch_size, shuffle=True)

    return data_loader


def prepare_loader(num_samples, nn_generator, nn_oracle) -> DataLoader:

    if config.general.oracle:
        return prepare_oracle_loader(num_samples, nn_generator, nn_oracle)

    return prepare_arxiv_loader(num_samples, nn_generator, nn_oracle)


def initialize() -> None:
    """
    This function loads the dataset of real samples to train the discriminator with. If oracle is set to True in the
    current configuration fake real samples of the oracle will be loaded instead of arxiv data. All directories used
    by the script will be created here. Not calling this function at the beginning of the script will lead to errors.
    """

    global oracle_dataset, arxiv_dataset

    log.init()
    print('Start initializing..')
    log.log.info('Start initializing..')

    if not os.path.exists(config.paths.app):
        os.makedirs(config.paths.app)

    if not os.path.exists(config.paths.synthetic_data):
        os.makedirs(config.paths.synthetic_data)

    if not os.path.exists(config.paths.oracle_data):
        os.makedirs(config.paths.oracle_data)

    if not os.path.exists(config.paths.arxiv_data) and not config.general.oracle:
        raise ValueError('Either train with Oracle or provide training samples at ' + config.paths.arxiv_data + '.')

    if not os.path.exists(config.paths.dump):
        open(config.paths.dump, 'w+')

    if not os.path.exists(config.paths.ray):
        os.makedirs(config.paths.ray)

    if not ray.is_initialized():
        if torch.cuda.is_available():
            ray.init(plasma_directory=config.paths.ray, memory=20000000000, object_store_memory=20000000000)
        else:
            ray.init(plasma_directory=config.paths.ray, memory=5000000000, object_store_memory=5000000000)

    if not config.general.oracle:
        arxiv_dataset = Dataset(config.paths.arxiv_data, label=config.general.label_real, recursive=True)

    else:
        oracle_dataset = Dataset(config.paths.oracle_data, label=config.general.label_real)

    print('Finished initializing.')
    log.log.info('Finished initializing.')


def finish(nn_policy, nn_discriminator, nn_oracle) -> None:

    directory = make_directory_with_timestamp()
    nn_policy.save(directory + '/policy-net.pt')
    nn_discriminator.save(directory + '/discriminator-net.pt')
    nn_oracle.save(directory + '/oracle-net.pt')

    log.finish_experiment(directory)

    evaluation = generator.sample(nn_policy, math.ceil(100 / config.general.batch_size))

    os.makedirs(directory + '/pngs')
    os.makedirs(directory + '/sequences')
    save_pngs(evaluation, directory + '/pngs')
    save_sequences(evaluation, directory + '/sequences')

    ray.shutdown()
