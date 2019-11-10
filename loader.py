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

        for sequence in samples:
            sequence_ids = []

            for onehot in sequence:
                sequence_ids.append(tokens.id(onehot))

            sequences.append(sequence_ids)
            strings.append(', '.join(str(s) for s in sequence_ids))

        all_sequences = '\n'.join(str(s) for s in sequences)
        f.write(all_sequences)


def make_directory_with_timestamp() -> str:
    """
    This function creates a directory named with the current time in the main app directory specified by the current
    configuration. This is useful to get unique directory names when saving experiment data, as the time stamp includes
    milliseconds.

    :return: The path to the directory created.
    """

    directory = config.paths.rdata + '/' + str(datetime.datetime.now())
    directory = directory.replace(':','-').replace(' ', '-')[:-7]
    os.makedirs(directory)

    return directory


def make_dataset(directory, nn_generator, label, num_batches) -> Dataset:
    """
    This function creates a Dataset of type dataset.Dataset with samples generated by the given generating net. The data
    gets stored in the given directory. Note that the dataset only saves image paths and not the images itsself. If the
    data gets removed the dataset becomes invalid.

    :param directory: The path to the directory in which the generated samples will be stored for the dataset.
    :param nn_generator: The net generating the data samples for the dataset.
    :param label: The label for the generated data, should be equal to label_synth in configs. Can be None.
    :param num_batches: The amount of batches generated with the generator net.
    :return: Returns a dataset of type dataset.Dataset.
    """

    clear_directory(directory)
    sequences = generator.sample(nn_generator, num_batches)
    save_pngs(sequences, directory)
    dataset = Dataset(directory, label)

    return dataset


def prepare_batch(batch) -> torch.Tensor:
    """
    This function prepares a batch of synthetic data and returns only a batch of images without a label. It stores the
    data in the synthetic data path of the script temporarily. It also deletes all contents of that directory
    beforehand. This function is useful to evaluate a single batch of sequences generated by the polcy net as the
    output fits as input to the discriminating net.

    :param batch: torch.Tensor of size (batch size, sequence length, input dimension) with sequences of onehot encodings
        to be converted to pngs and then loaded as image batch.
    :return: Returns a tensor of an image batch with size (batch size, 1, height, width).
    """

    clear_directory(config.paths.synthetic_data)
    save_pngs(batch, config.paths.synthetic_data)
    dataset = Dataset(config.paths.synthetic_data, config.general.label_synth)
    loader = DataLoader(dataset, config.general.batch_size)
    images = next(iter(loader))[0]  # (images, labels)
    images = images.to(config.general.device)

    return images


def prepare_oracle_loader(num_samples, nn_generator, nn_oracle) -> DataLoader:
    """
    This function prepares a torch DataLoader for oracle and generated data to equal amounts. The output will be of
    type (image batch, label batch). Consider reusing the same dataloader if computational costs are a concern.

    :param num_samples: The amount of samples the loader should hold.
    :param nn_generator: The generating net used to provide negative samples.
    :param nn_oracle: The oracle net providing the fake real samples.
    :return Returns a torch.util.DataLoader with output of type (image batch, label batch).
    """

    global oracle_dataset

    num_batches_each = math.ceil(num_samples / config.general.batch_size) // 2

    provided = len(oracle_dataset)
    missing = num_batches_each * config.general.batch_size - provided
    num_batches = math.ceil(missing / config.general.batch_size)
    samples = generator.sample(nn_oracle, num_batches)
    save_pngs(samples, config.paths.oracle_data)
    oracle_dataset = Dataset(config.paths.oracle_data, config.general.label_real)

    samples = generator.sample(nn_generator, num_batches_each)
    save_pngs(samples, config.paths.synthetic_data)
    dataset = Dataset(config.paths.synthetic_data, config.general.label_synth)

    dataset.append(oracle_dataset.inorder(num_batches_each * config.general.batch_size))
    data_loader = DataLoader(dataset, config.general.batch_size)

    return data_loader


def prepare_arxiv_loader(num_samples, nn_generator=None) -> DataLoader:
    """
    This function prepares a torch DataLoader for the arxiv data in the arxiv-samples directory. If a generator is
    provided, arxiv samples and synthetic samples generated by the generator will be mixed to equal amounts. In both
    samples the loader will hold num_samples of datapoints (image, label). The loader will be set to shuffle=True to
    make sure positive and negative samples do get mixed. Every time this function is called new arxiv samples will be
    loaded until the whole dataset has been iterated through.

    :param num_samples: The amount of samples the loader should hold.
    :param nn_generator: The generator used to generate negative samples additionally to the arxiv data, if provided.
    :return: Returns a torch.util.DataLoader with shuffle=True.
    """

    dataset = Dataset()

    if nn_generator is not None:
        num_batches = math.ceil((num_samples / config.general.batch_size) / 2)
        dataset = make_dataset(config.paths.synthetic_data, nn_generator, config.general.label_synth, num_batches)

        arxiv_samples = arxiv_dataset.inorder(num_samples // 2)

    else:
        arxiv_samples = arxiv_dataset.inorder(num_samples)

    dataset.append(arxiv_samples)
    data_loader = DataLoader(dataset, config.general.batch_size, shuffle=True)

    return data_loader


def prepare_loader(num_samples, nn_generator, nn_oracle) -> DataLoader:
    """
    This function returns a torch.util.DataLoader either with arxiv or oracle samples dependig on the current config.
    For documentation look at prepare_arxiv_loader() and prepare_oracle_loader().

    :param num_samples: The amount of samples the loader should hold.
    :param nn_generator: If provided, will be used to generate negative samples.
    :param nn_oracle: If provided, will be used to generate fake positive samples.
    :return: Returns a torch.util.DataLoader with samples (image,label) to iterate on.
    """

    if config.general.oracle:
        return prepare_oracle_loader(num_samples, nn_generator, nn_oracle)

    return prepare_arxiv_loader(num_samples, nn_generator)


def initialize() -> None:
    """
    This function sets up arxiv and oracle datasets dependant on the configurations. Ray gets initialized and the shared
    plasma object store will be created at config.paths.ray. All directories used by the script will be created here.
    Not calling this function at the beginning of the script will lead to errors.
    """

    global oracle_dataset, arxiv_dataset

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

    if not os.path.exists(config.paths.policies):
        os.makedirs(config.paths.policies)

    else:
        clear_directory(config.paths.policies)

    if not os.path.exists(config.paths.log):
        open(config.paths.log, 'w')

    if not ray.is_initialized():
        if torch.cuda.is_available():
            ray.init(plasma_directory=config.paths.ray, memory=20000000000, object_store_memory=20000000000)
        else:
            ray.init(plasma_directory=config.paths.ray, memory=5000000000, object_store_memory=5000000000)

    if not config.general.oracle:
        arxiv_dataset = Dataset(config.paths.arxiv_data, label=config.general.label_real, recursive=True)

    else:
        oracle_dataset = Dataset(config.paths.oracle_data, label=config.general.label_real)

    log.init()
    log.start_experiment()


def finish(nn_policy, nn_discriminator, nn_oracle) -> None:
    """
    This function creates a directory with the current timestamp at the application path set in the configurations.
    All experimental result data of a run such as weight parameters and log files will be saved. Additionally 100
    example images and sequences will be saved in this directory.

    :param nn_policy: The policy net used in the experiment and for which the example data should be generated.
    :param nn_discriminator: The discriminating net used in the experiment.
    :param nn_oracle: The oracle net used in the experiment, in case oracle training has been used.
    """
    directory = make_directory_with_timestamp()
    nn_policy.save(directory + '/policy-net.pt')
    nn_discriminator.save(directory + '/discriminator-net.pt')
    nn_oracle.save(directory + '/oracle-net.pt')

    evaluation = generator.sample(nn_policy, math.ceil(100 / config.general.batch_size))

    os.makedirs(directory + '/pngs')
    os.makedirs(directory + '/sequences')
    save_pngs(evaluation, directory + '/pngs')
    save_sequences(evaluation, directory + '/sequences')

    log.finish_experiment(directory)


def shutdown() -> None:
    """
    This function shuts down the ray connection and frees the shared memory used by the ray plasma object store. Should
    be called once at the end of the script.
    """

    ray.shutdown()
