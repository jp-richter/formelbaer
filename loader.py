from dataset import Dataset 

import constants as const

import converter
import nn_policy
import math
import generator


arxiv_data = None
oracle_data = None
device = None


def device():

	return device


def refresh():

    shutil.rmtree(const.DIRECTORY_GENERATED_DATA)
    os.makedirs(const.DIRECTORY_GENERATED_DATA)


def save_pngs(samples, directory):

	converter.convert_to_png(samples, directory)


def get_pos_neg_loader(synthetic_samples):

	refresh()

	save_pngs(synthetic_samples, const.DIRECTORY_GENERATED_DATA)
    data = Dataset(const.DIRECTORY_GENERATED_DATA, label=const.LABEL_SYNTH)

	if const.ORACLE:
		data.append(oracle_data.inordner(const.ADVERSARIAL_BATCHSIZE))

	else:
    	data.append(arxiv_data.random(const.ADVERSARIAL_BATCHSIZE))


    loader = DataLoader(data, const.ADVERSARIAL_BATCHSIZE)

    return loader


def load_single_batch(synthetic_samples):

	refresh()

	save_pngs(synthetic_samples, const.DIRECTORY_GENERATED_DATA)
	data = Dataset(DIRECTORY_GENERATED_DATA, label=const.LABEL_SYNTH)
	loader = DataLoader(data, const.ADVERSARIAL_BATCHSIZE)

	return next(iter(loader))


def get_experiment_directory():

    directory = constants.DIRECTORY_APPLICATION + '/'+ str(datetime.datetime.now())
    os.makedirs(directory)

    return directory


def initialize(device):
	global arxiv_data, oracle_data, device

	# make missing directories

	if os.path.exists(const.DIRECTORY_SFB_CLUSTER_ARXIV_DATA):
		const.DIRECTORY_ARXIV_DATA = const.DIRECTORY_SFB_CLUSTER_ARXIV_DATA

	if not os.path.exists(const.DIRECTORY_APPLICATION):
		os.makedirs(const.DIRECTORY_APPLICATION)

	if not os.path.exists(const.DIRECTORY_GENERATED_DATA):
		os.makedirs(const.DIRECTORY_GENERATED_DATA)

	if not os.path.exists(const.DIRECTORY_ORACLE_DATA):
		os.makedirs(const.DIRECTORY_ORACLE_DATA)

	if not os.path.exists(DIRECTORY_ARXIV_DATA):
		pass

	device = device

	# load positive example data either from oracle or arxiv

	if const.ORACLE:

	    # save oracle net with random weights
	    if not os.path.exists(const.FILE_ORACLE):
	        nn_policy.Oracle().save(const.FILE_ORACLE) 

	    # store samples from oracle distribution for adversarial training
	    samplesize = len([name for name in os.listdir(const.DIRECTORY_ORACLE_DATA) 
	    	if os.path.isfile(os.path.join(DIRECTORY_ORACLE_DATA, name))])

	    missing = const.ORACLE_SAMPLESIZE - samplesize
        batch_num = math.ceil(missing / const.ADVERSARIAL_BATCHSIZE)

        samples = generator.sample(nn_oracle, batch_num, const.ADVERSARIAL_SEQUENCE_LENGTH)
        save_pngs(samples, const.DIRECTORY_ORACLE_DATA)

        oracle_data = Dataset(const.DIRECTORY_ORACLE_DATA, label=const.LABEL_ARXIV)

    else:

        arxiv_data = Dataset(const.DIRECTORY_ARXIV_DATA, label=const.LABEL_ARXIV, recursive=True)
