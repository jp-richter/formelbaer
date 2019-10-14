from dataset import Dataset 

import constants as const

import converter


arxiv_data = None
device = None


def initialize(device):
	global arxiv_data, device

	if os.path.exists(const.DIRECTORY_SFB_CLUSTER_ARXIV_DATA):
		const.DIRECTORY_ARXIV_DATA = const.DIRECTORY_SFB_CLUSTER_ARXIV_DATA

	if not os.path.exists(const.DIRECTORY_APPLICATION):
		os.makedirs(const.DIRECTORY_APPLICATION)

	if not os.path.exists(const.DIRECTORY_GENERATED_DATA):
		os.makedirs(const.DIRECTORY_GENERATED_DATA)

	if not os.path.exists(DIRECTORY_ARXIV_DATA):
		raise ValueError()

	device = device
	arxiv_data = Dataset(arxiv_samples_directory, label=const.LABEL_ARXIV, recursive=True)


def device():

	return device


def refresh():

    shutil.rmtree(const.DIRECTORY_GENERATED_DATA)
    os.makedirs(const.DIRECTORY_GENERATED_DATA)


def save_pngs(samples, directory=const.DIRECTORY_GENERATED_DATA):

	converter.convert_to_png(samples, directory)


def get_pos_neg_loader(synthetic_samples):

	refresh()

	save_pngs(synthetic_samples)
    data = Dataset(DIRECTORY_GENERATED_DATA, label=const.LABEL_SYNTH)
    data.append(arxiv_data.random(amount=len(data)))
    loader = DataLoader(synth_data, const.ADVERSARIAL_BATCHSIZE)

    return loader


def load_single_batch(synthetic_samples):

	refresh()

	save_pngs(synthetic_samples)
	data = Dataset(DIRECTORY_GENERATED_DATA, label=const.LABEL_SYNTH)
	loader = DataLoader(data, const.ADVERSARIAL_BATCHSIZE)

	return next(iter(loader))


def get_experiment_directory():

    directory = constants.DIRECTORY_APPLICATION + '/'+ str(datetime.datetime.now())
    os.makedirs(directory)

    return directory
