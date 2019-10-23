import torch
import os
import PIL
import random
import torchvision

from torchvision import transforms


class Dataset(torchvision.datasets.vision.VisionDataset):
    """
    A custom Dataset class which inherits from torchvision.datasets.vision.VisionDataset. In contrast to the torch
    dataset classes it does not require the data of a single lable to be located in a specific subdirectory. Instead
    it allows to load the data of a directory and attach a specified label to it. To have data with different labels
    in a single dataset it is possible to request an arbitrary amount of samples from one instance and append it to
    another instance. The transforms of this dataset are specific to the current discriminator layout and hardcoded.
    This dataset only scans for .png and .pt i.e. tensors serialized with pickle files.
    """

    def __init__(self, folder, label=None, recursive=False) -> None:
        """
        The constructor of the Dataset class.

        :param folder: A path to the directory to be scanned for .png or .pt data.
        :param label: An int value specifiyng the label, for more information see the pytorch documentation about
            loss functions for classifiyng tasks. For this script synth label should always be 1 and real data label
            should always be 0.
        :param recursive: If this is set to True subdirectories will be included in the scan.
        """

        super(Dataset, self).__init__(folder)

        self.transform = transforms.Compose([
            lambda img: img.convert(mode='L'),  # fachprojekt uses LA
            transforms.CenterCrop((32, 333)),
            transforms.ToTensor()])
        # lambda img: img[0]]) # fachprojekt takes alpha channel (??)

        self.protocol = {
            '.png': lambda path: PIL.Image.open(path),
            '.pt': lambda path: torch.load(path)
        }

        self.samples = []
        self.index = 0

        self.__crawl__(folder, label, recursive)

    def __len__(self):
        """
        This function returns the length of the dataset.

        :return: Returns the length of the dataset.
        """

        return len(self.samples)

    def __getitem__(self, index):
        """
        This function accesses the element at the given index and returns a tensor representing the loaded image and
        its label.

        :param index: The index for the element returned.
        :return: Returns a tensor of size (32,333) representing an image in the dataset and its label in a tuple.
        """

        path, label, form = self.samples[index]
        image = self.protocol[form](path)
        image = self.transform(image)

        return image, label

    def __crawl__(self, folder, label, recursive):
        """
        This function scans a given directory for .png and .pt files and adds them with their respective label to the
        dataset.

        :param folder: The directory to be scanned for image data.
        :param label: The label of the scanned data.
        :param recursive: If set to True, subdirectories will be scanned too.
        """

        with os.scandir(folder) as iterator:
            for entry in iterator:

                if entry.is_file():
                    if entry.name.endswith('.png'):
                        self.samples.append((folder + '/' + entry.name, label, '.png'))
                    elif entry.name.endswith('.pt'):
                        self.samples.append((folder + '/' + entry.name, label, '.pt'))

                if entry.is_dir() and recursive:
                    self.__crawl__(folder + '/' + entry.name, label, recursive)

    def append(self, other):
        """
        This method appends the data of another instance of this class to the dataset. The raw data can be accessed
        by the random() and inorder() methods. Note that accessing the data by index does not work, because it will
        get transformed to image tensors while it is stored as (file path, label, format) internally.

        :param other: A list of (file path, label, format) tuples of another instance of this class.
        """

        self.samples += other

    def random(self, amount):
        """
        This method returns a given amount of data samples stored in this dataset. Might throw an error if the amount
        exceeds the length of the dataset. The data returned is chosen randomly from all data avalaible.

        :param amount: The amount of samples to return.
        :return: Returns data samples of (file path, label, format) with types (str, int, str).
        """

        return random.sample(self.samples, k=amount)

    def inorder(self, amount):
        """
        This method returns a given amount of data samples stored in this dataset. Might throw an error if the amount
        exceeds the length of the dataset. The data returned starts at index 0. For successive method calls it starts
        at the first index not returned at the last call. If the index pointer exceeds the length of the dataset it
        starts at 0 again.

        :param amount: The amount of samples to return.
        :return: Returns data samples of (file path, label, format) with types (str, int, str).
        """

        if self.index + amount >= len(self):
            samples = self.samples[self.index:]
            samples += self.samples[:(amount - (len(self) - self.index))]

            self.index = amount - (len(self) - self.index)

        else:
            samples = self.samples[self.index:self.index + amount]

            self.index += amount

        return samples
