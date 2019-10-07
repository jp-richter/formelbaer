import torch
import os
import PIL
import constants
import torchvision
import random

from torch.utils.data import DataLoader

_protocol = {
    '.png' : lambda path : PIL.Image.open(path),
    '.pt' : None
}


class Dataset(torchvision.datasets.vision.VisionDataset):

    def __init__(self, folder, label=None, transform=None):
        super(Dataset, self).__init__(folder,transform=transform)

        self.samples = []
        self.transform = transform

        with os.scandir(folder) as iterator:
            for entry in iterator:
                if entry.is_file():
                    if entry.name.endswith('.png'): 
                        self.samples.append((folder + '/' + entry.name, label, '.png'))
                    elif entry.name.endswith('.pt'):
                        self.samples.append((folder + '/' + entry.name, label, '.pt'))

    def __len__(self): 

        return len(self.samples)

    def __getitem__(self, index):

        path, label, form = self.samples[index]
        image = _protocol[form](path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, form

    def random(self, amount):

        return random.choices(self.samples, k=amount)

    def merge(self, data):

        self.samples += data 
