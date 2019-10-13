import torch
import os
import PIL
import random
import torchvision


_protocol = {
    '.png' : lambda path : PIL.Image.open(path),
    '.pt' : lambda path : torch.load(path)
}


class Dataset(torchvision.datasets.vision.VisionDataset):

    def __init__(self, folder, label=None, transform=None, recursive=False):
        super(Dataset, self).__init__(folder,transform=transform)

        self.samples = []
        self.transform = transform

        self.crawl(folder, label, recursive)

    def __len__(self): 

        return len(self.samples)

    def __getitem__(self, index):

        path, label, form = self.samples[index]
        image = _protocol[form](path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def crawl(self, folder, label, recursive):

        with os.scandir(folder) as iterator:
            for entry in iterator:
                
                if entry.is_file():
                    if entry.name.endswith('.png'): 
                        self.samples.append((folder + '/' + entry.name, label, '.png'))
                    elif entry.name.endswith('.pt'):
                        self.samples.append((folder + '/' + entry.name, label, '.pt'))

                if entry.is_dir() and recursive:
                    self.crawl(folder + '/' + entry.name, label, recursive)

    def append(self, other):

        self.samples += other

    def random(self, amount):

        return random.sample(self.samples, k=amount)
