import torch
import os
import PIL
import random
import torchvision


class Dataset(torchvision.datasets.vision.VisionDataset):

    def __init__(self, folder, label=None, recursive=False):
        super(Dataset, self).__init__(folder)

        self.transform = transforms.Compose([
            lambda img: img.convert(mode='LA'),
            transforms.CenterCrop((32, 333)),
            transforms.ToTensor(),
            lambda img: img[1]])

        self.protocol = {
            '.png' : lambda path : PIL.Image.open(path),
            '.pt' : lambda path : torch.load(path)
        }

        self.samples = []
        self.__crawl__(folder, label, recursive)

    def __len__(self): 

        return len(self.samples)

    def __getitem__(self, index):

        path, label, form = self.samples[index]
        image = self.protocol[form](path)
        image = self.transform(image)

        return image, float(label)

    def __crawl__(self, folder, label, recursive):

        with os.scandir(folder) as iterator:
            for entry in iterator:
                
                if entry.is_file():
                    if entry.name.endswith('.png'): 
                        self.samples.append((folder + '/' + entry.name, label, '.png'))
                    elif entry.name.endswith('.pt'):
                        self.samples.append((folder + '/' + entry.name, label, '.pt'))

                if entry.is_dir() and recursive:
                    self.__crawl__(folder + '/' + entry.name, label, recursive)

    def merge(self, other):

        self.samples += other

    def random(self, amount):

        return random.sample(self.samples, k=amount)
