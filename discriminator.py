import constants
import torch
import random
from nn_discriminator import Discriminator
from dataset import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


_model = Discriminator().to(constants.DEVICE)
_optimizer = torch.optim.Adam(_model.parameters(), lr=constants.CNN_LEARN_RATE)
_criterion = torch.nn.BCELoss()
_transform = transforms.Compose(
    [lambda img: img.convert(mode='LA'),
    transforms.CenterCrop((32, 333)),
    transforms.ToTensor(),
    lambda img: img[1]])
_arxiv_data = Dataset(constants.ARXIV, label=0, transform=_transform)


def rewards(folder):

    _model.eval()

    dataset = Dataset(folder=folder, label=1, transform=_transform)
    loader = DataLoader(dataset, constants.BATCH_SIZE)

    images, _ = next(iter(loader))
    images.to(constants.DEVICE)
    images = images[:,None,:,:]
    rewards = _model(images) 

    return rewards

def train():

    _model.train()
    _optimizer.zero_grad()

    generated_data = Dataset(constants.GENERATED, label=1, transform=_transform)
    generated_data.append(_arxiv_data.random())

    loader = DataLoader(generated_data, constants.BATCH_SIZE)
    iterator = iter(loader)

    for images, labels in iterator:
        images = images[:,None,:,:]

        images.to(constants.DEVICE)
        labels.to(constants.DEVICE)

        outputs = _model(images)
        loss = _criterion(outputs, labels.float()[:,None])

        loss.backward()
        _optimizer.step()
