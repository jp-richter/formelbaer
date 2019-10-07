from nn_discriminator import Discriminator
import constants
import torch
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
_positives = Dataset(constants.ARXIV, label=0, transform=_transform)


def rewards(folder):

    _model.eval()

    # TODO mal irgendwie sinnvoll mit den loadern umgehen

    dataset = Dataset(folder=folder, transform=_transform)
    loader = torch.utils.data.DataLoader(dataset, constants.GRU_BATCH_SIZE,drop_last=True)
    images, _, _ = next(iter(loader))
    images = images[:,None,:,:]
    rewards = _model(images) 

    return rewards

def train():

    _model.train()
    _optimizer.zero_grad()

    # TODO uberpruefen ob man loader seltener initialisieren und daten 
    # mehr auf einmal laden kann - mehr als nur einen batch?

    negatives = Dataset(constants.GENERATED, label=1, transform=_transform)
    negatives.merge(_positives.random(constants.GRU_BATCH_SIZE))

    loader = DataLoader(negatives, 2*constants.GRU_BATCH_SIZE,shuffle=True,num_workers=2,drop_last=True)
    images, labels, _ = next(iter(loader))
    images = images[:,None,:,:]

    images.to(constants.DEVICE)
    # labels.to(constants.DEVICE)

    outputs = _model(images)
    loss = _criterion(outputs, labels.float()[:,None])

    loss.backward()
    _optimizer.step()
