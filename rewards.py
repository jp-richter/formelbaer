from nn_discriminator import Discriminator
import constants
import torch
from PIL import Image
from torchvision import transforms, datasets


_model = Discriminator().to(constants.DEVICE)
_optimizer = torch.optim.Adam(_model.parameters(), lr=constants.CNN_LEARN_RATE)
_criterion = torch.nn.CrossEntropyLoss()
_loading = lambda path: Image.open(path)
_transform = transforms.Compose(
    [lambda img: img.convert(mode='LA'),
    transforms.CenterCrop((32, 333)),
    transforms.ToTensor(),
    lambda img: img[1]])


def rewards(folder):

    _model.eval()

    batch = datasets.ImageFolder(folder, transform=_transform,loader=_loading)
    loader = torch.utils.data.DataLoader(batch,constants.GRU_BATCH_SIZE)
    images, _ = next(iter(loader))
    images = images[:,None,:,:]
    rewards = _model(images) 

    return rewards

def train():

    _model.train()
    _optimizer.zero_grad()

    loader = None
    images = next(iter(loader))
    inputs, labels = images[0].to(constants.DEVICE), images[1].to(constants.DEVICE)

    outputs = _model(inputs)
    loss = _criterion(outputs, labels)

    loss.backward()
    _optimizer.step()
