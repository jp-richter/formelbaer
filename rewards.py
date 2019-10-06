from nn_discriminator import Discriminator
import constants
import torch
from PIL import Image
from torchvision import transforms, datasets


_model = Discriminator().to(constants.DEVICE)
_optimizer = torch.optim.Adam(_model.parameters(), lr=constants.CNN_LEARN_RATE)
_transform = transforms.Compose(
    [lambda img: img.convert(mode='LA'),
    transforms.CenterCrop((32, 333)),
    transforms.ToTensor(),
    lambda img: img[1]])


def rewards(folder):

    _model.eval()

    loading = lambda path: Image.open(path)

    batch = datasets.ImageFolder(folder, transform=_transform,loader=loading)
    loader = torch.utils.data.DataLoader(batch,constants.GRU_BATCH_SIZE)
    images, labels = next(iter(loader))
    images = images[:,None,:,:]

    with torch.no_grad():
        rewards = _model(images)

    return rewards

def train(negatives):

    pass
