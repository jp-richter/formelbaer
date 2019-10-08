import application
import torch
import random
from nn_discriminator import Discriminator, learnrate
from dataset import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = Discriminator().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learnrate)
criterion = torch.nn.BCELoss()
transform = transforms.Compose(
    [lambda img: img.convert(mode='LA'),
    transforms.CenterCrop((32, 333)),
    transforms.ToTensor(),
    lambda img: img[1]])
arxiv_data = Dataset(application.arxiv_dir, label=0, transform=transform)
running_loss = 0.0


def rewards(folder, batch_size):

    model.eval()

    dataset = Dataset(folder=folder, label=1, transform=transform)
    loader = DataLoader(dataset, batch_size)

    images, _ = next(iter(loader))
    images.to(device)
    images = images[:,None,:,:]
    rewards = model(images)

    return rewards[:,1] # [:,1] probability that p ~ real distribution

def train(folder, batch_size):
    global running_loss

    model.train()
    optimizer.zero_grad()

    generated_data = Dataset(folder, label=1, transform=transform)
    generated_data.append(arxiv_data.random(amount=batch_size))

    loader = DataLoader(generated_data, batch_size)
    iterator = iter(loader)

    for images, labels in iterator:
        images = images[:,None,:,:]

        images.to(device)
        labels.to(device)

        outputs = model(images)
        loss = criterion(outputs[:,1], labels.float())

        running_loss += loss.item()

        loss.backward()
        optimizer.step()
