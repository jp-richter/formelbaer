import application
import torch
import random
import constants as c
from nn_discriminator import Discriminator
from dataset import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


running_loss = 0.0

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = Discriminator().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=c.DISCRIMINATOR_LEARNRATE)
criterion = torch.nn.BCELoss()

transform = transforms.Compose([
    lambda img: img.convert(mode='LA'),
    transforms.CenterCrop((32, 333)),
    transforms.ToTensor(),
    lambda img: img[1]])
arxiv_label = 0
generated_label = 1
arxiv_data = Dataset(c.DIRECTORY_ARXIV_DATA, label=arxiv_label, transform=transform, recursive=True)


def rewards(folder=c.DIRECTORY_GENERATED_DATA):

    model.eval()

    dataset = Dataset(folder=folder, label=generated_label, transform=transform)
    batchsize = len(dataset)
    loader = DataLoader(dataset, batchsize)

    images, _ = next(iter(loader))
    images.to(device)
    images = images[:,None,:,:]
    rewards = model(images)

    return rewards[:,0] # [:,0] P(x ~ arxiv)

def train(folder=c.DIRECTORY_GENERATED_DATA):
    global running_loss

    model.train()
    optimizer.zero_grad()

    generated_data = Dataset(folder, label=generated_label, transform=transform)
    generated_data.append(arxiv_data.random(amount=len(generated_data)))

    loader = DataLoader(generated_data, half*2)
    images, labels = next(iter(loader))

    images = images[:,None,:,:]

    images.to(device)
    labels.to(device)

    outputs = model(images) 
    loss = criterion(outputs[:,1], labels.float())

    # output[:,0] P(x ~ arxiv)
    # output[:,1] P(x ~ generator)

    running_loss += loss.item()

    loss.backward()
    optimizer.step()

def save_parameters(folder):

    file = folder + 'discriminator_parameters.pt'
    torch.save(model.state_dict(), file)


def load_parameters(folder):

    file = folder + 'discriminator_parameters.pt'
    model.load_state_dict(torch.load(file))

