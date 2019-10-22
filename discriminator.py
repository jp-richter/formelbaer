import config as cfg
import torch
from torch import nn


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.pool2x4 = nn.MaxPool2d((2, 4))
        self.pool3x3 = nn.MaxPool2d((3, 3))
        self.pool2d = nn.AvgPool2d((1, 15))

        in_ch = 1
        out_ch = 32

        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3))
        self.conv2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 5))
        self.conv3 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3))

        self.fc3 = nn.Linear(5 * 32, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32,2)

        self.selu = nn.SELU()
        self.softmax = nn.Softmax(dim=0)

        self.running_loss = 0.0

    def forward(self,x):

        out = self.conv1(x)
        out = self.selu(out)
        out = self.pool2x4(out)

        out = self.conv2(out)
        out = self.selu(out)
        out = self.pool2x4(out)

        out = self.conv3(out)
        out = self.selu(out)
        out = self.pool3x3(out)

        out = out.view(-1, 5 * 32)
        out = self.fc3(out)
        out = self.selu(out)

        out = self.fc4(out)
        out = self.selu(out)

        out = self.fc5(out)
        out = self.softmax(out)

        return out

    def save(self, file):

        torch.save(self.state_dict(), file)

    def load(self, file):

        self.load_state_dict(torch.load(file))


def evaluate(nn_discriminator, image_batch):

    # image_batch: (batch_size, height, width)
    rewards = nn_discriminator(image_batch)

    return rewards[:,0][:,None] # [:,0] P(x ~ arxiv / oracle)


def update(nn_discriminator, d_opt, d_crit, loader):

    for images, labels in loader:

        images.to(cfg.app_cfg.device)
        labels.to(cfg.app_cfg.device)

        outputs = nn_discriminator(images)
        loss = d_crit(outputs[:,1], labels.float())

        # output[:,0] P(x ~ arxiv / oracle)
        # output[:,1] P(x ~ generator)

        nn_discriminator.running_loss += loss.item()

        loss.backward()
        d_opt.step()