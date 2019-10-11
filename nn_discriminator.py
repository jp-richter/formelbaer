import torch
import math
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


def gelu(x):
	"""
	Implementation of the gelu activation function.
	For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
	0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
	Also see https://arxiv.org/abs/1606.08415
	"""
	
	return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
