import torch
import numpy
import matplotlib.pyplot as plt


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.input_dim = 10
        self.hidden_dim = 50
        self.output_dim = 10

        self.lin1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.lin2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin3 = torch.nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        out = self.lin1(x)
        out = self.relu(out)

        out = self.lin2(out)
        out = self.relu(out)

        out = self.lin3(out)
        out = self.relu(out)

        out = self.softmax(out)

        return out


def main():
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)

    random = torch.randn(10)
    rewards = torch.tensor([100, 50, 30, 20, 10, 10, 10, 10, 10, 10]).float()

    for i in range(1000):
        opt.zero_grad()
        policy = net(random)

        # USE LOG
        policy = torch.log(policy)

        reward = torch.sum(policy * rewards)
        loss = -reward
        loss.backward()
        opt.step()

        if i % 100 == 0:
            print(reward)

    policy = net(random)
    chart_single(numpy.arange(0, 10, 1), numpy.array(policy.tolist()))


def chart_single(x, y):
    figure, axis = plt.subplots()
    axis.bar(x, y, width=0.5)
    plt.show()


def gradient():
    rewards = [100, 1]
    # theta = (0.5, 0.5)

    # f(theta) = softmax(theta) * rewards


if __name__ == '__main__':
    main()
