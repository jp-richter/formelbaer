import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import re


# example: x_values = np.arrage(0,100,10)
#          y_values = np.array(..)

def plot2d(x_values, x_limit, y_values, y_limit, x_label, y_label, title, fontsize):
    figure, axis = plt.subplots()
    axis.plot(x_values, y_values)

    plt.xlim(xmin=x_limit)
    plt.ylim(ymin=y_limit)
    axis.set(xlabel=x_label, ylabel=y_label)
    plt.title(title, fontsize=fontsize)

    axis.grid()
    plt.show()

    return figure


# example: x_values = np.arrage(0,100,10)
#          y_values = np.array(..)
#          z_values = np.array(..) size(x,y)

def plot3d(x_values, x_limit, y_values, y_limit, z_values, z_limit, x_label, y_label, z_label, title, fontsize):
    x, y = np.meshgrid(x_values, y_values)

    figure = plt.figure()
    axis = figure.gca(projection='3d')

    surface = axis.plot_surface(x, y, z_values, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    axis.set_xlim(x_limit)
    axis.set_ylim(y_limit)
    axis.set_zlim(z_limit)

    plt.gca().invert_xaxis()
    figure.colorbar(surface, shrink=0.5, aspect=5)

    figure.title(title)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)

    plt.show()

    return figure


def save_plot(figure, path):
    figure.savefig(path)


def plot(result_log, target):
    with open(result_log, 'r') as file:
        string = file.read()

    targets = {
        'greward': r'Generator\sReward\sas\sSequence:\s.*',
        'gloss': r'Generator\sLoss\sas\sSequence:\s.*',
        'gprediction': r'Generator\sPrediction\sas\sSequence:\s.*',
        'dloss': r'Discriminator\sLoss\sas\sSequence:\s.*'
    }

    targets_substrings = {
        'greward': lambda s: s[30:],
        'gloss': lambda s: s[28],
        'gprediction': lambda s: s[34],
        'dloss': lambda s: s[32]
    }

    pattern = re.compile(targets[target])
    result = []

    for match in re.finditer(pattern, string):

