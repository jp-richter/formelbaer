import matplotlib.pyplot as plt
from matplotlib import cm
import numpy
import re
import sys


# example: x_values = np.arrage(0,100,10)
#          y_values = np.array(..)

def single_plot2d(x_values, y_values, x_label, y_label, title, fontsize):
    figure, axis = plt.subplots()
    axis.plot(x_values, y_values)

    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    axis.set(xlabel=x_label, ylabel=y_label)
    plt.title(title, fontsize=fontsize)

    axis.grid()
    # plt.show()

    return figure, axis


def multiple_plot2d(values, x_label, y_label, legend, title, fontsize):
    figure, axis = plt.subplots()
    lines = ['b', 'r', 'y']

    for (x,y), line, label in zip(values, lines, legend):
        axis.plot(x, y, line, label=label, linewidth=0.3)

    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.title(title, fontsize=fontsize)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)

    leg = plt.legend()
    # get the individual lines inside legend and set line width
    for line in leg.get_lines():
        line.set_linewidth(1)

    # get label texts inside legend and set font size
    for text in leg.get_texts():
        text.set_fontsize('x-large')

    axis.grid()
    # plt.show()

    return figure, axis


# example: x_values = np.arrage(0,100,10)
#          y_values = np.array(..)
#          z_values = np.array(..) size(x,y)

def plot3d(x_values, x_limit, y_values, y_limit, z_values, z_limit, x_label, y_label, z_label, title, fontsize):
    x, y = numpy.meshgrid(x_values, y_values)

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

    # plt.show()

    return figure, axis


def save_plot(figure, path):
    figure.savefig(path)


def parse(filepath, target):
    with open(filepath, 'r') as file:
        string = file.read()

    targets = {
        'greward': r'Generator\sReward\sas\sSequence:\s.*',
        'gloss': r'Generator\sLoss\sas\sSequence:\s.*',
        'gprediction': r'Generator\sPrediction\sas\sSequence:\s.*',
        'dloss': r'Discriminator\sLoss\sas\sSequence:\s.*'
    }

    targets_substrings = {
        'greward': lambda s: s[30:],
        'gloss': lambda s: s[28:],
        'gprediction': lambda s: s[34:],
        'dloss': lambda s: s[32:]
    }

    pattern = re.compile(targets[target])
    result = []

    for match in re.finditer(pattern, string):
        result.append(match.group())

    assert len(result) == 1
    result = targets_substrings[target](result[0])
    ls = result.split(',')
    ls = [float(n) for n in ls]

    return ls


def plot(filepath):
    targets = ['greward', 'gloss', 'gprediction'] # prediction
    targets_labels = {
        'greward': 'Generator Reward',
        'gloss': 'Generator Loss',
        'gprediction': 'Generator Prediction'
    }
    results = []

    # single plots
    for t in targets:
        numbers = parse(filepath, t)
        x = numpy.arange(0, len(numbers), 1)
        y = numpy.array(numbers)

        # save single plot
        figure, _ = single_plot2d(x, y, 'Step', targets_labels[t], '', 12)
        save_plot(figure, filepath[:-11] + '{}_plot.png'.format(t))

        # numbers are to small in comparison
        if not t == 'gprediction':
            results.append((x,y))

    # plot all on same surface
    legend = ['Generator Reward', 'Generator Loss']
    figure, axis = multiple_plot2d(results, 'Step', '', legend, '', 12)
    save_plot(figure, filepath[:-11] + 'generator_plot.png')

    numbers = parse(filepath, 'dloss')
    x = numpy.arange(0,len(numbers),1)
    y = numpy.array(numbers)
    figure, _ = single_plot2d(x, y, 'Epoch', 'Discriminator Loss', '', 12)
    save_plot(figure, filepath[:-11] + 'discriminator_plot.png')


if __name__ == '__main__':
    assert len(sys.argv) == 2
    _, filepath = sys.argv
    plot(filepath)
