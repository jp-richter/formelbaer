# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


# example: x_values = np.arrage(0,100,10)
#          y_values = np.array(..)

def plot2d(x_values, x_limit, y_values, y_limit, x_label, y_label, title, fontsize):
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values)

    plt.xlim(xmin=x_limit)
    plt.ylim(ymin=y_limit)
    ax.set(xlabel=x_label, ylabel=y_label)
    plt.title(title, fontsize=fontsize)

    ax.grid()
    plt.show()

    return fig


def plot3d(x_values, x_limit, y_values, y_limit, z_values, z_limit,  x_label, y_label, z_label, title, fontsize):
    x = x_values
    y = y_values
    z = z_values



def save_plot(figure, path):
    figure.savefig(path)


def plot_runtime_with_fixed_steps_per_iteration(d_steps, g_steps):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    seq_len_axis = np.arange(5, 20, 1)  # [5, 6 ..]
    mc_trials_axis = np.arange(5, 20, 1)

    X, Y = np.meshgrid(seq_len_axis, mc_trials_axis)  # [5][5], [5][6], ..

    results = np.zeros((len(seq_len_axis), len(mc_trials_axis)))

    for i in range(len(seq_len_axis)):
        for j in range(len(mc_trials_axis)):
            runtime = estimate_runtime_per_iteration(d_steps, g_steps, seq_len_axis[i], mc_trials_axis[j])
            results[i][j] = ((runtime // 1000) // 60)

    surf = ax.plot_surface(X, Y, results, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_zlim(0, 20)
    plt.gca().invert_xaxis()
    fig.colorbar(surf, shrink=0.5, aspect=5)

    fig.suptitle('Runtime per iteration with gsteps = {} and dsteps = {} in min'.format(g_steps, d_steps), fontsize=14)
    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Montecarlo Trials', fontsize=12)

    plt.show()

