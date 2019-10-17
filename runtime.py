# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from datetime import datetime, timedelta
from enum import Enum


class Options(Enum):

	ITERATION = 1
	GSTEP = 2
	DSTEP  = 3
	LOADBATCH = 4
	GUPDATE = 5
	DUPDATE = 6
	STEP = 7
	MONTECARLO = 8


timers = {}
runtimes = {} # contains tuples (average, # of measurements)


def start_timer(keyword):

	start = datetime.now()
	runtimes[keyword] = start


def read_timer(keyword):

	start = runtimes[keyword]
	stop = datetime.now()
	time = stop - start

	if not keyword in runtimes.keys():
		runtimes[keyword] = (time, 1)

	else:
		average, num_times = runtimes[keyword]
		temp = average * num_times + time
		average = temp / (num_times) + 1
		runtimes[keyword] = (average, num_times + 1)

	return time


def estimate(keyword):

	if not keyword in runtimes.keys():
		return None

	return runtimes[keyword]


# fig.savefig('test.jpg')
# plt.gca().invert_xaxis()


def plot_runtime_with_fixed_steps_per_iteration(d_steps, g_steps):

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	seq_len_axis = np.arange(5, 20, 1) # [5, 6 ..]
	mc_trials_axis = np.arange(5, 20, 1)

	X, Y = np.meshgrid(seq_len_axis, mc_trials_axis) # [5][5], [5][6], ..

	results = np.zeros((len(seq_len_axis),len(mc_trials_axis)))
	
	for i in range(len(seq_len_axis)):
		for j in range(len(mc_trials_axis)):
			runtime = estimate_runtime_per_iteration(d_steps,g_steps,seq_len_axis[i], mc_trials_axis[j])
			results[i][j] = ((runtime // 1000) // 60)

	surf = ax.plot_surface(X, Y, results, cmap=cm.coolwarm,linewidth=0, antialiased=False)

	ax.set_zlim(0, 20)
	plt.gca().invert_xaxis()
	fig.colorbar(surf, shrink=0.5, aspect=5)

	fig.suptitle('Runtime per iteration with gsteps = {} and dsteps = {} in min'.format(g_steps, d_steps), fontsize=14)
	plt.xlabel('Sequence Length', fontsize=12)
	plt.ylabel('Montecarlo Trials', fontsize=12)

	plt.show()


def plot_runtime_per_iteration(d_steps, g_steps, seq_len, montecarlo_trials):

	iterations = np.arange(0,200,10)
	runtimes = np.zeros((len(iterations)))

	for i in range(len(runtimes)):
		runtime = estimate_runtime_per_iteration(d_steps, g_steps, seq_len, montecarlo_trials) * iterations[i]
		runtimes[i] = ((runtime // 1000) // 60) // 60

	fig, ax = plt.subplots()
	ax.plot(iterations, runtimes)

	plt.xlim(xmin=0) 
	plt.ylim(ymin=0) 

	ax.set(xlabel='iterations', ylabel='runtime (h)',
	       title='Runtime with {} dsteps, {} gsteps, {} sequence length and {} montecarlo runs'.format(
	       	d_steps, g_steps, seq_len, montecarlo_trials))
	plt.title('Runtime with {} dsteps, {} gsteps, {} sequence length and {} montecarlo runs'.format(
	       	d_steps, g_steps, seq_len, montecarlo_trials), fontsize=9)
	ax.grid()

	plt.show()


def estimate_runtime_per_iteration(d_steps, g_steps, seq_len, montecarlo_trials):

	fixcosts = 500

	d_sampling = 5 * d_steps * seq_len
	d_load_and_update = 4220 * d_steps

	g_single_steps = 4 * g_steps * seq_len
	g_rollout_steps = 5 * g_steps * montecarlo_trials * (seq_len**2 - seq_len) / 2
	g_load_and_reward = 2980 * g_steps * seq_len * montecarlo_trials
	g_update = 490 * g_steps

	d_total = d_sampling + d_load_and_update
	g_total = g_single_steps + g_rollout_steps + g_load_and_reward + g_update

	return fixcosts + d_total + g_total


def estimate_finish_time(iterations, d_steps, g_steps, seq_len, montecarlo_trials):

	runtime = iterations * estimate_runtime_per_iteration(d_steps,g_steps,seq_len,montecarlo_trials)

	date = datetime.now()
	date = date + timedelta(milliseconds=runtime)

	return 'Estimated finish time: ' + str(date)[:-10]


# plot_runtime_per_iteration(2,1,16,16)
