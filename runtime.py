# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np



# fig.savefig('test.jpg')
# plt.gca().invert_xaxis()


def plot_runtime_with_fixed_steps_and_length(d_steps, g_steps, seq_len):

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	iterations_axis = np.arange(50, 200, 10) # [5, 6 ..]
	mc_trials_axis = np.arange(5, 20, 1)

	X, Y = np.meshgrid(iterations_axis, mc_trials_axis) # [5][5], [5][6], ..

	results = np.zeros((len(iterations_axis),len(mc_trials_axis)))
	
	for i in range(len(iterations_axis)):
		for j in range(len(mc_trials_axis)):
			results[i][j] = estimate_runtime_per_iteration(d_steps,g_steps,iterations_axis[i], mc_trials_axis[j])

	surf = ax.plot_surface(X, Y, results, cmap=cm.coolwarm,linewidth=0, antialiased=False)

	# Customize the z axis.
	ax.set_zlim(50000, 2000000)
	# ax.zaxis.set_major_locator(LinearLocator(10))
	#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)

	fig.suptitle('Runtime per iteration with gsteps = {} and dsteps = {} in ms'.format(g_steps, d_steps), fontsize=14)
	plt.xlabel('Iterations', fontsize=12)
	plt.ylabel('Montecarlo Trials', fontsize=12)

	plt.show()


def plot_runtime_with_fixed_steps_and_trials(d_steps, g_steps, trials):

	pass


def plot_runtime_with_fixed_steps_per_iteration(d_steps, g_steps):

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	seq_len_axis = np.arange(5, 20, 1) # [5, 6 ..]
	mc_trials_axis = np.arange(5, 20, 1)

	X, Y = np.meshgrid(seq_len_axis, mc_trials_axis) # [5][5], [5][6], ..

	results = np.zeros((len(seq_len_axis),len(mc_trials_axis)))
	
	for i in range(len(seq_len_axis)):
		for j in range(len(mc_trials_axis)):
			results[i][j] = estimate_runtime_per_iteration(d_steps,g_steps,seq_len_axis[i], mc_trials_axis[j])

	surf = ax.plot_surface(X, Y, results, cmap=cm.coolwarm,linewidth=0, antialiased=False)

	# Customize the z axis.
	ax.set_zlim(0, 30000)
	#ax.zaxis.set_major_locator(LinearLocator(10))
	#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)

	fig.suptitle('Runtime per iteration with gsteps = {} and dsteps = {} in ms'.format(g_steps, d_steps), fontsize=14)
	plt.xlabel('Sequence Length', fontsize=12)
	plt.ylabel('Montecarlo Trials', fontsize=12)

	plt.show()


def plot_runtime_per_iteration(d_steps, g_steps, seq_len, montecarlo_trials):

	iterations = np.arange(0,200,10)
	runtimes = np.zeros((len(iterations)))

	for i in range(len(runtimes)):
		runtimes[i] = (estimate_runtime_per_iteration(d_steps, g_steps, seq_len, montecarlo_trials) * iterations[i] // 1000) // 60

	fig, ax = plt.subplots()
	ax.plot(iterations, runtimes)

	plt.xlim(xmin=0) 
	plt.ylim(ymin=0) 

	ax.set(xlabel='iterations', ylabel='runtime (ms)',
	       title='About as simple as it gets, folks')
	ax.grid()

	plt.show()


def estimate_runtime_per_iteration(d_steps, g_steps, seq_len, montecarlo_trials):

	fixcosts = 500

	# discriminator
	sampling = 5 * d_steps * seq_len
	load_samples_and_update = 4220 * d_steps

	# generator
	steps = 4 * g_steps * seq_len
	rollout_steps = 5 * g_steps * montecarlo_trials * (seq_len**2 - seq_len) / 2
	update = 490 * g_steps

	return fixcosts + sampling + load_samples_and_update + steps + rollout_steps + update

#plot_runtime_per_iteration(2,1,16,16)

print(estimate_runtime_per_iteration(2,1,16,16) // 1000)
