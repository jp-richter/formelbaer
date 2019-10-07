from tokens import *
from numpy import random
import tree
import tokens
import torch


#
# test: latex translation of example trees
#


# sequence = [37,146,28,150,26,90,136]

# expr = tree.parse(sequence)

# print(expr.latex())
# print(expr.string())

# target = tree.sequence(expr)

# print(target)


#
# simulation of generative policy network
#

# 1. accept input vector with fixed length of one tokens onehot encoding length 
# 2. generate random policy over token choices to simulate generative process

# n = 10 # ensure that generation terminates

# def simulate_generator(state):
# 	global n

# 	length = len(Token)

# 	if n == 0:
# 		policy = [0] * length
# 		policy[147] = 1 # letter z
# 		return policy
# 	else:
# 		n -= 1

# 	policy = random.uniform(size=length)

# 	# normalize random values
# 	total = sum(policy)
# 	policy = [f / total for f in policy]

# 	return policy 


#
# simulation of discriminator network
#

# 1. accept image file of latex expression generated with tex2png
# 2. return a random reward

# def simulate_discriminator(image_file):
# 	return random.random_integers(low=0,high=10,size=1)


#
# png und tex2png test
#

# test = 'mein test string 3 = 5 \\sigma'
# name = 'test'
# desktop = os.path.normpath(os.path.expanduser("~/Desktop"))

# tex2png(test, desktop + '/' + name + '.tex')
