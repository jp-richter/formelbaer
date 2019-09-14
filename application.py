from numpy.random import choice
from tokens import *
from tree import *

class Application:

	def policy(self, state):
		# state -> generator network
		# generator network -> policy vector
		pass

	def decision(self, policy):
		choices = list(Token)

		assert(len(choices) == len(policy))
		action  = choice(choices, p=policy)

		return action

	def generate(self, tree, action):
		node = Tree()
		node.token = action

		assert(tree.next())
		unsaturated = tree.next()
		unsaturated.children.append(node)

		return tree

	def reward(self, tree):
		latex = tree.latex()

		# benutz png tool darauf
		# input fuers zweite netz
		# output reward
		pass

	def update_policy(self):
		pass

	def update_generator(self):
		pass

	def main(self):

		# get policy
		# make decision
		# generate tree
		# repeat until tree is saturated

		# calculate reward
		# update policy

		# update generator every n cycles

		pass