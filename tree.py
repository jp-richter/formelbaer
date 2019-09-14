from tokens import Token

class Tree:

	def __init__(self):
		self.token = None 
		self.children = []

	def leaf(self):
		return not self.children

	def size(self):
		return sum(map(size, self.children), 1)

	def next(self):
		if self.saturated():
			return False

		if self.leaf():
			return self

		for child in self.children:
			if not child.saturated():
				return child.next()

		raise SystemError()

	def saturated(self):
		for child in children:
			if not child.saturated():
				return False

		return self.token.arity == length(children)

	def clone(self):
		tree = Tree()
		tree.token = self.token
		tree.children = list(map(clone, self.children))

		return tree

	def string(self):
		tmp = self.token.name + '['

		for child in self.children:
			tmp += child.string() + ','

		if not self.children:
			return tmp[:-1]

		return tmp[:-1] + ']'

	def latex(self):
		info = self.token.value

		assert(info.arity == len(self.children))
		assert(info.arity >= 0 and info.arity <= 3)

		# stdl string does not allow (generic) partial string formatting
		if info.arity == 0:
			return info.latex 

		if info.arity == 1:
			return info.latex.format(self.children[0].latex())

		if info.arity == 2:
			return info.latex.format(self.children[0].latex(),self.children[1].latex())

		if info.arity == 3:
			return info.latex.format(self.children[0].latex(),self.children[1].latex(),self.children[2].latex())
