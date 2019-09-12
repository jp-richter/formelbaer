class Tree:

	def __init__(self):
		self.type = None 
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

		return self.type.arity == length(children)

	def clone(self):
		tree = Tree()
		tree.type = self.type
		tree.children = list(map(clone, self.children))

		return tree


	def string(self):
		tmp = self.expression.name + '['

		for child in children:
			tmp += child.string() + ','

		return tmp[:-1] + ']'
