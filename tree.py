from itertools import chain
import tokens


class Tree:

    def __init__(self, token):

        assert token

        self.token = token
        self.children = []

    def __iter__(self):

        yield self

        for generator in chain(map(iter, self.children)):
            for child in generator:
                yield child

    def leaf(self):

        return not self.children

    def size(self):

        return sum(map(self.size, self.children), 1)

    def append(self, node):

        if self.saturated():
            return False

        for child in self.children:
            if child.append(node):
                return True

        if not self.token.arity == len(self.children):
            self.children.append(node)
            return True

        raise SystemError()

    def saturated(self):

        for child in self.children:
            if not child.saturated():
                return False

        return self.token.arity == len(self.children)

    def clone(self):

        tree = Tree(self.token)
        tree.children = list(map(self.clone, self.children))

        return tree

    def string(self):

        tmp = self.token.name + '['
        for child in self.children:
            tmp += child.string() + ','

        if not self.children:
            return tmp[:-1]

        return tmp[:-1] + ']'

    def latex(self):

        stub = Tree(tokens.TokenInfo(None, 0, None, ''))

        for _ in range(self.token.arity - len(self.children)):
            self.children.append(stub)

        # stdl string does not allow generic partial string formatting
        if self.token.arity == 0:
            return self.token.latex

        if self.token.arity == 1:
            return self.token.latex.format(self.children[0].latex())

        if self.token.arity == 2:
            return self.token.latex.format(self.children[0].latex(), self.children[1].latex())

        if self.token.arity == 3:
            return self.token.latex.format(self.children[0].latex(), self.children[1].latex(), self.children[2].latex())


def sequence_to_tree(sequence):
    root = Tree(tokens.get(sequence[0]))

    for code in sequence[1:]:
        node = Tree(tokens.get(code))
        saturated = not root.append(node)

        if saturated:
            break

    return root


def tree_to_sequence(tree):
    sequence = []

    for node in iter(tree):
        id = tokens.id(node.token.onehot)
        sequence.append(id)

    return sequence


def batch_to_tree(batch):
    trees = []
    for sample in batch:
        sequence = []

        for onehot in sample:
            sequence.append(tokens.id(onehot))

        trees.append(sequence_to_tree(sequence))

    return trees
