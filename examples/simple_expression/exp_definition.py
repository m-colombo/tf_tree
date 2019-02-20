import random
from tensorflow_trees.definition import TreeDefinition, NodeDefinition, Tree
import tensorflow as tf
import typing as T


def create_num_value(max_value):
    """Build a Value class that handle numbers in [0, max_value] encoded as 1ofk"""

    size = max_value + 1

    class NumValue(NodeDefinition.Value):
        representation_shape = size
        class_value = True

        @staticmethod
        def representation_to_abstract_batch(t: tf.Tensor):
            return (tf.argmax(t, axis=-1)).numpy()

        @staticmethod
        def abstract_to_representation_batch(v: T.List[T.Any]):
            return tf.one_hot(v, size, axis=-1)

    return NumValue


class OpValue(NodeDefinition.Value):
    representation_shape = 2
    class_value = True

    @staticmethod
    def representation_to_abstract_batch(t: tf.Tensor):
        ops = ['+', '-']
        return ops[(tf.argmax(t, axis=-1)).numpy()[0]]

    @staticmethod
    def abstract_to_representation_batch(v: T.List[T.Any]):
        return tf.one_hot(list(map(lambda x: 0 if x == '+' else 1, v)), 2, axis=-1)


class BinaryExpressionTreeGen:
    def __init__(self, max_value):

        self.NumValue = create_num_value(max_value)

        self.tree_def = TreeDefinition(
            node_types=[
                NodeDefinition("add_bin", may_root=True, arity=NodeDefinition.FixedArity(2)),
                NodeDefinition("sub_bin", may_root=True, arity=NodeDefinition.FixedArity(2)),
                NodeDefinition("num_value", may_root=True, arity=NodeDefinition.FixedArity(0), value_type=self.NumValue)

            ])

        self.leaf_values = list(range(0, max_value+1))
        self.node_types = self.tree_def.node_types

    def generate(self, max_depth):
        """Generate a random arithmetic expression tree, using just binary plus and minus
            Args:
                max_depth: integer > 0
            Returns:
                expression tree where leaves are int.
        """

        if max_depth == 1:  # recursion base case
            v = random.sample(self.leaf_values, 1)[0]
            return Tree(node_type_id='num_value', value=self.NumValue(abstract_value=v))

        elif max_depth > 1:
            types = self.node_types
            node_type = random.sample(types, 1)[0]

            if node_type.id == 'num_value':
                return self.generate(1)

            else:
                return Tree(node_type.id, children=[
                    self.generate(max_depth - 1),
                    self.generate(max_depth - 1)], value=None)

    def evaluate(self, et):
        """Evaluate the result of the arithmetic expression
            Args:
                et: expression tree
            Returns:
                an integer, the result
        """

        if et.node_type_id == 'num_value':
            return et.value.abstract_value
        elif et.node_type_id == 'sub_bin':
            return self.evaluate(et.children[0]) - self.evaluate(et.children[1])
        elif et.node_type_id == 'add_bin':
            return self.evaluate(et.children[0]) + self.evaluate(et.children[1])


class LabelledBinaryExpressionTreeGen(BinaryExpressionTreeGen):

    def __init__(self, max_value):
        super(LabelledBinaryExpressionTreeGen, self).__init__(max_value)
        self.NumValue = create_num_value(max_value)

        self.tree_def = TreeDefinition(
            node_types=[
                NodeDefinition("op_bin", may_root=True, arity=NodeDefinition.FixedArity(2), value_type=OpValue),
                NodeDefinition("num_value", may_root=True, arity=NodeDefinition.FixedArity(0), value_type=self.NumValue)

            ])

        self.node_types = self.tree_def.node_types

    def generate(self, max_depth):
        """Generate a random arithmetic expression tree, using just binary plus and minus
            Args:
                max_depth: integer > 0
            Returns:
                expression tree where leaves are int.
        """

        if max_depth == 1:  # recursion base case
            v = random.sample(self.leaf_values, 1)[0]
            return Tree(node_type_id='num_value', value=self.NumValue(abstract_value=v))

        elif max_depth > 1:
            types = self.node_types
            node_type = random.sample(types, 1)[0]

            if node_type.id == 'num_value':
                return self.generate(1)

            else:
                o = random.sample(['+', '-'], 1)[0]
                return Tree(node_type.id, children=[
                    self.generate(max_depth - 1),
                    self.generate(max_depth - 1)], value=OpValue(abstract_value=o))

    def evaluate(self, et):
        """Evaluate the result of the arithmetic expression
            Args:
                et: expression tree
            Returns:
                an integer, the result
        """

        if et.node_type_id == 'num_value':
            return et.value.abstract_value
        elif et.node_type_id == 'op_bin' and et.value.abstract_value == '-':
            return self.evaluate(et.children[0]) - self.evaluate(et.children[1])
        elif et.node_type_id == 'op_bin' and et.value.abstract_value == '+':
            return self.evaluate(et.children[0]) + self.evaluate(et.children[1])


class NaryExpressionTreeGen(BinaryExpressionTreeGen):
    def __init__(self, max_value, max_arity):
        super(NaryExpressionTreeGen, self).__init__(max_value)

        self.tree_def = TreeDefinition(
            node_types=[
                NodeDefinition("add_n", may_root=True,
                               arity=NodeDefinition.VariableArity(min_value=2, max_value=max_arity),
                               value_type=None),
                NodeDefinition("sub_bin", may_root=True, arity=NodeDefinition.FixedArity(2),
                               value_type=None),
                NodeDefinition("num_value", may_root=True, arity=NodeDefinition.FixedArity(0),
                               value_type=self.NumValue)
            ]
        )

        self.node_types = self.tree_def.node_types

    def generate(self, max_depth):
        """Generate a random arithmetic expression tree, using just n-ary plus and binary minus
            Args:
                max_depth: integer > 0
            Returns:
                expression tree where leaves are int.
        """

        if max_depth == 1: # recursion base case
            v = random.sample(self.leaf_values, 1)[0]
            return Tree(node_type_id='num_value', value=self.NumValue(abstract_value=v))

        elif max_depth > 1:
            types = self.node_types
            node_type = random.sample(types, 1)[0]

            if node_type.id == 'num_value':
                return self.generate(1)

            elif node_type.id == 'add_n':
                n = random.randint(node_type.arity.min_value, node_type.arity.max_value)
                return Tree(node_type.id, children=[self.generate(max_depth - 1) for _ in range(n)])

            elif node_type.id == 'sub_bin':
                return Tree(node_type.id, children=[
                    self.generate(max_depth - 1),
                    self.generate(max_depth - 1)])

    def evaluate(self, t: Tree):
        if len(t.children) > 0:
            if t.node_type_id == 'add_n':
                return sum(map(self.evaluate, t.children))
            if t.node_type_id == 'sub_bin':
                return self.evaluate(t.children[0]) - self.evaluate(t.children[1])
        else:
            return t.value.abstract_value


class LabelledNaryExpressionTreeGen(NaryExpressionTreeGen):
    def __init__(self, max_value, max_arity):
        super(LabelledNaryExpressionTreeGen, self).__init__(max_value, max_arity)

        self.tree_def = TreeDefinition(
            node_types=[
                NodeDefinition("op_n", may_root=True,
                               arity=NodeDefinition.VariableArity(min_value=2, max_value=max_arity),
                               value_type=OpValue),
                NodeDefinition("num_value", may_root=True, arity=NodeDefinition.FixedArity(0),
                               value_type=self.NumValue)
            ]
        )
        self.node_types = self.tree_def.node_types

    def generate(self, max_depth):
        """Generate a random arithmetic expression tree, using just n-ary plus and binary minus
            Args:
                max_depth: integer > 0
            Returns:
                expression tree where leaves are int.
        """

        if max_depth == 1: # recursion base case
            v = random.sample(self.leaf_values, 1)[0]
            return Tree(node_type_id='num_value', value=self.NumValue(abstract_value=v))

        elif max_depth > 1:
            types = self.node_types
            node_type = random.sample(types, 1)[0]
            o = random.sample(['+', '-'], 1)[0]

            if node_type.id == 'num_value':
                return self.generate(1)

            elif node_type.id == 'op_n' and o == "+":
                n = random.randint(node_type.arity.min_value, node_type.arity.max_value)
                return Tree(node_type.id, children=[self.generate(max_depth - 1) for _ in range(n)],
                            value=OpValue(abstract_value='+'))

            elif node_type.id == 'op_n' and o == "-":
                return Tree(node_type.id, children=[
                    self.generate(max_depth - 1),
                    self.generate(max_depth - 1)],
                            value=OpValue(abstract_value='-'))

    def evaluate(self, t: Tree):
        if len(t.children) > 0:
            if t.node_type_id == 'op_n' and t.value.abstract_value == "+":
                return sum(map(self.evaluate, t.children))
            if t.node_type_id == 'sub_bin' and t.value.abstract_value == '-':
                return self.evaluate(t.children[0]) - self.evaluate(t.children[1])
        else:
            return t.value.abstract_value
