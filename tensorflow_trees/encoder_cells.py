import tensorflow as tf

from tensorflow_trees.definition import NodeDefinition
from tensorflow_trees.miscellaneas import interpolate_layers_size

class _GatedModel(tf.keras.Model):
    def __init__(self, *, model: tf.keras.Model = None, node_def: NodeDefinition = None,
                 embedding_size: int = None, maximum_input_arity: int = None,
                 input_gate=False, output_gate=True, **kwargs):
        super(_GatedModel, self).__init__(**kwargs)

        if maximum_input_arity is None:
            maximum_input_arity = node_def.arity.value
            if maximum_input_arity is None:
                raise ValueError("unknown maximum_input_arity")

        input_size = embedding_size * maximum_input_arity + \
                     (node_def.value_type.representation_shape if node_def.value_type else 0)

        self.embedding_size = embedding_size

        if input_gate:
            # TODO should be the same network for every element ?
            self.input_gate = _NullableInputDenseLayer(input_size=input_size,
                                                       output_size=maximum_input_arity + (1 if node_def.value_type else 0))
        else:
            self.input_gate = None

        if output_gate:
            # TODO should be the same network for every element ?
            self.output_gate = _NullableInputDenseLayer(input_size=input_size + embedding_size, output_size=maximum_input_arity + 1)
        else:
            self.output_gate = None

        self.model = model

    def __call__(self, x, *args, **kwargs):
        """

        :param x: [
            packed_children: [batch_size, <= embedding_size * maximum_arity],
            values: [batch_size, value_size]
        ]
        if there's no value => value_size=0
        :return: [batch_size, embedding_size]
        """

        # extract inputs info
        packed_children, values = x
        batch_size = packed_children.shape[0]

        number_of_inputs = packed_children.shape[1].value // self.embedding_size
        children = tf.reshape(packed_children, [batch_size, number_of_inputs, self.embedding_size])
        packed_input = tf.concat([values, packed_children], axis=-1)

        # rescale inputs
        if self.input_gate:
            value_coef = 1 if values.shape[1] > 0 else 0
            gatings = tf.nn.softmax(self.input_gate(packed_input)[:, :number_of_inputs + value_coef], axis=-1)
            children_scaled = tf.reshape(
                children * tf.reshape(
                    gatings[:, value_coef:number_of_inputs+value_coef],
                    [batch_size, number_of_inputs, 1]),
                [batch_size, -1])

            values_scaled = values * tf.reshape(gatings[:, 0], [-1, 1])
            inputs = tf.concat([children_scaled, values_scaled], axis=-1)

        else:
            inputs = tf.concat([packed_children, values], axis=-1)

        output = self.model(inputs)

        # output gate: linearly combine output and inputs
        if self.output_gate:
            gatings = tf.nn.softmax(self.output_gate(tf.concat([packed_input, output], axis=-1))[:, :number_of_inputs+1])
            children_scaled = children * tf.reshape(gatings[:, 1:], [batch_size, number_of_inputs, 1])
            output_scaled = tf.reshape(output * tf.reshape(gatings[:, 0], [-1, 1]), [batch_size, 1, self.embedding_size])
            output = tf.reduce_sum(tf.concat([children_scaled, output_scaled], axis=1), axis=1)

        return output


class FixedArityNodeEmbedder(tf.keras.Model):

    def __init__(self, *,
                 node_definition: NodeDefinition = None,
                 embedding_size: int = None,
                 hidden_cell_coef: float = 0.5,
                 activation=tf.nn.tanh,
                 stacked_layers: int = 2,
                 input_gate=False,
                 output_gate=True,
                 **kwargs):

        super(FixedArityNodeEmbedder, self).__init__(**kwargs)

        self.cells = tf.keras.Sequential([
            tf.keras.layers.Dense(s, activation=activation)
            for s in interpolate_layers_size(
                input_size=node_definition.arity.value * embedding_size + (node_definition.value_type.representation_shape if node_definition.value_type else 0), # TODO might be too big
                output_size=embedding_size,
                layers=stacked_layers)
            ])  # TODO consider hidden_cell_coef to build fatter or thinner

        tf.logging.warning('trees:encoder:simpleCellBuilder:FixedArityNodeEmbedder\tnot using hidden_cell_coef')

        self.gated_model = _GatedModel(
            model=self.cells,
            node_def=node_definition,
            embedding_size=embedding_size,
            maximum_input_arity=node_definition.arity.value,
            input_gate=input_gate,
            output_gate=output_gate)

    def call(self, x, *args, **kwargs):
        """

        :param x: (
            packed_children: [batch_size, arity*embedding_size],
            values: [batch_size, value_size]
        )
        if no value => value_size = 0
        :return:
        """

        return self.gated_model(x, *args, **kwargs)


class VariableArityNodeEmbedder(tf.keras.Model):
    def __init__(self, *,
                 node_def: NodeDefinition = None,
                 maximum_input_arity: int = None,
                 embedding_size: int = None,
                 hidden_cell_coef: float = 0.5,
                 activation=tf.nn.tanh,
                 stacked_layers: int = 2,
                 input_gate=False,
                 output_gate=True,
                 **kwargs):
        super(VariableArityNodeEmbedder, self).__init__(**kwargs)

        tf.logging.warning('trees:encoder:simpleCellBuilder:VariableArityNodeEmbedder\tnot using hidden_cell_coef')
        input_size = embedding_size * maximum_input_arity + (node_def.value_type.representation_shape if node_def.value_type else 0)
        cells_size = list(interpolate_layers_size(input_size, embedding_size, stacked_layers))

        cells = [_NullableInputDenseLayer(
            input_size=input_size,
            output_activation=activation,
            output_size=cells_size[0])]

        for s in cells_size[1:]:
            cells.append(tf.keras.layers.Dense(s, activation))

        model = tf.keras.Sequential(cells)
        self.gated_model = _GatedModel(
            model=model,
            node_def=node_def,
            embedding_size=embedding_size,
            maximum_input_arity=maximum_input_arity,
            input_gate=input_gate,
            output_gate=output_gate
        )

    def __call__(self, x, *args, **kwargs):
        """

        :param x: (
            packed_children: [batch_size, arity*embedding_size],
            values: [batch_size, value_size]
        )
        if no value => value_size = 0
        :return:
        """
        return self.gated_model(x, *args, **kwargs)


class _NullableInputDenseLayer(tf.keras.layers.Layer):
    """ Build a dense layer which is optimized for right-0-padded input """

    def __init__(self, *, input_size: int = None,
                 output_activation=lambda x: x, output_size: int = None,
                 **kwargs):
        super(_NullableInputDenseLayer, self).__init__(**kwargs)

        self.hidden_kernel = self.add_weight(name='hidden_kernel',
                                             shape=[input_size, output_size],
                                             initializer='random_normal',
                                             trainable=True)

        self.hidden_bias = self.add_weight(name='hidden_bias',
                                           shape=(1, output_size),  # 1 is for broadcasting
                                           initializer='random_normal',
                                           trainable=True)

        self.output_activation = output_activation
        self.output_size = output_size

    def call(self, x, *args, **kwargs):
        """

        :param x: zero padded input [batch,  <= input_size]
        :return: [batch, output_size]
        """

        n = x.shape[1].value

        hidden_activation = self.output_activation(tf.matmul(x, self.hidden_kernel[:n]) + self.hidden_bias)

        return hidden_activation

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_size
