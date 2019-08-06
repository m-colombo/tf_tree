import tensorflow as tf
import typing as T

from tensorflow_trees.definition import NodeDefinition


class FixedArityNodeEmbedder(tf.keras.Model):

    # TODO consider also a coefficient for fatter/thinner interpolation and its skewness
    @staticmethod
    def _layers_size_linear(input_size: int, output_size: int, layers: int):
        for i in range(layers):
            yield int(input_size + (output_size - input_size) * (i + 1) / (layers))

    def __init__(self, *,
                 node_definition: NodeDefinition = None,
                 # arity: int = None,
                 embedding_size: int = None,
                 hidden_cell_coef: float = 0.5,
                 activation=tf.nn.tanh,
                 stacked_layers: int = 1,
                 input_gate=False,
                 output_gate=True,
                 **kwargs):

        super(FixedArityNodeEmbedder, self).__init__(**kwargs)

        self.node_def = node_definition
        # self.arity = arity
        self.embedding_size = embedding_size
        self.stacked_layers = stacked_layers
        self.hidden_cell_coef = hidden_cell_coef
        self.activation = activation

        if input_gate:
            self.input_gate = tf.keras.layers.Dense(
                units=1 + self.node_def.arity.value,   # TODO should be the same network for every element ?
                activation=None)
        else:
            self.input_gate = None

        if output_gate:
            self.output_gate = tf.keras.layers.Dense(
                units=1 + self.node_def.arity.value,   # TODO should be the same network for every element ?
                activation=None)
        else:
            self.output_gate = None

        self.cells = tf.keras.Sequential([
            tf.keras.layers.Dense(s,activation=activation)
            for s in FixedArityNodeEmbedder._layers_size_linear(
                input_size=self.node_def.arity.value * self.embedding_size + self.node_def.value_type.representation_shape if self.node_def.value_type else 0,
                output_size=embedding_size,
                layers=stacked_layers)
            ])  # TODO consider hidden_cell_coef to build fatter or thinner

        tf.logging.warning('trees:encoder:simpleCellBuilder:\tnot using hidden_cell_coef')


    def call(self, x, *args, **kwargs):
        """

        :param x: (
            packed_children: [batch_size, arity*embedding_size],
            values: [batch_size, value_size]
        )
        :param args:
        :param kwargs:
        :return:
        """

        # extract inputs info
        packed_children, values = x
        batch_size = packed_children.shape[0]

        # rescale inputs
        if self.input_gate:
            gatings = tf.nn.softmax(self.input_gate(tf.concat([packed_children, values], axis=-1)))
            children = tf.reshape(packed_children, [batch_size, self.node_def.arity.value, self.embedding_size])
            children_scaled = children * tf.reshape(gatings[:, 1:], [batch_size, self.node_def.arity.value, 1])
            values_scaled = values * tf.reshape(gatings[:, 0], [-1, 1])
            inputs = tf.concat([tf.reshape(children_scaled, [batch_size, -1]), values_scaled], axis=-1)
        else:
            inputs = tf.concat([packed_children, values], axis=-1)

        output = self.cells(inputs)

        # output gate: linearly combine output and inputs
        if self.output_gate:
            children = tf.reshape(packed_children, [batch_size, self.node_def.arity.value, self.embedding_size])
            gatings = tf.nn.softmax(self.output_gate(tf.concat([packed_children, values, output], axis=-1)))
            children_scaled = children * tf.reshape(gatings[:, 1:], [batch_size, self.node_def.arity.value, 1])
            output_scaled = tf.reshape(output * tf.reshape(gatings[:, 0], [-1, 1]), [batch_size, 1, self.embedding_size])
            output = tf.reduce_sum(tf.concat([children_scaled, output_scaled], axis=1), axis=1)

        return output


class NullableInputDenseLayer(tf.keras.layers.Layer):
    """ Build a dense layer which is optimized for left-0-padded input """

    def __init__(self, *, input_size: int = None,
                 hidden_activation=None, hidden_size: int = None,
                 **kwargs):
        super(NullableInputDenseLayer, self).__init__(**kwargs)

        self.hidden_activation = hidden_activation
        self.hidden_size = hidden_size
        self.input_size = input_size

    def build(self, input_shape):
        """

        :param input_shape: [batch, <=input_size]
        :return:
        """
        self.hidden_kernel = self.add_weight(name='hidden_kernel',
                                             shape=[self.input_size, self.hidden_size],
                                             initializer='random_normal',
                                             trainable=True)

        self.hidden_bias= self.add_weight(name='hidden_bias',
                                          shape=(1, self.hidden_size),    # 1 is for broadcasting
                                          initializer='random_normal',
                                          trainable=True)

        super(NullableInputDenseLayer, self).build(input_shape)

    def call(self, x, *args, **kwargs):
        """

        :param x: zero padded input [batch,  <= input_size]
        :return: [batch, hidden_size]
        """

        n = x.shape[1].value

        hidden_activation = self.hidden_activation(tf.matmul(x, self.hidden_kernel[:n]) + self.hidden_bias)

        return hidden_activation

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.hidden_size)


class GatedNullableInput(tf.keras.Model):

    def __init__(self, *, stacked_layers: int = 1, embedding_size: int = None, output_model_builder: tf.keras.Model = None, maximum_input_size: int = None, **kwargs):
        """

        :param embedding_size:
        :param output_model_builder:
        :param maximum_input_size: maximum number of inputs
        :param kwargs:
        """

        self.embedding_size = embedding_size
        self.output_model_builder = output_model_builder
        self.maximum_input_size = maximum_input_size
        self.stacked_layers = stacked_layers

        super(GatedNullableInput, self).__init__(**kwargs)

    def build(self, input_shape):
        """

        :param input_shape: supposed to be [[batch, children], [batch, values]]
        :return:
        """

        self.gating_f = NullableInputDenseLayer(input_size=self.maximum_input_size + self.embedding_size,
                                                hidden_activation=tf.nn.leaky_relu,
                                                hidden_size=self.maximum_input_size // self.embedding_size + 1)

        self.output_model = self.output_model_builder()

        super(GatedNullableInput, self).build(input_shape)

    def call(self, x, *args, **kwargs):
        """

        :param x: a list [
            zero padded input [batch,  <= input_size * emb],
            [batch, values]
            ]
        :return: [clones, batch, output_size]
        """
        children, values = x
        concat = tf.concat([children, values], axis=-1)

        output = self.output_model(concat)

        number_of_input = children.shape[1].value // self.embedding_size
        gating_inp = tf.concat([output, concat], axis=-1)
        gatings = tf.nn.softmax(self.gating_f(gating_inp)[:, :number_of_input+1], axis=1)
        weighted = tf.reshape(tf.concat([children, output], axis=-1), [children.shape[0], number_of_input + 1, -1]) * tf.expand_dims(gatings, -1)
        return tf.reduce_sum(weighted, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.embedding_size)
