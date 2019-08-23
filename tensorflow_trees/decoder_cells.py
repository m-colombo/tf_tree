import tensorflow as tf
import typing as T

from tensorflow_trees.definition import NodeDefinition
from tensorflow_trees.miscellaneas import interpolate_layers_size


class _GatedModel(tf.keras.Model):
    def __init__(self, *, embedding_size: int, model=None, **kwargs):
        """

        :param model: [batch_size, input_size] -> [batch_size, embedding_size * number_of_children]
        :param embedding_size:
        :param kwargs:
        """
        super(_GatedModel, self).__init__(**kwargs)

        self.embedding_size = embedding_size

        self.model = model
        self.output_gate = tf.keras.layers.Dense(1, activation=tf.sigmoid)

    def call(self, x, *args, **kwargs):
        """

        :param x: [batch_size, embedding_size + other_infos_size] if model else [batch_size, embedding_size * number_of_children]
        :return: [batch_size, embedding_size * arity]
        """

        if self.model is None:
            outputs = x
        else:
            outputs = self.model(x, *args, **kwargs)

        arity = outputs.shape[1].value // self.embedding_size
        batch_size = x.shape[0].value

        parent_embs = x[:, :self.embedding_size]
        childrens = tf.reshape(outputs, [batch_size, arity, self.embedding_size])

        gating_inp = tf.reshape(
            tf.concat([childrens,
                       tf.tile(tf.expand_dims(x, axis=1), [1, arity, 1])], axis=-1),
            [batch_size * arity, -1])

        gatings = self.output_gate(gating_inp)

        corrected = \
            gatings * tf.reshape(childrens, [batch_size * arity, -1]) + \
            (1 - gatings) * \
                tf.reshape(
                    tf.tile(tf.expand_dims(parent_embs, axis=1), [1, arity, 1]),
                    [batch_size * arity, -1])

        return tf.reshape(corrected, [batch_size, arity * self.embedding_size])


class FixedArityNodeDecoder(tf.keras.Model):
    def __init__(self, *, activation=None, hidden_coef: float= None, embedding_size: int = None,
                 node_def: NodeDefinition = None, stacked_layers: int = 2,
                 output_gate=True,
                 **kwargs):
        super(FixedArityNodeDecoder, self).__init__(**kwargs)

        self.embedding_size = embedding_size
        self.node_def = node_def
        self.activation = activation
        self.stacked_layers = stacked_layers
        self.output_gate = output_gate

        # self.model = None

    def build(self, input_shape):
        tf.logging.warning("trees:decoder:cells:FixedArityNodeDecoder\tnot using hidden_cell_coef")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(u, activation=self.activation)
            for u in interpolate_layers_size(
                input_shape[1].value,
                self.embedding_size * self.node_def.arity.value,
                self.stacked_layers  # TODO exploit hidden_coef for interpolation fatness
            )
        ])

        if self.output_gate:
            self.model = _GatedModel(model=model, embedding_size=self.embedding_size)
        else:
            self.model = model

    def call(self, x, *args, **kwargs):
        """

        :param x: [batch_size, embedding_size + other_infos_size]
        :return: [batch_size, embedding_size * arity]
        """
        return self.model(x)


class VariableArityNodeDecoder(tf.keras.Model):
    def __init__(self, *, activation, maximum_children, embedding_size, stacked_layers: int = 2, output_gate=True, **kwargs):
        super(VariableArityNodeDecoder, self).__init__(**kwargs)

        self.activation = activation
        self.embedding_size = embedding_size
        self.stacked_layers = stacked_layers
        self.maximum_children = maximum_children

        tf.logging.warning("trees:decoder:cells:VariableArityNodeDecoder	not using hidden_cell_coef")

        self.model = None
        if output_gate:
            self.output_gate = _GatedModel(embedding_size=self.embedding_size)

    def build(self, input_shape):
        self.model = tf.keras.Sequential([
            ParallelDense(
                activation=self.activation,
                size=s,
                parallel_clones=self.maximum_children
            )
            for s in interpolate_layers_size(
                input_shape[1].value,
                self.embedding_size,
                self.stacked_layers)
        ])

    def call(self, x, n: int, *args, **kwargs):
        """

        :param x: [batch, input_size]
        :param n: number of children to decode. It takes the n first children from left.
        :return: [n, batch, embedding_size]
        """
        x_ = tf.tile(tf.expand_dims(x, axis=0), [n, 1, 1])  # [clones, batch, input]
        output = self.model(x_)

        if self.output_gate is not None:
            output = self.output_gate(tf.reshape(tf.transpose(output, [1, 0, 2]), [x.shape[0], -1]))
            output = tf.transpose(tf.reshape(output, [x.shape[0], n,  self.embedding_size]), [1, 0, 2])

        return output


class ParallelDense(tf.keras.layers.Layer):
    """Computes parallely many dense layers with the same shape, or possibly only a prefix of them"""
    def __init__(self, *, activation, size, parallel_clones, **kwargs):
        super(ParallelDense, self).__init__(**kwargs)
        self.activation = activation
        self.parallel_clones = parallel_clones
        self.size = size

        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        """

        :param input_shape: [n <= parallel_clones, batch, input_size]
        :return:
        """
        super(ParallelDense, self).build(input_shape)

        self.kernel = self.add_weight(name='kernel',
                                      shape=[self.parallel_clones, input_shape[2].value, self.size],
                                      initializer='random_normal',
                                      trainable=True)

        self.bias = self.add_weight(name='bias',
                                    shape=(self.parallel_clones, 1, self.size),    # 1 is for broadcasting
                                    initializer='random_normal',
                                    trainable=True)

    def call(self, x, *args, **kwargs):
        """
        Parallely computes the first n parallel dense layers
        :param x: input [n <= parallel_clones, batch, input_size]
        :return: [n, batch, output_size]
                """
        n = x.shape[0]
        #  [clones, batch, input] * [clones, input, output] => [clones, batch, output]
        return self.activation(tf.matmul(x, self.kernel[:n]) + self.bias[:n])

