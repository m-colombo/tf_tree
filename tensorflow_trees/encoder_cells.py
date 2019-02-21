import tensorflow as tf
import typing as T

from tensorflow_trees.definition import NodeDefinition


class GatedFixedArityNodeEmbedder(tf.keras.Model):

    def __init__(self, *, activation=None, hidden_coef: float= None, embedding_size: int = None,
                 arity: int = None,
                 **kwargs):
        super(GatedFixedArityNodeEmbedder, self).__init__(**kwargs)

        self.activation = activation
        self.hidden_coef = hidden_coef
        self.embedding_size = embedding_size
        self.arity = arity

    def build(self, input_shape):
        """

        :param input_shape: supposed to be [[batch, children], [batch, values]]
        :return:
        """
        children_shape, value_shape = input_shape
        total_input_size = children_shape[1].value + value_shape[1].value

        self.gating_f = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1 + self.arity, activation=tf.sigmoid)])

        size = min(int((total_input_size + self.embedding_size) * self.hidden_coef), self.embedding_size)
        self.output_f = tf.keras.Sequential([
            tf.keras.layers.Dense(size,
                                  activation=self.activation, name='/1'),
            tf.keras.layers.Dense(size,
                                  activation=self.activation, name='/2a'),
            tf.keras.layers.Dense(self.embedding_size, activation=self.activation, name='/2')
        ])

        super(GatedFixedArityNodeEmbedder, self).build(input_shape)

    def call(self, x, *args, **kwargs):
        """

        :param x: a list[
            zero padded children input [batch,  <= input_size],
            parent values [batch, value_size]
            ]
        :return: [clones, batch, output_size]
        """
        children, values = x
        concat = tf.concat([children, values], axis=-1)
        output = self.output_f(concat)  # [batch, emb]

        # output gatings only on children embeddings (value embedding size might be different)
        # out = g * out + (g1 * c1 + g2 * c2 ...)
        childrens = tf.reshape(children, [children.shape[0], self.arity, -1])  # [batch, arity, children_emb]
        gatings = tf.expand_dims(tf.nn.softmax(self.gating_f(concat), axis=-1), axis=-1)
        corrected = tf.concat([childrens, tf.expand_dims(output, axis=1)], axis=1) * gatings
        return tf.reduce_sum(corrected, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)


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

    def __init__(self, *, embedding_size: int = None, output_model_builder: tf.keras.Model = None, maximum_input_size: int = None, **kwargs):
        """

        :param embedding_size:
        :param output_model_builder:
        :param maximum_input_size: maximum number of inputs
        :param kwargs:
        """

        self.embedding_size = embedding_size
        self.output_model_builder = output_model_builder
        self.maximum_input_size = maximum_input_size

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
