import tensorflow as tf
import typing as T

from tensorflow_trees.definition import NodeDefinition


class GatedFixedArityNodeEmbedder(tf.keras.Model):
    """ Build a dense 2-layer which is optimized for left-0-padded input """

    def __init__(self, *, activation=None, hidden_coef: float= None, embedding_size: int = None,
                 arity: int = None,
                 **kwargs):
        super(GatedFixedArityNodeEmbedder, self).__init__(**kwargs)

        self.activation = activation
        self.hidden_coef = hidden_coef
        self.embedding_size = embedding_size
        self.arity = arity

    def build(self, input_shape):

        # self.gating_f.build(input_shape)
        self.gating_f = tf.keras.Sequential([
            # tf.keras.layers.Dense(units=int(input_shape[1].value * self.hidden_coef), activation=tf.sigmoid),
            tf.keras.layers.Dense(units=1 + self.arity, activation=tf.sigmoid)])

        self.output_f = tf.keras.Sequential([
            tf.keras.layers.Dense(min(int((input_shape[1].value + self.embedding_size) * self.hidden_coef), self.embedding_size),
                                  activation=self.activation, name='/1'),
            tf.keras.layers.Dense(self.embedding_size, activation=self.activation, name='/2')
        ])

        # self.output_f.build(input_shape)

        super(GatedFixedArityNodeEmbedder, self).build(input_shape)

    def call(self, x, *args, **kwargs):
        """

        :param x: zero padded input [batch,  <= input_size]
        :return: [clones, batch, output_size]
        """
        output = self.output_f(x)  # [batch, emb]
        childrens = tf.reshape(x, [x.shape[0], self.arity, -1])  # [batch, arity, children_emb]
        gatings = tf.expand_dims(tf.nn.softmax(self.gating_f(tf.concat([x, output], axis=-1)), axis=-1), axis=-1)
        corrected = tf.concat([childrens, tf.expand_dims(output, axis=1)], axis=1) * gatings
        return tf.reduce_sum(corrected, axis=1)


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)


class GatedValueMerger(tf.keras.Model):
    # TODO dunno why it doesn't work
    def __init__(self, *, activation=None, hidden_coef: float= None, embedding_size: int = None,
                 **kwargs):
        super(GatedValueMerger, self).__init__(**kwargs)

        self.activation = activation
        self.hidden_coef = hidden_coef
        self.embedding_size = embedding_size


    # def build(self, input_shape):
    #     # self.gating_f = tf.keras.Sequential([
    #     #     # tf.keras.layers.Dense(units=int(input_shape[1].value * self.hidden_coef), activation=tf.sigmoid),
    #     #     tf.keras.layers.Dense(units=1 , activation=tf.sigmoid)])
    #     #
    #     # self.output_f = tf.keras.Sequential([
    #     #     tf.keras.layers.Dense(int((input_shape[0].value + self.embedding_size) * self.hidden_coef),
    #     #                           activation=self.activation, input_shape=input_shape, name='/1'),
    #     #     tf.keras.layers.Dense(self.embedding_size, activation=self.activation, name='/2')
    #     # ])
    #     super(GatedValueMerger, self).build(input_shape)

    # def call(self, x, *args, **kwargs):
    #     """
    #
    #     :param x: zero padded input [batch,  <= input_size]
    #     :return: [clones, batch, output_size]
    #     """
    #     # output = self.output_f(x)  # [batch, emb]
    #     # gatings = tf.nn.softmax(self.gating_f(tf.concat([x, output], axis=-1)), axis=-1)
    #     # corrected = output * gatings + (1 - gatings) * x[:, :self.embedding_size]
    #     # return corrected
    #     return tf.zeros([x.shape[0],self.embedding_size])

    # def compute_output_shape(self, input_shape):
    #     return (input_shape[0], self.embedding_size)


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

        :param x: zero padded input [batch,  <= input_size, emb]
        :return: [clones, batch, output_size]
        """
        n = x.shape[1].value

        hidden_activation = self.hidden_activation(tf.matmul(x, self.hidden_kernel[:n]) + self.hidden_bias)

        return hidden_activation

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.hidden_size)


class GatedNullableInput(tf.keras.Model):

    def __init__(self, *, embedding_size: int = None, output_model_builder: tf.keras.Model = None, input_size: int = None, **kwargs):

        self.embedding_size = embedding_size
        self.output_model_builder = output_model_builder
        self.input_size = input_size

        super(GatedNullableInput, self).__init__(**kwargs)

    def build(self, input_shape):

        self.gating_f = NullableInputDenseLayer(input_size=self.input_size + self.embedding_size, hidden_activation=tf.nn.leaky_relu, hidden_size=self.input_size // self.embedding_size + 1)
        self.output_model = self.output_model_builder()
        # self.gating_f = tf.keras.layers.Dense(units=input_shape[1].value // self.embedding_size + 1, activation=tf.sigmoid, name="gate")

        super(GatedNullableInput, self).build(input_shape)

    def call(self, x, *args, **kwargs):
        """

        :param x: zero padded input [batch,  <= input_size * emb]
        :return: [clones, batch, output_size]
        """

        output = self.output_model(x)
        number_of_input = x.shape[1].value // self.embedding_size
        gating_inp = tf.concat([output, x], axis=-1)
        gatings = tf.nn.softmax(self.gating_f(gating_inp)[:, :number_of_input+1], axis=1)
        weighted = tf.reshape(gating_inp, [x.shape[0], number_of_input + 1, -1]) * tf.expand_dims(gatings, -1)
        return tf.reduce_sum(weighted, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.embedding_size)
