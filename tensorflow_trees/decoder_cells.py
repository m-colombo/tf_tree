import tensorflow as tf
import typing as T

from tensorflow_trees.definition import NodeDefinition


class GatedFixedArityNodeDecoder(tf.keras.Model):
    """ Build a dense 2-layer which is optimized for left-0-padded input """

    def __init__(self, *, activation=None, hidden_coef: float= None, embedding_size: int = None,
                 arity: int = None,
                 **kwargs):

        self.activation = activation
        self.hidden_coef = hidden_coef
        self.embedding_size = embedding_size
        self.arity = arity
        super(GatedFixedArityNodeDecoder, self).__init__(**kwargs)

    def build(self, input_shape):

        self.gating_f = tf.keras.Sequential([
            # tf.keras.layers.Dense(units=int(input_shape[1].value * self.hidden_coef), activation=tf.sigmoid),
                                             tf.keras.layers.Dense(units=1, activation=tf.sigmoid)])
        self.output_f = tf.keras.Sequential([
            tf.keras.layers.Dense(int((input_shape[1].value + self.embedding_size * self.arity) * self.hidden_coef),
                                  activation=self.activation, input_shape=input_shape),
            tf.keras.layers.Dense(self.embedding_size * self.arity, activation=self.activation)
        ])
        super(GatedFixedArityNodeDecoder, self).build(input_shape)

    def call(self, x, *args, **kwargs):
        """

        :param x: zero padded input [batch,  <= input_size]
        :return: [clones, batch, output_size]
        """
        parent_embs = x[:,:self.embedding_size] # TODO check - concat order is importand
        output = self.output_f(x)  # [batch, emb * arity]
        childrens = tf.reshape(output, [x.shape[0], self.arity, -1])  # [batch, arity, children_emb]
        gating_inp = tf.reshape(tf.concat([childrens, tf.tile(tf.expand_dims(x, axis=1), [1,self.arity, 1])], axis=-1), [x.shape[0] * self.arity, -1])
        gatings = self.gating_f(gating_inp)
        corrected = tf.reshape(childrens, [x.shape[0] * self.arity, -1]) * gatings + (1 - gatings) * tf.reshape(tf.tile(tf.expand_dims(parent_embs, axis=1), [1,self.arity, 1]), [x.shape[0] * self.arity, -1])
        return tf.reshape(corrected, [x.shape[0], -1])


class ParallelDense(tf.keras.layers.Layer):
    """ Build n dense (two layer) parallel (independent) layers """

    def __init__(self, activation, hidden_size: int, output_size: int, parallel_clones: int, gated: bool = False, **kwargs):
        self.activation = activation
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.parallel_clones = parallel_clones
        self.gated = gated

        super(ParallelDense, self).__init__(**kwargs)

    def build(self, input_shape):

        self.hidden_kernel = self.add_weight(name='hidden_kernel',
                                             shape=[self.parallel_clones, input_shape[1].value, self.hidden_size],
                                             initializer='random_normal',
                                             trainable=True)

        self.hidden_bias= self.add_weight(name='hidden_bias',
                                          shape=(self.parallel_clones, 1, self.hidden_size),    # 1 is for broadcasting
                                          initializer='random_normal',
                                          trainable=True)

        self.out_kernel = self.add_weight(name='out_kernel',
                                          shape=(self.parallel_clones, self.hidden_size, self.output_size),
                                          initializer='random_normal',
                                          trainable=True)

        self.out_bias= self.add_weight(name='out_bias',
                                             shape=(self.parallel_clones, 1, self.output_size), # 1 is for broadcasting
                                             initializer='random_normal',
                                             trainable=True)

        if self.gated:
            self.gate_kernel = self.add_weight(name='gate_kernel',
                                               shape=[ (self.parallel_clones * self.output_size) + input_shape[1].value, self.parallel_clones],
                                               initializer='random_normal',
                                               trainable=True)

            self.gate_bias = self.add_weight(name='gate_bias',
                                             shape=[self.parallel_clones],
                                             initializer='random_normal',
                                             trainable=True)

        super(ParallelDense, self).build(input_shape)

    def call(self, x, n:int):
        """

        :param x: input [batch, input_size]
        :param n: compute only the first n clones
        :return: [clones, batch, output_size]
        """
        x_ = tf.tile(tf.expand_dims(x, axis=0), [n, 1, 1])    # [clones, batch, input]
        #  [clones, batch, input] * [clones, input, hidden] = [clones, batch, hidden]
        hidden_activation = self.activation(tf.matmul(x_, self.hidden_kernel[:n]) + self.hidden_bias[:n])

        # [clones, batch, hidden] * [clones, hidden, output] = [clones, batch, output]
        output = self.activation(tf.matmul(hidden_activation, self.out_kernel[:n]) + self.out_bias[:n])

        if self.gated:
            gate_inp = tf.concat([x, tf.reshape(output, [x.shape[0], -1])], axis=-1)
            gate = tf.nn.softmax(tf.nn.leaky_relu(tf.matmul(gate_inp, self.gate_kernel[:gate_inp.shape[1], :n]) + self.gate_bias[:n]), axis=-1)
            gate = tf.reshape(gate, [n, -1, 1])
            corrected = tf.reshape(x, [1, x.shape[0], -1])[:, :, :self.output_size] * gate + (1-gate) * output
            return corrected

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)   # TODO this is wrong, can't be used in Sequential composition

