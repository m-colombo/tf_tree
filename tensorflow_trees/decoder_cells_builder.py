import tensorflow as tf
import typing as T

from tensorflow_trees.definition import NodeDefinition


class GatedFixedArityNodeDecoder(tf.keras.Model):
    """ Build a dense 2-layer which is optimized for left-0-padded input """

    def __init__(self, _no_positional=None, activation=None, hidden_coef: float= None, embedding_size: int = None,
                 arity: int = None,
                 **kwargs):

        if _no_positional != None:
            raise ValueError("Positional argument not allowed!")

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


class DecoderCellsBuilder:
    """ Define interfaces and simple implementations for a cells builder, factory used for the decoder modules.
           - Distrib : generates a distribution over nodes types
           - Value infl. : projects from the embedding space to some value space
           - Node infl. : from parent embedding generates children embeddings (actual shapes depends on node arity)
       """

    def __init__(self,
                 distrib_builder: T.Callable[[T.Tuple[int, int], T.Optional[str]], tf.keras.Model],
                 value_inflater_builder: T.Callable[[NodeDefinition, T.Optional[str]], tf.keras.Model],
                 node_inflater_builder: T.Callable[[NodeDefinition, "Decoder", T.Optional[str]], tf.keras.Model]):
        """Simple implementation which just use callables, avoiding superfluous inheritance

        :param distrib_builder: see self.build_distrib_cell
        :param value_inflater_builder: see self.build_value_inflater
        :param node_inflater_builder: see self.build_node_inflater
        """
        self._distrib_builder = distrib_builder
        self._value_inflater_builder = value_inflater_builder
        self._node_inflater_builder = node_inflater_builder

    def build_distrib_cell(self, output_size: (int, int), decoder: "Decoder", name=None) -> tf.keras.Model:
        """Build a distribution cell that given an embedding returns a output_size[0] concatenated probability vector of size output_size[1]"""
        m = self._distrib_builder(output_size, decoder, name)
        # setattr(m, 'compiled_call', tf.contrib.eager.defun(m))   # it's actually slower
        setattr(m, 'compiled_call', m.__call__)
        return m

    def build_value_inflater(self, node_def: NodeDefinition, decoder: "Decoder", name=None) -> tf.keras.Model:
        """Build a cell that projects an embedding in the node value space"""
        m = self._value_inflater_builder(node_def, decoder, name)
        # setattr(m, 'compiled_call', tf.contrib.eager.defun(m))   # it's actually slower
        setattr(m, 'compiled_call', m.__call__)
        return m

    def build_node_inflater(self, node_def: NodeDefinition, decoder: "Decoder", name=None) -> tf.keras.Model:
        """Build a cell that given parent embedding returns children embeddings.
            - for FixedArity nodes the output is the concat of all the chlidren embeddings
            - for VariableArity nodes it's a RNN - take (state_embedding) as input and returns (child_embedding, new_state_embedding)
        """
        m = self._node_inflater_builder(node_def, decoder, name)
        if type(m) == tuple:
            for mi in m:
                # setattr(mi, 'compiled_call', tf.contrib.eager.defun(mi))   # it's actually slower
                setattr(mi, 'compiled_call', mi.__call__)
        else:
            # setattr(m, 'compiled_call', tf.contrib.eager.defun(m))   # it's actually slower
            setattr(m, 'compiled_call', m.__call__)

        return m

    @staticmethod
    def simple_distrib_cell_builder(hidden_coef, activation=tf.nn.relu):
        def f(output_size: (int, int), decoder: Decoder, name=None):
            total_output_size = output_size[0] * output_size[1]

            size1 = int((total_output_size + decoder.embedding_size) * hidden_coef)
            size2 = int((size1 + decoder.embedding_size) * hidden_coef)

            return tf.keras.Sequential([
                tf.keras.layers.Dense(300, activation=activation),
                # tf.keras.layers.Dense(200, activation=activation),
                tf.keras.layers.Dense(output_size[0] * output_size[1]),
                tf.keras.layers.Lambda(lambda x: tf.reshape(x, [-1, output_size[1]]), output_shape=(output_size[1],)),
                tf.keras.layers.Softmax()
            ], name=name)
        return f

    @staticmethod
    def simple_node_inflater_builder(hidden_coef: float, activation=tf.nn.relu, gate=True):
        def f(node_def: NodeDefinition, decoder, name=None):
            if type(node_def.arity) == NodeDefinition.FixedArity:
                if gate:
                    return GatedFixedArityNodeDecoder(activation=activation, hidden_coef=hidden_coef,
                                                  embedding_size=decoder.embedding_size,
                                                  arity=node_def.arity.value, name=name)
                else:
                    return tf.keras.Sequential([
                                        tf.keras.layers.Dense(
                        int(hidden_coef * decoder.embedding_size * node_def.arity.value), activation=activation),
                                        tf.keras.layers.Dense(decoder.embedding_size * node_def.arity.value,
                                                               activation=activation),
                    ], name=name)
            elif type(node_def.arity) == NodeDefinition.VariableArity and not decoder.use_flat_strategy:
                return tf.keras.Sequential([
                    tf.keras.layers.Dense(int(decoder.embedding_size * 2 * hidden_coef), activation=activation),
                    tf.keras.layers.Dense(decoder.embedding_size * 2, activation=activation),
                ], name=name)
            elif type(node_def.arity) == NodeDefinition.VariableArity and decoder.use_flat_strategy:
                return ParallelDense(activation=activation, hidden_size=decoder.embedding_size,
                                     output_size=decoder.embedding_size,
                                     parallel_clones=decoder.cut_arity, gated=gate, name=name), \
                       tf.keras.Sequential([
                           tf.keras.layers.Dense(int(decoder.embedding_size * (1+hidden_coef*0.5)), activation=activation),
                           tf.keras.layers.Dense(decoder.embedding_size, activation=activation),
                       ], name=name+'_extra')
        return f

    @staticmethod
    def simple_1ofk_value_inflater_builder(hidden_coef, activation=tf.nn.relu):
        def f(node_def: NodeDefinition, decoder: "Decoder", name=None):
            size1 = int((node_def.value_type.representation_shape + decoder.embedding_size) * hidden_coef)
            size2 = int((size1 + decoder.embedding_size) * hidden_coef)
            return tf.keras.Sequential([
                tf.keras.layers.Dense(2*size1, activation=activation),
                # tf.keras.layers.Dense(size2, activation=activation),
                tf.keras.layers.Dense(node_def.value_type.representation_shape),
                tf.keras.layers.Softmax()
            ], name=name)
        return f

    @staticmethod
    def simple_dense_value_inflater_builder(hidden_coef, activation=tf.nn.relu):
        def f(node_def: NodeDefinition, decoder: "Decoder", name=None):
            size1 = int((node_def.value_type.representation_shape + decoder.embedding_size) * hidden_coef)
            size2 = int((size1 + decoder.embedding_size) * hidden_coef)
            return tf.keras.Sequential([
                tf.keras.layers.Dense(size1, activation=activation),
                tf.keras.layers.Dense(size2, activation=activation),
                tf.keras.layers.Dense(node_def.value_type.representation_shape),
            ], name=name)
        return f

    @staticmethod
    def node_map(map):
        def f(node_def, *args, **kwargs):
            return map[node_def.id](node_def, *args, **kwargs)
        return f

