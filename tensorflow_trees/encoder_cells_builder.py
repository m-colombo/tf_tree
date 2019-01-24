import tensorflow as tf
import typing as T

from tensorflow_trees.definition import NodeDefinition


class GatedFixedArityNodeEmbedder(tf.keras.Model):
    """ Build a dense 2-layer which is optimized for left-0-padded input """

    def __init__(self, _no_positional=None, activation=None, hidden_coef: float= None, embedding_size: int = None,
                 arity: int = None,
                 **kwargs):
        super(GatedFixedArityNodeEmbedder, self).__init__(**kwargs)

        if _no_positional != None:
            raise ValueError("Positional argument not allowed!")

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
    def __init__(self, _no_positional=None, activation=None, hidden_coef: float= None, embedding_size: int = None,
                 **kwargs):
        super(GatedValueMerger, self).__init__(**kwargs)
        if _no_positional != None:
            raise ValueError("Positional argument not allowed!")

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

    def __init__(self, _no_positional=None, input_size: int = None,
                 hidden_activation=None, hidden_size: int = None,
                 **kwargs):
        super(NullableInputDenseLayer, self).__init__(**kwargs)

        if _no_positional != None:
            raise ValueError("Positional argument not allowed!")

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

    def __init__(self, _no_positional=None, embedding_size:int=None, output_model_builder: tf.keras.Model = None, input_size: int = None, **kwargs):

        if _no_positional != None:
            raise ValueError("Positional argument not allowed!")

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


class EncoderCellsBuilder:
    """ Define interfaces and simple implementations for a cells builder, factory used for the encoder modules.
        - Cells: merge children embeddings into parent embedding
        - Embedder: project leaves value into embedding space
        - Merger: merge node value and node embedding into embedding space.
    """

    def __init__(self,
                 cell_builder: T.Callable[[NodeDefinition, "Encoder", T.Union[str, None]], tf.keras.Model],
                 embedder_builder: T.Callable[[NodeDefinition, int, T.Union[str, None]], tf.keras.Model],
                 merger_builder: T.Callable[[NodeDefinition, int, T.Union[str, None]], tf.keras.Model]):
        """Simple implementation which just use callables, avoiding superfluous inheritance

        :param cell_builder: see CellsBuilder.build_cell doc
        :param embedder_builder: see CellsBuilder.build_embedder doc
        :param merger_builder: see CellsBuilder.build_merger doc
        """
        self._cell_builder = cell_builder
        self._embedder_builder = embedder_builder
        self._merger_builder = merger_builder

    def build_cell(self, parent_def: NodeDefinition, encoder: "Encoder", name=None) -> tf.keras.Model:
        """A cell is something that merge children embeddings into the parent embedding.
        Actual input shape shape depends from parent arity.
            - Fixed arity: input size = total children * embedding_size
            - Variable arity: input_size = 2 * embedding_size """

        m = self._cell_builder(parent_def, encoder, name)
        if type(m) == tuple:
            for mi in m:
                # setattr(mi, 'compiled_call', tf.contrib.eager.defun(mi))   # it's actually slower
                setattr(mi, 'optimized_call', mi.__call__)
        else:
            # setattr(m, 'compiled_call', tf.contrib.eager.defun(m))   # it's actually slower
            setattr(m, 'optimized_call', m.__call__)

        return m

    def build_embedder(self, leaf_def: NodeDefinition, embedding_size: int, name=None):
        """An embedder is something that projects leaf value into the embedding space"""
        m = self._embedder_builder(leaf_def, embedding_size, name)
        # setattr(m, 'optimized_call', tf.contrib.eager.defun(m))   # it's actually slower
        setattr(m, 'optimized_call', m)
        return m

    def build_merger(self, node_def: NodeDefinition, embedding_size: int, name=None):
        """A merger is something that take a node value, its embedding and merge them into a single embedding"""
        m = self._merger_builder(node_def, embedding_size, name)
        # setattr(m, 'optimized_call', tf.contrib.eager.defun(m))   # it's actually slower
        setattr(m, 'optimized_call', m)
        return m

    @staticmethod
    def simple_categorical_embedder_builder(hidden_coef: int, activation=tf.nn.leaky_relu):
        def f(leaf_def: NodeDefinition, embedding_size, name=None):
            return tf.keras.Sequential([
                tf.keras.layers.Dense(int(embedding_size * hidden_coef),
                                      activation=activation,
                                      input_shape=(leaf_def.value_type.representation_shape,),
                                      name=name+'/1'),
                tf.keras.layers.Dense(embedding_size,
                                      activation=activation,
                                      name=name+"/2")
            ])
        return f

    @staticmethod
    def simple_dense_embedder_builder(activation=tf.nn.leaky_relu):
        def f(leaf_def: NodeDefinition, embedding_size, name=None):

            return tf.keras.layers.Dense(embedding_size,
                                         activation=activation,
                                         input_shape=(leaf_def.value_type.representation_shape,),
                                         name=name)
        return f

    @staticmethod
    def simple_categorical_merger_builder(hidden_coef, activation=tf.nn.leaky_relu):
        """
        """
        def f(node_def: NodeDefinition, embedding_size, name=None):
            input_size = embedding_size + node_def.value_type.representation_shape
            # return GatedValueMerger(activation=activation, embedding_size=embedding_size, hidden_coef=hidden_coef, name=name)
            return tf.keras.Sequential([
                tf.keras.layers.Dense(int((input_size + embedding_size) * hidden_coef),
                                      activation=activation,
                                      input_shape=(input_size,),
                                      name=name + '/1'),
                tf.keras.layers.Dense(embedding_size,
                                      activation=activation,
                                      name=name + "/2")])
        return f

    @staticmethod
    def simple_cell_builder(hidden_coef, activation=tf.nn.leaky_relu, gate=True):
        def f(node_def: NodeDefinition, encoder: 'Encoder', name=None):
            if type(node_def.arity) == node_def.VariableArity:
                if not encoder.use_flat_strategy:
                    # TODO use rnn/lstm
                    input_shape = (encoder.embedding_size*2,)
                    if gate:
                        return GatedFixedArityNodeEmbedder(activation=activation, hidden_coef=hidden_coef, embedding_size=encoder.embedding_size, arity=2)
                else:
                    output_model_builder = lambda :tf.keras.Sequential([
                        NullableInputDenseLayer(input_size=encoder.embedding_size * (encoder.cut_arity + 1),  # 1 is the summarization of extra children
                                                hidden_activation=activation, hidden_size=encoder.embedding_size * int(encoder.cut_arity**hidden_coef)),
                        tf.keras.layers.Dense(encoder.embedding_size, activation=activation)
                    ], name=name)

                    return GatedNullableInput(output_model_builder=output_model_builder,
                                              input_size=encoder.embedding_size * (encoder.cut_arity + 1),
                                              embedding_size=encoder.embedding_size,
                                              name=name) if gate else output_model_builder(),\
                            tf.keras.Sequential([
                                # tf.keras.layers.Reshape([encoder.max_arity - encoder.cut_arity, encoder.embedding_size]),
                                tf.keras.layers.Dense(int(encoder.embedding_size * hidden_coef), activation=activation, input_shape=(encoder.embedding_size,),
                                                      name=name + '/extra_attention/1'),
                                tf.keras.layers.Dense(1, name=name + '/extra_attention/2')
                            ])
                    # input_shape = (encoder.embedding_size * encoder.max_arity, )
            elif type(node_def.arity) == node_def.FixedArity:
                if gate:
                    return GatedFixedArityNodeEmbedder(activation=activation, hidden_coef=hidden_coef, embedding_size=encoder.embedding_size, arity=node_def.arity.value)
                else:
                    return tf.keras.Sequential([
                        tf.keras.layers.Dense(int((encoder.embedding_size) * hidden_coef),
                                              activation=activation, name='/1'),
                        tf.keras.layers.Dense(encoder.embedding_size, activation=activation, name='/2')
                    ])

            return tf.keras.Sequential([
                tf.keras.layers.Dense(int((input_shape[0] + encoder.embedding_size) * hidden_coef), activation=activation, input_shape=input_shape, name=name+'/1'),
                tf.keras.layers.Dense(encoder.embedding_size, activation=activation, name=name+'/2')
            ])
        return f

