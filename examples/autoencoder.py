from tree_encoder import Encoder, EncoderCellsBuilder
from tree_decoder import Decoder, DecoderCellsBuilder
from batch import BatchOfTreesForEncoding, BatchOfTreesForDecoding
from simple_expression import BinaryExpressionTreeGen, NaryExpressionTreeGen

import tensorflow as tf
import tensorflow.contrib.eager as tfe


def define_flags():

    ##########
    # Tree characteristics
    ##########

    tf.flags.DEFINE_bool(
        "fixed_arity",
        default=False,
        help="Whether to employ trees whose nodes have always the same number of children" )

    tf.flags.DEFINE_integer(
        "max_depth",
        default=4,
        help="Maximum tree depth")

    tf.flags.DEFINE_integer(
        "max_arity",
        default=3,
        help="Maximum tree arity")

    tf.flags.DEFINE_integer(
        "max_node_count",
        default=120,
        help="Maximum total node count")

    ##########
    # Model parameters
    ##########

    tf.flags.DEFINE_integer(
        "cut_arity",
        default=2,
        help="Children exceeding this cardinality are generated/embedded by the same cell."
             "Only makes sense when using trees with a variable arity and employing the FLAT strategy ")

    tf.flags.DEFINE_integer(
        "embedding_size",
        default=100,
        help="Size of the embedding used during tree processing")

    tf.flags.DEFINE_string(
        "activation",
        default='leaky_relu',
        help="activation used where there are no particular constraints")

    tf.flags.DEFINE_string(
        "enc_variable_arity_strategy",
        default="FLAT",
        help="FLAT or REC")

    tf.flags.DEFINE_string(
        "dec_variable_arity_strategy",
        default="FLAT",
        help="FLAT or REC")

    tf.flags.DEFINE_float(
        "hidden_cell_coef",
        default=0.3,
        help="user to linear regres from input-output size to compute hidden size")

    tf.flags.DEFINE_boolean(
        "encoder_gate",
        default=True,
        help="")

    tf.flags.DEFINE_boolean(
        "decoder_gate",
        default=False,
        help="")

    ##########
    # Learning configuration
    ##########

    tf.flags.DEFINE_integer(
        "max_iter",
        default=2000,
        help="Maximum number of iteration to train")

    tf.flags.DEFINE_integer(
        "check_every",
        default=20,
        help="How often (iterations) to check performances")

    tf.flags.DEFINE_integer(
        "batch_size",
        default=64,
        help="")

FLAGS = tf.flags.FLAGS


def main(argv=None):

    #########
    # DATA
    #########

    if FLAGS.fixed_arity:
        tree_gen = BinaryExpressionTreeGen(0,9)
    else:
        tree_gen = NaryExpressionTreeGen(0, 9,FLAGS.max_arity)

    def get_batch():
        return [tree_gen.generate(FLAGS.max_depth) for _ in range(FLAGS.batch_size)]

    #########
    # MODEL
    #########

    activation = getattr(tf.nn, FLAGS.activation)

    encoder = Encoder(tree_def=tree_gen.tree_def,
                      embedding_size=FLAGS.embedding_size,
                      cut_arity=FLAGS.cut_arity, max_arity=FLAGS.max_arity,
                      variable_arity_strategy=FLAGS.enc_variable_arity_strategy,
                      cellsbuilder=EncoderCellsBuilder(
                            EncoderCellsBuilder.simple_cell_builder(hidden_coef=FLAGS.hidden_cell_coef,
                                                                    activation=activation, gate=FLAGS.encoder_gate),
                            EncoderCellsBuilder.simple_dense_embedder_builder(activation=activation),
                            EncoderCellsBuilder.simple_categorical_merger_builder(hidden_coef=FLAGS.hidden_cell_coef,
                                                                                  activation=activation)),
                      name='encoder')

    decoder = Decoder(tree_def=tree_gen.tree_def,
                      embedding_size=FLAGS.embedding_size,
                      max_node_count=FLAGS.max_node_count,
                      max_depth=FLAGS.max_depth,
                      max_arity=FLAGS.max_arity,
                      cut_arity=FLAGS.cut_arity,
                      cellbuilder=DecoderCellsBuilder(
                          DecoderCellsBuilder.simple_distrib_cell_builder(FLAGS.hidden_cell_coef, activation=activation),
                          DecoderCellsBuilder.simple_1ofk_value_inflater_builder(0.5, activation=tf.tanh),
                          DecoderCellsBuilder.simple_node_inflater_builder(FLAGS.hidden_cell_coef,
                                                                           activation=activation,
                                                                           gate=FLAGS.decoder_gate)),
                      variable_arity_strategy=FLAGS.dec_variable_arity_strategy,
                      attention=False)

    ###########
    # TRAINING
    ###########

    optimizer = tf.train.AdamOptimizer()

    for i in range(FLAGS.max_iter):
        with tfe.GradientTape() as tape:
            x = get_batch()
            batch_enc = BatchOfTreesForEncoding(x, FLAGS.embedding_size)
            encodings = encoder(batch_enc)
            batch_dec = BatchOfTreesForDecoding(encodings, tree_gen.tree_def, x)
            decoded = decoder(batch_dec)

            loss_struct, loss_val = batch_dec.reconstruction_loss()
            loss = loss_struct + loss_val

        variables = encoder.variables + decoder.variables
        grad = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grad, variables), global_step=tf.train.get_or_create_global_step())

        if i % FLAGS.check_every == 0:
            print("{0}:\t{1:.2f}".format(i, loss))


if __name__ == "__main__":
    define_flags()
    tfe.run()
