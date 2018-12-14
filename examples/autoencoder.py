from tree_encoder import Encoder, EncoderCellsBuilder
from tree_decoder import Decoder, DecoderCellsBuilder
from batch import BatchOfTreesForEncoding, BatchOfTreesForDecoding
from simple_expression import BinaryExpressionTreeGen, NaryExpressionTreeGen
from definition import Tree

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow.contrib.summary as tfs

import os
import json


def define_flags():

    ##########
    # Tree characteristics
    ##########

    tf.flags.DEFINE_bool(
        "fixed_arity",
        default=True,
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
        default=5000,
        help="Maximum number of iteration to train")

    tf.flags.DEFINE_integer(
        "check_every",
        default=20,
        help="How often (iterations) to check performances")

    tf.flags.DEFINE_integer(
        "batch_size",
        default=64,
        help="")

    ##########
    # Checkpoints and Logging
    ##########

    tf.flags.DEFINE_string(
        "model_dir",
        default="/tmp/tree_autoencoder/test",
        help="Directory to put the model summaries, parameters and checkpoint.")

    tf.flags.DEFINE_boolean(
        "restore",
        default=False,
        help="Whether to restore a previously saved model")

    tf.flags.DEFINE_boolean(
        "overwrite",
        default=False,
        help="Whether to overwrite existing model directory")


FLAGS = tf.flags.FLAGS


def main(argv=None):

    #########
    # Checkpoints and Summaries
    #########

    if tf.gfile.Exists(FLAGS.model_dir):
        if FLAGS.overwrite:
            tf.logging.warn("Deleting old log directory at {}".format(FLAGS.model_dir))
            tf.gfile.DeleteRecursively(FLAGS.model_dir)
            tf.gfile.MakeDirs(FLAGS.model_dir)
        elif not FLAGS.restore:
            raise ValueError("Log dir already exists!")
    else:
        tf.gfile.MakeDirs(FLAGS.model_dir)

    summary_writer = tfs.create_file_writer(FLAGS.model_dir, flush_millis=1000)
    summary_writer.set_as_default()
    print("Summaries in " + FLAGS.model_dir)

    if not FLAGS.restore:
        with open(os.path.join(FLAGS.model_dir, "flags.json"), 'w') as f:
            json.dump(FLAGS.flag_values_dict(), f)
    else:
        with open(os.path.join(FLAGS.model_dir, "flags.json")) as f:
            info = json.load(f)
            override_flags = ["embedding_size", "activation", "hidden_cell_coef",
                              "fixed_arity", "max_arity", "cut_arity", "max_node_count",
                              "enc_variable_arity_strategy", "dec_variable_arity_strategy",
                              "encoder_gate", "decoder_gate"]
        for f in override_flags:
            setattr(FLAGS, f, info[f])

    #########
    # DATA
    #########

    if FLAGS.fixed_arity:
        tree_gen = BinaryExpressionTreeGen(0, 9)
    else:
        tree_gen = NaryExpressionTreeGen(0, 9, FLAGS.max_arity)

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

    with tfs.always_record_summaries():
        for i in range(FLAGS.max_iter):
            with tfe.GradientTape() as tape:
                xs = get_batch()
                batch_enc = BatchOfTreesForEncoding(xs, FLAGS.embedding_size)
                encodings = encoder(batch_enc)
                batch_dec = BatchOfTreesForDecoding(encodings, tree_gen.tree_def, target_trees=xs)
                decoded = decoder(batch_dec)

                loss_struct, loss_val = batch_dec.reconstruction_loss()
                loss = loss_struct + loss_val

            variables = encoder.variables + decoder.variables
            grad = tape.gradient(loss, variables)

            gnorm = tf.global_norm(grad)
            tfs.scalar("grad/global_norm", gnorm)

            optimizer.apply_gradients(zip(grad, variables), global_step=tf.train.get_or_create_global_step())

            if i % FLAGS.check_every == 0:

                batch_dec_unsuperv = BatchOfTreesForDecoding(encodings, tree_gen.tree_def)
                decoded_unsuperv = decoder(batch_dec_unsuperv)

                _, _, v_avg_sup, v_acc_sup = Tree.compare_trees(xs, decoded)
                s_avg, s_acc, v_avg, v_acc = Tree.compare_trees(xs, decoded_unsuperv)

                print("{0}:\t{1:.3f}".format(i, loss))

                tfs.scalar("loss/struct", loss_struct)
                tfs.scalar("loss/val", loss_val)
                tfs.scalar("loss/loss", loss)

                tfs.scalar("overlaps/supervised/value_avg", v_avg_sup)
                tfs.scalar("overlaps/supervised/value_acc", v_acc_sup)

                tfs.scalar("overlaps/unsupervised/struct_avg", s_avg)
                tfs.scalar("overlaps/unsupervised/struct_acc", s_acc)
                tfs.scalar("overlaps/unsupervised/value_avg", v_avg)
                tfs.scalar("overlaps/unsupervised/value_acc", v_acc)


if __name__ == "__main__":
    define_flags()
    tfe.run()
