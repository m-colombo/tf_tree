from tensorflow_trees.tree_encoder import Encoder, EncoderCellsBuilder
from tensorflow_trees.tree_decoder import Decoder, DecoderCellsBuilder
from tensorflow_trees.simple_expression import BinaryExpressionTreeGen, NaryExpressionTreeGen
from tensorflow_trees.definition import Tree

from examples.flags_definition import *

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow.contrib.summary as tfs

import os
import json

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
                      variable_arity_strategy=FLAGS.dec_variable_arity_strategy)

    ###########
    # TRAINING
    ###########

    optimizer = tf.train.AdamOptimizer()

    with tfs.always_record_summaries():
        for i in range(FLAGS.max_iter):
            with tfe.GradientTape() as tape:
                xs = get_batch()
                batch_enc = encoder(xs)
                batch_dec = decoder(encodings=batch_enc.get_root_embeddings(), targets=xs)

                loss_struct, loss_val = batch_dec.reconstruction_loss()
                loss = loss_struct + loss_val

            variables = encoder.variables + decoder.variables
            grad = tape.gradient(loss, variables)

            gnorm = tf.global_norm(grad)
            grad, _ = tf.clip_by_global_norm(grad, 0.02, gnorm)

            tfs.scalar("norms/grad", gnorm)

            optimizer.apply_gradients(zip(grad, variables), global_step=tf.train.get_or_create_global_step())

            if i % FLAGS.check_every == 0:

                batch_unsuperv = decoder(encodings=batch_enc.get_root_embeddings())

                _, _, v_avg_sup, v_acc_sup = Tree.compare_trees(xs, batch_dec.decoded_trees)
                s_avg, s_acc, v_avg, v_acc = Tree.compare_trees(xs, batch_unsuperv.decoded_trees)

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
    define_common_flags()
    define_encoder_flags()
    define_decoder_flags()
    tfe.run()
