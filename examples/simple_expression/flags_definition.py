import tensorflow as tf


def define_common_flags():

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
        default='tanh',
        help="activation used where there are no particular constraints")

    tf.flags.DEFINE_float(
        "hidden_cell_coef",
        default=0.3,
        help="user to linear regres from input-output size to compute hidden size")

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
        "overwrite",
        default=False,
        help="Whether to overwrite existing model directory")


def define_encoder_flags():
    tf.flags.DEFINE_boolean(
        "encoder_ogate",
        default=True,
        help="output gate")

    tf.flags.DEFINE_boolean(
        "encoder_igate",
        default=False,
        help="input gate")

    tf.flags.DEFINE_string(
        "enc_variable_arity_strategy",
        default="FLAT",
        help="FLAT or REC")

    tf.flags.DEFINE_integer(
        "enc_cell_depth",
        default=2,
        help="how many layers does a cell have, including the output layer"
    )


def define_decoder_flags():
    tf.flags.DEFINE_integer(
        "dec_cell_depth",
        default=2,
        help="how many layers does a cell have, including the output layer"
    )

    tf.flags.DEFINE_boolean(
        "decoder_gate",
        default=False,
        help="")

    tf.flags.DEFINE_string(
        "dec_variable_arity_strategy",
        default="FLAT",
        help="FLAT or REC")
