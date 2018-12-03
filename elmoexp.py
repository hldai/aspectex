import tensorflow as tf
import os
from bilm.model import BidirectionalLanguageModel
from bilm.data import TokenBatcher


def __train(vocab_file, options_file, weight_file, token_embedding_file):
    batcher = TokenBatcher(vocab_file)

    # Build the biLM graph.
    bilm = BidirectionalLanguageModel(
        options_file,
        weight_file,
        use_character_inputs=False,
        embedding_weight_file=token_embedding_file
    )

    # Input placeholders to the biLM.
    context_token_ids = tf.placeholder('int32', shape=(None, None))
    question_token_ids = tf.placeholder('int32', shape=(None, None))
    # Get ops to compute the LM embeddings.
    context_embeddings_op = bilm(context_token_ids)
    question_embeddings_op = bilm(question_token_ids)

    context_embeds = context_embeddings_op['lm_embeddings']
    context_mask = context_embeddings_op['mask']
    q_embeds = question_embeddings_op['lm_embeddings']
    q_mask = question_embeddings_op['mask']

    with tf.Session() as sess:
        # It is necessary to initialize variables once before running inference.
        sess.run(tf.global_variables_initializer())
        context_ids = batcher.batch_sentences(tokenized_context)
        question_ids = batcher.batch_sentences(tokenized_question)

        # Compute ELMo representations (here for the input only, for simplicity).
        context_embeds_vals, context_mask_vals, q_embeds_vals, q_mask_vals = sess.run(
            [context_embeds, context_mask, q_embeds, q_mask],
            feed_dict={context_token_ids: context_ids,
                       question_token_ids: question_ids}
        )
    #     print(context_mask_vals.shape)
    #
    # exit()

    n_lm_layers = context_embeds_vals.shape[1]
    lm_dim = context_embeds_vals.shape[3]

    context_embed_input = tf.placeholder(tf.float32, [None, None, None, None], 'context_embed')
    context_mask_input = tf.placeholder(tf.int32, [None, None], 'context_mask')
    q_embed_input = tf.placeholder(tf.float32, [None, None, None, None], 'q_embed')
    q_mask_input = tf.placeholder(tf.int32, [None, None], 'q_mask')

    context_input = {
        'lm_embeddings': context_embed_input,
        'mask': context_mask_input
    }

    q_input = {
        'lm_embeddings': q_embed_input,
        'mask': q_mask_input
    }

    # context_input = context_embeddings_op
    # q_input = question_embeddings_op

    # Get an op to compute ELMo (weighted average of the internal biLM layers)
    # Our SQuAD model includes ELMo at both the input and output layers
    # of the task GRU, so we need 4x ELMo representations for the question
    # and context at each of the input and output.
    # We use the same ELMo weights for both the question and context
    # at each of the input and output.
    elmo_context_input = weight_layers('input', context_input, n_lm_layers, lm_dim, l2_coef=0.0)
    with tf.variable_scope('', reuse=True):
        # the reuse=True scope reuses weights from the context for the question
        elmo_question_input = weight_layers(
            'input', q_input, n_lm_layers, lm_dim, l2_coef=0.0
        )

    elmo_context_output = weight_layers(
        'output', context_input, n_lm_layers, lm_dim, l2_coef=0.0
    )
    with tf.variable_scope('', reuse=True):
        # the reuse=True scope reuses weights from the context for the question
        elmo_question_output = weight_layers(
            'output', q_input, n_lm_layers, lm_dim, l2_coef=0.0
        )

    with tf.Session() as sess:
        # It is necessary to initialize variables once before running inference.
        sess.run(tf.global_variables_initializer())

        # Create batches of data.
        context_ids = batcher.batch_sentences(tokenized_context)
        question_ids = batcher.batch_sentences(tokenized_question)

        # Compute ELMo representations (here for the input only, for simplicity).
        elmo_context_input_, elmo_question_input_ = sess.run(
            [elmo_context_input['weighted_op'], elmo_question_input['weighted_op']],
            # feed_dict={context_token_ids: context_ids,
            #            question_token_ids: question_ids}
            feed_dict={context_embed_input: context_embeds_vals,
                       context_mask_input: context_mask_vals,
                       q_embed_input: q_embeds_vals,
                       q_mask_input: q_mask_vals}
        )

        print(elmo_context_input_)


vocab_file = 'd:/data/res/elmo/tests/vocab_small.txt'
datadir = os.path.join('d:/data/res/elmo/tests', 'fixtures', 'model')
options_file = os.path.join(datadir, 'options.json')
weight_file = os.path.join(datadir, 'lm_weights.hdf5')
token_embedding_file = 'd:/data/res/elmo/tests/elmo_token_embeddings.hdf5'
__train(vocab_file, options_file, weight_file, token_embedding_file)
