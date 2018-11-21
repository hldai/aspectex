import tensorflow as tf
import numpy as np
from tensorflow.contrib.crf import crf_unary_score, CrfForwardRnnCell
from tensorflow.python.ops import rnn
from tensorflow.contrib.framework import smart_cond


def log_partition_func_z(n_tags, unary_score_mat, binary_score_mat):
    trans_mat_t_reshaped = tf.reshape(tf.transpose(binary_score_mat), [-1, n_tags, n_tags])

    tmp_z_val = unary_score_mat[:, 0, :]
    i_while = tf.constant(1)

    def loop_cond(i, cur_tmp_z_val):
        return tf.less(i, tf.shape(unary_score_mat)[1])

    def partition_func_calc_step(i, cur_tmp_z_val):
        cur_tmp_z_val_reshaped = tf.reshape(cur_tmp_z_val, [-1, 1, n_tags])
        unary_vals_reshaped = tf.reshape(unary_score_mat[:, i, :], [-1, n_tags, 1])
        cur_tmp_z_val = cur_tmp_z_val_reshaped + trans_mat_t_reshaped + unary_vals_reshaped
        cur_tmp_z_val = tf.reduce_logsumexp(cur_tmp_z_val, axis=2)
        return i + 1, cur_tmp_z_val

    _, z_val_at_labels = tf.while_loop(
        loop_cond, partition_func_calc_step, (i_while, tmp_z_val))
    log_z = tf.reduce_logsumexp(z_val_at_labels, axis=1)
    return log_z


def log_crf_score_loop(y_seqs_batch, unary_score_mat, binary_score_mat, batch_size):
    tmp_idxs = tf.constant(np.arange(0, batch_size, dtype=np.int32), tf.int32, shape=[batch_size, 1])
    y_seq_with_idxs = tf.concat([tmp_idxs, tf.reshape(y_seqs_batch[:, 0], (batch_size, 1))], axis=1)
    score = tf.gather_nd(unary_score_mat[:, 0, :], y_seq_with_idxs)
    i_while = tf.constant(1)

    def loop_cond(i, cur_score):
        return tf.less(i, tf.shape(unary_score_mat)[1])

    # batch_size = y_seqs_batch.shape[0]

    def crf_score_step(i, cur_score):
        y_seq_with_idxs = tf.concat([tmp_idxs, tf.reshape(y_seqs_batch[:, i], (batch_size, 1))], axis=1)
        unary_scores_y = tf.gather_nd(unary_score_mat[:, i, :], y_seq_with_idxs)

        bin_score_idxs = tf.gather(y_seqs_batch, [i - 1, i], axis=1)
        bin_scores_y = tf.gather_nd(binary_score_mat, bin_score_idxs)

        cur_score += unary_scores_y + bin_scores_y
        return i + 1, cur_score

    _, score = tf.while_loop(loop_cond, crf_score_step, (i_while, score))
    return score


def crf_log_likelihood_dhl(y_seqs_batch, unary_score_mat, binary_score_mat, batch_size, n_tags):
    log_score = log_crf_score_loop(y_seqs_batch, unary_score_mat, binary_score_mat, batch_size)
    log_z = log_partition_func_z(n_tags, unary_score_mat, binary_score_mat)
    return log_score - log_z


def crf_beta_backward(inputs, transition_params):
    batch_size = tf.shape(inputs)[0]
    seq_len = tf.shape(inputs)[1]
    n_tags = tf.shape(inputs)[2]

    def _single_seq_fn():
        return tf.ones([batch_size, 1, n_tags]) * -10000

    def _multi_seq_fn():
        trans_mat_t = tf.transpose(transition_params)

        def scan_step_backward(prev_betas, inputs):
            prev_betas_ex = tf.expand_dims(prev_betas, 2)
            inputs_ex = tf.expand_dims(inputs, 2)
            trans_scores = prev_betas_ex + trans_mat_t + inputs_ex
            new_batas = tf.reduce_logsumexp(trans_scores, 1)

            # new_batas = tf.reduce_logsumexp(trans_scores, 2)
            # trans_scores = prev_betas_ex + trans_mat_t
            # new_batas = inputs + tf.reduce_logsumexp(trans_scores, 2)
            return new_batas

        elems = tf.reverse(tf.transpose(inputs, [1, 0, 2]), [0])
        # init_val = tf.ones([batch_size, n_tags]) * -10000
        init_val = tf.zeros([batch_size, n_tags])
        rest_inputs = elems[:-1]
        betas_m = tf.scan(scan_step_backward, rest_inputs, initializer=init_val)
        betas_m = tf.concat([tf.expand_dims(init_val, 0), betas_m], axis=0)
        betas_m = tf.reverse(betas_m, [0])
        return betas_m

    betas = smart_cond(
        pred=tf.equal(seq_len, 1),
        true_fn=_single_seq_fn,
        false_fn=_multi_seq_fn)

    return betas


def crf_log_norm_forward_with_scan(inputs, transition_params):
    def scan_step_forward(inputs, cur_unary_scores):
        inputs_ex = tf.expand_dims(inputs, 2)
        trans_scores = inputs_ex + transition_params
        new_alphas = cur_unary_scores + tf.reduce_logsumexp(trans_scores, 1)
        return new_alphas

    elems = tf.transpose(inputs, [1, 0, 2])
    init_val = elems[0]
    rest_inputs = elems[1:]
    alphas = tf.scan(scan_step_forward, rest_inputs, initializer=init_val)
    alphas = tf.concat([tf.expand_dims(init_val, 0), alphas], axis=0)
    log_z = tf.reduce_logsumexp(alphas[-1], 1)
    return log_z, alphas


def crf_log_norm_forward(inputs, sequence_lengths, transition_params):
    first_input = tf.slice(inputs, [0, 0, 0], [-1, 1, -1])
    first_input = tf.squeeze(first_input, [1])

    def _single_seq_fn():
        log_norm = tf.reduce_logsumexp(first_input, [1])
        # Mask `log_norm` of the sequences with length <= zero.
        log_norm = tf.where(tf.less_equal(sequence_lengths, 0),
                            tf.zeros_like(log_norm),
                            log_norm)
        return log_norm, inputs

    def _multi_seq_fn():
        """Forward computation of alpha values."""
        rest_of_input = tf.slice(inputs, [0, 1, 0], [-1, -1, -1])

        # Compute the alpha values in the forward algorithm in order to get the
        # partition function.
        forward_cell = CrfForwardRnnCell(transition_params)
        # Sequence length is not allowed to be less than zero.
        sequence_lengths_less_one = tf.maximum(
            tf.constant(0, dtype=sequence_lengths.dtype),
            sequence_lengths - 1)
        all_alphas, alphas_final = rnn.dynamic_rnn(
            cell=forward_cell,
            inputs=rest_of_input,
            sequence_length=sequence_lengths_less_one,
            initial_state=first_input,
            dtype=tf.float32)
        log_norm = tf.reduce_logsumexp(alphas_final, [1])
        # Mask `log_norm` of the sequences with length <= zero.
        log_norm = tf.where(tf.less_equal(sequence_lengths, 0),
                            tf.zeros_like(log_norm),
                            log_norm)
        return log_norm, all_alphas

    log_norm_z, alphas = smart_cond(
        pred=tf.equal(tf.shape(inputs)[1], 1),
        true_fn=_single_seq_fn,
        false_fn=_multi_seq_fn)

    return log_norm_z, alphas
