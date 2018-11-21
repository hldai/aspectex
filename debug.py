# from seqItem import *
import numpy as np
from utils import utils
import tensorflow as tf
from models import crf
from tensorflow.contrib.crf import crf_log_likelihood, crf_log_norm


def __test():
    np.random.seed(152)
    batch_size = 3
    seq_len = 7
    n_tags = 5
    logits = tf.constant(np.random.uniform(0, 2, (batch_size, seq_len, n_tags)), tf.float32)
    trans_mat = tf.constant(np.random.uniform(0, 1, (n_tags, n_tags)), tf.float32)
    trans_mat = tf.nn.softmax(trans_mat)

    y_seq_vals = np.random.randint(0, n_tags, (batch_size, seq_len))
    y_seq = tf.placeholder(tf.int32, shape=[None, seq_len])
    # y_seq = tf.constant(y_seq_vals, np.int32)
    # tmp_idxs = tf.constant(np.arange(0, batch_size, dtype=np.int32), tf.int32, shape=[batch_size, 1])
    # y_seq_with_idxs = tf.concat([tmp_idxs, tf.reshape(y_seq[:, 0], (batch_size, 1))], axis=1)
    # tmp_idxs = tf.constant()
    # gather_nd_idxs = tf.gather(y_seq_with_idxs, [0, 1], axis=1)
    # unary_scores_y = tf.gather_nd(logits[:, 0, :], y_seq_with_idxs)
    # bin_score_idxs = tf.gather(y_seq, [0, 1], axis=1)
    # bin_scores_y = tf.gather_nd(trans_mat, bin_score_idxs)

    # f0 = logits[:, 0, :]
    # f0_re = tf.reshape(f0, [-1, 1, n_tags])
    # logits1_re = tf.reshape(logits[:, 1, :], [-1, n_tags, 1])
    # f1 = f0_re + trans_mat_t_reshaped + logits1_re
    # f1_sum = tf.reduce_logsumexp(f1, axis=2)

    # score_mat = tf.gather(trans_mat, y_seq)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # log_z = crf.log_partition_func_z(n_tags, logits, trans_mat)
    # log_score = crf.crf_log_likelihood_dhl(y_seq, logits, trans_mat, batch_size, n_tags)

    seq_lens = tf.constant(np.array([seq_len for _ in range(batch_size)], np.int32), tf.int32)
    # log_likelihood, _ = crf_log_likelihood(logits, y_seq, seq_lens, trans_mat)
    # log_z, alphas = crf.crf_log_norm_forward(logits, seq_lens, trans_mat)
    # log_z_cal = tf.reduce_logsumexp(alphas, 2)

    # log_z, alphas = crf.crf_log_norm_forward(logits, seq_lens, trans_mat)
    log_z, alphas = crf.crf_log_norm_forward_with_scan(logits, trans_mat)
    # alphas = tf.transpose(alphas, [1, 0, 2])

    betas = crf.crf_beta_backward(logits, trans_mat)

    # alpha_betas = tf.concat([alphas, betas], 2)
    alpha_betas = alphas + betas
    valids = tf.reduce_logsumexp(alpha_betas, 2)

    # log_z = crf_log_norm(logits, seq_lens, trans_mat)

    sess = tf.Session()
    # vals = sess.run([alphas, betas, alpha_betas, valids, log_z], feed_dict={y_seq: y_seq_vals})
    vals = sess.run([alphas, betas, log_z, valids], feed_dict={y_seq: y_seq_vals})
    for i, v in enumerate(vals):
        print('val', i)
        print(v)


# __test()
vals = np.random.randint(0, 3, [2, 3])
print(vals)
exp_vals = np.exp(vals)
print(exp_vals)
print(exp_vals / np.expand_dims(np.sum(exp_vals, axis=1), 1))

print(np.max(vals, axis=1))
exp_vals = np.exp(vals - np.expand_dims(np.max(vals, axis=1), 1))
print(exp_vals / np.expand_dims(np.sum(exp_vals, axis=1), 1))

