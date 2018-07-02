import numpy as np
import tensorflow as tf
import config
from utils import utils


def __build_data(sent_text_file, correctness_file, word_vecs):
    y_true = utils.read_lines(correctness_file)
    y_true = np.asarray([int(yi) for yi in y_true], np.int32)

    sent_texts = utils.read_lines(sent_text_file)

    vec_dim = next(iter(word_vecs.values())).shape[0]
    print('word vec dim: {}'.format(vec_dim))
    X = np.zeros((len(sent_texts), vec_dim), np.float32)
    for i, sent_text in enumerate(sent_texts):
        words = sent_text.split(' ')
        wcnt = 0
        for w in words:
            vec = word_vecs.get(w, None)
            if vec is not None:
                X[i] += vec
                wcnt += 1
        X[i] /= wcnt
    return X, y_true


def __train(X_train, y_true_train, X_test, y_true_test):
    batch_size = 5
    n_epoches = 100
    n_batches = X_train.shape[0] // batch_size

    dim = X_train.shape[1]
    X = tf.placeholder(tf.float32, [None, dim])
    y = tf.placeholder(tf.float32)

    W = tf.Variable(tf.random_uniform([dim, 1]))
    b = tf.Variable(tf.random_uniform([1, 1]))

    f = tf.matmul(X, W) + b
    f = tf.reshape(f, [-1])
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f, labels=y))
    pred = tf.round(tf.sigmoid(f))
    correctness = tf.cast(tf.equal(pred, y), tf.float32)
    accuracy = tf.reduce_mean(correctness)

    opt = tf.train.GradientDescentOptimizer(0.01)
    trainer = opt.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for epoch in range(n_epoches):
        losses = list()
        for batch_idx in range(n_batches):
            X_batch = X_train[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            y_batch = y_true_train[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            _, loss_val = sess.run([trainer, loss], feed_dict={
                X: X_batch,
                y: y_batch
            })
            losses.append(loss_val)

        y_pred_val = sess.run(pred, feed_dict={
            X: X_test
        })


correctness_file = 'd:/data/aspect/semeval14/deprnn-correctness.txt'
print('loading {} ...'.format(config.WORD_VEC_FILE))
word_vecs = utils.load_word_vec_file(config.WORD_VEC_FILE)
print('done')
n_train = 600
X_input, y_true = __build_data(config.SE14_LAPTOP_TEST_SENT_TOK_TEXTS_FILE, correctness_file, word_vecs)
perm = np.random.permutation(len(y_true))
idxs_train, idxs_test = perm[:n_train], perm[n_train:]
X_train = X_input[idxs_train]
y_train_true = y_true[idxs_train]
X_test = X_input[idxs_test]
y_test_true = y_true[idxs_test]
__train(X_train, y_train_true, X_test, y_test_true)
