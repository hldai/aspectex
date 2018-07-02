import numpy as np
import tensorflow as tf
import config
from utils import utils


class Judge:
    def __init__(self, dim):
        self.X = tf.placeholder(tf.float32, [None, dim])
        self.y = tf.placeholder(tf.float32)

        W = tf.Variable(tf.random_uniform([dim, 1]))
        b = tf.Variable(tf.random_uniform([1, 1]))

        f = tf.matmul(self.X, W) + b
        f = tf.reshape(f, [-1])
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f, labels=self.y)
                                   ) + 0.01 * tf.nn.l2_loss(W)
        self.pred = tf.round(tf.sigmoid(f))
        self.correctness = tf.cast(tf.equal(self.pred, self.y), tf.float32)
        self.accuracy = tf.reduce_mean(self.correctness)

    def train(self, lr, n_epoch, batch_size, X_train, y_true_train, X_val, y_val_test):
        n_batches = X_train.shape[0] // batch_size
        print('{} batches'.format(n_batches))
        opt = tf.train.GradientDescentOptimizer(lr)
        trainer = opt.minimize(self.loss)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        for epoch in range(n_epoch):
            losses = list()
            for batch_idx in range(n_batches):
                X_batch = X_train[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                y_batch = y_true_train[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                _, loss_val = sess.run([trainer, self.loss], feed_dict={
                    self.X: X_batch,
                    self.y: y_batch
                })
                losses.append(loss_val)

            accuracy_val = sess.run(self.accuracy, feed_dict={
                self.X: X_val,
                self.y: y_val_test
            })
            print(sum(losses), accuracy_val)


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


def __train():
    sent_tok_texts_file = 'd:/data/aspect/semeval14/judge_data/laptops_jtest_texts_tok.txt'
    correctness_file = 'd:/data/aspect/semeval14/judge_data/test_correctness.txt'
    print('loading {} ...'.format(config.WORD_VEC_FILE))
    word_vecs = utils.load_word_vec_file(config.WORD_VEC_FILE)
    print('done')
    n_train = 800
    X_input, y_true = __build_data(sent_tok_texts_file, correctness_file, word_vecs)
    perm = np.random.permutation(len(y_true))
    idxs_train, idxs_test = perm[:n_train], perm[n_train:]
    X_train = X_input[idxs_train]
    X_test = X_input[idxs_test]
    y_train_true = y_true[idxs_train]
    y_test_true = y_true[idxs_test]
    # __train_model(X_train, y_train_true, X_test, y_test_true)
    judge = Judge(X_train.shape[1])
    judge.train(0.01, 100, 5, X_train, y_train_true, X_test, y_test_true)


def __build_cnn_data(sent_tok_texts_file, word_vecs, correctness_file):
    sent_texts = utils.read_lines(sent_tok_texts_file)
    data_vocab = set()
    for text in sent_texts:
        words = text.split(' ')
        for w in words:
            data_vocab.add(w)


def __train_cnn():
    sent_tok_texts_file = 'd:/data/aspect/semeval14/judge_data/laptops_jtest_texts_tok.txt'
    correctness_file = 'd:/data/aspect/semeval14/judge_data/test_correctness.txt'
    word_vecs = utils.load_word_vec_file(config.WORD_VEC_FILE)
    __build_cnn_data(sent_tok_texts_file, word_vecs, correctness_file)


# __train()
__train_cnn()
