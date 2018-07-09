import numpy as np
import pickle
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


def __gen_word_vec_file_for_data(sent_tok_texts_file, word_vecs, dst_file):
    sent_texts = utils.read_lines(sent_tok_texts_file)
    data_vocab = set()
    for text in sent_texts:
        words = text.split(' ')
        for w in words:
            if w in word_vecs:
                data_vocab.add(w)

    vec_dim = next(iter(word_vecs.values())).shape[0]
    data_vocab = list(data_vocab)
    n_words = len(data_vocab)
    print(n_words, vec_dim)
    word_vec_matrix = np.zeros((n_words + 1, vec_dim), np.float32)
    word_vec_matrix[0] = np.random.uniform(-1, 1, vec_dim)
    for i in range(1, n_words + 1):
        word = data_vocab[i - 1]
        word_vec_matrix[i] = word_vecs[word]
    with open(dst_file, 'wb') as fout:
        pickle.dump((data_vocab, word_vec_matrix), fout, protocol=pickle.HIGHEST_PROTOCOL)


def __build_cnn_data(sent_tok_texts_file, vocab, correctness_file):
    sent_texts = utils.read_lines(sent_tok_texts_file)
    max_sent_len = 0
    sent_words_list = list()
    for text in sent_texts:
        words = text.split(' ')
        sent_words_list.append(words)
        if len(words) > max_sent_len:
            max_sent_len = len(words)
    print('max sent len: {}'.format(max_sent_len))

    # Note: first row in word_vec_matrix is not for a word
    word_idx_dict = {w: i + 1 for i, w in enumerate(vocab)}
    n_sents = len(sent_texts)
    X = np.zeros((n_sents, max_sent_len), np.int32)
    for i, sent_words in enumerate(sent_words_list):
        for j, w in enumerate(sent_words):
            X[i][j] = word_idx_dict.get(w, 0)

    y_true_str = utils.read_lines(correctness_file)
    y_true = np.zeros((len(y_true_str), 2), np.int32)
    for i, yi in enumerate(y_true_str):
        yi = int(yi)
        if yi == 0:
            y_true[i][0] = 1
        else:
            y_true[i][1] = 1
    # y_true = np.asarray([int(yi) for yi in y_true], np.int32)
    return X, y_true


def __train_cnn():
    from models.textcnn import TextCNN

    sent_tok_texts_file = 'd:/data/aspect/semeval14/judge_data/laptops_jtest_texts_tok.txt'
    correctness_file = 'd:/data/aspect/semeval14/judge_data/test_correctness.txt'
    word_vec_data_file = 'd:/data/aspect/semeval14/judge_data/word_vecs.pkl'

    # word_vecs = utils.load_word_vec_file(config.WORD_VEC_FILE)
    # __gen_word_vec_file_for_data(sent_tok_texts_file, word_vecs, word_vec_data_file)

    with open(word_vec_data_file, 'rb') as f:
        vocab, word_vec_matrix = pickle.load(f)
    print('{} words, word vec dim: {}'.format(len(vocab), word_vec_matrix.shape[1]))
    X, y_true = __build_cnn_data(sent_tok_texts_file, vocab, correctness_file)
    n_train = 600
    X_train = X[:n_train]
    y_true_train = y_true[:n_train]
    X_test = X[n_train:]
    y_true_test = y_true[n_train:]

    batch_size = 5
    n_batch = len(X_train) // batch_size
    n_epoch = 100
    filter_sizes = [2, 3, 4, 5]
    n_filters = 20
    l2_reg_lamb = 0.1
    cnnmodel = TextCNN(
        X.shape[1], 2, len(vocab), word_vec_matrix.shape[1], filter_sizes, n_filters, l2_reg_lamb, word_vec_matrix)

    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = optimizer.minimize(cnnmodel.loss)
    # grads_and_vars = optimizer.compute_gradients(cnnmodel.loss)
    # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
        losses = list()
        for batch_idx in range(n_batch):
            X_batch = X_train[batch_size * batch_idx: batch_size * (batch_idx + 1)]
            y_true_batch = y_true_train[batch_size * batch_idx: batch_size * (batch_idx + 1)]
            feed_dict = {
                cnnmodel.input_x: X_batch,
                cnnmodel.input_y: y_true_batch,
                cnnmodel.dropout_keep_prob: 0.5
            }
            _, loss = sess.run([train_op, cnnmodel.loss], feed_dict)
            losses.append(loss)
        y_pred = sess.run(cnnmodel.predictions, feed_dict={
            cnnmodel.input_x: X_test,
            cnnmodel.input_y: y_true_test,
            cnnmodel.dropout_keep_prob: 1.0
        })
        y_true_tmp = np.argmax(y_true_test, axis=1)
        n_neg, neg_hit_cnt = 0, 0
        for yi_true, yi_pred in zip(y_true_tmp, y_pred):
            if yi_true == 0:
                n_neg += 1
                if yi_pred == 0:
                    neg_hit_cnt += 1
        print(neg_hit_cnt / n_neg, np.sum(np.equal(y_pred, y_true_tmp)) / len(y_pred))
        # print(sum(losses))
        # print(y_pred)
        # print(y_true_tmp)
        # print(len(y_pred), len(y_true_tmp))


# __train()
__train_cnn()
