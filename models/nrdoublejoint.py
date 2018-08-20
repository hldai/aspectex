import tensorflow as tf
from collections import namedtuple
import logging

import numpy as np
from utils import utils
from utils.datautils import TrainData, ValidData


class NeuRuleDoubleJoint:
    # TrainData = namedtuple(
    #     "TrainData", ['word_idxs_list_train', 'labels_list_train', 'word_idxs_list_valid', 'labels_list_valid',
    #                   'valid_texts', 'aspects_true_list', 'opinions_true_list'])
    # TestData = namedtuple('TestData', ['word_idxs_list', 'texts', 'aspects_true_list', 'opinions_true_list'])

    def __init__(self, n_tags, word_embeddings, share_lstm, hidden_size_lstm=100,
                 batch_size=20, lr_method='adam', clip=-1, use_crf=True, model_file=None):
        logging.info('hidden_size_lstm={}, batch_size={}, lr_method={}'.format(
            hidden_size_lstm, batch_size, lr_method))

        self.n_tags_src = 3
        self.n_tags = n_tags
        self.hidden_size_lstm = hidden_size_lstm
        self.batch_size = batch_size
        self.lr_method = lr_method
        self.clip = clip
        self.saver = None
        self.share_lstm = share_lstm

        # self.n_words, self.dim_word = word_embeddings.shape
        self.use_crf = use_crf

        self.word_idxs = tf.placeholder(tf.int32, shape=[None, None], name='word_idxs')
        # self.rule_hidden_input = tf.placeholder(tf.float32, shape=[None, None, hidden_size_lstm * 2],
        #                                         name='rule_hidden_input')
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
        self.labels_src1 = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        self.labels_src2 = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        self.labels_tar = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

        self.__add_word_embedding_op(word_embeddings)
        # self.__setup_rule_lstm_hidden()
        self.__add_logits_op()
        self.__add_pred_op()
        self.__add_loss_op()
        self.__add_train_op(self.lr_method, self.lr, self.clip)
        self.__init_session(model_file)

    def __add_word_embedding_op(self, word_embeddings_val):
        with tf.variable_scope("words"):
            _word_embeddings = tf.constant(
                    word_embeddings_val,
                    name="_word_embeddings",
                    dtype=tf.float32)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_idxs, name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    def __add_logits_op(self):
        if self.share_lstm:
            with tf.variable_scope("bi-lstm"):
                cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
                cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, self.word_embeddings,
                        sequence_length=self.sequence_lengths, dtype=tf.float32)
                self.lstm_output = tf.concat([output_fw, output_bw], axis=-1)
                self.lstm_output = tf.nn.dropout(self.lstm_output, self.dropout)
                lstm_output1 = lstm_output2 = self.lstm_output
        else:
            with tf.variable_scope("bi-lstm-1"):
                cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
                cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, self.word_embeddings,
                        sequence_length=self.sequence_lengths, dtype=tf.float32)
                self.lstm_output1 = tf.concat([output_fw, output_bw], axis=-1)
                self.lstm_output1 = tf.nn.dropout(self.lstm_output1, self.dropout)
                lstm_output1 = self.lstm_output1

            with tf.variable_scope("bi-lstm-2"):
                cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
                cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, self.word_embeddings,
                        sequence_length=self.sequence_lengths, dtype=tf.float32)
                self.lstm_output2 = tf.concat([output_fw, output_bw], axis=-1)
                self.lstm_output2 = tf.nn.dropout(self.lstm_output2, self.dropout)
                lstm_output2 = self.lstm_output2

        with tf.variable_scope("proj-src1"):
            self.W_src1 = tf.get_variable("W", dtype=tf.float32, shape=[
                2 * self.hidden_size_lstm, self.n_tags_src])
            self.b_src1 = tf.get_variable(
                "b", shape=[self.n_tags_src], dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(lstm_output1)[1]
            output = tf.reshape(lstm_output1, [-1, 2 * self.hidden_size_lstm])
            pred = tf.matmul(output, self.W_src1) + self.b_src1
            self.logits_src1 = tf.reshape(pred, [-1, nsteps, self.n_tags_src])

        with tf.variable_scope("proj-src2"):
            self.W_src2 = tf.get_variable("W", dtype=tf.float32, shape=[
                2 * self.hidden_size_lstm, 3])
            self.b_src2 = tf.get_variable(
                "b", shape=[self.n_tags_src], dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(lstm_output2)[1]
            output = tf.reshape(lstm_output2, [-1, 2 * self.hidden_size_lstm])
            pred = tf.matmul(output, self.W_src2) + self.b_src2
            self.logits_src2 = tf.reshape(pred, [-1, nsteps, self.n_tags_src])

        with tf.variable_scope("proj-target"):
            input_size = 2 * self.hidden_size_lstm if self.share_lstm else 4 * self.hidden_size_lstm

            self.W_tar = tf.get_variable("W", dtype=tf.float32, shape=[input_size, self.n_tags])
            self.b_tar = tf.get_variable(
                "b", shape=[self.n_tags], dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(lstm_output1)[1]

            if self.share_lstm:
                output = lstm_output1
            else:
                output = tf.concat([lstm_output1, lstm_output2], axis=-1)
            output = tf.reshape(output, [-1, input_size])
            # pred = tf.matmul(output, self.W_tar) + self.b_tar
            pred = tf.matmul(output, self.W_tar) + self.b_tar
            self.logits_tar = tf.reshape(pred, [-1, nsteps, self.n_tags])

    def __add_pred_op(self):
        if not self.use_crf:
            self.labels_pred_src1 = tf.cast(tf.argmax(self.logits_src1, axis=-1), tf.int32)
            self.labels_pred_src2 = tf.cast(tf.argmax(self.logits_src2, axis=-1), tf.int32)
            self.labels_pred_tar = tf.cast(tf.argmax(self.logits_tar, axis=-1), tf.int32)

    def __add_loss_op(self):
        """Defines the loss"""
        if self.use_crf:
            with tf.variable_scope("crf-src1"):
                log_likelihood, self.trans_params_src1 = tf.contrib.crf.crf_log_likelihood(
                        self.logits_src1, self.labels_src1, self.sequence_lengths)
                self.loss_src1 = tf.reduce_mean(-log_likelihood)

            with tf.variable_scope("crf-src2"):
                log_likelihood, self.trans_params_src2 = tf.contrib.crf.crf_log_likelihood(
                        self.logits_src2, self.labels_src2, self.sequence_lengths)
                self.loss_src2 = tf.reduce_mean(-log_likelihood)

            with tf.variable_scope("crf-tar"):
                log_likelihood, self.trans_params_tar = tf.contrib.crf.crf_log_likelihood(
                        self.logits_tar, self.labels_tar, self.sequence_lengths)
                # self.loss_tar = tf.reduce_mean(-log_likelihood)
                self.loss_tar = tf.reduce_mean(-log_likelihood) + 0.001 * tf.nn.l2_loss(self.W_tar)
        else:
            assert False
            # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            #         logits=self.logits, labels=self.labels)
            # mask = tf.sequence_mask(self.sequence_lengths)
            # losses = tf.boolean_mask(losses, mask)
            # self.loss = tf.reduce_mean(losses)
            # self.loss = tf.reduce_mean(losses) + 0.01 * tf.nn.l2_loss(self.W)

        # for tensorboard
        # tf.summary.scalar("loss", self.loss)

    def __add_train_op(self, lr_method, lr, clip=-1):
        """Defines self.train_op that performs an update on a batch

        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: (tf.placeholder) tf.float32, learning rate
            loss: (tensor) tf.float32 loss to minimize
            clip: (python float) clipping of gradient. If < 0, no clipping

        """
        _lr_m = lr_method.lower()  # lower to make sure

        with tf.variable_scope("train_step"):
            if _lr_m == 'adam':  # sgd method
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if clip > 0:  # gradient clipping if clip is positive
                grads, vs = zip(*optimizer.compute_gradients(self.loss_src1))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op_src1 = optimizer.apply_gradients(zip(grads, vs))

                grads, vs = zip(*optimizer.compute_gradients(self.loss_src2))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op_src2 = optimizer.apply_gradients(zip(grads, vs))

                grads, vs = zip(*optimizer.compute_gradients(self.loss_tar))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op_tar = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op_src1 = optimizer.minimize(self.loss_src1)
                self.train_op_src2 = optimizer.minimize(self.loss_src2)
                self.train_op_tar = optimizer.minimize(self.loss_tar)

    def __init_session(self, model_file):
        """Defines self.sess and initialize the variables"""
        # self.logger.info("Initializing tf session")
        self.sess = tf.Session()
        if model_file is None:
            # self.saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())
            # self.saver.restore(self.sess, rule_model_file)
        else:
            tf.train.Saver().restore(self.sess, model_file)

    def get_feed_dict(self, word_idx_seqs, task, label_seqs=None, lr=None, dropout=None):
        word_idx_seqs = [list(word_idxs) for word_idxs in word_idx_seqs]
        word_ids, sequence_lengths = utils.pad_sequences(word_idx_seqs, 0)

        # build feed dictionary
        feed = {
            self.word_idxs: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if label_seqs is not None:
            label_seqs = [list(labels) for labels in label_seqs]
            labels, _ = utils.pad_sequences(label_seqs, 0)
            if task == 'src1':
                feed[self.labels_src1] = labels
            elif task == 'src2':
                feed[self.labels_src2] = labels
            else:
                feed[self.labels_tar] = labels

        feed[self.lr] = lr
        feed[self.dropout] = dropout

        return feed, sequence_lengths

    def __train_batch(self, word_idxs_list_train, labels_list_train, batch_idx, lr, dropout, task):
        word_idxs_list_batch = word_idxs_list_train[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
        labels_list_batch = labels_list_train[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
        feed_dict, _ = self.get_feed_dict(word_idxs_list_batch, task,
                                          label_seqs=labels_list_batch, lr=lr, dropout=dropout)

        if task == 'src1':
            _, train_loss = self.sess.run(
                [self.train_op_src1, self.loss_src1], feed_dict=feed_dict)
        elif task == 'src2':
            _, train_loss = self.sess.run(
                [self.train_op_src2, self.loss_src2], feed_dict=feed_dict)
        else:
            _, train_loss = self.sess.run(
                [self.train_op_tar, self.loss_tar], feed_dict=feed_dict)
        return train_loss

    def pre_train(self, data_train_s1: TrainData, data_valid_s1: ValidData, data_train_s2: TrainData,
                  data_valid_s2: ValidData, vocab, n_epochs,
                  lr=0.001, dropout=0.5, save_file=None):
        logging.info('pretrain, n_epochs={}, lr={}, dropout={}'.format(n_epochs, lr, dropout))
        if save_file is not None and self.saver is None:
            self.saver = tf.train.Saver()

        n_train_src1 = len(data_train_s1.word_idxs_list)
        n_batches_src1 = (n_train_src1 + self.batch_size - 1) // self.batch_size

        n_train_src2 = len(data_train_s2.word_idxs_list)
        n_batches_src2 = (n_train_src2 + self.batch_size - 1) // self.batch_size

        best_f1 = 0
        best_f11, best_f12 = 0, 0
        batch_idx_src1, batch_idx_src2 = 0, 0
        for epoch in range(n_epochs):
            # losses_src, losses_seg_src = list(), list()
            for i in range(n_batches_src1):
                train_loss_src1 = self.__train_batch(
                    data_train_s1.word_idxs_list, data_train_s1.labels_list, batch_idx_src1, lr, dropout, 'src1')
                batch_idx_src1 = batch_idx_src1 + 1 if batch_idx_src1 + 1 < n_batches_src1 else 0
                train_loss_src2 = self.__train_batch(
                    data_train_s2.word_idxs_list, data_train_s2.labels_list, batch_idx_src2, lr, dropout, 'src2')
                batch_idx_src2 = batch_idx_src2 + 1 if batch_idx_src2 + 1 < n_batches_src2 else 0

                if (i + 1) % 100 == 0:
                    p1, r1, f11, _, _, _ = self.evaluate(
                        data_valid_s1.word_idxs_list, data_valid_s1.tok_texts,
                        data_valid_s1.aspects_true_list, 'src1')

                    p2, r2, f12, _, _, _ = self.evaluate(
                        data_valid_s2.word_idxs_list, data_valid_s2.tok_texts,
                        data_valid_s2.opinions_true_list, 'src2')

                    logging.info('src1, p={:.4f}, r={:.4f}, f1={:.4f}; src2, p={:.4f}, r={:.4f}, f1={:.4f}'.format(
                        p1, r1, f11, p2, r2, f12
                    ))

                    if f11 + f12 > best_f1:
                        best_f1 = f11 + f12
                    # if f11 >= best_f11 and f12 >= best_f12:
                    #     best_f11 = f11
                    #     best_f12 = f12
                        if self.saver is not None:
                            self.saver.save(self.sess, save_file)
                            # print('model saved to {}'.format(save_file))
                            logging.info('model saved to {}'.format(save_file))

    def join_train(self, data_train_s1: TrainData, data_valid_s1: ValidData, data_train_s2: TrainData,
                   data_valid_s2: ValidData, data_train_t: TrainData, data_valid_t: ValidData,
                   data_test_t: ValidData, n_epochs=10, lr=0.001, dropout=0.5, save_file=None):
        logging.info('n_epochs={}, lr={}, dropout={}'.format(n_epochs, lr, dropout))
        if save_file is not None and self.saver is None:
            self.saver = tf.train.Saver()

        n_train_src1, n_batches_src1, n_train_src2, n_batches_src2 = 0, 0, 0, 0
        n_train_src1 = len(data_train_s1.word_idxs_list)
        n_batches_src1 = (n_train_src1 + self.batch_size - 1) // self.batch_size

        n_train_src2 = len(data_train_s2.word_idxs_list)
        n_batches_src2 = (n_train_src2 + self.batch_size - 1) // self.batch_size

        n_train_tar = len(data_train_t.word_idxs_list)
        n_batches_tar = (n_train_tar + self.batch_size - 1) // self.batch_size

        best_f1_a, best_f1_o, best_f1 = 0, 0, 0
        batch_idx_src1, batch_idx_src2 = 0, 0
        for epoch in range(n_epochs):
            # losses_src, losses_seg_src = list(), list()
            losses_tar, losses_seg_tar = list(), list()
            for i in range(n_batches_tar):
                train_loss_tar = self.__train_batch(
                    data_train_t.word_idxs_list_train, data_train_t.labels_list_train, i, lr, dropout, 'tar')
                losses_tar.append(train_loss_tar)

                train_loss_src1 = self.__train_batch(
                    data_train_s1.word_idxs_list, data_train_s1.labels_list, batch_idx_src1,
                    lr, dropout, 'src1'
                )
                batch_idx_src1 = batch_idx_src1 + 1 if batch_idx_src1 + 1 < n_batches_src1 else 0.
                train_loss_src2 = self.__train_batch(
                    data_train_s2.word_idxs_list, data_train_s2.labels_list, batch_idx_src2,
                    lr, dropout, 'src2'
                )
                batch_idx_src2 = batch_idx_src2 + 1 if batch_idx_src2 + 1 < n_batches_src2 else 0
            loss_tar = sum(losses_tar)

            # metrics = self.run_evaluate(dev)
            aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r, opinion_f1 = self.evaluate(
                data_valid_t.word_idxs_list_valid, data_valid_t.valid_texts, data_valid_t.aspects_true_list,
                'tar', data_valid_t.opinions_true_list)
            # print('iter {}, loss={:.4f}, p={:.4f}, r={:.4f}, f1={:.4f}, best_f1={:.4f}'.format(
            #     epoch, loss_tar, p, r, f1, best_f1))
            logging.info('iter {}, loss={:.4f}, p={:.4f}, r={:.4f}, f1={:.4f}, best_f1={:.4f},'
                         ' p={:.4f}, r={:.4f}, f1={:.4f}, best_f1={:.4f}'.format(
                epoch, loss_tar, aspect_p, aspect_r, aspect_f1, best_f1_a, opinion_p, opinion_r,
                opinion_f1, best_f1_o))

            if aspect_f1 > best_f1_a:
                best_f1_a = aspect_f1
            if opinion_f1 > best_f1_o:
                best_f1_o = opinion_f1
            if aspect_f1 + opinion_f1 > best_f1:
                best_f1 = aspect_f1 + opinion_f1
                if self.saver is not None:
                    self.saver.save(self.sess, save_file)
                    # print('model saved to {}'.format(save_file))
                    logging.info('model saved to {}'.format(save_file))

            p1, r1, f11, _, _, _ = self.evaluate(
                data_valid_s1.word_idxs_list, data_valid_s1.labels_list,
                data_valid_s1.tok_texts, data_valid_s1.aspects_true_list, 'src1')

            p2, r2, f12, _, _, _ = self.evaluate(
                data_valid_s2.word_idxs_list, data_valid_s2.labels_list,
                data_valid_s2.tok_texts, data_valid_s2.aspects_true_list, 'src2')
            logging.info('src1, p={:.4f}, r={:.4f}, f1={:.4f}; src2, p={:.4f}, r={:.4f}, f1={:.4f}'.format(
                p1, r1, f11, p2, r2, f12
            ))

    def train(self, data_train: TrainData, data_valid: ValidData, data_test: ValidData, vocab,
              n_epochs=10, lr=0.001, dropout=0.5, save_file=None):
        logging.info('n_epochs={}, lr={}, dropout={}'.format(n_epochs, lr, dropout))
        if save_file is not None and self.saver is None:
            self.saver = tf.train.Saver()

        n_train = len(data_train.word_idxs_list)
        n_batches = (n_train + self.batch_size - 1) // self.batch_size

        best_f1_sum = 0
        for epoch in range(n_epochs):
            losses, losses_seg = list(), list()
            for i in range(n_batches):
                train_loss_tar = self.__train_batch(
                    data_train.word_idxs_list, data_train.labels_list, i, lr, dropout, 'tar')
                losses.append(train_loss_tar)

            loss = sum(losses)
            # metrics = self.run_evaluate(dev)
            aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r, opinion_f1 = self.evaluate(
                data_valid.word_idxs_list, data_valid.tok_texts, data_valid.aspects_true_list,
                'tar', data_valid.opinions_true_list)

            logging.info('iter {}, loss={:.4f}, p={:.4f}, r={:.4f}, f1={:.4f};'
                         ' p={:.4f}, r={:.4f}, f1={:.4f}; best_f1_sum={:.4f}'.format(
                epoch, loss, aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r,
                opinion_f1, best_f1_sum))

            # if aspect_f1 + opinion_f1 > best_f1_sum:
            if True:
                best_f1_sum = aspect_f1 + opinion_f1

                aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r, opinion_f1 = self.evaluate(
                    data_test.word_idxs_list, data_test.tok_texts, data_test.aspects_true_list,
                    'tar', data_test.opinions_true_list)
                # print('iter {}, loss={:.4f}, p={:.4f}, r={:.4f}, f1={:.4f}, best_f1={:.4f}'.format(
                #     epoch, loss_tar, p, r, f1, best_f1))
                logging.info('Test, p={:.4f}, r={:.4f}, f1={:.4f}; p={:.4f}, r={:.4f}, f1={:.4f}'.format(
                    aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r,
                    opinion_f1))

                if self.saver is not None:
                    self.saver.save(self.sess, save_file)
                    # print('model saved to {}'.format(save_file))
                    logging.info('model saved to {}'.format(save_file))

    def predict_batch(self, word_idxs, task):
        fd, sequence_lengths = self.get_feed_dict(word_idxs, task, dropout=1.0)

        if task == 'src1':
            logits = self.logits_src1
            trans_params = self.trans_params_src1
        elif task == 'src2':
            logits = self.logits_src2
            trans_params = self.trans_params_src2
        else:
            logits = self.logits_tar
            trans_params = self.trans_params_tar

        # get tag scores and transition params of CRF
        viterbi_sequences = []
        logits_val, trans_params_val = self.sess.run(
                [logits, trans_params], feed_dict=fd)

        # iterate over the sentences because no batching in vitervi_decode
        for logit, sequence_length in zip(logits_val, sequence_lengths):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params_val)
            viterbi_sequences += [viterbi_seq]

        return viterbi_sequences, sequence_lengths

    def get_terms_from_label_list(self, labels, tok_text, label_beg, label_in):
        terms = list()
        words = tok_text.split(' ')
        # print(labels_pred)
        # print(len(words), len(labels_pred))
        assert len(words) == len(labels)

        p = 0
        while p < len(words):
            yi = labels[p]
            if yi == label_beg:
                pright = p
                while pright + 1 < len(words) and labels[pright + 1] == label_in:
                    pright += 1
                terms.append(' '.join(words[p: pright + 1]))
                p = pright + 1
            else:
                p += 1
        return terms

    def evaluate(self, word_idxs_list_valid, test_texts, terms_true_list, task,
                 opinions_ture_list=None):
        aspect_true_cnt, aspect_sys_cnt, aspect_hit_cnt = 0, 0, 0
        opinion_true_cnt, opinion_sys_cnt, opinion_hit_cnt = 0, 0, 0
        error_sents, error_terms = list(), list()
        correct_sent_idxs = list()
        for sent_idx, (word_idxs, text, terms_true) in enumerate(zip(
                word_idxs_list_valid, test_texts, terms_true_list)):
            labels_pred, sequence_lengths = self.predict_batch([word_idxs], task)
            labels_pred = labels_pred[0]

            aspect_terms_sys = self.get_terms_from_label_list(labels_pred, text, 1, 2)

            new_hit_cnt = utils.count_hit(terms_true, aspect_terms_sys)
            aspect_true_cnt += len(terms_true)
            aspect_sys_cnt += len(aspect_terms_sys)
            aspect_hit_cnt += new_hit_cnt
            if new_hit_cnt == aspect_true_cnt:
                correct_sent_idxs.append(sent_idx)

            if opinions_ture_list is None:
                continue

            opinion_terms_sys = self.get_terms_from_label_list(labels_pred, text, 3, 4)
            opinion_terms_true = opinions_ture_list[sent_idx]

            new_hit_cnt = utils.count_hit(opinion_terms_true, opinion_terms_sys)
            opinion_hit_cnt += new_hit_cnt
            opinion_true_cnt += len(opinion_terms_true)
            opinion_sys_cnt += len(opinion_terms_sys)

        # save_json_objs(error_sents, 'd:/data/aspect/semeval14/error-sents.txt')
        # with open('d:/data/aspect/semeval14/error-sents.txt', 'w', encoding='utf-8') as fout:
        #     for sent, terms in zip(error_sents, error_terms):
                # terms_true = [t['term'].lower() for t in sent['terms']] if 'terms' in sent else list()
                # fout.write('{}\n{}\n\n'.format(sent['text'], terms))
        # with open('d:/data/aspect/semeval14/lstmcrf-correct.txt', 'w', encoding='utf-8') as fout:
        #     fout.write('\n'.join([str(i) for i in correct_sent_idxs]))

        aspect_p, aspect_r, aspect_f1 = utils.prf1(aspect_true_cnt, aspect_sys_cnt, aspect_hit_cnt)
        if opinions_ture_list is None:
            return aspect_p, aspect_r, aspect_f1, 0, 0, 0

        opinion_p, opinion_r, opinion_f1 = utils.prf1(opinion_true_cnt, opinion_sys_cnt, opinion_hit_cnt)
        return aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r, opinion_f1
