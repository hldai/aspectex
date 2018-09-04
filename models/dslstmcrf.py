import tensorflow as tf
import logging
from utils import utils
from utils.datautils import TrainData, ValidData
from utils.modelutils import evaluate_ao_extraction


class DSLSTMCRF:
    def __init__(self, word_embeddings, n_tags_src=3, n_tags_tar=5, hidden_size_lstm=300, batch_size=20,
                 train_word_embeddings=False, lr_method='adam', model_file=None):
        self.n_tags_src = n_tags_src
        self.n_tags_tar = n_tags_tar
        self.vals_word_embeddings = word_embeddings
        self.hidden_size_lstm = hidden_size_lstm
        self.batch_size = batch_size
        self.lr_method = lr_method
        self.saver = None

        self.n_words, self.dim_word = word_embeddings.shape

        self.word_idxs = tf.placeholder(tf.int32, shape=[None, None], name='word_idxs')
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        self.manual_feat = None

        self.__add_word_embedding_op(train_word_embeddings)
        self.__add_logits_op()
        self.__add_loss_op()
        self.__add_train_op(self.lr_method, self.lr)
        self.__init_session(model_file)

    def __add_word_embedding_op(self, train_word_embeddings):
        with tf.variable_scope("words"):
            if self.vals_word_embeddings is None:
                # self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.n_words, self.dim_word])
            else:
                # _word_embeddings = tf.Variable(
                #         self.vals_word_embeddings,
                #         name="_word_embeddings",
                #         dtype=tf.float32,
                #         trainable=train_word_embeddings)
                _word_embeddings = tf.constant(
                        self.vals_word_embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_idxs, name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    def __add_logits_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
            # shape of self.output_fw: (batch_size, sequence_len, self.hidden_size_lstm)
            (self.output_fw, self.output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            self.lstm_output = tf.concat([self.output_fw, self.output_bw], axis=-1)
            # if self.manual_feat is not None:
            #     self.lstm_output = tf.concat([self.lstm_output, self.manual_feat], axis=-1)
            self.lstm_output = tf.nn.dropout(self.lstm_output, self.dropout)

        with tf.variable_scope("proj-src-a"):
            dim_tmp = 2 * self.hidden_size_lstm
            self.W_src_a = tf.get_variable("W", dtype=tf.float32, shape=[dim_tmp, self.n_tags_src])
            self.b_src_a = tf.get_variable(
                "b", shape=[self.n_tags_src], dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(self.lstm_output)[1]
            output = tf.reshape(self.lstm_output, [-1, dim_tmp])
            pred = tf.matmul(output, self.W_src_a) + self.b_src_a
            self.logits_src_a = tf.reshape(pred, [-1, nsteps, self.n_tags_src])

        with tf.variable_scope('proj-src-o'):
            dim_tmp = 2 * self.hidden_size_lstm
            self.W_src_o = tf.get_variable("W", dtype=tf.float32, shape=[dim_tmp, self.n_tags_src])
            self.b_src_o = tf.get_variable(
                "b", shape=[self.n_tags_src], dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(self.lstm_output)[1]
            output = tf.reshape(self.lstm_output, [-1, dim_tmp])
            pred = tf.matmul(output, self.W_src_o) + self.b_src_o
            self.logits_src_o = tf.reshape(pred, [-1, nsteps, self.n_tags_src])

        with tf.variable_scope('proj-tar'):
            dim_tmp = 2 * self.hidden_size_lstm
            self.W_tar = tf.get_variable("W", dtype=tf.float32, shape=[dim_tmp, self.n_tags_tar])
            self.b_tar = tf.get_variable(
                "b", shape=[self.n_tags_tar], dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(self.lstm_output)[1]
            output = tf.reshape(self.lstm_output, [-1, dim_tmp])
            pred = tf.matmul(output, self.W_tar) + self.b_tar
            self.logits_tar = tf.reshape(pred, [-1, nsteps, self.n_tags_tar])

    def __add_loss_op(self):
        with tf.variable_scope("crf-src-a"):
            log_likelihood, self.trans_params_src_a = tf.contrib.crf.crf_log_likelihood(
                    self.logits_src_a, self.labels, self.sequence_lengths)
            # self.loss = tf.reduce_mean(-log_likelihood)
            self.loss_src_a = tf.reduce_mean(-log_likelihood) + 0.001 * tf.nn.l2_loss(self.W_src_a)
        with tf.variable_scope("crf-src-o"):
            log_likelihood, self.trans_params_src_o = tf.contrib.crf.crf_log_likelihood(
                    self.logits_src_o, self.labels, self.sequence_lengths)
            # self.loss = tf.reduce_mean(-log_likelihood)
            self.loss_src_o = tf.reduce_mean(-log_likelihood) + 0.001 * tf.nn.l2_loss(self.W_src_o)
        with tf.variable_scope("crf-tar"):
            log_likelihood, self.trans_params_tar = tf.contrib.crf.crf_log_likelihood(
                    self.logits_tar, self.labels, self.sequence_lengths)
            # self.loss = tf.reduce_mean(-log_likelihood)
            self.loss_tar = tf.reduce_mean(-log_likelihood) + 0.001 * tf.nn.l2_loss(self.W_tar)

    def __add_train_op(self, lr_method, lr):
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

            self.train_op_src_a = optimizer.minimize(self.loss_src_a)
            self.train_op_src_o = optimizer.minimize(self.loss_src_o)
            self.train_op_tar = optimizer.minimize(self.loss_tar)

    def __init_session(self, model_file):
        """Defines self.sess and initialize the variables"""
        # self.logger.info("Initializing tf session")
        self.sess = tf.Session()
        if model_file is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            tf.train.Saver().restore(self.sess, model_file)

    def get_W_b(self):
        return self.sess.run([self.W_tar, self.b_tar])

    def get_feed_dict(self, word_idx_seqs, label_seqs=None, lr=None, dropout=None, manual_feat=None):
        word_idx_seqs = [list(word_idxs) for word_idxs in word_idx_seqs]
        word_ids, sequence_lengths = utils.pad_sequences(word_idx_seqs, 0)

        # print(len(word_ids))
        # build feed dictionary
        feed = {
            self.word_idxs: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if label_seqs is not None:
            label_seqs = [list(labels) for labels in label_seqs]
            labels, _ = utils.pad_sequences(label_seqs, 0)
            feed[self.labels] = labels

        if self.manual_feat is not None:
            manual_feat, lens = utils.pad_feat_sequence(manual_feat, manual_feat[0].shape[1])
            feed[self.manual_feat] = manual_feat

        feed[self.lr] = lr
        feed[self.dropout] = dropout

        return feed, sequence_lengths

    def calc_hidden_vecs(self, word_idxs_list, batch_size):
        n_samples = len(word_idxs_list)
        n_batches = (n_samples + batch_size - 1) // batch_size

        hidden_vecs_batch_list = list()
        for i in range(n_batches):
            word_idxs_list_batch = word_idxs_list[i * batch_size: (i + 1) * batch_size]
            fd, sequence_lengths = self.get_feed_dict(word_idxs_list_batch, dropout=1.0)
            hidden_vecs_batch = self.sess.run(self.lstm_output, feed_dict=fd)
            hidden_vecs_batch_list.append(hidden_vecs_batch)
        return hidden_vecs_batch_list

    def __train_batch(self, word_idxs_list_train, labels_list_train, batch_idx, lr, dropout, task):
        word_idxs_list_batch = word_idxs_list_train[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
        labels_list_batch = labels_list_train[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
        feed_dict, _ = self.get_feed_dict(word_idxs_list_batch,
                                          label_seqs=labels_list_batch, lr=lr, dropout=dropout)

        if task == 'src-a':
            _, train_loss = self.sess.run(
                [self.train_op_src_a, self.loss_src_a], feed_dict=feed_dict)
        elif task == 'src-o':
            _, train_loss = self.sess.run(
                [self.train_op_src_o, self.loss_src_o], feed_dict=feed_dict)
        else:
            _, train_loss = self.sess.run(
                [self.train_op_tar, self.loss_tar], feed_dict=feed_dict)
        return train_loss

    def pre_train(self, data_train_sa: TrainData, data_valid_sa: ValidData, data_train_so: TrainData,
                  data_valid_so: ValidData, vocab, n_epochs,
                  lr=0.001, dropout=0.5, save_file=None):
        logging.info('pretrain, n_epochs={}, lr={}, dropout={}'.format(n_epochs, lr, dropout))
        if save_file is not None and self.saver is None:
            self.saver = tf.train.Saver()

        n_train_sa = len(data_train_sa.word_idxs_list)
        n_batches_sa = (n_train_sa + self.batch_size - 1) // self.batch_size

        n_train_so = len(data_train_sa.word_idxs_list)
        n_batches_so = (n_train_so + self.batch_size - 1) // self.batch_size

        best_f1 = 0
        best_f11, best_f12 = 0, 0
        batch_idx_sa, batch_idx_so = 0, 0
        for epoch in range(n_epochs):
            # losses_src, losses_seg_src = list(), list()
            for i in range(n_batches_sa):
                train_loss_sa = self.__train_batch(
                    data_train_sa.word_idxs_list, data_train_sa.labels_list, batch_idx_sa, lr, dropout, 'src-a')
                batch_idx_sa = batch_idx_sa + 1 if batch_idx_sa + 1 < n_batches_sa else 0
                train_loss_so = self.__train_batch(
                    data_train_so.word_idxs_list, data_train_so.labels_list, batch_idx_so, lr, dropout, 'src-o')
                batch_idx_so = batch_idx_so + 1 if batch_idx_so + 1 < n_batches_so else 0

                if (i + 1) % 100 == 0:
                    p1, r1, f11, _, _, _ = self.evaluate(
                        data_valid_sa.word_idxs_list, data_valid_sa.tok_texts,
                        data_valid_sa.aspects_true_list, 'src-a', is_train=False)

                    p2, r2, f12, _, _, _ = self.evaluate(
                        data_valid_so.word_idxs_list, data_valid_so.tok_texts,
                        data_valid_so.opinions_true_list, 'src-o', is_train=False)

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

    def train(self, data_train: TrainData, data_valid: ValidData, data_test: ValidData, vocab,
              n_epochs=10, lr=0.001, dropout=0.5, save_file=None):
        logging.info('n_epochs={}, lr={}, dropout={}'.format(n_epochs, lr, dropout))
        if save_file is not None and self.saver is None:
            self.saver = tf.train.Saver()

        n_train = len(data_train.word_idxs_list)
        n_batches = (n_train + self.batch_size - 1) // self.batch_size

        best_f1_sum = 0
        best_a_f1, best_o_f1 = 0, 0
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

            # if True:
            # if aspect_f1 + opinion_f1 > best_f1_sum:
            if aspect_f1 > best_a_f1 and opinion_f1 > best_o_f1:
                best_a_f1 = aspect_f1
                best_o_f1 = opinion_f1
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

    def joint_train(self, data_train_s1: TrainData, data_valid_s1: ValidData, data_train_s2: TrainData,
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
                    data_train_t.word_idxs_list, data_train_t.labels_list, i, lr, dropout, 'tar')
                losses_tar.append(train_loss_tar)

                train_loss_src1 = self.__train_batch(
                    data_train_s1.word_idxs_list, data_train_s1.labels_list, batch_idx_src1,
                    lr, dropout, 'src-a'
                )
                batch_idx_src1 = batch_idx_src1 + 1 if batch_idx_src1 + 1 < n_batches_src1 else 0
                train_loss_src2 = self.__train_batch(
                    data_train_s2.word_idxs_list, data_train_s2.labels_list, batch_idx_src2,
                    lr, dropout, 'src-o'
                )
                batch_idx_src2 = batch_idx_src2 + 1 if batch_idx_src2 + 1 < n_batches_src2 else 0
            loss_tar = sum(losses_tar)

            # metrics = self.run_evaluate(dev)
            aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r, opinion_f1 = self.evaluate(
                data_valid_t.word_idxs_list, data_valid_t.tok_texts, data_valid_t.aspects_true_list,
                'tar', data_valid_t.opinions_true_list)
            # print('iter {}, loss={:.4f}, p={:.4f}, r={:.4f}, f1={:.4f}, best_f1={:.4f}'.format(
            #     epoch, loss_tar, p, r, f1, best_f1))
            logging.info('iter {}, loss={:.4f}, p={:.4f}, r={:.4f}, f1={:.4f}, best_f1={:.4f},'
                         ' p={:.4f}, r={:.4f}, f1={:.4f}, best_f1={:.4f}'.format(
                epoch, loss_tar, aspect_p, aspect_r, aspect_f1, best_f1_a, opinion_p, opinion_r,
                opinion_f1, best_f1_o))

            # if aspect_f1 + opinion_f1 > best_f1:
            if aspect_f1 > best_f1_a and opinion_f1 > best_f1_o:
                best_f1_a = aspect_f1
                best_f1_o = opinion_f1
                best_f1 = aspect_f1 + opinion_f1

                aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r, opinion_f1 = self.evaluate(
                    data_test_t.word_idxs_list, data_test_t.tok_texts, data_test_t.aspects_true_list,
                    'tar', data_test_t.opinions_true_list)

                logging.info('Test, p={:.4f}, r={:.4f}, f1={:.4f},'
                             ' p={:.4f}, r={:.4f}, f1={:.4f}'.format(
                    aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r, opinion_f1))

                if self.saver is not None:
                    self.saver.save(self.sess, save_file)
                    # print('model saved to {}'.format(save_file))
                    logging.info('model saved to {}'.format(save_file))

            p1, r1, f11, _, _, _ = self.evaluate(
                data_valid_s1.word_idxs_list, data_valid_s1.tok_texts,
                data_valid_s1.aspects_true_list, 'src-a')

            p2, r2, f12, _, _, _ = self.evaluate(
                data_valid_s2.word_idxs_list, data_valid_s2.tok_texts,
                data_valid_s2.opinions_true_list, 'src-o')
            logging.info('src1, p={:.4f}, r={:.4f}, f1={:.4f}; src2, p={:.4f}, r={:.4f}, f1={:.4f}'.format(
                p1, r1, f11, p2, r2, f12
            ))

    # def train(self, word_idxs_list_train, labels_list_train, word_idxs_list_valid, labels_list_valid,
    #           vocab, valid_texts, aspects_true_list, opinions_true_list, train_feat_list=None, valid_feat_list=None,
    #           n_epochs=10, lr=0.001, dropout=0.5, save_file=None, error_file=None):
    #     if save_file is not None and self.saver is None:
    #         self.saver = tf.train.Saver()
    #
    #     n_train = len(word_idxs_list_train)
    #     n_batches = (n_train + self.batch_size - 1) // self.batch_size
    #
    #     best_f1_a, best_f1_o, best_f1_sum = 0, 0, 0
    #     for epoch in range(n_epochs):
    #         best_f1_a, best_f1_o, best_f1_sum = self.__train_epoch(
    #             epoch, n_batches, word_idxs_list_train, labels_list_train, word_idxs_list_valid, labels_list_valid,
    #             valid_texts, aspects_true_list, opinions_true_list, lr, dropout,
    #             best_f1_a, best_f1_o, best_f1_sum, train_feat_list=train_feat_list, valid_feat_list=valid_feat_list,
    #             error_file=error_file, save_model_file=save_file)

    def predict_all(self, word_idxs_list, feat_list):
        label_seq_list = list()
        for i, word_idxs in enumerate(word_idxs_list):
            feat_seq_batch = None if feat_list is None else [feat_list[i]]
            label_seq, lens = self.predict_batch([word_idxs], feat_seq_batch)
            label_seq_list.append(label_seq[0])
        return label_seq_list

    def predict_batch(self, word_idxs_list, task):
        fd, sequence_lengths = self.get_feed_dict(word_idxs_list, dropout=1.0)

        # get tag scores and transition params of CRF
        viterbi_sequences = []
        if task == 'src-a':
            logits, trans_params = self.sess.run(
                    [self.logits_src_a, self.trans_params_src_a], feed_dict=fd)
        elif task == 'src-o':
            logits, trans_params = self.sess.run(
                    [self.logits_src_o, self.trans_params_src_o], feed_dict=fd)
        else:
            logits, trans_params = self.sess.run(
                    [self.logits_tar, self.trans_params_tar], feed_dict=fd)

        # iterate over the sentences because no batching in vitervi_decode
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params)
            viterbi_sequences += [viterbi_seq]

        return viterbi_sequences, sequence_lengths

    def evaluate(self, word_idxs_list_valid, test_texts, terms_true_list, task,
                 opinions_ture_list=None, is_train=True):
        aspect_true_cnt, aspect_sys_cnt, aspect_hit_cnt = 0, 0, 0
        opinion_true_cnt, opinion_sys_cnt, opinion_hit_cnt = 0, 0, 0
        error_sents, error_terms = list(), list()
        correct_sent_idxs = list()
        for sent_idx, (word_idxs, text, terms_true) in enumerate(zip(
                word_idxs_list_valid, test_texts, terms_true_list)):
            labels_pred, sequence_lengths = self.predict_batch([word_idxs], task)
            labels_pred = labels_pred[0]

            aspect_terms_sys = utils.get_terms_from_label_list(labels_pred, text, 1, 2)

            new_hit_cnt = utils.count_hit(terms_true, aspect_terms_sys)
            aspect_true_cnt += len(terms_true)
            aspect_sys_cnt += len(aspect_terms_sys)
            aspect_hit_cnt += new_hit_cnt
            if new_hit_cnt == aspect_true_cnt:
                correct_sent_idxs.append(sent_idx)

            if opinions_ture_list is None:
                continue

            if is_train:
                opinion_terms_sys = utils.get_terms_from_label_list(labels_pred, text, 3, 4)
            else:
                opinion_terms_sys = utils.get_terms_from_label_list(labels_pred, text, 1, 2)
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
