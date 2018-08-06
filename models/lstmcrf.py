import tensorflow as tf
import logging
from utils.utils import pad_sequences, pad_feat_sequence
from utils.modelutils import evaluate_ao_extraction


class LSTMCRF:
    def __init__(self, n_tags, word_embeddings, hidden_size_lstm=300, batch_size=20, train_word_embeddings=False,
                 lr_method='adam', clip=-1, use_crf=True, manual_feat_len=0, model_file=None):
        self.n_tags = n_tags
        self.vals_word_embeddings = word_embeddings
        self.hidden_size_lstm = hidden_size_lstm
        self.batch_size = batch_size
        self.lr_method = lr_method
        self.clip = clip
        self.saver = None
        self.manual_feat_len = manual_feat_len

        self.n_words, self.dim_word = word_embeddings.shape
        self.use_crf = use_crf

        self.word_idxs = tf.placeholder(tf.int32, shape=[None, None], name='word_idxs')
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        self.manual_feat = None
        if manual_feat_len > 0:
            self.manual_feat = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name='manual_feat')

        self.__add_word_embedding_op(train_word_embeddings)
        self.__add_logits_op()
        self.__add_pred_op()
        self.__add_loss_op()
        self.__add_train_op(self.lr_method, self.lr, self.loss, self.clip)
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
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
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

        with tf.variable_scope("proj"):
            dim_tmp = 2 * self.hidden_size_lstm + self.manual_feat_len
            self.W = tf.get_variable("W", dtype=tf.float32, shape=[dim_tmp, self.n_tags])

            self.b = tf.get_variable(
                "b", shape=[self.n_tags], dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(self.lstm_output)[1]
            if self.manual_feat is not None:
                output = tf.concat([self.lstm_output, self.manual_feat], axis=-1)
            else:
                output = self.lstm_output
            output = tf.reshape(output, [-1, dim_tmp])
            pred = tf.matmul(output, self.W) + self.b
            self.logits = tf.reshape(pred, [-1, nsteps, self.n_tags])

    def __add_pred_op(self):
        if not self.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

    def __add_loss_op(self):
        """Defines the loss"""
        if self.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params  # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
            # self.loss = tf.reduce_mean(-log_likelihood) + 0.001 * tf.nn.l2_loss(self.W)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)
            # self.loss = tf.reduce_mean(losses) + 0.01 * tf.nn.l2_loss(self.W)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)

    def __add_train_op(self, lr_method, lr, loss, clip=-1):
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
                grads, vs = zip(*optimizer.compute_gradients(loss))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)

    def __init_session(self, model_file):
        """Defines self.sess and initialize the variables"""
        # self.logger.info("Initializing tf session")
        self.sess = tf.Session()
        if model_file is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            tf.train.Saver().restore(self.sess, model_file)

    def get_W_b(self):
        return self.sess.run([self.W, self.b])

    def get_feed_dict(self, word_idx_seqs, label_seqs=None, lr=None, dropout=None, manual_feat=None):
        word_idx_seqs = [list(word_idxs) for word_idxs in word_idx_seqs]
        word_ids, sequence_lengths = pad_sequences(word_idx_seqs, 0)

        # print(len(word_ids))
        # build feed dictionary
        feed = {
            self.word_idxs: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if label_seqs is not None:
            label_seqs = [list(labels) for labels in label_seqs]
            labels, _ = pad_sequences(label_seqs, 0)
            feed[self.labels] = labels

        if self.manual_feat is not None:
            manual_feat, lens = pad_feat_sequence(manual_feat, manual_feat[0].shape[1])
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

    def __train_epoch(self, epoch, n_batches, word_idxs_list_train, labels_list_train, word_idxs_list_valid,
                      labels_list_valid, valid_texts, aspects_true_list, opinions_true_list,
                      lr, dropout, best_f1_a, best_f1_o, best_f1_sum,
                      train_feat_list=None, valid_feat_list=None,
                      error_file=None, save_model_file=None):
        losses, losses_seg = list(), list()
        for i in range(n_batches):
            word_idxs_list_batch = word_idxs_list_train[i * self.batch_size: (i + 1) * self.batch_size]
            feat_list_batch = None
            if train_feat_list is not None:
                feat_list_batch = train_feat_list[i * self.batch_size: (i + 1) * self.batch_size]
            labels_list_batch = labels_list_train[i * self.batch_size: (i + 1) * self.batch_size]
            feed_dict, _ = self.get_feed_dict(word_idxs_list_batch, labels_list_batch, lr, dropout, feat_list_batch)
            # _, train_loss = self.sess.run(
            #     [self.train_op, self.loss], feed_dict=feed_dict)
            _, train_loss, lstm_output = self.sess.run(
                [self.train_op, self.loss, self.lstm_output], feed_dict=feed_dict)
            losses.append(train_loss)
            losses_seg.append(train_loss)

            if (i + 1) % (5000 // self.batch_size) == 0:
                loss_val = sum(losses_seg)
                pred_label_seq_list = self.predict_all(word_idxs_list_valid, valid_feat_list)
                tmp_result = evaluate_ao_extraction(
                    labels_list_valid, pred_label_seq_list, valid_texts, aspects_true_list, opinions_true_list,
                    error_file
                )
                a_p, a_r, a_f1, o_p, o_r, o_f1 = tmp_result
                # p, r, f1 = self.evaluate(word_idxs_list_valid, labels_list_valid, vocab,
                #                          valid_texts, aspects_true_list)
                logging.info('iter={}, loss={:.4f}, p={:.4f}, r={:.4f}, f1={:.4f}, best_f1={:.4f}; '
                             'p={:.4f}, r={:.4f}, f1={:.4f}, best_f1={:.4f}'.format(
                    epoch, loss_val, a_p, a_r, a_f1, best_f1_a, o_p, o_r, o_f1, best_f1_o))
                losses_seg = list()

                if a_f1 + o_f1 > best_f1_sum:
                    best_f1_sum = a_f1 + o_f1
                    if self.saver is not None:
                        self.saver.save(self.sess, save_model_file)
                        # print('model saved to {}'.format(save_file))
                        logging.info('model saved to {}'.format(save_model_file))
                if a_f1 > best_f1_a:
                    best_f1_a = a_f1
                if o_f1 > best_f1_o:
                    best_f1_o = o_f1
        # print('iter {}, loss={}'.format(epoch, sum(losses)))
        # metrics = self.run_evaluate(dev)
        loss_val = sum(losses)
        pred_label_seq_list = self.predict_all(word_idxs_list_valid, valid_feat_list)
        tmp_result = evaluate_ao_extraction(
            labels_list_valid, pred_label_seq_list, valid_texts, aspects_true_list, opinions_true_list, error_file
        )
        # if opinions_true_list is None:
        #     a_p, a_r, a_f1 = tmp_result
        #     o_p, o_r, o_f1 = 0, 0, 0
        # else:
        a_p, a_r, a_f1, o_p, o_r, o_f1 = tmp_result
        # p, r, f1 = self.evaluate(word_idxs_list_valid, labels_list_valid, vocab, valid_texts, aspects_true_list)
        logging.info('iter={}, loss={:.4f}, p={:.4f}, r={:.4f}, f1={:.4f}, best_f1={:.4f}; '
                     'p={:.4f}, r={:.4f}, f1={:.4f}, best_f1={:.4f}'.format(
            epoch, loss_val, a_p, a_r, a_f1, best_f1_a, o_p, o_r, o_f1, best_f1_o))
        if a_f1 + o_f1 > best_f1_sum:
            best_f1_sum = a_f1 + o_f1
            if self.saver is not None:
                self.saver.save(self.sess, save_model_file)
                logging.info('model saved to {}'.format(save_model_file))
        if a_f1 > best_f1_a:
            best_f1_a = a_f1
        if o_f1 > best_f1_o:
            best_f1_o = o_f1

        return best_f1_a, best_f1_o, best_f1_sum

    def train(self, word_idxs_list_train, labels_list_train, word_idxs_list_valid, labels_list_valid,
              vocab, valid_texts, aspects_true_list, opinions_true_list, train_feat_list=None, valid_feat_list=None,
              n_epochs=10, lr=0.001, dropout=0.5, save_file=None, error_file=None):
        if save_file is not None and self.saver is None:
            self.saver = tf.train.Saver()

        n_train = len(word_idxs_list_train)
        n_batches = (n_train + self.batch_size - 1) // self.batch_size

        best_f1_a, best_f1_o, best_f1_sum = 0, 0, 0
        for epoch in range(n_epochs):
            best_f1_a, best_f1_o, best_f1_sum = self.__train_epoch(
                epoch, n_batches, word_idxs_list_train, labels_list_train, word_idxs_list_valid, labels_list_valid,
                valid_texts, aspects_true_list, opinions_true_list, lr, dropout,
                best_f1_a, best_f1_o, best_f1_sum, train_feat_list=train_feat_list, valid_feat_list=valid_feat_list,
                error_file=error_file, save_model_file=save_file)

    def predict_all(self, word_idxs_list, feat_list):
        label_seq_list = list()
        for i, word_idxs in enumerate(word_idxs_list):
            feat_seq_batch = None if feat_list is None else [feat_list[i]]
            label_seq, lens = self.predict_batch([word_idxs], feat_seq_batch)
            label_seq_list.append(label_seq[0])
        return label_seq_list

    def predict_batch(self, word_idxs_list, feat_list):
        fd, sequence_lengths = self.get_feed_dict(word_idxs_list, dropout=1.0, manual_feat=feat_list)

        # get tag scores and transition params of CRF
        viterbi_sequences = []
        logits, trans_params = self.sess.run(
                [self.logits, self.trans_params], feed_dict=fd)

        # iterate over the sentences because no batching in vitervi_decode
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params)
            viterbi_sequences += [viterbi_seq]

        return viterbi_sequences, sequence_lengths
