import tensorflow as tf
import numpy as np
from models import crf
from tensorflow.contrib.crf import crf_log_norm, crf_sequence_score
import logging
from utils import utils
from utils.modelutils import evaluate_ao_extraction
from utils.datautils import TrainData, ValidData


def np_softmax(vals):
    exp_vals = np.exp(vals - np.expand_dims(np.max(vals, axis=1), 1))
    return exp_vals / np.expand_dims(np.sum(exp_vals, axis=1), 1)


class NeuCRFAutoEncoder:
    def __init__(self, n_tags, word_embeddings, hidden_size_lstm=100, batch_size=5, train_word_embeddings=False,
                 lr_method='adam', manual_feat_len=0, model_file=None):
        self.n_tags = n_tags
        self.vals_word_embeddings = word_embeddings
        self.hidden_size_lstm = hidden_size_lstm
        self.batch_size = batch_size
        self.lr_method = lr_method
        self.saver = None
        self.manual_feat_len = manual_feat_len

        self.n_words, self.dim_word = word_embeddings.shape

        self.word_idxs = tf.placeholder(tf.int32, shape=[None, None], name='word_idxs')
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        self.manual_feat = None
        if manual_feat_len > 0:
            self.manual_feat = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name='manual_feat')

        self.crf_bin_score_mat = tf.Variable(
            np.ones((n_tags, n_tags), np.float32) / (n_tags * n_tags), dtype=tf.float32)

        self.__add_word_embedding_op(train_word_embeddings)
        self.__add_logits_op()
        self.__add_decoder()
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
                _word_embeddings = tf.Variable(
                        self.vals_word_embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=train_word_embeddings)
                # _word_embeddings = tf.constant(
                #         self.vals_word_embeddings,
                #         name="_word_embeddings",
                #         dtype=tf.float32)

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

    def __add_decoder(self):
        self.yw_prob_val_mat_val = np.random.uniform(
            -1, 1, size=(self.n_tags, self.n_words)).astype(np.float32)

        self.yw_prob_val_mat = tf.placeholder(tf.float32, shape=[self.n_tags, self.n_words], name="yw_prob_mat")
        self.yw_prob_mat = tf.transpose(tf.nn.softmax(self.yw_prob_val_mat))
        self.decoder_log_probs = tf.log(tf.gather(self.yw_prob_mat, self.word_idxs))

    def __add_loss_op(self):
        # log_likelihood, _ = crf_log_likelihood(
        #     self.logits, self.labels, self.sequence_lengths, self.crf_bin_score_mat)
        # self.loss_encoder = tf.reduce_mean(-log_likelihood)

        self.unary_score_input = self.logits + self.decoder_log_probs
        # self.supervised_seq_score = crf_sequence_score(
        #     self.unary_score_input, self.labels, self.sequence_lengths, self.crf_bin_score_mat)
        self.supervised_seq_score = crf_sequence_score(
            self.logits, self.labels, self.sequence_lengths, self.crf_bin_score_mat)

        # self.log_norm_unsupervised = crf_log_norm(
        #     unary_score_input, self.sequence_lengths, self.crf_bin_score_mat)
        self.log_norm_unsupervised, self.alphas = crf.crf_log_norm_forward_with_scan(
            self.unary_score_input, self.crf_bin_score_mat)

        self.betas = crf.crf_beta_backward(self.unary_score_input, self.crf_bin_score_mat)
        # self.valids = tf.reduce_logsumexp(self.alphas + self.betas, 2)

        self.log_partition_z = crf_log_norm(self.logits, self.sequence_lengths, self.crf_bin_score_mat)

        self.loss_l = tf.reduce_mean(-(self.supervised_seq_score - self.log_partition_z))
        self.loss_u = tf.reduce_mean(-(self.log_norm_unsupervised - self.log_partition_z))
        self.loss_supervised = self.loss_l + self.loss_u

        # tf.summary.scalar("loss", self.loss)

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

            # self.train_op_supervised = optimizer.minimize(self.loss_supervised)
            self.train_op_supervised = optimizer.minimize(self.loss_l)
            self.train_op_unsupervised = optimizer.minimize(self.loss_u)

    def __init_session(self, model_file):
        """Defines self.sess and initialize the variables"""
        # self.logger.info("Initializing tf session")
        self.sess = tf.Session()
        if model_file is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            tf.train.Saver().restore(self.sess, model_file)

    def update_decoder_probs_mat(self, word_seqs_l, label_seqs, word_seqs_u):
        new_theta_table = np.zeros((self.n_tags, self.n_words))
        # directly optimize the decoder
        for i, word_seq in enumerate(word_seqs_l):
            for xi, y in zip(word_seq, label_seqs[i]):
                # print(y, xi)
                new_theta_table[y, xi] += 1
        # for word_seq in word_seqs_u:
        #     feed_dict = {
        #         self.word_idxs: np.expand_dims(word_seq, 0),
        #         self.dropout: 1.0,
        #         self.yw_prob_val_mat: self.yw_prob_val_mat_val,
        #         self.sequence_lengths: np.array([len(word_seq)], np.int32)
        #     }
        #     # feed_dict[self.word_idxs] = np.expand_dims(word_seq, 0)
        #     alphas, betas, z = self.sess.run(
        #         [self.alphas, self.betas, self.log_norm_unsupervised], feed_dict=feed_dict)
        #     # print(valids)
        #     # print(z)
        #     # print()
        #     for t in range(len(word_seq)):
        #         expected_count = np.exp((alphas[t] + betas[t] - z))
        #         word_id = word_seq[t]
        #         new_theta_table[:, word_id] += np.squeeze(expected_count)

        # new_theta_table = npsoftmax(new_theta_table)
        # print('npsm')
        new_theta_table = np_softmax(new_theta_table)
        self.yw_prob_val_mat_val = new_theta_table
        # print(self.yw_prob_val_mat_val)
        # np.savetxt('d:/data/tmp/tmp.txt', self.yw_prob_val_mat_val)
        # print('save')
        # for v in self.yw_prob_val_mat_val:

    def get_W_b(self):
        return self.sess.run([self.W, self.b])

    def get_feed_dict(self, word_idx_seqs, label_seqs=None, lr=None, dropout=None, manual_feat=None):
        word_idx_seqs = [list(word_idxs) for word_idxs in word_idx_seqs]
        word_ids, sequence_lengths = utils.pad_sequences(word_idx_seqs, 0, fixed_len=True)

        # print(len(word_ids))
        # build feed dictionary
        feed = {
            self.word_idxs: word_ids,
            self.sequence_lengths: sequence_lengths,
            self.yw_prob_val_mat: self.yw_prob_val_mat_val
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

    def __train_batch_supervised(self, word_idxs_list_train, labels_list_train, batch_idx, lr, dropout):
        word_idxs_list_batch = word_idxs_list_train[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
        labels_list_batch = labels_list_train[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
        feed_dict, _ = self.get_feed_dict(word_idxs_list_batch,
                                          label_seqs=labels_list_batch, lr=lr, dropout=dropout)

        # _, train_loss = self.sess.run(
        #     [self.train_op_supervised, self.loss_supervised],
        #     feed_dict=feed_dict)
        _, train_loss = self.sess.run(
            [self.train_op_supervised, self.loss_l],
            feed_dict=feed_dict)
        return train_loss

    def __train_batch_unsupervised(self, word_idx_seqs, batch_idx, lr, dropout):
        word_idx_seqs_batch = word_idx_seqs[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
        feed_dict, _ = self.get_feed_dict(word_idx_seqs_batch, lr=lr, dropout=dropout)
        _, loss = self.sess.run([self.train_op_unsupervised, self.loss_u], feed_dict=feed_dict)
        return loss

    def test_model(self, train_data: TrainData):
        i = 0
        lr = 0.001
        dropout = 0.5
        word_idxs_list_batch = train_data.word_idxs_list[i * self.batch_size: (i + 1) * self.batch_size]
        # feat_list_batch = None
        # labels_list_batch = train_data.labels_list[i * self.batch_size: (i + 1) * self.batch_size]
        # feed_dict, _ = self.get_feed_dict(word_idxs_list_batch, labels_list_batch, lr, dropout, feat_list_batch)
        # _, loss = self.sess.run([self.train_op_supervised, self.loss_supervised], feed_dict=feed_dict)
        loss = self.__train_batch_supervised(train_data.word_idxs_list, train_data.labels_list, 1, lr, dropout)
        print(loss)

    def train(self, data_train: TrainData, data_valid: ValidData, data_test: ValidData, unsupervised_word_seqs,
              n_epochs=10, lr=0.001, dropout=0.5, save_file=None, dst_aspects_file=None, dst_opinions_file=None):
        logging.info('n_epochs={}, lr={}, dropout={}'.format(n_epochs, lr, dropout))
        if save_file is not None and self.saver is None:
            self.saver = tf.train.Saver()

        n_train_l = len(data_train.word_idxs_list)
        n_batches_l = (n_train_l + self.batch_size - 1) // self.batch_size  # n_batches supervised
        n_train_u = len(unsupervised_word_seqs)
        n_batches_u = (n_train_u + self.batch_size - 1) // self.batch_size  # n_batches unsupervised

        best_f1_sum = 0
        best_a_f1, best_o_f1 = 0, 0
        for epoch in range(n_epochs):
            # self.update_decoder_probs_mat(data_train.word_idxs_list, data_train.labels_list, unsupervised_word_seqs)

            losses_l, losses_u, losses_seg = list(), list(), list()
            for i in range(n_batches_l):
                train_loss_l = self.__train_batch_supervised(
                    data_train.word_idxs_list, data_train.labels_list, i, lr, dropout)
                losses_l.append(train_loss_l)

            # for i in range(n_batches_u):
            #     train_loss_u = self.__train_batch_unsupervised(unsupervised_word_seqs, i, lr, dropout)
            #     losses_u.append(train_loss_u)

            loss_l = sum(losses_l)
            loss_u = sum(losses_u)
            # print(loss_l, loss_u)

            # self.update_decoder_probs_mat(data_train.word_idxs_list, data_train.labels_list, unsupervised_word_seqs)

            aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r, opinion_f1 = self.evaluate(
                data_valid.word_idxs_list, data_valid.tok_texts, data_valid.aspects_true_list,
                data_valid.opinions_true_list
            )

            logging.info(
                'iter {}, l_l={:.4f}, l_u={:.4f} p={:.4f}, r={:.4f}, f1={:.4f};'
                ' p={:.4f}, r={:.4f}, f1={:.4f}; best_f1_sum={:.4f}'.format(
                    epoch, loss_l, loss_u, aspect_p, aspect_r, aspect_f1, opinion_p,
                    opinion_r, opinion_f1, best_f1_sum))

            # if True:
            # if aspect_f1 + opinion_f1 > best_f1_sum:
            # if aspect_f1 > best_a_f1 and opinion_f1 > best_o_f1:
            #     best_f1_sum = aspect_f1 + opinion_f1
            #     best_a_f1 = aspect_f1
            #     best_o_f1 = opinion_f1
            #
            #     aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r, opinion_f1 = self.evaluate(
            #         data_test.word_idxs_list, data_test.tok_texts, data_test.aspects_true_list,
            #         'tar', data_test.opinions_true_list)
            #     # print('iter {}, loss={:.4f}, p={:.4f}, r={:.4f}, f1={:.4f}, best_f1={:.4f}'.format(
            #     #     epoch, loss_tar, p, r, f1, best_f1))
            #     logging.info('Test, p={:.4f}, r={:.4f}, f1={:.4f}; p={:.4f}, r={:.4f}, f1={:.4f}'.format(
            #         aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r,
            #         opinion_f1))
            #
            #     if self.saver is not None:
            #         self.saver.save(self.sess, save_file)
            #         # print('model saved to {}'.format(save_file))
            #         logging.info('model saved to {}'.format(save_file))

    def predict_all(self, word_idxs_list, feat_list):
        label_seq_list = list()
        for i, word_idxs in enumerate(word_idxs_list):
            feat_seq_batch = None if feat_list is None else [feat_list[i]]
            label_seq, lens = self.predict_batch([word_idxs])
            label_seq_list.append(label_seq[0])
        return label_seq_list

    def predict_batch(self, word_idxs_list):
        fd, sequence_lengths = self.get_feed_dict(word_idxs_list, dropout=1.0)

        # get tag scores and transition params of CRF
        viterbi_sequences = []
        # logits, trans_params = self.sess.run(
        #         [self.unary_score_input, self.crf_bin_score_mat], feed_dict=fd)
        logits, trans_params = self.sess.run(
                [self.logits, self.crf_bin_score_mat], feed_dict=fd)

        # iterate over the sentences because no batching in vitervi_decode
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params)
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

    def evaluate(self, word_idxs_list_valid, test_texts, terms_true_list,
                 opinions_ture_list=None, dst_aspects_file=None, dst_opinions_file=None):
        aspect_true_cnt, aspect_sys_cnt, aspect_hit_cnt = 0, 0, 0
        opinion_true_cnt, opinion_sys_cnt, opinion_hit_cnt = 0, 0, 0
        error_sents, error_terms = list(), list()
        correct_sent_idxs = list()
        aspect_terms_sys_list, opinion_terms_sys_list = list(), list()
        for sent_idx, (word_idxs, text, terms_true) in enumerate(zip(
                word_idxs_list_valid, test_texts, terms_true_list)):
            labels_pred, sequence_lengths = self.predict_batch([word_idxs])
            labels_pred = labels_pred[0]

            aspect_terms_sys = self.get_terms_from_label_list(labels_pred, text, 1, 2)
            # print(aspect_terms_sys)

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

        if dst_aspects_file is not None:
            utils.write_terms_list(aspect_terms_sys_list, dst_aspects_file)
        if dst_opinions_file is not None:
            utils.write_terms_list(opinion_terms_sys_list, dst_opinions_file)

        aspect_p, aspect_r, aspect_f1 = utils.prf1(aspect_true_cnt, aspect_sys_cnt, aspect_hit_cnt)
        if opinions_ture_list is None:
            return aspect_p, aspect_r, aspect_f1, 0, 0, 0

        opinion_p, opinion_r, opinion_f1 = utils.prf1(opinion_true_cnt, opinion_sys_cnt, opinion_hit_cnt)
        return aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r, opinion_f1
