import tensorflow as tf
import numpy as np
import logging
from utils import utils
from utils.modelutils import evaluate_ao_extraction
from utils.bldatautils import TrainDataBert, ValidDataBert, ValidDataBertOL
from models.robert import Robert


class BertLSTMCRF:
    def __init__(self, n_tags, word_embed_dim, learning_rate=0.001, hidden_size_lstm=300, batch_size=5,
                 lr_method='adam', manual_feat_len=0, model_file=None):
        self.n_tags = n_tags
        self.hidden_size_lstm = hidden_size_lstm
        self.batch_size = batch_size
        self.lr_method = lr_method
        self.saver = None
        self.manual_feat_len = manual_feat_len
        self.init_learning_rate = learning_rate

        self.word_embed_pad = np.random.normal(size=word_embed_dim)

        self.word_embeddings_input = tf.placeholder(
            tf.float32, shape=[None, None, word_embed_dim], name='word_embeddings')
        self.word_embeddings = self.word_embeddings_input
        # self.word_embeddings = tf.nn.dropout(self.word_embeddings_input, self.dropout)

        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        self.manual_feat = None
        if manual_feat_len > 0:
            self.manual_feat = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name='manual_feat')

        self.__add_logits_op()
        self.__add_loss_op()
        self.__add_train_op(self.lr_method, self.lr, self.loss)
        self.__init_session(model_file)

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

    def __add_loss_op(self):
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.labels, self.sequence_lengths)
        self.trans_params = trans_params  # need to evaluate it for decoding
        self.loss = tf.reduce_mean(-log_likelihood)
        # self.loss = tf.reduce_mean(-log_likelihood) + 0.001 * tf.nn.l2_loss(self.W)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)

    def __add_train_op(self, lr_method, lr, loss):
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

    def get_feed_dict(self, word_embeddings, label_seqs=None, lr=None, dropout=None):
        word_embed_seqs, sequence_lengths = utils.pad_embed_sequences(word_embeddings, self.word_embed_pad)
        word_embed_seqs = np.array(word_embed_seqs, np.float32)
        # print(word_embed_seqs.shape)

        # print(len(word_ids))
        # build feed dictionary
        feed = {
            self.word_embeddings_input: word_embed_seqs,
            self.sequence_lengths: sequence_lengths
        }

        if label_seqs is not None:
            label_seqs = [list(labels) for labels in label_seqs]
            labels, _ = utils.pad_sequences(label_seqs, 0)
            feed[self.labels] = labels

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

    def __train_batch(self, word_embed_seqs, label_seqs, batch_idx, lr, dropout):
        # print(self.batch_size)
        word_embed_seqs_batch = word_embed_seqs[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
        label_seqs_batch = label_seqs[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
        feed_dict, _ = self.get_feed_dict(word_embed_seqs_batch,
                                          label_seqs=label_seqs_batch, lr=lr, dropout=dropout)

        _, train_loss = self.sess.run(
            [self.train_op, self.loss], feed_dict=feed_dict)
        return train_loss

    def get_feed_dict_ol(self, embed_arr, seq_lens, lr, dropout, label_seqs=None):
        feed = {
            self.word_embeddings_input: embed_arr,
            self.sequence_lengths: seq_lens
        }

        if label_seqs is not None:
            feed[self.labels] = label_seqs

        feed[self.lr] = lr
        feed[self.dropout] = dropout
        return feed

    def __evaluate_ol(self, tfrec_dataset_valid, token_seqs, at_true_list, ot_true_list, robert_model):
        next_valid_example = tfrec_dataset_valid.make_one_shot_iterator().get_next()
        all_preds = list()
        idx = 0
        while True:
            try:
                features = self.sess.run(next_valid_example)
                all_layers = self.sess.run(robert_model.all_layers, feed_dict={
                    robert_model.input_ids: features["input_ids"], robert_model.input_mask: features["input_mask"],
                    robert_model.segment_ids: features["segment_ids"], robert_model.label_ids: features["label_ids"],
                    robert_model.hidden_dropout: 1.0, robert_model.attention_dropout: 1.0
                })
                seq_embeds = np.concatenate(
                    [all_layers[-1], all_layers[-2], all_layers[-3], all_layers[-4]], axis=-1)
                idx += 1
                seq_lens = np.squeeze(features['seq_len'])
                max_seq_len = np.max(seq_lens)
                embed_arr = seq_embeds[:, :max_seq_len, :]

                preds = self.predict_batch_ol(embed_arr, seq_lens)

                for y_pred in preds:
                    all_preds.append(y_pred[1:])
            except tf.errors.OutOfRangeError:
                break
        assert len(all_preds) == len(at_true_list)
        token_seqs = [token_seq[1:len(token_seq) - 1] for token_seq in token_seqs]
        (a_p, a_r, a_f1, o_p, o_r, o_f1
         ) = utils.prf1_for_terms(all_preds, token_seqs, at_true_list, ot_true_list)
        return a_p, a_r, a_f1, o_p, o_r, o_f1

    def train_ol(self, robert_model: Robert, train_tfrec_file, valid_tfrec_file, test_tfrec_file, seq_length,
                 n_train, data_valid: ValidDataBertOL, data_test: ValidDataBertOL,
                 n_epochs=10, lr=0.001, dropout=0.5):
        from models import robert

        logging.info('n_epochs={}, lr={}, dropout={}'.format(n_epochs, lr, dropout))

        n_batches = (n_train + self.batch_size - 1) // self.batch_size

        dataset_train = robert.get_dataset(train_tfrec_file, self.batch_size, True, seq_length)
        dataset_valid = robert.get_dataset(valid_tfrec_file, 8, False, seq_length)
        dataset_test = robert.get_dataset(test_tfrec_file, 8, False, seq_length)
        next_train_example = dataset_train.make_one_shot_iterator().get_next()
        best_f1_sum = 0
        for epoch in range(n_epochs):
            losses = list()
            for i in range(n_batches):
                features = self.sess.run(next_train_example)
                all_layers = self.sess.run(robert_model.all_layers, feed_dict={
                    robert_model.input_ids: features["input_ids"], robert_model.input_mask: features["input_mask"],
                    robert_model.segment_ids: features["segment_ids"], robert_model.label_ids: features["label_ids"],
                    robert_model.hidden_dropout: 1.0, robert_model.attention_dropout: 1.0
                })
                seq_embeds = np.concatenate(
                    [all_layers[-1], all_layers[-2], all_layers[-3], all_layers[-4]], axis=-1)
                # print(all_layers[-1].shape)
                # print(seq_embeds.shape)
                seq_lens = np.squeeze(features['seq_len'])
                # print(seq_lens)
                # print(all_layers[-1])
                # print(seq_embeds)
                # exit()
                max_seq_len = np.max(seq_lens)
                embed_arr = seq_embeds[:, :max_seq_len, :]
                label_seqs = features["label_ids"][:, :max_seq_len]
                feed_dict = self.get_feed_dict_ol(embed_arr, seq_lens, lr, dropout, label_seqs)
                _, train_loss = self.sess.run(
                    [self.train_op, self.loss], feed_dict=feed_dict)
                losses.append(train_loss)
            loss = sum(losses)

            a_p_v, a_r_v, a_f1_v, o_p_v, o_r_v, o_f1_v = self.__evaluate_ol(
                dataset_valid, data_valid.token_seqs, data_valid.aspects_true_list,
                data_valid.opinions_true_list, robert_model)
            logging.info(
                'iter {}, loss={:.4f}, p={:.4f}, r={:.4f}, f1={:.4f};'
                ' p={:.4f}, r={:.4f}, f1={:.4f}; best_f1_sum={:.4f}'.format(
                    epoch, loss, a_p_v, a_r_v, a_f1_v, o_p_v, o_r_v,
                    o_f1_v, best_f1_sum))
            if a_f1_v + o_f1_v > best_f1_sum:
                best_f1_sum = a_f1_v + o_f1_v
                a_p_t, a_r_t, a_f1_t, o_p_t, o_r_t, o_f1_t = self.__evaluate_ol(
                    dataset_test, data_test.token_seqs, data_test.aspects_true_list,
                    data_test.opinions_true_list, robert_model)
                logging.info(
                    'Test, p={:.4f}, r={:.4f}, a_f1={:.4f};'
                    ' p={:.4f}, r={:.4f}, o_f1={:.4f}'.format(
                        a_p_t, a_r_t, a_f1_t, o_p_t, o_r_t, o_f1_t, best_f1_sum))

    def train(self, data_train: TrainDataBert, data_valid: ValidDataBert, data_test: ValidDataBert,
              n_epochs=10, lr=0.001, dropout=0.5, save_file=None, dst_aspects_file=None, dst_opinions_file=None):
        logging.info('n_epochs={}, lr={}, dropout={}'.format(n_epochs, lr, dropout))
        if save_file is not None and self.saver is None:
            self.saver = tf.train.Saver()

        n_train = len(data_train.label_seqs)
        n_batches = (n_train + self.batch_size - 1) // self.batch_size

        best_f1_sum = 0
        best_a_f1, best_o_f1 = 0, 0
        for epoch in range(n_epochs):
            losses, losses_seg = list(), list()
            for i in range(n_batches):
                train_loss = self.__train_batch(
                    data_train.word_embed_seqs, data_train.label_seqs, i, lr, dropout)
                losses.append(train_loss)

            loss = sum(losses)
            # print(loss)
            # metrics = self.run_evaluate(dev)
            aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r, opinion_f1 = self.evaluate(
                data_valid.word_embed_seqs, data_valid.tok_texts, data_valid.aspects_true_list,
                data_valid.opinions_true_list)

            logging.info('iter {}, loss={:.4f}, p={:.4f}, r={:.4f}, f1={:.4f};'
                         ' p={:.4f}, r={:.4f}, f1={:.4f}; best_f1_sum={:.4f}'.format(
                epoch, loss, aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r,
                opinion_f1, best_f1_sum))

            # # if True:
            if aspect_f1 + opinion_f1 > best_f1_sum:
            # if aspect_f1 > best_a_f1 and opinion_f1 > best_o_f1:
                best_f1_sum = aspect_f1 + opinion_f1
                best_a_f1 = aspect_f1
                best_o_f1 = opinion_f1

                aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r, opinion_f1 = self.evaluate(
                    data_test.word_embed_seqs, data_test.tok_texts, data_test.aspects_true_list,
                    data_test.opinions_true_list)
                # print('iter {}, loss={:.4f}, p={:.4f}, r={:.4f}, f1={:.4f}, best_f1={:.4f}'.format(
                #     epoch, loss_tar, p, r, f1, best_f1))
                logging.info('Test, p={:.4f}, r={:.4f}, f1={:.4f}; p={:.4f}, r={:.4f}, f1={:.4f}'.format(
                    aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r,
                    opinion_f1))

                if self.saver is not None:
                    self.saver.save(self.sess, save_file)
                    # print('model saved to {}'.format(save_file))
                    logging.info('model saved to {}'.format(save_file))

    def predict_all(self, word_idxs_list):
        label_seq_list = list()
        for i, word_idxs in enumerate(word_idxs_list):
            label_seq, lens = self.predict_batch([word_idxs])
            label_seq_list.append(label_seq[0])
        return label_seq_list

    def predict_batch_ol(self, token_embed_arr, seq_lens):
        fd = self.get_feed_dict_ol(token_embed_arr, seq_lens, lr=None, dropout=1.0)

        # get tag scores and transition params of CRF
        viterbi_sequences = []
        logits, trans_params = self.sess.run(
                [self.logits, self.trans_params], feed_dict=fd)

        # iterate over the sentences because no batching in vitervi_decode
        for logit, sequence_length in zip(logits, seq_lens):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params)
            viterbi_sequences += [viterbi_seq]

        return viterbi_sequences

    def predict_batch(self, token_embed_seqs):
        fd, sequence_lengths = self.get_feed_dict(token_embed_seqs, dropout=1.0)

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

    def evaluate(self, word_embed_seqs, test_texts, terms_true_list,
                 opinions_ture_list=None, dst_aspects_file=None, dst_opinions_file=None):
        aspect_true_cnt, aspect_sys_cnt, aspect_hit_cnt = 0, 0, 0
        opinion_true_cnt, opinion_sys_cnt, opinion_hit_cnt = 0, 0, 0
        error_sents, error_terms = list(), list()
        correct_sent_idxs = list()
        aspect_terms_sys_list, opinion_terms_sys_list = list(), list()
        for sent_idx, (word_embed_seq, text, terms_true) in enumerate(zip(
                word_embed_seqs, test_texts, terms_true_list)):
            labels_pred, sequence_lengths = self.predict_batch([word_embed_seq])
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

        if dst_aspects_file is not None:
            utils.write_terms_list(aspect_terms_sys_list, dst_aspects_file)
        if dst_opinions_file is not None:
            utils.write_terms_list(opinion_terms_sys_list, dst_opinions_file)

        aspect_p, aspect_r, aspect_f1 = utils.prf1(aspect_true_cnt, aspect_sys_cnt, aspect_hit_cnt)
        if opinions_ture_list is None:
            return aspect_p, aspect_r, aspect_f1, 0, 0, 0

        opinion_p, opinion_r, opinion_f1 = utils.prf1(opinion_true_cnt, opinion_sys_cnt, opinion_hit_cnt)
        return aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r, opinion_f1
