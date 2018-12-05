import tensorflow as tf
import numpy as np
import logging
from utils import utils, datautils
from utils.modelutils import evaluate_ao_extraction
from utils.bldatautils import TrainDataBert, ValidDataBert, ValidDataBertOL
from models.robert import Robert


class BertNRDJ:
    def __init__(self, n_tags, word_embed_dim, learning_rate=0.001, hidden_size_lstm=300, batch_size=5,
                 lr_method='adam', manual_feat_len=0, model_file=None, n_lstm_layers=1):
        self.n_tags_src = 3
        self.n_tags = n_tags
        self.hidden_size_lstm = hidden_size_lstm
        self.batch_size = batch_size
        self.lr_method = lr_method
        self.saver = None
        self.manual_feat_len = manual_feat_len
        self.init_learning_rate = learning_rate
        self.n_lstm_layers = n_lstm_layers

        self.word_embed_pad = np.random.normal(size=word_embed_dim)

        self.word_embeddings_input = tf.placeholder(
            tf.float32, shape=[None, None, word_embed_dim], name='word_embeddings')
        self.word_embeddings = self.word_embeddings_input
        # self.word_embeddings = tf.nn.dropout(self.word_embeddings_input, self.dropout)

        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
        self.labels_src1 = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        self.labels_src2 = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        self.labels_tar = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

        self.__add_logits_op()
        self.__add_loss_op()
        self.__add_train_op(self.lr_method, self.lr)
        self.__init_session(model_file)

    def __add_logits_op(self):
        lstm_output1 = self.word_embeddings
        for i in range(self.n_lstm_layers):
            with tf.variable_scope("bi-lstm-{}".format(i + 1)):
                self.cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
                cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    self.cell_fw, cell_bw, lstm_output1,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
                self.lstm_output1 = tf.concat([output_fw, output_bw], axis=-1)
                self.lstm_output1 = tf.nn.dropout(self.lstm_output1, self.dropout)
                lstm_output1 = self.lstm_output1

        lstm_output2 = self.word_embeddings
        for i in range(self.n_lstm_layers):
            with tf.variable_scope("bi-lstm-{}".format(self.n_lstm_layers + i + 1)):
                cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
                cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size_lstm)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, lstm_output2,
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
                2 * self.hidden_size_lstm, self.n_tags_src])
            self.b_src2 = tf.get_variable(
                "b", shape=[self.n_tags_src], dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(lstm_output2)[1]
            output = tf.reshape(lstm_output2, [-1, 2 * self.hidden_size_lstm])
            pred = tf.matmul(output, self.W_src2) + self.b_src2
            self.logits_src2 = tf.reshape(pred, [-1, nsteps, self.n_tags_src])

        with tf.variable_scope("proj-target"):
            input_size = 4 * self.hidden_size_lstm

            self.W_tar = tf.get_variable("W", dtype=tf.float32, shape=[input_size, self.n_tags])
            self.b_tar = tf.get_variable(
                "b", shape=[self.n_tags], dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(lstm_output1)[1]

            output = tf.concat([lstm_output1, lstm_output2], axis=-1)
            output = tf.reshape(output, [-1, input_size])
            # pred = tf.matmul(output, self.W_tar) + self.b_tar
            pred = tf.matmul(output, self.W_tar) + self.b_tar
            self.logits_tar = tf.reshape(pred, [-1, nsteps, self.n_tags])

    def __add_loss_op(self):
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

            self.train_op_src1 = optimizer.minimize(self.loss_src1)
            self.train_op_src2 = optimizer.minimize(self.loss_src2)
            self.train_op_tar = optimizer.minimize(self.loss_tar)

    def __init_session(self, model_file):
        """Defines self.sess and initialize the variables"""
        # self.logger.info("Initializing tf session")
        self.sess = tf.Session()
        if model_file is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            tf.train.Saver().restore(self.sess, model_file)

    def get_feed_dict_ol(self, embed_arr, seq_lens, lr, dropout, task, label_seqs=None):
        feed = {
            self.word_embeddings_input: embed_arr,
            self.sequence_lengths: seq_lens
        }

        # if label_seqs is not None:
        #     feed[self.labels] = label_seqs

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
        return feed

    def predict_batch_ol(self, token_embed_arr, seq_lens, task):
        fd = self.get_feed_dict_ol(token_embed_arr, seq_lens, lr=None, task=task, dropout=1.0)

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
        logits, trans_params = self.sess.run(
                [logits, trans_params], feed_dict=fd)

        # iterate over the sentences because no batching in vitervi_decode
        for logit, sequence_length in zip(logits, seq_lens):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params)
            viterbi_sequences += [viterbi_seq]

        return viterbi_sequences

    def __get_batch_input(self, all_layers, seq_lens):
        seq_embeds = np.concatenate(
            [all_layers[-1], all_layers[-2], all_layers[-3], all_layers[-4]], axis=-1)
        # seq_embeds = np.concatenate(
        #     [all_layers[-1], all_layers[-2]], axis=-1)
        # seq_embeds = all_layers[-1]
        seq_lens = np.squeeze(seq_lens, axis=-1)
        max_seq_len = np.max(seq_lens)
        embed_arr = seq_embeds[:, :max_seq_len, :]
        return embed_arr, seq_lens

    def __get_all_inputs_from_tfrec(self, robert_model, tfrec_dataset):
        next_example = tfrec_dataset.make_one_shot_iterator().get_next()
        embed_arr_list, seq_lens_list = list(), list()
        while True:
            try:
                features = self.sess.run(next_example)
                all_layers = self.sess.run(robert_model.all_layers, feed_dict={
                    robert_model.input_ids: features["input_ids"], robert_model.input_mask: features["input_mask"],
                    robert_model.segment_ids: features["segment_ids"], robert_model.label_ids: features["label_ids"],
                    robert_model.hidden_dropout: 1.0, robert_model.attention_dropout: 1.0
                })
                embed_arr, seq_lens = self.__get_batch_input(all_layers, features['seq_len'])
                embed_arr_list.append(embed_arr)
                seq_lens_list.append(seq_lens)
            except tf.errors.OutOfRangeError:
                break
        return embed_arr_list, seq_lens_list

    def __evaluate_ol(self, embed_arr_list, seq_lens_list, token_seqs, at_true_list, ot_true_list):
        all_preds = list()
        for embed_arr, seq_lens in zip(embed_arr_list, seq_lens_list):
            preds = self.predict_batch_ol(embed_arr, seq_lens, 'tar')
            for y_pred in preds:
                all_preds.append(y_pred[1:])

        assert len(all_preds) == len(at_true_list)
        token_seqs = [token_seq[1:len(token_seq) - 1] for token_seq in token_seqs]
        (a_p, a_r, a_f1, o_p, o_r, o_f1
         ) = utils.prf1_for_terms(all_preds, token_seqs, at_true_list, ot_true_list)
        return a_p, a_r, a_f1, o_p, o_r, o_f1

    def __evaluate_single_term_type(
            self, embed_arr_list, seq_lens_list, token_seqs, terms_list, task):
        all_preds = list()
        for embed_arr, seq_lens in zip(embed_arr_list, seq_lens_list):
            preds = self.predict_batch_ol(embed_arr, seq_lens, task)

            for y_pred in preds:
                all_preds.append(y_pred[1:])

        assert len(all_preds) == len(terms_list)
        token_seqs = [token_seq[1:len(token_seq) - 1] for token_seq in token_seqs]
        p, r, f1 = utils.prf1_for_single_term_type(all_preds, token_seqs, terms_list)
        return p, r, f1

    def __train_batch(self, robert_model, next_train_example, lr, dropout, task):
        features = self.sess.run(next_train_example)
        all_layers = self.sess.run(robert_model.all_layers, feed_dict={
            robert_model.input_ids: features["input_ids"], robert_model.input_mask: features["input_mask"],
            robert_model.segment_ids: features["segment_ids"], robert_model.label_ids: features["label_ids"],
            robert_model.hidden_dropout: 1.0, robert_model.attention_dropout: 1.0
        })
        # seq_embeds = np.concatenate(
        #     [all_layers[-1], all_layers[-2], all_layers[-3], all_layers[-4]], axis=-1)
        # seq_lens = np.squeeze(features['seq_len'], axis=-1)
        # embed_arr = seq_embeds[:, :max_seq_len, :]
        embed_arr, seq_lens = self.__get_batch_input(all_layers, features['seq_len'])
        max_seq_len = np.max(seq_lens)
        label_seqs = features["label_ids"][:, :max_seq_len]

        feed_dict = self.get_feed_dict_ol(embed_arr, seq_lens, lr, dropout, task=task, label_seqs=label_seqs)
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

    def pretrain(
            self, robert_model: Robert, train_aspect_tfrec_file, valid_aspect_tfrec_file,
            train_opinion_tfrec_file, valid_opinion_tfrec_file, valid_tokens_file,
            seq_length, valid_aspect_terms_list, valid_opinion_terms_list, n_steps, batch_size,
            lr=0.001, dropout=0.5, save_file=None):
        from models import robert

        if save_file is not None and self.saver is None:
            self.saver = tf.train.Saver()

        token_seqs_valid = datautils.read_tokens_file(valid_tokens_file)

        dataset_aspect_train = robert.get_dataset(train_aspect_tfrec_file, batch_size, True, seq_length)
        dataset_opinion_train = robert.get_dataset(train_opinion_tfrec_file, batch_size, True, seq_length)
        dataset_aspect_valid = robert.get_dataset(valid_aspect_tfrec_file, 8, False, seq_length)
        dataset_opinion_valid = robert.get_dataset(valid_opinion_tfrec_file, 8, False, seq_length)
        next_aspect_train_example = dataset_aspect_train.make_one_shot_iterator().get_next()
        next_opinion_train_example = dataset_opinion_train.make_one_shot_iterator().get_next()
        best_f1_sum = 0
        losses_aspect, losses_opinion = list(), list()
        valid_a_embed_arr_list, valid_a_seq_lens_list = self.__get_all_inputs_from_tfrec(
            robert_model, dataset_aspect_valid)
        valid_o_embed_arr_list, valid_o_seq_lens_list = self.__get_all_inputs_from_tfrec(
            robert_model, dataset_opinion_valid)
        for step in range(n_steps):
            train_loss_apect = self.__train_batch(robert_model, next_aspect_train_example, lr, dropout, 'src1')
            losses_aspect.append(train_loss_apect)
            train_loss_opinion = self.__train_batch(robert_model, next_opinion_train_example, lr, dropout, 'src2')
            losses_opinion.append(train_loss_opinion)
            if (step + 1) % 100 == 0:
                loss_aspect, loss_opinion = sum(losses_aspect), sum(losses_opinion)
                losses_aspect, losses_opinion = list(), list()
                a_p, a_r, a_f1 = self.__evaluate_single_term_type(
                    valid_a_embed_arr_list, valid_a_seq_lens_list, token_seqs_valid, valid_aspect_terms_list, 'src1')
                o_p, o_r, o_f1 = self.__evaluate_single_term_type(
                    valid_o_embed_arr_list, valid_o_seq_lens_list, token_seqs_valid, valid_opinion_terms_list, 'src2')
                logging.info(
                    'step={}, al={:.4f}, ol={:.4f}, p={:.4f}, r={:.4f}, '
                    'a_f1={:.4f}, p={:.4f}, r={:.4f}, o_f1={:.4f}'.format(
                        step + 1, loss_aspect, loss_opinion, a_p, a_r, a_f1, o_p, o_r, o_f1
                    ))
                if a_f1 + o_f1 > best_f1_sum:
                    best_f1_sum = a_f1 + o_f1
                    if self.saver is not None:
                        self.saver.save(self.sess, save_file)
                        # print('model saved to {}'.format(save_file))
                        logging.info('model saved to {}'.format(save_file))

    def train(
            self, robert_model: Robert, train_tfrec_file, valid_tfrec_file, test_tfrec_file, seq_length,
            n_train, data_valid: ValidDataBertOL, data_test: ValidDataBertOL,
            n_epochs=100, lr=0.001, dropout=0.5, start_eval_spoch=0):
        from models import robert

        logging.info('n_epochs={}, lr={}, dropout={}'.format(n_epochs, lr, dropout))

        n_batches = (n_train + self.batch_size - 1) // self.batch_size

        dataset_valid = robert.get_dataset(valid_tfrec_file, 8, False, seq_length)
        valid_embed_arr_list, valid_seq_lens_list = self.__get_all_inputs_from_tfrec(robert_model, dataset_valid)
        dataset_test = robert.get_dataset(test_tfrec_file, 8, False, seq_length)
        test_embed_arr_list, test_seq_lens_list = self.__get_all_inputs_from_tfrec(robert_model, dataset_test)
        dataset_train = robert.get_dataset(train_tfrec_file, self.batch_size, True, seq_length)
        next_train_example = dataset_train.make_one_shot_iterator().get_next()
        best_f1_sum = 0
        for epoch in range(n_epochs):
            losses = list()
            for i in range(n_batches):
                train_loss = self.__train_batch(robert_model, next_train_example, lr, dropout, 'tar')
                losses.append(train_loss)
            loss = sum(losses)

            a_p_v, a_r_v, a_f1_v, o_p_v, o_r_v, o_f1_v = self.__evaluate_ol(
                valid_embed_arr_list, valid_seq_lens_list, data_valid.token_seqs, data_valid.aspects_true_list,
                data_valid.opinions_true_list)
            logging.info(
                'iter {}, loss={:.4f}, p={:.4f}, r={:.4f}, f1={:.4f};'
                ' p={:.4f}, r={:.4f}, f1={:.4f}; best_f1_sum={:.4f}'.format(
                    epoch, loss, a_p_v, a_r_v, a_f1_v, o_p_v, o_r_v,
                    o_f1_v, best_f1_sum))
            # if epoch >= start_eval_spoch and a_f1_v + o_f1_v > best_f1_sum:
            if True:
                best_f1_sum = a_f1_v + o_f1_v
                a_p_t, a_r_t, a_f1_t, o_p_t, o_r_t, o_f1_t = self.__evaluate_ol(
                    test_embed_arr_list, test_seq_lens_list, data_test.token_seqs, data_test.aspects_true_list,
                    data_test.opinions_true_list)
                logging.info(
                    'Test, p={:.4f}, r={:.4f}, a_f1={:.4f};'
                    ' p={:.4f}, r={:.4f}, o_f1={:.4f}'.format(
                        a_p_t, a_r_t, a_f1_t, o_p_t, o_r_t, o_f1_t, best_f1_sum))