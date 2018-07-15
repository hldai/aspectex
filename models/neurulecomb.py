import tensorflow as tf
from utils.utils import pad_sequences


class NeuRuleComb:
    def __init__(self, n_tags, word_embeddings, hidden_size_lstm=300, batch_size=20, train_word_embeddings=False,
                 lr_method='adam', clip=-1, use_crf=True, model_file=None):
        self.n_tags = n_tags
        self.vals_word_embeddings = word_embeddings
        self.hidden_size_lstm = hidden_size_lstm
        self.batch_size = batch_size
        self.lr_method = lr_method
        self.clip = clip
        self.saver = None

        self.n_words, self.dim_word = word_embeddings.shape
        self.use_crf = use_crf

        self.word_idxs = tf.placeholder(tf.int32, shape=[None, None], name='word_idxs')
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

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
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            self.W = tf.get_variable("W", dtype=tf.float32, shape=[
                2 * self.hidden_size_lstm, self.n_tags])

            b = tf.get_variable("b", shape=[self.n_tags],
                                dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * self.hidden_size_lstm])
            pred = tf.matmul(output, self.W) + b
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
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, model_file)

    def get_feed_dict(self, word_idx_seqs, label_seqs=None, lr=None, dropout=None):
        word_idx_seqs = [list(word_idxs) for word_idxs in word_idx_seqs]
        word_ids, sequence_lengths = pad_sequences(word_idx_seqs, 0)

        # build feed dictionary
        feed = {
            self.word_idxs: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if label_seqs is not None:
            label_seqs = [list(labels) for labels in label_seqs]
            labels, _ = pad_sequences(label_seqs, 0)
            feed[self.labels] = labels

        feed[self.lr] = lr
        feed[self.dropout] = dropout

        return feed, sequence_lengths

    def train(self, word_idxs_list_train, labels_list_train, word_idxs_list_valid, labels_list_valid,
              vocab, valid_texts, terms_true_list, n_epochs=10, lr=0.001, dropout=0.5, save_file=None):
        if save_file is not None and self.saver is None:
            self.saver = tf.train.Saver()

        n_train = len(word_idxs_list_train)
        n_batches = (n_train + self.batch_size - 1) // self.batch_size

        for epoch in range(n_epochs):
            losses, losses_seg = list(), list()
            for i in range(n_batches):
                word_idxs_list_batch = word_idxs_list_train[i * self.batch_size: (i + 1) * self.batch_size]
                labels_list_batch = labels_list_train[i * self.batch_size: (i + 1) * self.batch_size]
                feed_dict, _ = self.get_feed_dict(word_idxs_list_batch, labels_list_batch, lr, dropout)
                _, train_loss = self.sess.run(
                    [self.train_op, self.loss], feed_dict=feed_dict)
                losses.append(train_loss)
                losses_seg.append(train_loss)

                if (i + 1) % (5000 // self.batch_size) == 0:
                    print('iter {}, batch {}, loss={}'.format(epoch, i, sum(losses_seg)))
                    self.evaluate(word_idxs_list_valid, labels_list_valid, vocab, valid_texts, terms_true_list)
                    losses_seg = list()

                    if self.saver is not None:
                        self.saver.save(self.sess, save_file)
                        print('model saved to {}'.format(save_file))
            print('iter {}, loss={}'.format(epoch, sum(losses)))
            # metrics = self.run_evaluate(dev)
            self.evaluate(word_idxs_list_valid, labels_list_valid, vocab, valid_texts, terms_true_list)

    def predict_batch(self, word_idxs):
        fd, sequence_lengths = self.get_feed_dict(word_idxs, dropout=1.0)

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

    def evaluate(self, word_idxs_list_valid, labels_list_valid, vocab, test_texts, terms_true_list):
        cnt_true, cnt_sys, cnt_hit = 0, 0, 0
        error_sents, error_terms = list(), list()
        correct_sent_idxs = list()
        sent_idx = 0
        for word_idxs, labels, text, terms_true in zip(
                word_idxs_list_valid, labels_list_valid, test_texts, terms_true_list):
            terms_sys = set()
            labels_pred, sequence_lengths = self.predict_batch([word_idxs])
            labels_pred = labels_pred[0]
            words = text.split(' ')
            # print(labels_pred)
            # print(len(words), len(labels_pred))
            assert len(words) == len(labels_pred)

            p = 0
            while p < len(words):
                yi = labels_pred[p]
                if yi == 1:
                    pright = p
                    while pright + 1 < len(words) and labels_pred[pright + 1] == 2:
                        pright += 1
                    terms_sys.add(' '.join(words[p: pright + 1]))
                    p = pright + 1
                else:
                    p += 1

            hit = True
            terms_not_hit = set()
            for t in terms_true:
                if t in terms_sys:
                    cnt_hit += 1
                else:
                    hit = False
                    terms_not_hit.add(t)
            cnt_true += len(terms_true)
            cnt_sys += len(terms_sys)
            if not hit:
                # error_sents.append(sent)
                error_terms.append(terms_not_hit)
            else:
                correct_sent_idxs.append(sent_idx)
            sent_idx += 1

        # save_json_objs(error_sents, 'd:/data/aspect/semeval14/error-sents.txt')
        with open('d:/data/aspect/semeval14/error-sents.txt', 'w', encoding='utf-8') as fout:
            for sent, terms in zip(error_sents, error_terms):
                # terms_true = [t['term'].lower() for t in sent['terms']] if 'terms' in sent else list()
                fout.write('{}\n{}\n\n'.format(sent['text'], terms))
        with open('d:/data/aspect/semeval14/lstmcrf-correct.txt', 'w', encoding='utf-8') as fout:
            fout.write('\n'.join([str(i) for i in correct_sent_idxs]))

        p = cnt_hit / cnt_sys
        r = cnt_hit / (cnt_true - 16)
        f1 = 2 * p * r / (p + r)
        print(p, r, f1, cnt_true)
        # p, r, f1 = set_evaluate(terms_true, terms_sys)
        # print(p, r, f1)
