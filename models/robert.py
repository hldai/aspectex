import tensorflow as tf
from models import bertmodel, optimization


def get_dataset(data_file, batch_size, is_train, seq_length):
    dataset = tf.data.TFRecordDataset(data_file)
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "seq_len": tf.FixedLenFeature([1], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    if is_train:
        dataset = dataset.repeat()
        # dataset = dataset.shuffle(buffer_size=100)

    drop_remainder = True if is_train else False
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return dataset


class Robert:
    def __init__(self, bert_config, n_labels, seq_length, is_train, init_checkpoint=None,
                 learning_rate=5e-5, n_train_steps=1000, n_warmup_steps=10,
                 dropout_val=0.9):
        self.input_ids = tf.placeholder(tf.int32, [None, seq_length])
        self.input_mask = tf.placeholder(tf.int32, [None, seq_length])
        self.segment_ids = tf.placeholder(tf.int32, [None, seq_length])
        self.label_ids = tf.placeholder(tf.int32, [None, seq_length])
        self.hidden_dropout = tf.placeholder(tf.float32, shape=[], name="hidden_dropout_prob")
        self.attention_dropout = tf.placeholder(
            tf.float32, shape=[], name="attention_probs_dropout_prob")
        self.dropout_val = dropout_val
        self.seq_len = seq_length
        self.n_labels = n_labels

        model = bertmodel.BertModel(
            config=bert_config,
            input_ids=self.input_ids,
            hidden_dropout_prob=self.hidden_dropout,
            attention_probs_dropout_prob=self.attention_dropout,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)

        self.all_layers = model.get_all_encoder_layers()
        self.output_layer = model.get_sequence_output()
        if is_train:
            self.__init_task_loss(
                learning_rate=learning_rate, n_train_steps=n_train_steps, n_warmup_steps=n_warmup_steps
            )

        if init_checkpoint:
            # use checkpoint
            tvars = tf.trainable_variables()
            initialized_variable_names = {}
            (assignment_map, initialized_variable_names
             ) = bertmodel.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    def __init_task_loss(self, learning_rate, n_train_steps, n_warmup_steps):
        hidden_size = self.output_layer.shape[-1].value

        output_weight = tf.get_variable(
            "output_weights", [self.n_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        output_bias = tf.get_variable(
            "output_bias", [self.n_labels], initializer=tf.zeros_initializer()
        )

        with tf.variable_scope("task-loss"):
            output_layer = tf.nn.dropout(self.output_layer, keep_prob=self.dropout_val)
            output_layer = tf.reshape(output_layer, [-1, hidden_size])
            logits = tf.matmul(output_layer, output_weight, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            self.logits = tf.reshape(logits, [-1, self.seq_len, self.n_labels])
            # mask = tf.cast(input_mask,tf.float32)
            # loss = tf.contrib.seq2seq.sequence_loss(logits,labels,mask)
            # return (loss, logits, predict)
            ##########################################################################
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(self.label_ids, depth=self.n_labels, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            per_example_loss = tf.dtypes.cast(self.input_mask, tf.float32) * per_example_loss
            self.task_loss = tf.reduce_sum(per_example_loss)
            probabilities = tf.nn.softmax(logits, axis=-1)
            self.predicts = tf.argmax(probabilities, axis=-1)

        self.train_op = optimization.create_optimizer(
            self.task_loss, learning_rate, n_train_steps, n_warmup_steps, False)

    def get_embedding(self, sess, input_ids, input_mask, segment_ids, label_ids):
        layer_outputs = sess.run(self.output_layer, feed_dict={
            self.input_ids: input_ids, self.input_mask: input_mask,
            self.segment_ids: segment_ids, self.label_ids: label_ids,
            self.hidden_dropout: 1.0, self.attention_dropout: 1.0
        })

        return layer_outputs
