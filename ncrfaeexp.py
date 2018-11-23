import pickle
from models.ncrfae import NeuCRFAutoEncoder
import config
from utils.loggingutils import init_logging
from utils import datautils
import logging
import datetime


if __name__ == '__main__':
    str_today = datetime.date.today().strftime('%y-%m-%d')
    init_logging('log/ncrfae-train-{}.log'.format(str_today), mode='a', to_stdout=True)

    word_vecs_file = config.SE14_LAPTOP_AMAZON_WORD_VEC_FILE
    n_tags = 5
    n_train = -1
    label_opinions = True
    train_valid_split_file = config.SE14_LAPTOP_TRAIN_VALID_SPLIT_FILE
    train_tok_texts_file = config.SE14_LAPTOP_TRAIN_TOK_TEXTS_FILE
    train_sents_file = config.SE14_LAPTOP_TRAIN_SENTS_FILE
    test_tok_texts_file = config.SE14_LAPTOP_TEST_TOK_TEXTS_FILE
    test_sents_file = config.SE14_LAPTOP_TEST_SENTS_FILE
    # unsupervised_tok_texts_file = config.SE14_LAPTOP_TRAIN_TOK_TEXTS_FILE
    unsupervised_tok_texts_file = config.AMAZON_TOK_TEXTS_FILE

    logging.info('word_vec_file: {}'.format(config.SE14_LAPTOP_AMAZON_WORD_VEC_FILE))
    print('loading data ...')
    with open(word_vecs_file, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)
        # print(vocab)

    word_idx_dict = {w: i + 1 for i, w in enumerate(vocab)}
    unsupervised_word_seqs = datautils.read_sents_to_word_idx_seqs(unsupervised_tok_texts_file, word_idx_dict)
    print(len(unsupervised_word_seqs), 'unsupervised sents')

    n_unsupervised_sents_used = 100
    unsupervised_word_seqs = unsupervised_word_seqs[:n_unsupervised_sents_used]
    logging.info('{} unsupervised sents used.'.format(n_unsupervised_sents_used))

    train_data, valid_data, test_data = datautils.get_data_semeval(
        train_sents_file, train_tok_texts_file, train_valid_split_file, test_sents_file, test_tok_texts_file,
        vocab, n_train, label_opinions)
    ncrfae = NeuCRFAutoEncoder(n_tags, word_vecs_matrix, batch_size=3, lr_method='adam')
    # ncrfae.test_model(train_data)
    ncrfae.train(train_data, valid_data, test_data, unsupervised_word_seqs, n_epochs=500, lr=0.001)
