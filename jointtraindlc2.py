import numpy as np
import pickle
import config
import datetime
from platform import platform
from models.dslstmcrf import DSLSTMCRF
from utils import utils, modelutils, datautils
from utils.loggingutils import init_logging
import tensorflow as tf
import logging


def __train_dlc(word_vecs_file, train_tok_texts_file, train_sents_file, train_valid_split_file, test_tok_texts_file,
                test_sents_file):
    init_logging('log/dlc-jtrain2-{}.log'.format(str_today), mode='a', to_stdout=True)

    # n_train = 1000
    n_train = -1
    label_opinions = True
    # label_opinions = False
    n_tags = 5 if label_opinions else 3
    # n_tags = 5 if task == 'train' else 3
    batch_size = 10
    lr = 0.001
    share_lstm = False

    print('loading data ...')
    with open(word_vecs_file, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)
    logging.info('word vec dim: {}, n_words={}'.format(word_vecs_matrix.shape[1], word_vecs_matrix.shape[0]))

    train_data_src1, valid_data_src1 = datautils.get_data_amazon(
        vocab, pre_aspect_terms_file, pre_tok_texts_file, 'aspect')
    train_data_src2, valid_data_src2 = datautils.get_data_amazon(
        vocab, pre_opinion_terms_file, pre_tok_texts_file, 'opinion')

    train_data, valid_data, test_data = datautils.get_data_semeval(
        train_sents_file, train_tok_texts_file, train_valid_split_file, test_sents_file, test_tok_texts_file,
        vocab, n_train, label_opinions)
    print('done')

    dlc = DSLSTMCRF(word_vecs_matrix, hidden_size_lstm=hidden_size_lstm,
                    model_file=None, batch_size=batch_size)

    dlc.joint_train(train_data_src1, valid_data_src1, train_data_src2, valid_data_src2,
                    train_data, valid_data, test_data, n_epochs=n_epochs, lr=lr)


str_today = datetime.date.today().strftime('%y-%m-%d')

dm = 'semeval15'
# dm = 'semeval14'
dataset_name = 'restaurant'
# dataset_name = 'laptops'
hidden_size_lstm = 100
n_epochs = 400

os_env = 'Windows' if platform().startswith('Windows') else 'Linux'

if dataset_name == 'laptops':
    # word_vecs_file = config.SE14_LAPTOP_GLOVE_WORD_VEC_FILE
    # word_vecs_file = config.SE14_LAPTOP_AMAZON_WORD_VEC_FILE
    pre_tok_texts_file = config.AMAZON_TOK_TEXTS_FILE
    pre_aspect_terms_file = config.AMAZON_RM_TERMS_FILE
    pre_opinion_terms_file = config.AMAZON_TERMS_TRUE4_FILE

    wv_dim = '100'
    if os_env == 'Windows':
        word_vecs_file = 'd:/data/aspect/semeval14/model-data/amazon-wv-nr-{}-sg-n10-w8-i30.pkl'.format(wv_dim)
    else:
        word_vecs_file = '/home/hldai/data/aspect/semeval14/model-data/amazon-wv-nr-{}-sg-n10-w8-i30.pkl'.format(wv_dim)

    train_valid_split_file = config.SE14_LAPTOP_TRAIN_VALID_SPLIT_FILE
    train_tok_texts_file = config.SE14_LAPTOP_TRAIN_TOK_TEXTS_FILE
    train_sents_file = config.SE14_LAPTOP_TRAIN_SENTS_FILE
    test_tok_texts_file = config.SE14_LAPTOP_TEST_TOK_TEXTS_FILE
    test_sents_file = config.SE14_LAPTOP_TEST_SENTS_FILE
else:
    # word_vecs_file = config.SE14_REST_GLOVE_WORD_VEC_FILE
    # pre_aspect_terms_file = 'd:/data/aspect/semeval14/restaurants/yelp-aspect-rule-result-r.txt'
    # aspect_terms_file = 'd:/data/aspect/semeval14/restaurant/yelp-aspect-rule-result-r1.txt'
    if os_env == 'Windows':
        pre_aspect_terms_file = 'd:/data/aspect/{}/restaurants/yelp-aspect-rm-rule-result.txt'.format(dm)
        pre_opinion_terms_file = 'd:/data/aspect/{}/restaurants/yelp-opinion-rule-result.txt'.format(dm)
        pre_tok_texts_file = 'd:/data/res/yelp-review-eng-tok-sents-round-9.txt'
    else:
        pre_aspect_terms_file = '/home/hldai/data/aspect/{}/restaurants/yelp-aspect-rm-rule-result.txt'.format(dm)
        pre_opinion_terms_file = '/home/hldai/data/aspect/{}/restaurants/yelp-opinion-rule-result.txt'.format(dm)
        pre_tok_texts_file = '/home/hldai/data/yelp/yelp-review-eng-tok-sents-round-9.txt'

    if dm == 'semeval14':
        train_valid_split_file = config.SE14_REST_TRAIN_VALID_SPLIT_FILE
        train_tok_texts_file = config.SE14_REST_TRAIN_TOK_TEXTS_FILE
        train_sents_file = config.SE14_REST_TRAIN_SENTS_FILE
        test_tok_texts_file = config.SE14_REST_TEST_TOK_TEXTS_FILE
        test_sents_file = config.SE14_REST_TEST_SENTS_FILE
        # word_vecs_file = config.SE14_REST_YELP_WORD_VEC_FILE
        word_vecs_file = '/home/hldai/data/aspect/semeval14/model-data/yelp-word-vecs-sg-100-n10-i20-w5.pkl'
    else:
        train_valid_split_file = config.SE15_REST_TRAIN_VALID_SPLIT_FILE
        train_tok_texts_file = config.SE15_REST_TRAIN_TOK_TEXTS_FILE
        train_sents_file = config.SE15_REST_TRAIN_SENTS_FILE
        test_tok_texts_file = config.SE15_REST_TEST_TOK_TEXTS_FILE
        test_sents_file = config.SE15_REST_TEST_SENTS_FILE
        # word_vecs_file = config.SE15_REST_YELP_WORD_VEC_FILE
        word_vecs_file = '/home/hldai/data/aspect/semeval15/model-data/yelp-word-vecs-sg-100-n10-i20-w5.pkl'

__train_dlc(word_vecs_file, train_tok_texts_file, train_sents_file, train_valid_split_file,
            test_tok_texts_file, test_sents_file)
