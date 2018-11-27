import numpy as np
import pickle
import os
import config
import datetime
from platform import platform
from models.nrdoublejoint import NeuRuleDoubleJoint
from utils import utils, modelutils, datautils
from utils.loggingutils import init_logging
import tensorflow as tf
import logging


def __pre_train_nrdj(word_vecs_file, tok_texts_file, aspect_terms_file, opinion_terms_file,
                     dst_model_file, task, load_model_file=None):
    init_logging('log/nrdj-pre-{}-{}.log'.format(utils.get_machine_name(), str_today), mode='a', to_stdout=True)

    # n_train = 1000
    n_train = -1
    label_opinions = True
    # label_opinions = False
    n_tags = 5 if label_opinions else 3
    # n_tags = 5 if task == 'train' else 3
    batch_size = 20
    lr = 0.001
    share_lstm = False

    logging.info(word_vecs_file)
    logging.info(aspect_terms_file)
    logging.info(opinion_terms_file)
    logging.info('dst: {}'.format(dst_model_file))

    print('loading data ...')
    with open(word_vecs_file, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)

    # train_data_src1, valid_data_src1 = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE1_FILE)
    # train_data_src1, valid_data_src1 = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE3_FILE)
    train_data_src1, valid_data_src1 = datautils.get_data_amazon(
        vocab, aspect_terms_file, tok_texts_file, 'aspect')
    train_data_src2, valid_data_src2 = datautils.get_data_amazon(
        vocab, opinion_terms_file, tok_texts_file, 'opinion')
    print('done')

    nrdj = NeuRuleDoubleJoint(n_tags, word_vecs_matrix, share_lstm,
                              hidden_size_lstm=hidden_size_lstm,
                              model_file=load_model_file, batch_size=batch_size)

    nrdj.pre_train(train_data_src1, valid_data_src1, train_data_src2, valid_data_src2, vocab,
                   n_epochs=30, lr=lr, save_file=dst_model_file)


def __train_nrdj(word_vecs_file, train_tok_texts_file, train_sents_file, train_valid_split_file, test_tok_texts_file,
                 test_sents_file, load_model_file, task):
    init_logging('log/nrdj-train-{}-{}.log'.format(utils.get_machine_name(), str_today), mode='a', to_stdout=True)

    dst_aspects_file = 'd:/data/aspect/semeval14/nrdj-aspects.txt'
    dst_opinions_file = 'd:/data/aspect/semeval14/nrdj-opinions.txt'

    # n_train = 1000
    n_train = -1
    label_opinions = True
    # label_opinions = False
    n_tags = 5 if label_opinions else 3
    # n_tags = 5 if task == 'train' else 3
    batch_size = 10
    lr = 0.001
    share_lstm = False

    logging.info(word_vecs_file)
    logging.info('load model {}'.format(load_model_file))
    logging.info(test_sents_file)

    print('loading data ...')
    with open(word_vecs_file, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)
    logging.info('word vec dim: {}, n_words={}'.format(word_vecs_matrix.shape[1], word_vecs_matrix.shape[0]))
    train_data, valid_data, test_data = datautils.get_data_semeval(
        train_sents_file, train_tok_texts_file, train_valid_split_file, test_sents_file, test_tok_texts_file,
        vocab, n_train, label_opinions)
    print('done')

    nrdj = NeuRuleDoubleJoint(n_tags, word_vecs_matrix, share_lstm,
                              hidden_size_lstm=hidden_size_lstm,
                              model_file=load_model_file,
                              batch_size=batch_size)

    nrdj.train(train_data, valid_data, test_data, vocab, n_epochs=n_epochs, lr=lr, dst_aspects_file=dst_aspects_file,
               dst_opinions_file=dst_opinions_file)


str_today = datetime.date.today().strftime('%y-%m-%d')

dm = 'semeval15'
# dm = 'semeval14'
dataset_name = 'restaurant'
# dataset_name = 'laptops'
hidden_size_lstm = 100
n_epochs = 200

os_env = 'Windows' if platform().startswith('Windows') else 'Linux'

if dataset_name == 'laptops':
    # word_vecs_file = config.SE14_LAPTOP_GLOVE_WORD_VEC_FILE
    # word_vecs_file = config.SE14_LAPTOP_AMAZON_WORD_VEC_FILE
    pre_tok_texts_file = config.AMAZON_TOK_TEXTS_FILE
    pre_aspect_terms_file = config.AMAZON_RM_TERMS_FILE
    pre_opinion_terms_file = config.AMAZON_TERMS_TRUE4_FILE

    wv_dim = '100'
    if os_env == 'Windows':
        word_vecs_file = 'd:/data/aspect/semeval14/model-data/amazon-wv-{}-sg-n10-w8-i30.pkl'.format(wv_dim)
        # rule_model_file = 'd:/data/aspect/semeval14/model-data/d{}/wv-{}.ckpl'.format(wv_dim, wv_dim)
        rule_model_file = config.LAPTOP_RULE_MODEL2_FILE
    else:
        word_vecs_file = '/home/hldai/data/aspect/semeval14/model-data/amazon-wv-{}-sg-n10-w8-i30.pkl'.format(wv_dim)
        rule_model_file = '/home/hldai/data/aspect/semeval14/model-data/d{}/wv-{}.ckpl'.format(wv_dim, wv_dim)

    train_valid_split_file = config.SE14_LAPTOP_TRAIN_VALID_SPLIT_FILE
    train_tok_texts_file = config.SE14_LAPTOP_TRAIN_TOK_TEXTS_FILE
    train_sents_file = config.SE14_LAPTOP_TRAIN_SENTS_FILE
    test_tok_texts_file = config.SE14_LAPTOP_TEST_TOK_TEXTS_FILE
    test_sents_file = config.SE14_LAPTOP_TEST_SENTS_FILE

# dataset = 'se15-restaurants'
dataset = 'se14-restaurants'

if dataset == 'se15-restaurants':
    rule_model_file = os.path.join(config.DATA_DIR_SE15, 'model-data/pretrain/yelpr9-rest-part0_04.ckpl')
elif dataset == 'se14-restaurants':
    rule_model_file = os.path.join(config.DATA_DIR_SE14, 'model-data/pretrain/yelpr9-rest-part0_04.ckpl')
else:
    rule_model_file = os.path.join(config.DATA_DIR_SE14, 'model-data/pretrain/amazon.ckpl')

dataset_files = config.DATA_FILES[dataset]
auto_labeled_data_files = config.DATA_FILES['restaurants-yelp']
if 'laptops' in dataset:
    auto_labeled_data_files = config.DATA_FILES['laptops-amazon']

# __pre_train_nrdj(word_vecs_file, pre_tok_texts_file, pre_aspect_terms_file,
#                  pre_opinion_terms_file, rule_model_file, 'both', load_model_file=rule_model_file)
# __pre_train_nrdj(word_vecs_file, pre_tok_texts_file, pre_aspect_terms_file,
#                  pre_opinion_terms_file, rule_model_file, 'both')
__pre_train_nrdj(
    dataset_files['word_vecs_file'], auto_labeled_data_files['sent_texts_file'],
    dataset_files['rule_aspect_result_file'], dataset_files['rule_opinion_result_file'], rule_model_file, 'both'
)
# __train_nrdj(word_vecs_file, train_tok_texts_file, train_sents_file, train_valid_split_file,
#              test_tok_texts_file, test_sents_file, rule_model_file, 'both')
