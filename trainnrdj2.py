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
                     dst_model_file, task, lamb, lstm_l2, train_word_embeddings=False, load_model_file=None):
    init_logging('log/{}-pre-{}-{}.log'.format(os.path.splitext(
        os.path.basename(__file__))[0], utils.get_machine_name(), str_today), mode='a', to_stdout=True)

    # n_train = 1000
    n_train = -1
    label_opinions = True
    # label_opinions = False
    n_tags = 5 if label_opinions else 3
    # n_tags = 5 if task == 'train' else 3
    batch_size = 32
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
    logging.info('train_word_embeddings={} lstm_l2={}'.format(train_word_embeddings, lstm_l2))

    nrdj = NeuRuleDoubleJoint(n_tags, word_vecs_matrix, share_lstm,
                              hidden_size_lstm=hidden_size_lstm, train_word_embeddings=train_word_embeddings,
                              lamb=lamb, lstm_l2_src=lstm_l2, model_file=load_model_file, batch_size=batch_size)

    nrdj.pre_train(train_data_src1, valid_data_src1, train_data_src2, valid_data_src2, vocab,
                   n_epochs=30, lr=lr, save_file=dst_model_file)


def __train_nrdj(word_vecs_file, train_tok_texts_file, train_sents_file, train_valid_split_file, test_tok_texts_file,
                 test_sents_file, load_model_file, task):
    init_logging('log/{}-train-{}-{}.log'.format(os.path.splitext(
        os.path.basename(__file__))[0], utils.get_machine_name(), str_today), mode='a', to_stdout=True)

    # dst_aspects_file = 'd:/data/aspect/semeval14/nrdj-aspects.txt'
    # dst_opinions_file = 'd:/data/aspect/semeval14/nrdj-opinions.txt'
    dst_aspects_file, dst_opinions_file = None, None

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

    nrdj = NeuRuleDoubleJoint(n_tags, word_vecs_matrix, share_lstm, hidden_size_lstm=hidden_size_lstm,
                              model_file=load_model_file, batch_size=batch_size)
    nrdj.train(train_data, valid_data, test_data, vocab, n_epochs=n_epochs, lr=lr, dst_aspects_file=dst_aspects_file,
               dst_opinions_file=dst_opinions_file)


if __name__ == '__main__':
    str_today = datetime.date.today().strftime('%y-%m-%d')

    hidden_size_lstm = 100
    n_epochs = 200
    train_word_embeddings = False

    # dataset = 'se15r'
    # dataset = 'se14r'
    dataset = 'se14l'

    lamb = 0.001
    lstm_l2_src = True

    if dataset == 'se15r':
        # rule_model_file = os.path.join(config.DATA_DIR_SE15, 'model-data/pretrain/yelpr9-rest-part0_04.ckpt')
        # word_vecs_file = os.path.join(config.DATA_DIR_SE15, 'model-data/yelp-w2v-sg-100-n10-i30-w5.pkl')
        # rule_model_file = os.path.join(config.DATA_DIR_SE15, 'model-data/pretrain/yelpr9-rest-part0_04-tmp.ckpt')
        # word_vecs_file = os.path.join(config.DATA_DIR_SE15, 'model-data/yelp-w2v-sg-100-n10-i30-w5.pkl')
        rule_model_file = os.path.join(config.SE15_DIR, 'model-data/pretrain/yelpr9-rest-part0_04-300d-100h-twv.ckpt')
        word_vecs_file = os.path.join(config.SE15_DIR, 'model-data/yelp-w2v-sg-300-n10-i30-w5.pkl')
    elif dataset == 'se14r':
        rule_model_file = os.path.join(config.SE14_DIR, 'model-data/pretrain/yelpr9-rest-part0_04-reg1e3.ckpt')
        word_vecs_file = os.path.join(config.SE14_DIR, 'model-data/yelp-w2v-sg-100-n10-i30-w5.pkl')
    else:
        rule_model_file = os.path.join(config.SE14_DIR, 'model-data/pretrain/amazon-100d-100h-twv.ckpt')
        # word_vecs_file = os.path.join(config.SE14_DIR, 'model-data/amazon-wv-300-sg-n10-w8-i30.pkl')
        word_vecs_file = os.path.join(config.SE14_DIR, 'model-data/amazon-wv-100-sg-n10-w8-i30.pkl')

    dataset_files = config.DATA_FILES[dataset]
    auto_labeled_data_files = config.DATA_FILES['restaurants-yelp']
    if dataset == 'se14l':
        auto_labeled_data_files = config.DATA_FILES['laptops-amazon']

    # __pre_train_nrdj(word_vecs_file, pre_tok_texts_file, pre_aspect_terms_file,
    #                  pre_opinion_terms_file, rule_model_file, 'both', load_model_file=rule_model_file)
    # __pre_train_nrdj(
    #     word_vecs_file, auto_labeled_data_files['sent_texts_file'],
    #     dataset_files['pretrain_aspect_terms_file'], dataset_files['pretrain_opinion_terms_file'],
    #     rule_model_file, 'both', lamb=lamb, lstm_l2=lstm_l2_src, train_word_embeddings=train_word_embeddings)
    __train_nrdj(word_vecs_file, dataset_files['train_tok_texts_file'], dataset_files['train_sents_file'],
                 dataset_files['train_valid_split_file'], dataset_files['test_tok_texts_file'],
                 dataset_files['test_sents_file'], None, 'both')
