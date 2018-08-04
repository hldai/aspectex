import numpy as np
import pickle
import config
import datetime
from models.nrdoublejoint import NeuRuleDoubleJoint
from models.nrdjmlp import NeuRuleDoubleJointMLP
from models.nrdjdeep import NeuRuleDoubleJointDeep
from utils import utils, modelutils, datautils
from utils.loggingutils import init_logging
import tensorflow as tf


def __train_nrdj_mlp_restaurant_pr():
    init_logging('log/nrdj-mlp-restaurant-{}.log'.format(str_today), mode='a', to_stdout=True)

    # n_train = 1000
    n_train = -1
    task = 'train'
    label_task = 'aspect'
    n_tags = 5 if label_task == 'both' else 3
    # n_tags = 5 if task == 'train' else 3
    batch_size = 20
    hidden_size_lstm = 100
    n_epochs = 500
    lr = 0.001
    share_lstm = False
    # load_pretrained_model = True
    load_pretrained_model = False
    # train_mode = 'target-only'
    train_mode = 'all'

    aspect_terms_p_file = 'd:/data/aspect/semeval14/restaurant/yelp-aspect-rule-result-p.txt'
    aspect_terms_r_file = 'd:/data/aspect/semeval14/restaurant/yelp-aspect-rule-result-r.txt'
    # opinion_terms_file = 'd:/data/aspect/semeval14/restaurant/yelp-opinion-rule-result.txt'
    yelp_tok_texts_file = 'd:/data/res/yelp-review-eng-tok-sents-round-9.txt'
    rule_model_file = 'd:/data/aspect/semeval14/tf-model/drest/yelp-nrdj.ckpl'
    # rule_model_file = None

    load_model_file = None
    if task == 'train' and load_pretrained_model:
        load_model_file = rule_model_file
    # save_model_file = None if task == 'train' else rule_model_file
    save_model_file = rule_model_file if task == 'pretrain' else None

    print('loading data ...')
    with open(config.SE14_REST_GLOVE_WORD_VEC_FILE, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)
    train_data_tar, valid_data_tar = datautils.get_data_semeval(
        config.SE14_REST_TRAIN_SENTS_FILE, config.SE14_REST_TRAIN_TOK_TEXTS_FILE,
        config.SE14_REST_TEST_SENTS_FILE, config.SE14_REST_TEST_TOK_TEXTS_FILE,
        vocab, n_train, label_task)
    # train_data_src1, valid_data_src1 = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE1_FILE)
    # train_data_src1, valid_data_src1 = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE3_FILE)
    train_data_src1, valid_data_src1 = datautils.get_data_amazon(
        vocab, aspect_terms_p_file, yelp_tok_texts_file, 'aspect')
    # train_data_src2, valid_data_src2 = datautils.get_data_amazon(
    #     vocab, aspect_terms_r_file, yelp_tok_texts_file, 'opinion')
    train_data_src2, valid_data_src2 = datautils.get_data_amazon(
        vocab, aspect_terms_r_file, yelp_tok_texts_file, 'aspect')
    # train_data_src2, valid_data_src2 = datautils.get_data_amazon(
    #     vocab, opinion_terms_file, yelp_tok_texts_file, 'opinion')
    print('done')

    nrdj = NeuRuleDoubleJointMLP(n_tags, word_vecs_matrix, share_lstm,
                                 hidden_size_lstm=hidden_size_lstm,
                                 model_file=load_model_file)

    nrj_train_data_src1 = nrj_train_data_src2 = None
    # if train_mode != 'target-only':
    nrj_train_data_src1 = NeuRuleDoubleJointMLP.TrainData(
        train_data_src1.word_idxs_list, train_data_src1.labels_list, valid_data_src1.word_idxs_list,
        valid_data_src1.labels_list, valid_data_src1.tok_texts, valid_data_src1.aspects_true_list, None
    )
    nrj_train_data_src2 = NeuRuleDoubleJointMLP.TrainData(
        train_data_src2.word_idxs_list, train_data_src2.labels_list, valid_data_src2.word_idxs_list,
        valid_data_src2.labels_list, valid_data_src2.tok_texts, valid_data_src2.aspects_true_list,
        None
    )

    nrj_train_data_tar = NeuRuleDoubleJointMLP.TrainData(
        train_data_tar.word_idxs_list, train_data_tar.labels_list, valid_data_tar.word_idxs_list,
        valid_data_tar.labels_list, valid_data_tar.tok_texts, valid_data_tar.aspects_true_list,
        valid_data_tar.opinions_true_list
    )
    nrdj.train(nrj_train_data_src1, nrj_train_data_src2, nrj_train_data_tar, vocab, train_mode,
               n_epochs=n_epochs, lr=lr)


def __train_nrdj_deep_restaurant_pr():
    init_logging('log/nrdj-deep-restaurant-{}.log'.format(str_today), mode='a', to_stdout=True)

    # n_train = 1000
    n_train = -1
    task = 'train'
    label_task = 'aspect'
    n_tags = 5 if label_task == 'both' else 3
    # n_tags = 5 if task == 'train' else 3
    batch_size = 20
    hidden_size_lstm = 100
    n_epochs = 500
    lr = 0.001
    share_lstm = False
    # load_pretrained_model = True
    load_pretrained_model = False
    # train_mode = 'target-only'
    train_mode = 'all'

    aspect_terms_p_file = 'd:/data/aspect/semeval14/restaurant/yelp-aspect-rule-result-p.txt'
    aspect_terms_r_file = 'd:/data/aspect/semeval14/restaurant/yelp-aspect-rule-result-r.txt'
    # opinion_terms_file = 'd:/data/aspect/semeval14/restaurant/yelp-opinion-rule-result.txt'
    yelp_tok_texts_file = 'd:/data/res/yelp-review-eng-tok-sents-round-9.txt'
    rule_model_file = 'd:/data/aspect/semeval14/tf-model/drest/yelp-nrdj.ckpl'
    # rule_model_file = None

    load_model_file = None
    if task == 'train' and load_pretrained_model:
        load_model_file = rule_model_file
    # save_model_file = None if task == 'train' else rule_model_file
    save_model_file = rule_model_file if task == 'pretrain' else None

    print('loading data ...')
    with open(config.SE14_REST_GLOVE_WORD_VEC_FILE, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)
    train_data_tar, valid_data_tar = datautils.get_data_semeval(
        config.SE14_REST_TRAIN_SENTS_FILE, config.SE14_REST_TRAIN_TOK_TEXTS_FILE,
        config.SE14_REST_TEST_SENTS_FILE, config.SE14_REST_TEST_TOK_TEXTS_FILE,
        vocab, n_train, label_task)
    # train_data_src1, valid_data_src1 = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE1_FILE)
    # train_data_src1, valid_data_src1 = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE3_FILE)
    train_data_src1, valid_data_src1 = datautils.get_data_amazon(
        vocab, aspect_terms_p_file, yelp_tok_texts_file, 'aspect')
    # train_data_src2, valid_data_src2 = datautils.get_data_amazon(
    #     vocab, aspect_terms_r_file, yelp_tok_texts_file, 'opinion')
    train_data_src2, valid_data_src2 = datautils.get_data_amazon(
        vocab, aspect_terms_r_file, yelp_tok_texts_file, 'aspect')
    # train_data_src2, valid_data_src2 = datautils.get_data_amazon(
    #     vocab, opinion_terms_file, yelp_tok_texts_file, 'opinion')
    print('done')

    nrdj = NeuRuleDoubleJointDeep(n_tags, word_vecs_matrix, share_lstm,
                                 hidden_size_lstm=hidden_size_lstm,
                                 model_file=load_model_file)

    nrj_train_data_src1 = nrj_train_data_src2 = None
    # if train_mode != 'target-only':
    nrj_train_data_src1 = NeuRuleDoubleJointDeep.TrainData(
        train_data_src1.word_idxs_list, train_data_src1.labels_list, valid_data_src1.word_idxs_list,
        valid_data_src1.labels_list, valid_data_src1.tok_texts, valid_data_src1.aspects_true_list, None
    )
    nrj_train_data_src2 = NeuRuleDoubleJointDeep.TrainData(
        train_data_src2.word_idxs_list, train_data_src2.labels_list, valid_data_src2.word_idxs_list,
        valid_data_src2.labels_list, valid_data_src2.tok_texts, valid_data_src2.aspects_true_list,
        None
    )

    nrj_train_data_tar = NeuRuleDoubleJointDeep.TrainData(
        train_data_tar.word_idxs_list, train_data_tar.labels_list, valid_data_tar.word_idxs_list,
        valid_data_tar.labels_list, valid_data_tar.tok_texts, valid_data_tar.aspects_true_list,
        valid_data_tar.opinions_true_list
    )
    nrdj.train(nrj_train_data_src1, nrj_train_data_src2, nrj_train_data_tar, vocab, train_mode,
               n_epochs=n_epochs, lr=lr)


str_today = datetime.date.today().strftime('%y-%m-%d')
# __train_nrdj_restaurant()
# __train_nrdj_restaurant_pr()
# __train_nrdj_mlp_restaurant_pr()
__train_nrdj_deep_restaurant_pr()
