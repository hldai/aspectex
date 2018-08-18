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
import logging


def __train_nrdj_restaurant_pr():
    init_logging('log/nrdj-restaurant-{}.log'.format(str_today), mode='a', to_stdout=True)

    # n_train = 1000
    n_train = -1
    # task = 'pretrain'
    task = 'train'
    label_task = 'aspect'
    n_tags = 5 if label_task == 'both' else 3
    # n_tags = 5 if task == 'train' else 3
    batch_size = 20
    hidden_size_lstm = 100
    n_epochs = 200
    lr = 0.001
    share_lstm = False
    # load_pretrained_model = True
    load_pretrained_model = False
    # train_mode = 'target-only'
    train_mode = 'all'
    src1_label, src2_label = 'aspect', 'opinion'

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
        vocab, aspect_terms_p_file, yelp_tok_texts_file, src1_label)
    # train_data_src2, valid_data_src2 = datautils.get_data_amazon(
    #     vocab, aspect_terms_r_file, yelp_tok_texts_file, 'opinion')
    train_data_src2, valid_data_src2 = datautils.get_data_amazon(
        vocab, aspect_terms_r_file, yelp_tok_texts_file, src2_label)
    # train_data_src2, valid_data_src2 = datautils.get_data_amazon(
    #     vocab, opinion_terms_file, yelp_tok_texts_file, 'opinion')
    print('done')

    nrdj = NeuRuleDoubleJoint(n_tags, word_vecs_matrix, share_lstm,
                              hidden_size_lstm=hidden_size_lstm,
                              model_file=load_model_file)

    nrj_train_data_src1 = nrj_train_data_src2 = None
    # if train_mode != 'target-only':
    nrj_train_data_src1 = NeuRuleDoubleJoint.TrainData(
        train_data_src1.word_idxs_list, train_data_src1.labels_list, valid_data_src1.word_idxs_list,
        valid_data_src1.labels_list, valid_data_src1.tok_texts, valid_data_src1.aspects_true_list, None
    )
    nrj_train_data_src2 = NeuRuleDoubleJoint.TrainData(
        train_data_src2.word_idxs_list, train_data_src2.labels_list, valid_data_src2.word_idxs_list,
        valid_data_src2.labels_list, valid_data_src2.tok_texts, None,
        valid_data_src2.opinions_true_list
    )

    nrj_train_data_tar = NeuRuleDoubleJoint.TrainData(
        train_data_tar.word_idxs_list, train_data_tar.labels_list, valid_data_tar.word_idxs_list,
        valid_data_tar.labels_list, valid_data_tar.tok_texts, valid_data_tar.aspects_true_list,
        valid_data_tar.opinions_true_list
    )

    if task == 'pretrain':
        nrdj.pre_train(nrj_train_data_src1, nrj_train_data_src2, vocab, n_epochs=n_epochs, lr=lr,
                       save_file=save_model_file)
    if task == 'train':
        nrdj.train(nrj_train_data_src1, nrj_train_data_src2, nrj_train_data_tar, vocab, train_mode,
                   n_epochs=n_epochs, lr=lr)


def __train_nrdj_joint_restaurant_pr():
    init_logging('log/nrdj-restaurant-{}.log'.format(str_today), mode='a', to_stdout=True)

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

    nrdj = NeuRuleDoubleJoint(n_tags, word_vecs_matrix, share_lstm,
                              hidden_size_lstm=hidden_size_lstm,
                              model_file=load_model_file)

    nrj_train_data_src1 = nrj_train_data_src2 = None
    # if train_mode != 'target-only':
    nrj_train_data_src1 = NeuRuleDoubleJoint.TrainData(
        train_data_src1.word_idxs_list, train_data_src1.labels_list, valid_data_src1.word_idxs_list,
        valid_data_src1.labels_list, valid_data_src1.tok_texts, valid_data_src1.aspects_true_list, None
    )
    nrj_train_data_src2 = NeuRuleDoubleJoint.TrainData(
        train_data_src2.word_idxs_list, train_data_src2.labels_list, valid_data_src2.word_idxs_list,
        valid_data_src2.labels_list, valid_data_src2.tok_texts, valid_data_src2.aspects_true_list,
        None
    )

    nrj_train_data_tar = NeuRuleDoubleJoint.TrainData(
        train_data_tar.word_idxs_list, train_data_tar.labels_list, valid_data_tar.word_idxs_list,
        valid_data_tar.labels_list, valid_data_tar.tok_texts, valid_data_tar.aspects_true_list,
        valid_data_tar.opinions_true_list
    )
    nrdj.train(nrj_train_data_src1, nrj_train_data_src2, nrj_train_data_tar, vocab, train_mode,
               n_epochs=n_epochs, lr=lr)


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
    share_lstm = True
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


def __pre_train_nrdj(word_vecs_file, tok_texts_file, aspect_terms_file, opinion_terms_file,
                     dst_model_file, task):
    init_logging('log/nrdj-pre-{}.log'.format(str_today), mode='a', to_stdout=True)

    # n_train = 1000
    n_train = -1
    label_opinions = True
    # label_opinions = False
    n_tags = 5 if label_opinions else 3
    # n_tags = 5 if task == 'train' else 3
    batch_size = 20
    lr = 0.001
    share_lstm = False

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
                              model_file=None)

    nrj_train_data_src1 = nrj_train_data_src2 = None
    # if train_mode != 'target-only':
    # nrj_train_data_src1 = NeuRuleDoubleJoint.TrainData(
    #     train_data_src1.word_idxs_list, train_data_src1.labels_list, valid_data_src1.word_idxs_list,
    #     valid_data_src1.labels_list, valid_data_src1.tok_texts, valid_data_src1.aspects_true_list, None
    # )
    # nrj_train_data_src2 = NeuRuleDoubleJoint.TrainData(
    #     train_data_src2.word_idxs_list, train_data_src2.labels_list, valid_data_src2.word_idxs_list,
    #     valid_data_src2.labels_list, valid_data_src2.tok_texts, None,
    #     valid_data_src2.opinions_true_list
    # )

    nrdj.pre_train(train_data_src1, valid_data_src1, train_data_src2, valid_data_src2, vocab,
                   n_epochs=n_epochs, lr=lr, save_file=dst_model_file)


def __train_nrdj(word_vecs_file, train_tok_texts_file, train_sents_file, train_valid_split_file, test_tok_texts_file,
                 test_sents_file, load_model_file, task):
    init_logging('log/nrdj-train-{}.log'.format(str_today), mode='a', to_stdout=True)

    # n_train = 1000
    n_train = -1
    label_opinions = True
    # label_opinions = False
    n_tags = 5 if label_opinions else 3
    # n_tags = 5 if task == 'train' else 3
    batch_size = 20
    lr = 0.001
    share_lstm = False

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
                              model_file=load_model_file)

    nrdj.train(train_data, valid_data, test_data, vocab, n_epochs=n_epochs, lr=lr)


str_today = datetime.date.today().strftime('%y-%m-%d')

dm = 'semeval15'
dataset_name = 'restaurant'
# dataset_name = 'laptops'
hidden_size_lstm = 100
n_epochs = 200

if dataset_name == 'laptops':
    # word_vecs_file = config.SE14_LAPTOP_GLOVE_WORD_VEC_FILE
    word_vecs_file = config.SE14_LAPTOP_AMAZON_WORD_VEC_FILE
    pre_tok_texts_file = config.AMAZON_TOK_TEXTS_FILE
    pre_aspect_terms_file = config.AMAZON_RM_TERMS_FILE
    pre_opinion_terms_file = config.AMAZON_TERMS_TRUE4_FILE
    rule_model_file = config.LAPTOP_RULE_MODEL2_FILE

    train_valid_split_file = config.SE14_LAPTOP_TRAIN_VALID_SPLIT_FILE
    train_tok_texts_file = config.SE14_LAPTOP_TRAIN_TOK_TEXTS_FILE
    train_sents_file = config.SE14_LAPTOP_TRAIN_SENTS_FILE
    test_tok_texts_file = config.SE14_LAPTOP_TEST_TOK_TEXTS_FILE
    test_sents_file = config.SE14_LAPTOP_TEST_SENTS_FILE
else:
    # word_vecs_file = config.SE14_REST_GLOVE_WORD_VEC_FILE
    # pre_aspect_terms_file = 'd:/data/aspect/semeval14/restaurants/yelp-aspect-rule-result-r.txt'
    # aspect_terms_file = 'd:/data/aspect/semeval14/restaurant/yelp-aspect-rule-result-r1.txt'
    pre_aspect_terms_file = 'd:/data/aspect/{}/restaurants/yelp-aspect-rm-rule-result.txt'.format(dm)
    pre_opinion_terms_file = 'd:/data/aspect/{}/restaurants/yelp-opinion-rule-result.txt'.format(dm)
    pre_tok_texts_file = 'd:/data/res/yelp-review-eng-tok-sents-round-9.txt'
    rule_model_file = 'd:/data/aspect/{}/tf-model/drest/yelp-nrdj.ckpl'.format(dm)
    # rule_model_file = 'd:/data/aspect/semeval14/tf-model/drest/yelp-nrdj.ckpl'

    if dm == 'semeval14':
        train_valid_split_file = config.SE14_REST_TRAIN_VALID_SPLIT_FILE
        train_tok_texts_file = config.SE14_REST_TRAIN_TOK_TEXTS_FILE
        train_sents_file = config.SE14_REST_TRAIN_SENTS_FILE
        test_tok_texts_file = config.SE14_REST_TEST_TOK_TEXTS_FILE
        test_sents_file = config.SE14_REST_TEST_SENTS_FILE
        word_vecs_file = config.SE14_REST_YELP_WORD_VEC_FILE
    else:
        train_valid_split_file = config.SE15_REST_TRAIN_VALID_SPLIT_FILE
        train_tok_texts_file = config.SE15_REST_TRAIN_TOK_TEXTS_FILE
        train_sents_file = config.SE15_REST_TRAIN_SENTS_FILE
        test_tok_texts_file = config.SE15_REST_TEST_TOK_TEXTS_FILE
        test_sents_file = config.SE15_REST_TEST_SENTS_FILE
        word_vecs_file = config.SE15_REST_YELP_WORD_VEC_FILE

__pre_train_nrdj(word_vecs_file, pre_tok_texts_file, pre_aspect_terms_file,
                 pre_opinion_terms_file, rule_model_file, 'both')
# __train_nrdj(word_vecs_file, train_tok_texts_file, train_sents_file, train_valid_split_file,
#              test_tok_texts_file, test_sents_file, rule_model_file, 'both')
# __train_nrdj_restaurant_pr()
# __train_nrdj_joint_restaurant_pr()
# __train_nrdj_mlp_restaurant_pr()
# __train_nrdj_deep_restaurant_pr()
