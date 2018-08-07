import numpy as np
import pickle
import config
import datetime
from models.lstmcrf import LSTMCRF
from models.nrcomb import NeuRuleComb
from models.nrdoublejoint import NeuRuleDoubleJoint
from models.nrjoint import NeuRuleJoint, NRJTrainData
from utils import utils, modelutils, datautils
from utils.loggingutils import init_logging
import tensorflow as tf


def __get_manual_feat(tok_texts_file, terms_file):
    tok_texts = utils.read_lines(tok_texts_file)
    terms_list = utils.load_json_objs(terms_file)
    feat_list = list()
    for terms_true, tok_text in zip(terms_list, tok_texts):
        words = tok_text.split(' ')
        label_seq = modelutils.label_sentence(words, terms_true)
        feat_seq = np.zeros([len(label_seq), 3], np.int32)
        for i, v in enumerate(label_seq):
            feat_seq[i][v] = 1
        feat_list.append(feat_seq)
    return feat_list


def __merge_feat_list(feat_list0, feat_list1):
    assert len(feat_list0) == len(feat_list1)
    feat_list_new = list()
    for feat0, feat1 in zip(feat_list0, feat_list1):
        assert feat0.shape[0] == feat1.shape[0]
        feat_list_new.append(np.concatenate([feat0, feat1], axis=1))
        # print(np.concatenate([feat0, feat1], axis=1))
    return feat_list_new


def __train_lstmcrf_manual_feat():
    init_logging('log/nrmf-{}.log'.format(str_today), mode='a', to_stdout=True)
    hidden_size_lstm = 100
    n_epochs = 200
    n_tags = 5

    train_aspect_rule_result_file = 'd:/data/aspect/semeval14/laptops/laptops-train-aspect-rule-result.txt'
    train_opinion_rule_result_file = 'd:/data/aspect/semeval14/laptops/laptops-train-opinion-rule-result.txt'
    valid_aspect_rule_result_file = 'd:/data/aspect/semeval14/laptops/laptops-test-aspect-rule-result.txt'
    valid_opinion_rule_result_file = 'd:/data/aspect/semeval14/laptops/laptops-test-opinion-rule-result.txt'

    print('loading data ...')
    with open(config.SE14_LAPTOP_GLOVE_WORD_VEC_FILE, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)
    train_data, valid_data = datautils.get_data_semeval(
        config.SE14_LAPTOP_TRAIN_SENTS_FILE, config.SE14_LAPTOP_TRAIN_TOK_TEXTS_FILE,
        config.SE14_LAPTOP_TEST_SENTS_FILE, config.SE14_LAPTOP_TEST_TOK_TEXTS_FILE,
        vocab, -1, 'both')

    train_aspect_feat_list = __get_manual_feat(
        config.SE14_LAPTOP_TRAIN_TOK_TEXTS_FILE, train_aspect_rule_result_file)
    train_opinion_feat_list = __get_manual_feat(
        config.SE14_LAPTOP_TRAIN_TOK_TEXTS_FILE, train_opinion_rule_result_file)
    train_feat_list = __merge_feat_list(train_aspect_feat_list, train_opinion_feat_list)

    valid_aspect_feat_list = __get_manual_feat(
        config.SE14_LAPTOP_TEST_TOK_TEXTS_FILE, valid_aspect_rule_result_file)
    valid_opinion_feat_list = __get_manual_feat(
        config.SE14_LAPTOP_TEST_TOK_TEXTS_FILE, valid_opinion_rule_result_file)
    valid_feat_list = __merge_feat_list(valid_aspect_feat_list, valid_opinion_feat_list)

    manual_feat_len = train_feat_list[0].shape[1]
    print('manual feat len: {}'.format(manual_feat_len))
    lstmcrf = LSTMCRF(n_tags, word_vecs_matrix, hidden_size_lstm=hidden_size_lstm, manual_feat_len=manual_feat_len)
    # print(valid_data.aspects_true_list)
    lstmcrf.train(train_data.word_idxs_list, train_data.labels_list, valid_data.word_idxs_list,
                  valid_data.labels_list, vocab, valid_data.tok_texts, valid_data.aspects_true_list,
                  valid_data.opinions_true_list, train_feat_list=train_feat_list, valid_feat_list=valid_feat_list,
                  n_epochs=n_epochs)


def __pretrain_lstmcrf(word_vecs_file, pre_tok_texts_file, pre_aspect_terms_file, pre_opinion_terms_file,
                       dst_model_file, task):
    init_logging('log/nr-pre-{}.log'.format(str_today), mode='a', to_stdout=True)

    n_tags = 5 if task == 'both' else 3

    print('loading data ...')
    with open(word_vecs_file, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)

    load_model_file = None
    save_model_file = dst_model_file
    if task == 'both':
        train_data, valid_data = datautils.get_data_amazon_ao(
            vocab, pre_aspect_terms_file, pre_opinion_terms_file, pre_tok_texts_file)
    elif task == 'aspect':
        train_data, valid_data = datautils.get_data_amazon(
            vocab, pre_aspect_terms_file, pre_tok_texts_file, task)
    else:
        train_data, valid_data = datautils.get_data_amazon(
            vocab, pre_opinion_terms_file, pre_tok_texts_file, task)
    print('done')

    # lstmcrf = LSTMCRF(n_tags, word_vecs_matrix, hidden_size_lstm=hidden_size_lstm)
    lstmcrf = LSTMCRF(n_tags, word_vecs_matrix, hidden_size_lstm=hidden_size_lstm, model_file=load_model_file)
    # print(valid_data.aspects_true_list)
    lstmcrf.train(train_data.word_idxs_list, train_data.labels_list, valid_data.word_idxs_list,
                  valid_data.labels_list, vocab, valid_data.tok_texts, valid_data.aspects_true_list,
                  valid_data.opinions_true_list, n_epochs=n_epochs, save_file=save_model_file)


def __train_lstmcrf(word_vecs_file, train_tok_texts_file, train_sents_file, test_tok_texts_file,
                    test_sents_file, load_model_file, task, error_file=None):
    init_logging('log/nr-{}.log'.format(str_today), mode='a', to_stdout=True)

    n_tags = 5 if task == 'both' else 3

    print('loading data ...')
    with open(word_vecs_file, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)

    save_model_file = None
    train_data, valid_data = datautils.get_data_semeval(
        train_sents_file, train_tok_texts_file,
        test_sents_file, test_tok_texts_file,
        vocab, -1, task)

    # train_data, valid_data = __get_data_semeval(vocab, -1)
    # train_data, valid_data = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE1_FILE)
    # train_data, valid_data = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE2_FILE)
    print('done')

    # lstmcrf = LSTMCRF(n_tags, word_vecs_matrix, hidden_size_lstm=hidden_size_lstm)
    lstmcrf = LSTMCRF(n_tags, word_vecs_matrix, hidden_size_lstm=hidden_size_lstm, model_file=load_model_file)
    # print(valid_data.aspects_true_list)
    lstmcrf.train(train_data.word_idxs_list, train_data.labels_list, valid_data.word_idxs_list,
                  valid_data.labels_list, vocab, valid_data.tok_texts, valid_data.aspects_true_list,
                  valid_data.opinions_true_list,
                  n_epochs=n_epochs, save_file=save_model_file, error_file=error_file)


def __train_neurule_joint():
    # n_train = 1000
    n_train = -1

    # model_file = 'd:/data/amazon/model/lstmcrfrule.ckpt'
    rule_model_file = config.LAPTOP_RULE_MODEL2_FILE
    save_model_file = None
    print('done')
    n_tags = 3
    batch_size = 20
    hidden_size_lstm = 100
    n_epochs = 500

    print('loading data ...')
    with open(config.SE14_LAPTOP_GLOVE_WORD_VEC_FILE, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)

    # train_data_src, valid_data_src = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE1_FILE)
    train_data_src, valid_data_src = datautils.get_data_amazon(vocab, config.AMAZON_TERMS_TRUE2_FILE)
    nrj_train_data_src = NRJTrainData(
        train_data_src.word_idxs_list, train_data_src.labels_list, valid_data_src.word_idxs_list,
        valid_data_src.labels_list, valid_data_src.tok_texts, valid_data_src.terms_true_list
    )

    train_data_tar, valid_data_tar = datautils.get_data_semeval(
        config.SE14_LAPTOP_TRAIN_SENTS_FILE, config.SE14_LAPTOP_TRAIN_TOK_TEXTS_FILE,
        config.SE14_LAPTOP_TEST_SENTS_FILE, config.SE14_LAPTOP_TEST_TOK_TEXTS_FILE,
        vocab, n_train)
    nrj_train_data_tar = NRJTrainData(
        train_data_tar.word_idxs_list, train_data_tar.labels_list, valid_data_tar.word_idxs_list,
        valid_data_tar.labels_list, valid_data_tar.tok_texts, valid_data_tar.terms_true_list
    )

    nrc = NeuRuleJoint(n_tags, word_vecs_matrix, hidden_size_lstm=hidden_size_lstm, model_file=None)
    nrc.train(nrj_train_data_src, nrj_train_data_tar, vocab, n_epochs=n_epochs)


def __train_neurule_double_joint():
    init_logging('log/nrdj-{}.log'.format(str_today), mode='a', to_stdout=True)

    # n_train = 1000
    n_train = -1
    # task = 'pretrain'
    task = 'train'
    label_opinions = True
    n_tags = 5 if label_opinions else 3
    # n_tags = 5 if task == 'train' else 3
    batch_size = 20
    lr = 0.001
    share_lstm = False
    train_mode = 'target-only'

    print('loading data ...')
    with open(config.SE14_LAPTOP_GLOVE_WORD_VEC_FILE, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)
    train_data_tar, valid_data_tar = datautils.get_data_semeval(
        config.SE14_LAPTOP_TRAIN_SENTS_FILE, config.SE14_LAPTOP_TRAIN_TOK_TEXTS_FILE,
        config.SE14_LAPTOP_TEST_SENTS_FILE, config.SE14_LAPTOP_TEST_TOK_TEXTS_FILE,
        vocab, n_train, label_opinions)
    # train_data_src1, valid_data_src1 = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE1_FILE)
    # train_data_src1, valid_data_src1 = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE3_FILE)
    train_data_src1, valid_data_src1 = datautils.get_data_amazon(
        vocab, config.AMAZON_TERMS_TRUE2_FILE, config.AMAZON_TOK_TEXTS_FILE, 'aspect')
    train_data_src2, valid_data_src2 = datautils.get_data_amazon(
        vocab, config.AMAZON_TERMS_TRUE4_FILE, config.AMAZON_TOK_TEXTS_FILE, 'opinion')
    rule_model_file = config.LAPTOP_NRDJ_RULE_MODEL_FILE if task == 'train' else None
    # rule_model_file = None
    pretrain_model_file = config.LAPTOP_NRDJ_RULE_MODEL_FILE
    save_model_file = config.LAPTOP_NRDJ_RULE_MODEL_FILE
    print('done')

    nrdj = NeuRuleDoubleJoint(n_tags, word_vecs_matrix, share_lstm,
                              hidden_size_lstm=hidden_size_lstm,
                              model_file=rule_model_file)

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
                       save_file=pretrain_model_file)
    if task == 'train':
        nrdj.train(nrj_train_data_src1, nrj_train_data_src2, nrj_train_data_tar, vocab, train_mode,
                   n_epochs=n_epochs, lr=lr)


str_today = datetime.date.today().strftime('%y-%m-%d')

dataset_name = 'restaurant'
method = 'lstm_crf'
hidden_size_lstm = 100
n_epochs = 200

if dataset_name == 'laptops':
    word_vecs_file = config.SE14_LAPTOP_WORD_VECS_FILE
    pre_tok_texts_file = config.AMAZON_TOK_TEXTS_FILE
    pre_aspect_terms_file = config.AMAZON_TERMS_TRUE2_FILE
    pre_opinion_terms_file = config.AMAZON_TERMS_TRUE4_FILE
    rule_model_file = config.LAPTOP_RULE_MODEL2_FILE

    train_tok_texts_file = config.SE14_LAPTOP_TRAIN_TOK_TEXTS_FILE
    train_sents_file = config.SE14_LAPTOP_TRAIN_SENTS_FILE
    test_tok_texts_file = config.SE14_LAPTOP_TEST_TOK_TEXTS_FILE
    test_sents_file = config.SE14_LAPTOP_TEST_SENTS_FILE
else:
    word_vecs_file = config.SE14_REST_GLOVE_WORD_VEC_FILE
    pre_aspect_terms_file = 'd:/data/aspect/semeval14/restaurant/yelp-aspect-rule-result-r.txt'
    # aspect_terms_file = 'd:/data/aspect/semeval14/restaurant/yelp-aspect-rule-result-r1.txt'
    pre_opinion_terms_file = 'd:/data/aspect/semeval14/restaurant/yelp-opinion-rule-result.txt'
    pre_tok_texts_file = 'd:/data/res/yelp-review-eng-tok-sents-round-9.txt'
    rule_model_file = 'd:/data/aspect/semeval14/tf-model/r1/restaurants-rule2.ckpl'

    train_tok_texts_file = config.SE14_REST_TRAIN_TOK_TEXTS_FILE
    train_sents_file = config.SE14_REST_TRAIN_SENTS_FILE
    test_tok_texts_file = config.SE14_REST_TEST_TOK_TEXTS_FILE
    test_sents_file = config.SE14_REST_TEST_SENTS_FILE

# __pretrain_lstmcrf(word_vecs_file, pre_tok_texts_file, pre_aspect_terms_file,
#                    pre_opinion_terms_file, rule_model_file, 'both')

__train_lstmcrf(word_vecs_file, train_tok_texts_file, train_sents_file, test_tok_texts_file,
                test_sents_file, rule_model_file, 'both')

# __train_lstmcrf_manual_feat()
# __train_lstm_crf_restaurant()
# __train_neurule_joint()
# __train_neurule_double_joint()
