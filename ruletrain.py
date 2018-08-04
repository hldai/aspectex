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


def __train_lstm_crf_restaurant():
    init_logging('log/nr-rest-{}.log'.format(str_today), mode='a', to_stdout=True)
    # task = 'train'
    task = 'pretrain'
    # label_task_pretrain = 'both'
    # label_task_train = 'both'
    label_task_pretrain = 'aspect'
    label_task_train = 'aspect'
    # load_pretrained_model = False
    load_pretrained_model = True
    hidden_size_lstm = 100
    n_epochs = 200
    n_tags = 3
    if label_task_pretrain == 'both' or label_task_train == 'both':
        n_tags = 5

    aspect_terms_file = 'd:/data/aspect/semeval14/restaurant/yelp-aspect-rule-result.txt'
    opinion_terms_file = 'd:/data/aspect/semeval14/restaurant/yelp-opinion-rule-result.txt'
    yelp_tok_texts_file = 'd:/data/res/yelp-review-eng-tok-sents-round-9.txt'
    rule_model_file = 'd:/data/aspect/semeval14/tf-model/r1/restaurants-rule2.ckpl'
    error_file = None

    load_model_file = None
    if task == 'train' and load_pretrained_model:
        load_model_file = rule_model_file
    # save_model_file = None if task == 'train' else rule_model_file
    save_model_file = rule_model_file if task == 'pretrain' else None

    print('loading data ...')
    with open(config.SE14_REST_GLOVE_WORD_VEC_FILE, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)
    if task == 'train':
        train_data, valid_data = datautils.get_data_semeval(
            config.SE14_REST_TRAIN_SENTS_FILE, config.SE14_REST_TRAIN_TOK_TEXTS_FILE,
            config.SE14_REST_TEST_SENTS_FILE, config.SE14_REST_TEST_TOK_TEXTS_FILE,
            vocab, -1, label_task_train)
    else:
        if label_task_pretrain == 'both':
            train_data, valid_data = datautils.get_data_amazon_ao(
                vocab, aspect_terms_file, opinion_terms_file, yelp_tok_texts_file)
        elif label_task_pretrain == 'aspect':
            train_data, valid_data = datautils.get_data_amazon(
                vocab, aspect_terms_file, yelp_tok_texts_file, label_task_pretrain)
        else:
            train_data, valid_data = datautils.get_data_amazon(
                vocab, opinion_terms_file, yelp_tok_texts_file, label_task_pretrain)
    print('done')

    print(len(vocab), 'words in vocab')
    # lstmcrf = LSTMCRF(n_tags, word_vecs_matrix, hidden_size_lstm=hidden_size_lstm)
    lstmcrf = LSTMCRF(n_tags, word_vecs_matrix, hidden_size_lstm=hidden_size_lstm, model_file=load_model_file)
    # print(valid_data.aspects_true_list)
    lstmcrf.train(train_data.word_idxs_list, train_data.labels_list, valid_data.word_idxs_list,
                  valid_data.labels_list, vocab, valid_data.tok_texts, valid_data.aspects_true_list,
                  valid_data.opinions_true_list,
                  n_epochs=n_epochs, save_file=save_model_file, error_file=error_file)


def __train_lstmcrf():
    init_logging('log/nr-{}.log'.format(str_today), mode='a', to_stdout=True)

    # task = 'pretrain'
    task = 'train'
    # label_task_pretrain = 'opinion'
    # label_task_pretrain = 'aspect'
    label_task_pretrain = 'both'
    label_task_train = 'both'
    hidden_size_lstm = 100
    n_epochs = 200
    n_tags = 3
    if label_task_pretrain == 'both' or label_task_train == 'both':
        n_tags = 5
    # n_tags = 5 if label_task == 'both' else 3
    # load_pretrained_model = False
    load_pretrained_model = True
    error_file = 'd:/data/aspect/semeval14/error-sents.txt' if task == 'train' else None
    # error_file = None

    aspect_terms_file = config.AMAZON_TERMS_TRUE2_FILE
    opinion_terms_file = config.AMAZON_TERMS_TRUE4_FILE
    rule_model_file = config.LAPTOP_RULE_MODEL2_FILE

    print('loading data ...')
    with open(config.SE14_LAPTOP_GLOVE_WORD_VEC_FILE, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)

    if task == 'train':
        load_model_file = rule_model_file
        save_model_file = None
        train_data, valid_data = datautils.get_data_semeval(
            config.SE14_LAPTOP_TRAIN_SENTS_FILE, config.SE14_LAPTOP_TRAIN_TOK_TEXTS_FILE,
            config.SE14_LAPTOP_TEST_SENTS_FILE, config.SE14_LAPTOP_TEST_TOK_TEXTS_FILE,
            vocab, -1, label_task_train)
    else:
        load_model_file = None
        save_model_file = rule_model_file
        if label_task_pretrain == 'both':
            train_data, valid_data = datautils.get_data_amazon_ao(
                vocab, aspect_terms_file, opinion_terms_file, config.AMAZON_TOK_TEXTS_FILE)
        elif label_task_pretrain == 'aspect':
            train_data, valid_data = datautils.get_data_amazon(
                vocab, aspect_terms_file, config.AMAZON_TOK_TEXTS_FILE, label_task_pretrain)
        else:
            train_data, valid_data = datautils.get_data_amazon(
                vocab, opinion_terms_file, config.AMAZON_TOK_TEXTS_FILE, label_task_pretrain)

    if not load_pretrained_model:
        load_model_file = None

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


def __train_neurule_comb():
    print('loading data ...')
    with open(config.SE14_LAPTOP_GLOVE_WORD_VEC_FILE, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)
    train_data, valid_data = datautils.get_data_semeval(vocab)
    # train_data, valid_data = __get_data_amazon(vocab)
    rule_model_file = config.LAPTOP_RULE_MODEL1_FILE
    save_model_file = None
    print('done')
    n_tags = 3
    batch_size = 20
    hidden_size_lstm = 100

    hidden_size_lstm_rule = 100
    lstmcrf = LSTMCRF(n_tags, word_vecs_matrix, batch_size=batch_size, hidden_size_lstm=hidden_size_lstm_rule,
                      model_file=rule_model_file)
    hidden_vecs_batch_list_train = lstmcrf.calc_hidden_vecs(train_data.word_idxs_list, batch_size)
    hidden_vecs_list_valid = lstmcrf.calc_hidden_vecs(valid_data.word_idxs_list, 1)
    W, b = lstmcrf.get_W_b()

    tf.reset_default_graph()
    nrc = NeuRuleComb(n_tags, word_vecs_matrix, rule_model_file, W, b, hidden_size_lstm=hidden_size_lstm,
                      hidden_size_lstm_rule=hidden_size_lstm_rule, model_file=None)
    nrc.train(train_data.word_idxs_list, train_data.labels_list, valid_data.word_idxs_list,
              valid_data.labels_list, vocab, valid_data.tok_texts, valid_data.terms_true_list,
              hidden_vecs_batch_list_train, hidden_vecs_list_valid,
              n_epochs=100, save_file=save_model_file)
    # nrc = NeuRuleJoint(n_tags, word_vecs_matrix, model_file=None)
    # nrc.train(train_data.word_idxs_list, train_data.labels_list, valid_data.word_idxs_list,
    #           valid_data.labels_list, vocab, valid_data.tok_texts, valid_data.terms_true_list,
    #           n_epochs=100, save_file=save_model_file)


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
    hidden_size_lstm = 100
    n_epochs = 200
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


def __train_nrdj_restaurant():
    init_logging('log/nrdj-restaurant-{}.log'.format(str_today), mode='a', to_stdout=True)

    # n_train = 1000
    n_train = -1
    task = 'pretrain'
    # task = 'train'
    label_opinions = True
    # label_opinions = False
    n_tags = 5 if label_opinions else 3
    # n_tags = 5 if task == 'train' else 3
    batch_size = 20
    hidden_size_lstm = 100
    n_epochs = 200
    lr = 0.001
    share_lstm = False
    load_pretrained_model = True
    train_mode = 'target-only'

    aspect_terms_file = 'd:/data/aspect/semeval14/restaurant/yelp-aspect-rule-result.txt'
    opinion_terms_file = 'd:/data/aspect/semeval14/restaurant/yelp-opinion-rule-result.txt'
    yelp_tok_texts_file = 'd:/data/res/yelp-review-tok-sents-round-9.txt'
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
        vocab, n_train, label_opinions)
    # train_data_src1, valid_data_src1 = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE1_FILE)
    # train_data_src1, valid_data_src1 = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE3_FILE)
    train_data_src1, valid_data_src1 = datautils.get_data_amazon(
        vocab, aspect_terms_file, yelp_tok_texts_file, 'aspect')
    train_data_src2, valid_data_src2 = datautils.get_data_amazon(
        vocab, opinion_terms_file, yelp_tok_texts_file, 'opinion')
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
    n_epochs = 200
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


str_today = datetime.date.today().strftime('%y-%m-%d')
# __train_lstmcrf_manual_feat()
# __train_lstmcrf()
# __train_lstm_crf_restaurant()
# __train_neurule_joint()
# __train_neurule_double_joint()
# __train_nrdj_restaurant()
# __train_nrdj_restaurant_pr()
__train_nrdj_joint_restaurant_pr()
