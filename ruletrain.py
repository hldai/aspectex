import numpy as np
import pickle
import config
import datetime
from collections import namedtuple
from models.lstmcrf import LSTMCRF
from models.nrcomb import NeuRuleComb
from models.nrdoublejoint import NeuRuleDoubleJoint
from models.nrjoint import NeuRuleJoint, NRJTrainData
from utils import utils
from utils.loggingutils import init_logging
import tensorflow as tf


TrainData = namedtuple("TrainData", ["labels_list", "word_idxs_list"])
ValidData = namedtuple("TestData", [
    "labels_list", "word_idxs_list", "tok_texts", "aspects_true_list", "opinions_true_list"])


def __find_sub_words_seq(words, sub_words):
    i, li, lj = 0, len(words), len(sub_words)
    while i + lj <= li:
        j = 0
        while j < lj:
            if words[i + j] != sub_words[j]:
                break
            j += 1
        if j == lj:
            return i
        i += 1
    return -1


def __label_words_with_terms(words, terms, label_val_beg, label_val_in, x):
    for term in terms:
        term_words = term.lower().split(' ')
        pbeg = __find_sub_words_seq(words, term_words)
        if pbeg == -1:
            # print(words)
            # print(terms)
            # print()
            continue
        x[pbeg] = label_val_beg
        for p in range(pbeg + 1, pbeg + len(term_words)):
            x[p] = label_val_in


# TODO some of the aspect_terms are not found
def __label_sentence(words, aspect_terms=None, opinion_terms=None):
    label_val_beg, label_val_in = 1, 2

    x = np.zeros(len(words), np.int32)
    if aspect_terms is not None:
        __label_words_with_terms(words, aspect_terms, label_val_beg, label_val_in, x)
        label_val_beg, label_val_in = 3, 4

    if opinion_terms is None:
        return x

    __label_words_with_terms(words, opinion_terms, label_val_beg, label_val_in, x)
    return x


def __get_word_idx_sequence(words_list, vocab):
    seq_list = list()
    word_idx_dict = {w: i + 1 for i, w in enumerate(vocab)}
    for words in words_list:
        seq_list.append([word_idx_dict.get(w, 0) for w in words])
    return seq_list


def __data_from_sents_file(filename, tok_text_file, vocab, task):
    sents = utils.load_json_objs(filename)
    texts = utils.read_lines(tok_text_file)

    words_list = [text.split(' ') for text in texts]
    len_max = max([len(words) for words in words_list])
    print('max sentence len:', len_max)

    labels_list = list()
    for sent_idx, (sent, sent_words) in enumerate(zip(sents, words_list)):
        aspect_terms, opinion_terms = None, None
        if task != 'opinion':
            aspect_objs = sent.get('terms', None)
            aspect_terms = [t['term'] for t in aspect_objs] if aspect_objs is not None else list()

        if task != 'aspect':
            opinion_terms = sent.get('opinions', list())

        x = __label_sentence(sent_words, aspect_terms, opinion_terms)
        labels_list.append(x)

    word_idxs_list = __get_word_idx_sequence(words_list, vocab)

    return labels_list, word_idxs_list


def __get_data_semeval(vocab, n_train, task):
    labels_list_train, word_idxs_list_train = __data_from_sents_file(
        config.SE14_LAPTOP_TRAIN_SENTS_FILE, config.SE14_LAPTOP_TRAIN_TOK_TEXTS_FILE, vocab, task)
    if n_train > -1:
        labels_list_train = labels_list_train[:n_train]
        word_idxs_list_train = word_idxs_list_train[:n_train]

    labels_list_test, word_idxs_list_test = __data_from_sents_file(
        config.SE14_LAPTOP_TEST_SENTS_FILE, config.SE14_LAPTOP_TEST_TOK_TEXTS_FILE, vocab, task)
    # exit()
    sents_test = utils.load_json_objs(config.SE14_LAPTOP_TEST_SENTS_FILE)
    aspect_terms_true_list_test = list() if task != 'opinion' else None
    opinion_terms_true_list_test = list() if task != 'aspect' else None
    for sent in sents_test:
        if aspect_terms_true_list_test is not None:
            aspect_terms_true_list_test.append(
                [t['term'].lower() for t in sent['terms']] if 'terms' in sent else list())
        if opinion_terms_true_list_test is not None:
            opinion_terms_true_list_test.append([w.lower() for w in sent.get('opinions', list())])
    test_texts = utils.read_lines(config.SE14_LAPTOP_TEST_TOK_TEXTS_FILE)

    train_data = TrainData(labels_list_train, word_idxs_list_train)
    valid_data = ValidData(labels_list_test, word_idxs_list_test, test_texts, aspect_terms_true_list_test,
                           opinion_terms_true_list_test)
    return train_data, valid_data


def __get_data_amazon_ao(vocab, aspect_terms_file, opinion_terms_file, tok_texts_file):
    aspect_terms_list = utils.load_json_objs(aspect_terms_file)
    opinion_terms_list = utils.load_json_objs(opinion_terms_file)
    tok_texts = utils.read_lines(tok_texts_file)
    assert len(aspect_terms_list) == len(tok_texts)
    assert len(opinion_terms_list) == len(tok_texts)

    word_idx_dict = {w: i + 1 for i, w in enumerate(vocab)}

    label_seq_list = list()
    word_idx_seq_list = list()
    for aspect_terms, opinion_terms, tok_text in zip(aspect_terms_list, opinion_terms_list, tok_texts):
        words = tok_text.split(' ')
        label_seq = __label_sentence(words, aspect_terms, opinion_terms)
        label_seq_list.append(label_seq)
        word_idx_seq_list.append([word_idx_dict.get(w, 0) for w in words])

    np.random.seed(3719)
    perm = np.random.permutation(len(label_seq_list))
    n_train = len(label_seq_list) - 2000
    idxs_train, idxs_valid = perm[:n_train], perm[n_train:]

    label_seq_list_train = [label_seq_list[idx] for idx in idxs_train]
    word_idx_seq_list_train = [word_idx_seq_list[idx] for idx in idxs_train]
    train_data = TrainData(label_seq_list_train, word_idx_seq_list_train)

    label_seq_list_valid = [label_seq_list[idx] for idx in idxs_valid]
    word_idx_seq_list_valid = [word_idx_seq_list[idx] for idx in idxs_valid]
    tok_texts_valid = [tok_texts[idx] for idx in idxs_valid]

    aspects_list_valid = [aspect_terms_list[idx] for idx in idxs_valid]
    opinions_list_valid = [opinion_terms_list[idx] for idx in idxs_valid]
    valid_data = ValidData(label_seq_list_valid, word_idx_seq_list_valid, tok_texts_valid,
                           aspects_list_valid, opinions_list_valid)

    return train_data, valid_data


def __get_data_amazon(vocab, true_terms_file, task):
    tok_texts_file = config.AMAZON_TOK_TEXTS_FILE
    terms_true_list = utils.load_json_objs(true_terms_file)
    tok_texts = utils.read_lines(tok_texts_file)
    assert len(terms_true_list) == len(tok_texts)

    word_idx_dict = {w: i + 1 for i, w in enumerate(vocab)}

    label_seq_list = list()
    word_idx_seq_list = list()
    for terms_true, tok_text in zip(terms_true_list, tok_texts):
        words = tok_text.split(' ')
        label_seq = __label_sentence(words, terms_true)
        label_seq_list.append(label_seq)
        word_idx_seq_list.append([word_idx_dict.get(w, 0) for w in words])

    np.random.seed(3719)
    perm = np.random.permutation(len(label_seq_list))
    n_train = len(label_seq_list) - 2000
    idxs_train, idxs_valid = perm[:n_train], perm[n_train:]

    label_seq_list_train = [label_seq_list[idx] for idx in idxs_train]
    word_idx_seq_list_train = [word_idx_seq_list[idx] for idx in idxs_train]
    train_data = TrainData(label_seq_list_train, word_idx_seq_list_train)

    label_seq_list_valid = [label_seq_list[idx] for idx in idxs_valid]
    word_idx_seq_list_valid = [word_idx_seq_list[idx] for idx in idxs_valid]
    tok_texts_valid = [tok_texts[idx] for idx in idxs_valid]
    terms_true_list_valid = [terms_true_list[idx] for idx in idxs_valid]
    aspect_true_list, opinion_true_list = None, None
    if task != 'opinion':
        aspect_true_list = terms_true_list_valid
    if task != 'aspect':
        opinion_true_list = terms_true_list_valid
    valid_data = ValidData(label_seq_list_valid, word_idx_seq_list_valid, tok_texts_valid,
                           aspect_true_list, opinion_true_list)

    return train_data, valid_data


def __get_manual_feat(tok_texts_file, terms_file):
    tok_texts = utils.read_lines(tok_texts_file)
    terms_list = utils.load_json_objs(terms_file)
    feat_list = list()
    for terms_true, tok_text in zip(terms_list, tok_texts):
        words = tok_text.split(' ')
        label_seq = __label_sentence(words, terms_true)
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

    train_aspect_rule_result_file = 'd:/data/aspect/semeval14/laptops-train-aspect-rule-result.txt'
    train_opinion_rule_result_file = 'd:/data/aspect/semeval14/laptops-train-opinion-rule-result.txt'
    valid_aspect_rule_result_file = 'd:/data/aspect/semeval14/laptops-test-aspect-rule-result.txt'
    valid_opinion_rule_result_file = 'd:/data/aspect/semeval14/laptops-test-opinion-rule-result.txt'

    print('loading data ...')
    with open(config.SE14_LAPTOP_GLOVE_WORD_VEC_FILE, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)
    train_data, valid_data = __get_data_semeval(vocab, -1, 'both')

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


def __train_lstmcrf():
    init_logging('log/nr-{}.log'.format(str_today), mode='a', to_stdout=True)

    # task = 'pretrain'
    task = 'train'
    # label_task_pretrain = 'opinion'
    # label_task_pretrain = 'aspect'
    label_task_pretrain = 'both'
    label_task_train = 'both'
    rule_id = 2
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

    # if rule_id == 1:
    #     rule_true_terms_file = config.AMAZON_TERMS_TRUE1_FILE
    #     rule_model_file = config.LAPTOP_RULE_MODEL1_FILE
    # elif rule_id == 2:
    #     rule_true_terms_file = config.AMAZON_TERMS_TRUE2_FILE
    #     rule_model_file = config.LAPTOP_RULE_MODEL2_FILE
    # elif rule_id == 3:
    #     rule_true_terms_file = config.AMAZON_TERMS_TRUE3_FILE
    #     rule_model_file = config.LAPTOP_RULE_MODEL3_FILE
    # else:
    #     rule_true_terms_file = config.AMAZON_TERMS_TRUE4_FILE
    #     rule_model_file = config.LAPTOP_RULE_MODEL4_FILE

    print('loading data ...')
    with open(config.SE14_LAPTOP_GLOVE_WORD_VEC_FILE, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)

    if task == 'train':
        load_model_file = rule_model_file
        save_model_file = None
        train_data, valid_data = __get_data_semeval(vocab, -1, label_task_train)
    else:
        load_model_file = None
        save_model_file = rule_model_file
        if label_task_pretrain == 'both':
            train_data, valid_data = __get_data_amazon_ao(
                vocab, aspect_terms_file, opinion_terms_file, config.AMAZON_TOK_TEXTS_FILE)
        elif label_task_pretrain == 'aspect':
            train_data, valid_data = __get_data_amazon(vocab, aspect_terms_file, label_task_pretrain)
        else:
            train_data, valid_data = __get_data_amazon(vocab, opinion_terms_file, label_task_pretrain)

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
    train_data, valid_data = __get_data_semeval(vocab)
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
    train_data_src, valid_data_src = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE2_FILE)
    nrj_train_data_src = NRJTrainData(
        train_data_src.word_idxs_list, train_data_src.labels_list, valid_data_src.word_idxs_list,
        valid_data_src.labels_list, valid_data_src.tok_texts, valid_data_src.terms_true_list
    )

    train_data_tar, valid_data_tar = __get_data_semeval(vocab, n_train)
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
    train_data_tar, valid_data_tar = __get_data_semeval(vocab, n_train, label_opinions)
    # train_data_src1, valid_data_src1 = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE1_FILE)
    # train_data_src1, valid_data_src1 = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE3_FILE)
    train_data_src1, valid_data_src1 = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE2_FILE, 'aspect')
    train_data_src2, valid_data_src2 = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE4_FILE, 'opinion')
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
__train_lstmcrf_manual_feat()
# __train_lstmcrf()
# __train_neurule_joint()
# __train_neurule_double_joint()
# __get_data_amazon(None)
