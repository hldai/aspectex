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
ValidData = namedtuple("TestData", ["labels_list", "word_idxs_list", "tok_texts", "terms_true_list"])


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


# TODO some of the aspect_terms are not found
def __label_sentence(words, aspect_terms):
    x = np.zeros(len(words), np.int32)
    for term in aspect_terms:
        term_words = term.lower().split(' ')
        pbeg = __find_sub_words_seq(words, term_words)
        if pbeg == -1:
            # print(term)
            # print(words)
            # print()
            continue
        x[pbeg] = 1
        for p in range(pbeg + 1, pbeg + len(term_words)):
            x[p] = 2
    return x


def __get_word_idx_sequence(words_list, vocab):
    seq_list = list()
    word_idx_dict = {w: i + 1 for i, w in enumerate(vocab)}
    for words in words_list:
        seq_list.append([word_idx_dict.get(w, 0) for w in words])
    return seq_list


def __data_from_sents_file(filename, tok_text_file, vocab):
    sents = utils.load_json_objs(filename)
    texts = utils.read_lines(tok_text_file)

    words_list = [text.split(' ') for text in texts]
    len_max = max([len(words) for words in words_list])
    print('max sentence len:', len_max)

    labels_list = list()
    for sent_idx, (sent, sent_words) in enumerate(zip(sents, words_list)):
        aspect_objs = sent.get('terms', None)
        aspect_terms = [t['term'] for t in aspect_objs] if aspect_objs is not None else list()
        x = __label_sentence(sent_words, aspect_terms)
        labels_list.append(x)

    word_idxs_list = __get_word_idx_sequence(words_list, vocab)

    return labels_list, word_idxs_list


def __get_data_semeval(vocab, n_train):
    labels_list_train, word_idxs_list_train = __data_from_sents_file(
        config.SE14_LAPTOP_TRAIN_SENTS_FILE, config.SE14_LAPTOP_TRAIN_TOK_TEXTS_FILE, vocab)
    if n_train > -1:
        labels_list_train = labels_list_train[:n_train]
        word_idxs_list_train = word_idxs_list_train[:n_train]

    labels_list_test, word_idxs_list_test = __data_from_sents_file(
        config.SE14_LAPTOP_TEST_SENTS_FILE, config.SE14_LAPTOP_TEST_TOK_TEXTS_FILE, vocab)
    sents_test = utils.load_json_objs(config.SE14_LAPTOP_TEST_SENTS_FILE)
    terms_true_list_test = list()
    for sent in sents_test:
        terms_true_list_test.append([t['term'].lower() for t in sent['terms']] if 'terms' in sent else list())
    test_texts = utils.read_lines(config.SE14_LAPTOP_TEST_TOK_TEXTS_FILE)

    train_data = TrainData(labels_list_train, word_idxs_list_train)
    valid_data = ValidData(labels_list_test, word_idxs_list_test, test_texts, terms_true_list_test)
    return train_data, valid_data


def __get_data_amazon(vocab, true_terms_file):
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
    valid_data = ValidData(label_seq_list_valid, word_idx_seq_list_valid, tok_texts_valid, terms_true_list_valid)

    return train_data, valid_data


def __train():
    print('loading data ...')
    with open(config.SE14_LAPTOP_GLOVE_WORD_VEC_FILE, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)
    train_data, valid_data = __get_data_semeval(vocab, -1)
    # train_data, valid_data = __get_data_amazon(vocab)
    # model_file = 'd:/data/amazon/model/lstmcrfrule.ckpt'
    model_file = config.LAPTOP_RULE_MODEL_FILE
    save_model_file = None
    print('done')
    n_tags = 3
    hidden_size_lstm = 100
    # lstmcrf = LSTMCRF(n_tags, word_vecs_matrix, hidden_size_lstm=hidden_size_lstm)
    lstmcrf = LSTMCRF(n_tags, word_vecs_matrix, hidden_size_lstm=hidden_size_lstm, model_file=model_file)
    lstmcrf.train(train_data.word_idxs_list, train_data.labels_list, valid_data.word_idxs_list,
                  valid_data.labels_list, vocab, valid_data.tok_texts, valid_data.terms_true_list,
                  n_epochs=100, save_file=save_model_file)


def __train_neurule_comb():
    print('loading data ...')
    with open(config.SE14_LAPTOP_GLOVE_WORD_VEC_FILE, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)
    train_data, valid_data = __get_data_semeval(vocab)
    # train_data, valid_data = __get_data_amazon(vocab)
    # model_file = 'd:/data/amazon/model/lstmcrfrule.ckpt'
    rule_model_file = config.LAPTOP_RULE_MODEL_FILE
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

    print('loading data ...')
    with open(config.SE14_LAPTOP_GLOVE_WORD_VEC_FILE, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)
    train_data_tar, valid_data_tar = __get_data_semeval(vocab, n_train)
    train_data_src, valid_data_src = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE1_FILE)
    # train_data_src, valid_data_src = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE2_FILE)
    # model_file = 'd:/data/amazon/model/lstmcrfrule.ckpt'
    rule_model_file = config.LAPTOP_RULE_MODEL_FILE
    save_model_file = None
    print('done')
    n_tags = 3
    batch_size = 20
    hidden_size_lstm = 100
    n_epochs = 500
    share_W = False
    nrc = NeuRuleJoint(n_tags, word_vecs_matrix, share_W, hidden_size_lstm=hidden_size_lstm, model_file=None)
    nrj_train_data_src = NRJTrainData(
        train_data_src.word_idxs_list, train_data_src.labels_list, valid_data_src.word_idxs_list,
        valid_data_src.labels_list, valid_data_src.tok_texts, valid_data_src.terms_true_list
    )
    nrj_train_data_tar = NRJTrainData(
        train_data_tar.word_idxs_list, train_data_tar.labels_list, valid_data_tar.word_idxs_list,
        valid_data_tar.labels_list, valid_data_tar.tok_texts, valid_data_tar.terms_true_list
    )
    nrc.train(nrj_train_data_src, nrj_train_data_tar, vocab, n_epochs=n_epochs)


def __train_neurule_double_joint():
    init_logging('log/nrdj-{}.log'.format(str_today), mode='a', to_stdout=True)

    # n_train = 1000
    n_train = -1

    print('loading data ...')
    with open(config.SE14_LAPTOP_GLOVE_WORD_VEC_FILE, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)
    train_data_tar, valid_data_tar = __get_data_semeval(vocab, n_train)
    train_data_src1, valid_data_src1 = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE1_FILE)
    train_data_src2, valid_data_src2 = __get_data_amazon(vocab, config.AMAZON_TERMS_TRUE2_FILE)
    # model_file = 'd:/data/amazon/model/lstmcrfrule.ckpt'
    rule_model_file = config.LAPTOP_NRDJ_RULE_MODEL_FILE
    save_model_file = None
    print('done')
    n_tags = 3
    batch_size = 20
    hidden_size_lstm = 100
    n_epochs = 500
    lr = 0.01
    nrdj = NeuRuleDoubleJoint(n_tags, word_vecs_matrix, hidden_size_lstm=hidden_size_lstm,
                              model_file=None)

    nrj_train_data_src1 = NRJTrainData(
        train_data_src1.word_idxs_list, train_data_src1.labels_list, valid_data_src1.word_idxs_list,
        valid_data_src1.labels_list, valid_data_src1.tok_texts, valid_data_src1.terms_true_list
    )
    nrj_train_data_src2 = NRJTrainData(
        train_data_src2.word_idxs_list, train_data_src2.labels_list, valid_data_src2.word_idxs_list,
        valid_data_src2.labels_list, valid_data_src2.tok_texts, valid_data_src2.terms_true_list
    )

    nrdj.pre_train(nrj_train_data_src1, nrj_train_data_src2, vocab, n_epochs=n_epochs, lr=lr,
                   save_file=rule_model_file)

    # nrj_train_data_tar = NRJTrainData(
    #     train_data_tar.word_idxs_list, train_data_tar.labels_list, valid_data_tar.word_idxs_list,
    #     valid_data_tar.labels_list, valid_data_tar.tok_texts, valid_data_tar.terms_true_list
    # )
    # nrdj.train(nrj_train_data_src1, nrj_train_data_src2, nrj_train_data_tar, vocab, n_epochs=n_epochs, lr=lr)


str_today = datetime.date.today().strftime('%y-%m-%d')
# __train()
# __train_neurule_joint()
__train_neurule_double_joint()
# __get_data_amazon(None)
