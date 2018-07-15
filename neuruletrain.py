import numpy as np
import pickle
import config
from collections import namedtuple
from models.lstmcrf import LSTMCRF
from utils import utils


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


def __get_data_semeval(vocab):
    labels_list_train, word_idxs_list_train = __data_from_sents_file(
        config.SE14_LAPTOP_TRAIN_SENTS_FILE, config.SE14_LAPTOP_TRAIN_TOK_TEXTS_FILE, vocab)
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


def __get_data_amazon(vocab):
    tok_texts_file = config.AMAZON_TOK_TEXTS_FILE
    terms_true_file = config.AMAZON_TERMS_TRUE_FILE
    terms_true_list = utils.load_json_objs(terms_true_file)
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
    # train_data, valid_data = __get_data_semeval(vocab)
    train_data, valid_data = __get_data_amazon(vocab)
    # model_file = 'd:/data/amazon/model/lstmcrfrule.ckpt'
    model_file = config.LAPTOP_RULE_MODEL_FILE
    print('done')
    n_tags = 3
    hidden_size_lstm = 100
    lstmcrf = LSTMCRF(n_tags, word_vecs_matrix, hidden_size_lstm=hidden_size_lstm)
    # lstmcrf = LSTMCRF(n_tags, word_vecs_matrix, model_file=model_file)
    lstmcrf.train(train_data.word_idxs_list, train_data.labels_list, valid_data.word_idxs_list,
                  valid_data.labels_list, vocab, valid_data.tok_texts, valid_data.terms_true_list,
                  n_epochs=100, save_file=model_file)


__train()
# __get_data_amazon(None)
