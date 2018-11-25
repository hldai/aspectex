import numpy as np
from collections import namedtuple
import json
from utils import utils, modelutils
import config


TrainData = namedtuple("TrainData", ["labels_list", "word_idxs_list"])
ValidData = namedtuple("ValidData", [
    "labels_list", "word_idxs_list", "tok_texts", "aspects_true_list", "opinions_true_list"])


# TODO use this function in more places
def load_terms_list(sents_file, to_lower):
    aspect_terms_list, opinion_terms_list = list(), list()
    f = open(sents_file, encoding='utf-8')
    for line in f:
        sent = json.loads(line)
        aspect_objs = sent.get('terms', None)
        aspect_terms = [t['term'].lower() for t in aspect_objs] if aspect_objs is not None else list()
        opinion_terms = sent.get('opinions', list())
        opinion_terms = [t.lower() for t in opinion_terms]

        aspect_terms_list.append(aspect_terms)
        opinion_terms_list.append(opinion_terms)
    f.close()
    return aspect_terms_list, opinion_terms_list


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
def label_sentence(words, aspect_terms=None, opinion_terms=None):
    label_val_beg, label_val_in = 1, 2

    x = np.zeros(len(words), np.int32)
    if aspect_terms is not None:
        __label_words_with_terms(words, aspect_terms, label_val_beg, label_val_in, x)
        label_val_beg, label_val_in = 3, 4

    if opinion_terms is None:
        return x

    __label_words_with_terms(words, opinion_terms, label_val_beg, label_val_in, x)
    return x


# TODO EMPTY and UNKNOWN tokens for vocab
def __get_word_idx_sequence(words_list, vocab):
    seq_list = list()
    word_idx_dict = {w: i + 1 for i, w in enumerate(vocab)}
    for words in words_list:
        seq_list.append([word_idx_dict.get(w, 0) for w in words])
    return seq_list


def data_from_sents_file(sents, tok_texts, vocab, task):
    words_list = [text.split(' ') for text in tok_texts]
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

        x = label_sentence(sent_words, aspect_terms, opinion_terms)
        labels_list.append(x)

    word_idxs_list = __get_word_idx_sequence(words_list, vocab)

    return labels_list, word_idxs_list


def read_sents_to_word_idx_seqs(tok_texts_file, word_idx_dict):
    texts = utils.read_lines(tok_texts_file)
    word_idx_seq_list = list()
    for sent_text in texts:
        words = sent_text.strip().split(' ')
        word_idx_seq_list.append([word_idx_dict.get(w, 0) for w in words])
    return word_idx_seq_list


def __get_valid_data(sents, tok_texts, vocab, task):
    labels_list_test, word_idxs_list_test = data_from_sents_file(sents, tok_texts, vocab, task)
    # exit()

    aspect_terms_true_list = list() if task != 'opinion' else None
    opinion_terms_true_list = list() if task != 'aspect' else None
    for sent in sents:
        if aspect_terms_true_list is not None:
            aspect_terms_true_list.append(
                [t['term'].lower() for t in sent['terms']] if 'terms' in sent else list())
        if opinion_terms_true_list is not None:
            opinion_terms_true_list.append([w.lower() for w in sent.get('opinions', list())])

    return ValidData(labels_list_test, word_idxs_list_test, tok_texts, aspect_terms_true_list,
                     opinion_terms_true_list)


def get_data_semeval(train_sents_file, train_tok_text_file, train_valid_split_file, test_sents_file,
                     test_tok_text_file, vocab, n_train, task):
    tvs_line = utils.read_lines(train_valid_split_file)[0]
    tvs_arr = [int(v) for v in tvs_line.split()]

    sents = utils.load_json_objs(train_sents_file)
    texts = utils.read_lines(train_tok_text_file)

    sents_train, texts_train, sents_valid, texts_valid = list(), list(), list(), list()
    for label, s, t in zip(tvs_arr, sents, texts):
        if label == 0:
            sents_train.append(s)
            texts_train.append(t)
        else:
            sents_valid.append(s)
            texts_valid.append(t)

    labels_list_train, word_idxs_list_train = data_from_sents_file(sents_train, texts_train, vocab, task)
    if n_train > -1:
        labels_list_train = labels_list_train[:n_train]
        word_idxs_list_train = word_idxs_list_train[:n_train]

    train_data = TrainData(labels_list_train, word_idxs_list_train)

    valid_data = __get_valid_data(sents_valid, texts_valid, vocab, task)

    sents_test = utils.load_json_objs(test_sents_file)
    texts_test = utils.read_lines(test_tok_text_file)
    test_data = __get_valid_data(sents_test, texts_test, vocab, task)
    return train_data, valid_data, test_data


def get_data_amazon_ao(vocab, aspect_terms_file, opinion_terms_file, tok_texts_file):
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
        label_seq = label_sentence(words, aspect_terms, opinion_terms)
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
    valid_data = ValidData(
        label_seq_list_valid, word_idx_seq_list_valid, tok_texts_valid, aspects_list_valid, opinions_list_valid)

    return train_data, valid_data


def get_data_amazon(vocab, true_terms_file, tok_texts_file, task):
    terms_true_list = utils.load_json_objs(true_terms_file)
    tok_texts = utils.read_lines(tok_texts_file)
    # print(len(terms_true_list), tok_texts_file, len(tok_texts))
    assert len(terms_true_list) == len(tok_texts)

    word_idx_dict = {w: i + 1 for i, w in enumerate(vocab)}

    label_seq_list = list()
    word_idx_seq_list = list()
    for terms_true, tok_text in zip(terms_true_list, tok_texts):
        words = tok_text.split(' ')
        label_seq = label_sentence(words, terms_true)
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
    valid_data = ValidData(
        label_seq_list_valid, word_idx_seq_list_valid, tok_texts_valid, aspect_true_list, opinion_true_list)

    return train_data, valid_data
