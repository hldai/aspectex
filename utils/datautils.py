import numpy as np
from collections import namedtuple
from utils import utils, modelutils
import config


TrainData = namedtuple("TrainData", ["labels_list", "word_idxs_list"])
ValidData = namedtuple("TestData", [
    "labels_list", "word_idxs_list", "tok_texts", "aspects_true_list", "opinions_true_list"])


def get_data_semeval(train_sents_file, train_tok_text_file, test_sents_file,
                     test_tok_text_file, vocab, n_train, task):
    sents = utils.load_json_objs(train_sents_file)
    texts = utils.read_lines(train_tok_text_file)
    labels_list_train, word_idxs_list_train = modelutils.data_from_sents_file(sents, texts, vocab, task)
    if n_train > -1:
        labels_list_train = labels_list_train[:n_train]
        word_idxs_list_train = word_idxs_list_train[:n_train]

    sents_test = utils.load_json_objs(test_sents_file)
    texts_test = utils.read_lines(test_tok_text_file)
    labels_list_test, word_idxs_list_test = modelutils.data_from_sents_file(sents_test, texts_test, vocab, task)
    # exit()

    aspect_terms_true_list_test = list() if task != 'opinion' else None
    opinion_terms_true_list_test = list() if task != 'aspect' else None
    for sent in sents_test:
        if aspect_terms_true_list_test is not None:
            aspect_terms_true_list_test.append(
                [t['term'].lower() for t in sent['terms']] if 'terms' in sent else list())
        if opinion_terms_true_list_test is not None:
            opinion_terms_true_list_test.append([w.lower() for w in sent.get('opinions', list())])

    train_data = TrainData(labels_list_train, word_idxs_list_train)
    valid_data = ValidData(labels_list_test, word_idxs_list_test, texts_test, aspect_terms_true_list_test,
                           opinion_terms_true_list_test)
    return train_data, valid_data


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
        label_seq = modelutils.label_sentence(words, aspect_terms, opinion_terms)
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
        label_seq = modelutils.label_sentence(words, terms_true)
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
