import numpy as np
from collections import namedtuple
import json
from utils import utils, modelutils
import config


TrainData = namedtuple("TrainData", ["labels_list", "word_idxs_list"])
ValidData = namedtuple("ValidData", ["texts", "labels_list", "word_idxs_list", "word_span_seqs", "tok_texts",
                                     "aspects_true_list", "opinions_true_list"])


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


def __label_words_with_terms_by_span(word_spans, term_spans, label_val_beg, label_val_in, x):
    for term_span in term_spans:
        is_first = True
        for i, wspan in enumerate(word_spans):
            if wspan[0] >= term_span[0] and wspan[1] <= term_span[1]:
                if is_first:
                    is_first = False
                    x[i] = label_val_beg
                else:
                    x[i] = label_val_in
            if wspan[0] > term_span[1]:
                break


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


# TODO some of the aspect_terms are not found
def label_sentence_by_span(words, word_spans, aspect_term_spans=None, opinion_terms=None):
    label_val_beg, label_val_in = 1, 2

    x = np.zeros(len(words), np.int32)
    if aspect_term_spans is not None:
        __label_words_with_terms_by_span(word_spans, aspect_term_spans, label_val_beg, label_val_in, x)
        label_val_beg, label_val_in = 3, 4

    if opinion_terms is None:
        return x

    __label_words_with_terms(words, opinion_terms, label_val_beg, label_val_in, x)
    return x


# TODO EMPTY and UNKNOWN tokens for vocab
def __get_word_idx_sequence(words_list, vocab):
    seq_list = list()
    word_idx_dict = {w: i + 1 for i, w in enumerate(vocab)}
    unk_id = len(vocab) - 1
    for words in words_list:
        seq_list.append([word_idx_dict.get(w, unk_id) for w in words])
    return seq_list


def data_from_sents_file(sents, tok_texts, word_span_seqs, vocab, task):
    words_list = [text.split(' ') for text in tok_texts]
    len_max = max([len(words) for words in words_list])
    print('max sentence len:', len_max)

    labels_list = list()
    for sent_idx, (sent, sent_words) in enumerate(zip(sents, words_list)):
        aspect_term_spans, aspect_terms, opinion_terms = None, None, None
        if task != 'opinion':
            aspect_objs = sent.get('terms', list())
            # aspect_terms = [t['term'] for t in aspect_objs]
            aspect_term_spans = [t['span'] for t in aspect_objs]

        if task != 'aspect':
            opinion_terms = sent.get('opinions', list())

        x = label_sentence_by_span(sent_words, word_span_seqs[sent_idx], aspect_term_spans, opinion_terms)
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


def __get_valid_data(sents, tok_texts, word_span_seqs, vocab, task):
    labels_list_test, word_idxs_list_test = data_from_sents_file(sents, tok_texts, word_span_seqs, vocab, task)
    # exit()

    aspect_terms_true_list = list() if task != 'opinion' else None
    opinion_terms_true_list = list() if task != 'aspect' else None
    texts = list()
    for sent in sents:
        texts.append(sent['text'])
        if aspect_terms_true_list is not None:
            aspect_terms_true_list.append(
                [t['term'].lower() for t in sent['terms']] if 'terms' in sent else list())
        if opinion_terms_true_list is not None:
            opinion_terms_true_list.append([w.lower() for w in sent.get('opinions', list())])

    return ValidData(texts, labels_list_test, word_idxs_list_test, word_span_seqs, tok_texts, aspect_terms_true_list,
                     opinion_terms_true_list)


def get_data_semeval(train_sents_file, train_tok_text_file, train_valid_split_file, test_sents_file,
                     test_tok_text_file, vocab, n_train, task):
    tvs_line = utils.read_lines(train_valid_split_file)[0]
    tvs_arr = [int(v) for v in tvs_line.split()]

    sents = utils.load_json_objs(train_sents_file)
    # texts = utils.read_lines(train_tok_text_file)
    tok_texts, word_span_seqs = load_token_pos_file(train_tok_text_file)

    sents_train, tok_texts_train, sents_valid, tok_texts_valid = list(), list(), list(), list()
    word_span_seqs_train, word_span_seqs_valid = list(), list()
    for label, s, t, span_seq in zip(tvs_arr, sents, tok_texts, word_span_seqs):
        if label == 0:
            sents_train.append(s)
            tok_texts_train.append(t)
            word_span_seqs_train.append(span_seq)
        else:
            sents_valid.append(s)
            tok_texts_valid.append(t)
            word_span_seqs_valid.append(span_seq)

    labels_list_train, word_idxs_list_train = data_from_sents_file(
        sents_train, tok_texts_train, word_span_seqs_train, vocab, task)
    if n_train > -1:
        labels_list_train = labels_list_train[:n_train]
        word_idxs_list_train = word_idxs_list_train[:n_train]

    train_data = TrainData(labels_list_train, word_idxs_list_train)

    valid_data = __get_valid_data(sents_valid, tok_texts_valid, word_span_seqs_valid, vocab, task)

    sents_test = utils.load_json_objs(test_sents_file)
    texts_test, word_span_seqs_test = load_token_pos_file(test_tok_text_file)
    test_data = __get_valid_data(sents_test, texts_test, word_span_seqs_test, vocab, task)
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


def read_tokens_file(tokens_file):
    token_seqs = list()
    with open(tokens_file, encoding='utf-8') as f:
        for line in f:
            token_seqs.append(line.strip().split(' '))
    return token_seqs


def gen_train_valid_sample_idxs_file(tok_texts_file, n_valid_samples, output_file):
    tok_texts = utils.read_lines(tok_texts_file)
    n_samples = len(tok_texts)
    np.random.seed(3719)
    perm = np.random.permutation(n_samples)
    n_train = n_samples - n_valid_samples
    idxs_train, idxs_valid = perm[:n_train], perm[n_train:]
    with open(output_file, 'w', encoding='utf-8') as fout:
        fout.write('{}\n'.format(' '.join([str(idx) for idx in idxs_train])))
        fout.write('{}\n'.format(' '.join([str(idx) for idx in idxs_valid])))


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
        None, label_seq_list_valid, word_idx_seq_list_valid, None, tok_texts_valid,
        aspect_true_list, opinion_true_list)

    return train_data, valid_data


def gen_yelp_review_sents(yelp_review_file, dst_file, biz_id_set=None):
    import re
    import nltk
    f = open(yelp_review_file, encoding='utf-8')
    fout = open(dst_file, 'w', encoding='utf-8', newline='\n')
    for i, line in enumerate(f):
        review = json.loads(line)
        if biz_id_set is not None and review['business_id'] not in biz_id_set:
            continue

        review_text = review['text']
        sents = nltk.sent_tokenize(review_text)
        # print(sents)
        for sent in sents:
            sent = sent.strip()
            if not sent:
                continue
            sent = re.sub(r'\s+', ' ', sent)
            fout.write('{}\n'.format(sent))
        if i % 10000 == 0:
            print(i)
        # if i > 10000:
        #     break
    f.close()
    fout.close()


def get_yelp_restaurant_reviews(yelp_review_file, yelp_biz_file, dst_file):
    restaurant_categories = {'Restaurants', 'Diners', 'Mexican', 'Fast Food', 'Food'}
    restaurant_biz_set = set()
    with open(yelp_biz_file, encoding='utf-8') as f:
        for i, line in enumerate(f):
            biz = json.loads(line)
            # print(biz)
            is_restaurant = False
            biz_categories = biz.get('categories', list())
            if not biz_categories:
                continue
            for cat in biz_categories:
                if cat in restaurant_categories:
                    is_restaurant = True
                    break
            if is_restaurant:
                restaurant_biz_set.add(biz['business_id'])

    gen_yelp_review_sents(yelp_review_file, dst_file, restaurant_biz_set)


def load_train_valid_idxs(train_valid_idxs_file):
    with open(train_valid_idxs_file, encoding='utf-8') as f:
        train_idxs = next(f).strip().split(' ')
        train_idxs = [int(idx) for idx in train_idxs]
        valid_idxs = next(f).strip().split(' ')
        valid_idxs = [int(idx) for idx in valid_idxs]
    return train_idxs, valid_idxs


def load_token_pos_file(filename):
    tok_texts, tok_span_seqs = list(), list()
    with open(filename, encoding='utf-8') as f:
        for line in f:
            tok_texts.append(line.strip())
            tok_spans_str = next(f).strip()
            vals = [int(v) for v in tok_spans_str.split(' ')]
            tok_span_seqs.append([(vals[2 * i], vals[2 * i + 1]) for i in range(len(vals) // 2)])
    return tok_texts, tok_span_seqs


def __add_unk_word(word_vecs_matrix):
    n_words = word_vecs_matrix.shape[0]
    dim = word_vecs_matrix.shape[1]
    word_vecs = np.zeros((n_words + 1, dim), np.float32)
    for i in range(n_words):
        word_vecs[i] = word_vecs_matrix[i]
    word_vecs[n_words] = np.random.normal(0, 0.1, dim)
    print(word_vecs[n_words])
    return word_vecs


def load_word_vecs(word_vecs_file, add_unknown_word=False):
    import pickle
    with open(word_vecs_file, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)
    if add_unknown_word:
        word_vecs_matrix = __add_unk_word(word_vecs_matrix)
        vocab.append('<UNK>')
    return vocab, word_vecs_matrix
