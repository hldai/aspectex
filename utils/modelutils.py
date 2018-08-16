from utils.utils import prf1, count_hit
import logging
import numpy as np


def filter_empty_dep_trees(trees):
    idxs_remove = set()
    for ind, tree in enumerate(trees):
        # the tree is empty
        if not tree.get_word_nodes():
            idxs_remove.add(ind)
        elif tree.get_node(0).is_word == 0:
            print(tree.get_words(), ind)
            idxs_remove.add(ind)

    keep_idxs = [idx for idx in range(len(trees)) if idx not in idxs_remove]
    return [t for i, t in enumerate(trees) if i in keep_idxs]


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


def __get_word_idx_sequence(words_list, vocab):
    seq_list = list()
    word_idx_dict = {w: i + 1 for i, w in enumerate(vocab)}
    for words in words_list:
        seq_list.append([word_idx_dict.get(w, 0) for w in words])
    return seq_list


def data_from_sents_file(sents, texts, vocab, task):
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

        x = label_sentence(sent_words, aspect_terms, opinion_terms)
        labels_list.append(x)

    word_idxs_list = __get_word_idx_sequence(words_list, vocab)

    return labels_list, word_idxs_list


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


def get_terms_from_label_list(labels, tok_text, label_beg, label_in):
    terms = list()
    words = tok_text.split(' ')
    # print(labels_pred)
    # print(len(words), len(labels_pred))
    assert len(words) == len(labels)

    p = 0
    while p < len(words):
        yi = labels[p]
        if yi == label_beg:
            pright = p
            while pright + 1 < len(words) and labels[pright + 1] == label_in:
                pright += 1
            terms.append(' '.join(words[p: pright + 1]))
            p = pright + 1
        else:
            p += 1
    return terms


def evaluate_ao_extraction(true_labels_list, pred_labels_list, test_texts, aspects_true_list,
                           opinions_ture_list=None, error_file=None):
    aspect_true_cnt, aspect_sys_cnt, aspect_hit_cnt = 0, 0, 0
    opinion_true_cnt, opinion_sys_cnt, opinion_hit_cnt = 0, 0, 0
    error_sents, error_terms_true, error_terms_sys = list(), list(), list()
    correct_sent_idxs = list()
    if aspects_true_list is not None:
        aspect_label_beg, aspect_label_in, opinion_label_beg, opinion_label_in = 1, 2, 3, 4
    else:
        aspect_label_beg, aspect_label_in, opinion_label_beg, opinion_label_in = 3, 4, 1, 2
    for sent_idx, (true_labels, pred_labels, text) in enumerate(zip(
            true_labels_list, pred_labels_list, test_texts)):
        if aspects_true_list is not None:
            aspects_true = aspects_true_list[sent_idx]
            aspect_terms_sys = get_terms_from_label_list(pred_labels, text, aspect_label_beg, aspect_label_in)

            new_hit_cnt = count_hit(aspects_true, aspect_terms_sys)
            aspect_true_cnt += len(aspects_true)
            aspect_sys_cnt += len(aspect_terms_sys)
            aspect_hit_cnt += new_hit_cnt
            if new_hit_cnt == aspect_true_cnt:
                correct_sent_idxs.append(sent_idx)

        if opinions_ture_list is None:
            continue

        opinion_terms_sys = get_terms_from_label_list(pred_labels, text, opinion_label_beg, opinion_label_in)
        opinion_terms_true = opinions_ture_list[sent_idx]

        new_hit_cnt = count_hit(opinion_terms_true, opinion_terms_sys)
        opinion_hit_cnt += new_hit_cnt
        opinion_true_cnt += len(opinion_terms_true)
        opinion_sys_cnt += len(opinion_terms_sys)

        if new_hit_cnt < len(opinion_terms_true):
            error_sents.append(text)
            error_terms_true.append(opinion_terms_true)
            error_terms_sys.append(opinion_terms_sys)

    # save_json_objs(error_sents, 'd:/data/aspect/semeval14/error-sents.txt')
    if error_file is not None:
        with open(error_file, 'w', encoding='utf-8') as fout:
            for sent, terms_true, terms_sys in zip(error_sents, error_terms_true, error_terms_sys):
                fout.write('{}\n{}\n{}\n\n'.format(sent, terms_true, terms_sys))
        logging.info('error written to {}'.format(error_file))
    # with open('d:/data/aspect/semeval14/lstmcrf-correct.txt', 'w', encoding='utf-8') as fout:
    #     fout.write('\n'.join([str(i) for i in correct_sent_idxs]))

    aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r, opinion_f1 = 0, 0, 0, 0, 0, 0
    if aspects_true_list is not None:
        aspect_p, aspect_r, aspect_f1 = prf1(aspect_true_cnt, aspect_sys_cnt, aspect_hit_cnt)

    if opinions_ture_list is not None:
        opinion_p, opinion_r, opinion_f1 = prf1(opinion_true_cnt, opinion_sys_cnt, opinion_hit_cnt)
    return aspect_p, aspect_r, aspect_f1, opinion_p, opinion_r, opinion_f1
