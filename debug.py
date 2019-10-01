# from seqItem import *
from utils import utils
# import tensorflow as tf
import numpy as np
# from tensorflow.contrib.crf import crf_log_likelihood, crf_log_norm
# from models import crf
import config
import xml.etree.ElementTree as ET


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


def label_sentence(text, words, word_spans, aspect_term_spans=None, opinion_terms=None):
    label_val_beg, label_val_in = 1, 2

    x = np.zeros(len(words), np.int32)
    if aspect_term_spans is not None:
        __label_words_with_terms_by_span(word_spans, aspect_term_spans, label_val_beg, label_val_in, x)
        label_val_beg, label_val_in = 3, 4

    if opinion_terms is None:
        return x

    __label_words_with_terms(words, opinion_terms, label_val_beg, label_val_in, x)
    return x


def recover_terms(text, word_spans, label_seq, label_beg, label_in):
    p = 0
    terms = list()
    while p < len(label_seq):
        if label_seq[p] == label_beg:
            pend = p + 1
            while pend < len(word_spans) and label_seq[pend] == label_in:
                pend += 1
            term_beg = word_spans[p][0]
            term_end = word_spans[pend - 1][1]
            # print(text[term_beg:term_end])
            terms.append(text[term_beg:term_end])
            p = pend
        else:
            p += 1
    return terms


test_sents = utils.load_json_objs(config.SE15R_FILES['test_sents_file'])
# terms_true_list = [s.get('opinions', list()) for s in test_sents]
terms_sys_list1 = utils.load_json_objs('d:/data/aspect/semeval15/nrdj-aspects-good.txt')
terms_sys_list2 = utils.load_json_objs('d:/data/aspect/semeval15/nrdj-aspects-bad.txt')
for terms1, terms2 in zip(terms_sys_list1, terms_sys_list2):
    hit_cnt = utils.count_hit(terms1, terms2)
    if hit_cnt < len(terms1):
        print(terms1, terms2)
