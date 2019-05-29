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


task = 'both'
dataset_files = config.DATA_FILES['se14l']
# train_sents_file = dataset_files['test_sents_file']
# train_tok_text_file = 'd:/data/aspect/semeval14/laptops/laptops_test_texts_tok_pos.txt'
train_sents_file = dataset_files['train_sents_file']
train_tok_text_file = dataset_files['train_tok_texts_file']
sents = utils.load_json_objs(train_sents_file)
# tok_texts = utils.read_lines(train_tok_text_file)

tok_texts, tok_span_seqs = list(), list()
with open(train_tok_text_file, encoding='utf-8') as f:
    for line in f:
        tok_texts.append(line.strip())
        tok_spans_str = next(f).strip()
        vals = [int(v) for v in tok_spans_str.split(' ')]
        tok_span_seqs.append([(vals[2 * i], vals[2 * i + 1]) for i in range(len(vals) // 2)])

words_list = [text.split(' ') for text in tok_texts]
len_max = max([len(words) for words in words_list])
print('max sentence len:', len_max)


labels_list = list()
for sent_idx, (sent, sent_words) in enumerate(zip(sents, words_list)):
    aspect_terms, opinion_terms = None, None
    aspect_term_spans = None
    if task != 'opinion':
        aspect_objs = sent.get('terms', list())
        aspect_terms = [t['term'] for t in aspect_objs]
        aspect_term_spans = [t['span'] for t in aspect_objs]
        # for x in aspect_objs:
        #     text = sent['text'].replace(' ', ' ')
        #     span = x['span']
        #     # print(x)
        #     # print(sent['text'][span[0]:span[1]])
        #     if text[span[0]:span[1]] != x['term']:
        #         print(x)
        #         print(sent['text'][span[0]:span[1]])

    if task != 'aspect':
        opinion_terms = sent.get('opinions', list())

    sent_text = sent['text']
    tok_spans = tok_span_seqs[sent_idx]
    x = label_sentence(sent['text'], sent_words, tok_spans, aspect_term_spans, opinion_terms)
    aspect_terms_rec = recover_terms(sent['text'], tok_spans, x, 1, 2)
    for t in aspect_terms:
        if t not in aspect_terms_rec:
            print('**', t, aspect_terms_rec)
            print(sent_text)
            print(sent_words)
            print(tok_spans)
            print()
    # for i, v in enumerate(x):
    #     if 3 > v > 0:
    #         print(sent_words[i])
