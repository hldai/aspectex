import numpy as np
import pickle
import config
from utils import utils


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

    label_seqs = list()
    for sent_idx, (sent, sent_words) in enumerate(zip(sents, words_list)):
        aspect_objs = sent.get('terms', None)
        aspect_terms = [t['term'] for t in aspect_objs] if aspect_objs is not None else list()
        x = __label_sentence(sent_words, aspect_terms)
        label_seqs.append(x)

    word_idxs_list = __get_word_idx_sequence(words_list, vocab)

    return label_seqs, word_idxs_list


def __get_data(word_vecs_file, sents_files):
    with open(word_vecs_file, 'rb') as f:
        vocab, word_vecs_matrix = pickle.load(f)

    label_seqs, word_idxs_list = __data_from_sents_file(
        config.SE14_LAPTOP_TRAIN_SENTS_FILE, config.SE14_LAPTOP_TRAIN_TOK_TEXTS_FILE, vocab)


__get_data(config.SE14_LAPTOP_GLOVE_WORD_VEC_FILE, [])
