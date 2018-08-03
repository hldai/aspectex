import numpy as np
import pickle
import tensorflow as tf
import config
from models import rules
from utils import utils


def __gen_aspect_noun_filter_dict_file(sents_file, tok_texts_file, pos_tags_file, common_words_file, dst_file):
    sents = utils.load_json_objs(sents_file)
    tok_texts = utils.read_lines(tok_texts_file)
    pos_tags_list = utils.load_pos_tags(pos_tags_file)
    term_sys_cnts, term_hit_cnts = dict(), dict()
    for sent_idx, (sent, tok_text, pos_tags) in enumerate(zip(sents, tok_texts, pos_tags_list)):
        sent_words = tok_text.split(' ')
        noun_phrases = rules.rec_rule1(sent_words, pos_tags, None)
        term_objs = sent.get('terms', list())
        terms_true = {term_obj['term'].lower() for term_obj in term_objs}
        for n in noun_phrases:
            sys_cnt = term_sys_cnts.get(n, 0)
            term_sys_cnts[n] = sys_cnt + 1
            if n in terms_true:
                hit_cnt = term_hit_cnts.get(n, 0)
                term_hit_cnts[n] = hit_cnt + 1

    common_words = utils.read_lines(common_words_file)
    filter_terms = set(common_words)
    for term, sys_cnt in term_sys_cnts.items():
        hit_cnt = term_hit_cnts.get(term, 0)
        # print(term, hit_cnt, sys_cnt)
        if hit_cnt / sys_cnt < 0.4:
            filter_terms.add(term)

    fout = open(dst_file, 'w', encoding='utf-8', newline='\n')
    for t in filter_terms:
        fout.write('{}\n'.format(t))
    fout.close()


sents_file = config.SE14_REST_TRAIN_SENTS_FILE
tok_texts_file = config.SE14_REST_TRAIN_TOK_TEXTS_FILE
common_words_file = 'd:/data/aspect/common-words.txt'
pos_tags_file = 'd:/data/aspect/semeval14/restaurant/restaurants-train-rule-pos.txt'
rest_aspect_nouns_filter_file = 'd:/data/aspect/semeval14/restaurant/aspect-nouns-filter.txt'
__gen_aspect_noun_filter_dict_file(
    sents_file, tok_texts_file, pos_tags_file, common_words_file, rest_aspect_nouns_filter_file)
