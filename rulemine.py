import config
from utils import utils
import pandas as pd


NOUN_POS_TAGS = {'NN', 'NNP', 'NNS', 'NNPS'}
VB_POS_TAGS = {'VB', 'VBN', 'VBP', 'VBZ', 'VBG', 'VBD'}


def __find_phrase_word_idx_span(phrase, sent_words):
    phrase_words = phrase.split()
    pleft = 0
    while pleft + len(phrase_words) <= len(sent_words):
        p = pleft
        while p - pleft < len(phrase_words) and sent_words[p] == phrase_words[p - pleft]:
            p += 1
        if p - pleft == len(phrase_words):
            return pleft, p
        pleft += 1
    return None


def __find_related_l2_dep_tags(related_dep_tag_idxs, dep_tags, pos_tags, sent_words, term_word_idx_span):
    related = list()
    for i in related_dep_tag_idxs:
        rel, gov, dep = dep_tags[i]
        igov, wgov = gov
        idep, wdep = dep
        for j, dep_tag_j in enumerate(dep_tags):
            if i == j:
                continue
            rel_j, gov_j, dep_j = dep_tag_j
            igov_j, wgov_j = gov_j
            idep_j, wdep_j = dep_j
            if igov == igov_j or igov == idep_j or idep == igov_j or idep == idep_j:
                # print(dep_tags[i], dep_tag_j)
                related.append((dep_tags[i], dep_tag_j))
    return related


def __patterns_from_l1_dep_tags(aspect_word_wc, related_dep_tags, pos_tags, term_word_idx_span, opinion_terms):
    widx_beg, widx_end = term_word_idx_span
    # print(related_dep_tags)
    patterns = set()
    for dep_tag in related_dep_tags:
        rel, gov, dep = dep_tag
        igov, wgov = gov
        idep, wdep = dep
        if widx_beg <= igov < widx_end:
            patterns.add((rel, aspect_word_wc, wdep))
            patterns.add((rel, aspect_word_wc, pos_tags[idep]))
            if wdep in opinion_terms:
                patterns.add((rel, aspect_word_wc, '_OP'))
        elif widx_beg <= idep < widx_end:
            patterns.add((rel, wgov, aspect_word_wc))
            patterns.add((rel, pos_tags[igov], aspect_word_wc))
            if wgov in opinion_terms:
                patterns.add((rel, '_OP', aspect_word_wc))
        else:
            patterns.add((rel, wgov, wdep))
            patterns.add((rel, pos_tags[igov], wdep))
            patterns.add((rel, wgov, pos_tags[idep]))
            if wgov in opinion_terms:
                patterns.add((rel, '_OP', wdep))
                patterns.add((rel, '_OP', pos_tags[idep]))
            if wdep in opinion_terms:
                patterns.add((rel, wgov, '_OP'))
                patterns.add((rel, pos_tags[igov], '_OP'))
    return patterns


def __patterns_from_l2_dep_tags(aspect_word_wc, related_dep_tag_tups, pos_tags, term_word_idx_span, opinion_terms):
    # widx_beg, widx_end = term_word_idx_span
    patterns = set()
    for dep_tag_i, dep_tag_j in related_dep_tag_tups:
        patterns_i = __patterns_from_l1_dep_tags(
            aspect_word_wc, [dep_tag_i], pos_tags, term_word_idx_span, opinion_terms)
        patterns_j = __patterns_from_l1_dep_tags(
            aspect_word_wc, [dep_tag_j], pos_tags, term_word_idx_span, opinion_terms)
        for pi in patterns_i:
            for pj in patterns_j:
                if pi < pj:
                    patterns.add((pi, pj))
                else:
                    patterns.add((pj, pi))
    return patterns


def __word_legal_by_freq(w: str, word_freq_dict):
    return w == '_A' or w == '_OP' or w.isupper() or word_freq_dict.get(w, 0) > 10


def __l1_pattern_legal(pattern, word_freq_dict):
    rel, gov, dep = pattern
    if gov != '_A' and gov != '_OP' and not gov.isupper() and word_freq_dict.get(gov, 0) < 10:
        return False
    if dep != '_A' and dep != '_OP' and not dep.isupper() and word_freq_dict.get(dep, 0) < 10:
        return False
    return True


def __filter_l1_patterns(patterns, word_freq_dict):
    patterns_new = list()
    for p in patterns:
        if __l1_pattern_legal(p, word_freq_dict):
            patterns_new.append(p)
    return patterns_new


def __filter_l2_patterns(patterns, word_freq_dict):
    patterns_new = list()
    for p in patterns:
        pl, pr = p
        if not __l1_pattern_legal(pl, word_freq_dict) or not __l1_pattern_legal(pr, word_freq_dict):
            continue
        patterns_new.append(p)
    return patterns_new


def __find_related_dep_patterns(aspect_word_wc, dep_tags, pos_tags, sent_words, term_word_idx_span, opinion_terms):
    widx_beg, widx_end = term_word_idx_span
    # print(' '.join(sent_words[widx_beg: widx_end]))
    # for widx in range(widx_beg, widx_end):
    related_dep_tag_idxs = set()
    for i, dep_tag in enumerate(dep_tags):
        rel, gov, dep = dep_tag
        igov, wgov = gov
        idep, wdep = dep
        if not widx_beg <= igov < widx_end and not widx_beg <= idep < widx_end:
            continue
        if widx_beg <= igov < widx_end and widx_beg <= idep < widx_end:
            continue
        # print(dep_tag)
        related_dep_tag_idxs.add(i)
    # print(related_dep_tag_idxs)
    patterns_l1 = __patterns_from_l1_dep_tags(aspect_word_wc,
        [dep_tags[idx] for idx in related_dep_tag_idxs], pos_tags, term_word_idx_span, opinion_terms)
    # print(patterns_new)
    related_l2 = __find_related_l2_dep_tags(related_dep_tag_idxs, dep_tags, pos_tags, sent_words, term_word_idx_span)
    patterns_l2 = __patterns_from_l2_dep_tags(aspect_word_wc, related_l2, pos_tags, term_word_idx_span, opinion_terms)
    return patterns_l1, patterns_l2


def __get_word_cnts_dict(word_cnts_file):
    with open(word_cnts_file, encoding='utf-8') as f:
        df = pd.read_csv(f)
    word_cnt_dict = dict()
    for w, cnt, _ in df.itertuples(False, None):
        word_cnt_dict[w] = cnt
    return word_cnt_dict


def __get_term_pos_type(term_pos_tags):
    for t in term_pos_tags:
        if t in NOUN_POS_TAGS:
            return 'N'
    for t in term_pos_tags:
        if t in VB_POS_TAGS:
            return 'V'
    return None


def __find_rule_candidates(dep_tags_list, pos_tags_list, aspect_terms_list, opinion_terms_vocab, word_cnts_file):
    word_freq_dict = __get_word_cnts_dict(word_cnts_file)

    # sents = utils.load_json_objs(sents_file)
    cnt_miss, cnt_patterns = 0, 0
    patterns_l1_cnts, patterns_l2_cnts = dict(), dict()
    for sent_idx, (dep_tags, pos_tags) in enumerate(zip(dep_tags_list, pos_tags_list)):
        assert len(dep_tags) == len(pos_tags)
        sent_words = [dep_tup[2][1] for dep_tup in dep_tags]

        aspect_terms = aspect_terms_list[sent_idx]
        for term in aspect_terms:
            idx_span = __find_phrase_word_idx_span(term, sent_words)
            if idx_span is None:
                cnt_miss += 1
                continue

            term_pos_tags = set([pos_tags[i] for i in range(idx_span[0], idx_span[1])])
            term_pos_type = __get_term_pos_type(term_pos_tags)
            if term_pos_type is None:
                # print(term)
                continue

            aspect_word_wc = '_A{}'.format(term_pos_type)

            patterns_l1_new, patterns_l2_new = __find_related_dep_patterns(aspect_word_wc,
                dep_tags, pos_tags, sent_words, idx_span, opinion_terms_vocab)

            patterns_l1_new = __filter_l1_patterns(patterns_l1_new, word_freq_dict)
            patterns_l2_new = __filter_l2_patterns(patterns_l2_new, word_freq_dict)

            for p in patterns_l1_new:
                cnt = patterns_l1_cnts.get(p, 0)
                patterns_l1_cnts[p] = cnt + 1
            for p in patterns_l2_new:
                cnt = patterns_l2_cnts.get(p, 0)
                patterns_l2_cnts[p] = cnt + 1

            # patterns_l1.update(patterns_l1_new)
            # patterns_l2.update(patterns_l2_new)
        # if sent_idx >= 100:
        #     break

    # patterns_l1, patterns_l2 = set(), set()
    patterns_l1 = {p for p, cnt in patterns_l1_cnts.items() if cnt > 10}
    patterns_l2 = {p for p, cnt in patterns_l2_cnts.items() if cnt > 10}

    print(cnt_miss, 'terms missed')
    return patterns_l1, patterns_l2


def __match_l1_pattern(pattern, dep_tags, pos_tags, opinion_terms_vocab):
    pass


def __filter_patterns_through_matching(patterns_l1, patterns_l2, dep_tags_list, pos_tags_list, opinion_terms_vocab):
    pass


def __gen_aspect_patterns(dep_tags_file, pos_tags_file, sents_file, opinion_terms_file, word_cnts_file):
    dep_tags_list = utils.load_dep_tags_list(dep_tags_file)
    pos_tags_list = utils.load_pos_tags(pos_tags_file)
    opinion_terms_vocab = set(utils.read_lines(opinion_terms_file))
    sents = utils.load_json_objs(sents_file)
    aspect_terms_list = list()
    for sent in sents:
        aspect_terms_list.append([t['term'].lower() for t in sent.get('terms', list())])
    patterns_l1, patterns_l2 = __find_rule_candidates(
        dep_tags_list, pos_tags_list, aspect_terms_list, opinion_terms_vocab, word_cnts_file)
    print(len(patterns_l1), 'l1 patterns', len(patterns_l2), 'l2 patterns')


dep_tags_file = 'd:/data/aspect/semeval14/laptops/laptops-train-rule-dep.txt'
pos_tags_file = 'd:/data/aspect/semeval14/laptops/laptops-train-rule-pos.txt'
sent_texts_file = 'd:/data/aspect/semeval14/laptops/laptops_train_texts.txt'
opinion_terms_file = 'd:/data/aspect/semeval14/opinion-terms-full.txt'
word_cnts_file = 'd:/data/aspect/semeval14/laptops/word_cnts.txt'
# train_sents_file = config.SE14_LAPTOP_TRAIN_SENTS_FILE
sents_file = config.SE14_LAPTOP_TRAIN_SENTS_FILE

__gen_aspect_patterns(dep_tags_file, pos_tags_file, sents_file, opinion_terms_file, word_cnts_file)
