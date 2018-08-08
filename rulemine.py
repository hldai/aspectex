import pandas as pd
from collections import namedtuple
import config
from utils import utils
from models import rules, rulescommon


RuleMineData = namedtuple('RuleMineData', ['dep_tag_seqs', 'pos_tag_seqs', 'sents'])

# NOUN_POS_TAGS = {'NN', 'NNP', 'NNS', 'NNPS'}
# VB_POS_TAGS = {'VB', 'VBN', 'VBP', 'VBZ', 'VBG', 'VBD'}


# def __get_noun_phrase(dep_tags, pos_tags, base_word_idxs):
#     words = [tup[2][1] for tup in dep_tags]
#     phrase_word_idxs = set(base_word_idxs)
#
#     ileft = min(phrase_word_idxs)
#     iright = max(phrase_word_idxs)
#     ileft_new, iright_new = ileft, iright
#     while ileft_new > 0:
#         if pos_tags[ileft_new - 1] in NOUN_POS_TAGS:
#             ileft_new -= 1
#         else:
#             break
#     while iright_new < len(pos_tags) - 1:
#         if pos_tags[iright_new + 1] in {'NN', 'NNP', 'NNS', 'CD'}:
#             iright_new += 1
#         else:
#             break
#
#     phrase = ' '.join([words[widx] for widx in range(ileft_new, iright_new + 1)])
#     return phrase


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
        # print(dep_tag_i, dep_tag_j)
        # print(patterns_i, patterns_j)

        if dep_tag_i[1][0] == dep_tag_j[1][0] or dep_tag_i[1][0] == dep_tag_j[2][0]:
            patterns_i = {(tup, 1) for tup in patterns_i}
        else:
            patterns_i = {(tup, 2) for tup in patterns_i}

        if dep_tag_j[1][0] == dep_tag_i[1][0] or dep_tag_j[1][0] == dep_tag_i[2][0]:
            patterns_j = {(tup, 1) for tup in patterns_j}
        else:
            patterns_j = {(tup, 2) for tup in patterns_j}
        # print(patterns_i, patterns_j)
        # print()

        for pi in patterns_i:
            for pj in patterns_j:
                if pi[0][pi[1]] != pj[0][pj[1]]:
                    # print(pi, pj)
                    continue
                if pi < pj:
                    patterns.add((pi, pj))
                else:
                    patterns.add((pj, pi))
    return patterns


def __word_legal_by_freq(w: str, word_freq_dict):
    return w == '_A' or w == '_OP' or w.isupper() or word_freq_dict.get(w, 0) > 10


def __l1_pattern_legal(pattern, word_freq_dict):
    rel, gov, dep = pattern
    if not gov.startswith('_A') and gov != '_OP' and not gov.isupper() and word_freq_dict.get(gov, 0) < 10:
        return False
    if not dep.startswith('_A') and dep != '_OP' and not dep.isupper() and word_freq_dict.get(dep, 0) < 10:
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
        if not __l1_pattern_legal(pl[0], word_freq_dict) or not __l1_pattern_legal(pr[0], word_freq_dict):
            # print(p)
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
        if t in rulescommon.NOUN_POS_TAGS:
            return 'N'
    for t in term_pos_tags:
        if t in rulescommon.VB_POS_TAGS:
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


# def __match_pattern_word(pw, w, pos_tag, opinion_terms_vocab):
#     if pw == '_AV' and pos_tag in rulescommon.VB_POS_TAGS:
#         return True
#     if pw == '_AN' and pos_tag in rulescommon.NOUN_POS_TAGS:
#         return True
#     if pw == '_OP' and w in opinion_terms_vocab:
#         return True
#     if pw.isupper() and pos_tag == pw:
#         return True
#     return pw == w


# def __match_l1_pattern(pattern, dep_tag, pos_tags, opinion_terms_vocab):
#     prel, pgov, pdep = pattern
#     rel, (igov, wgov), (idep, wdep) = dep_tag
#     if rel != prel:
#         return False
#     return __match_pattern_word(pgov, wgov, pos_tags[igov], opinion_terms_vocab) and __match_pattern_word(
#         pdep, wdep, pos_tags[idep], opinion_terms_vocab)


# def __get_l1_pattern_matched_dep_tags(pattern, dep_tags, pos_tags, opinion_terms_vocab):
#     matched_idxs = list()
#     for i, dep_tag in enumerate(dep_tags):
#         if __match_l1_pattern(pattern, dep_tag, pos_tags, opinion_terms_vocab):
#             matched_idxs.append(i)
#     return matched_idxs


# def __get_aspect_term_from_matched_pattern(pattern, dep_tags, pos_tags, matched_dep_tag_idx):
#     if pattern[1].startswith('_A'):
#         aspect_position = 1
#     elif pattern[2].startswith('_A'):
#         aspect_position = 2
#     else:
#         return None
#
#     dep_tag = dep_tags[matched_dep_tag_idx]
#     widx, w = dep_tag[aspect_position]
#     if pattern[aspect_position] == '_AV':
#         return w
#     else:
#         return rulescommon.get_noun_phrase_from_seed(dep_tags, pos_tags, [widx])


# def __find_terms_by_l1_pattern(pattern, dep_tags, pos_tags, opinion_terms_vocab, filter_terms_vocab):
#     terms = list()
#     matched_dep_tag_idxs = __get_l1_pattern_matched_dep_tags(pattern, dep_tags, pos_tags, opinion_terms_vocab)
#     for idx in matched_dep_tag_idxs:
#         term = __get_aspect_term_from_matched_pattern(pattern, dep_tags, pos_tags, idx)
#
#         if term in filter_terms_vocab:
#             continue
#         terms.append(term)
#     return terms


# def __find_terms_by_l2_pattern(pattern, dep_tags, pos_tags, opinion_terms_vocab, filter_terms_vocab):
#     (pl, ipl), (pr, ipr) = pattern
#     terms = list()
#     matched_dep_tag_idxs = __get_l1_pattern_matched_dep_tags(pl, dep_tags, pos_tags, opinion_terms_vocab)
#     for idx in matched_dep_tag_idxs:
#         dep_tag_l = dep_tags[idx]
#         sw_idx = dep_tag_l[ipl][0]  # index of the shared word
#
#         for j, dep_tag_r in enumerate(dep_tags):
#             if dep_tag_r[ipr][0] != sw_idx:
#                 continue
#             if not __match_l1_pattern(pr, dep_tag_r, pos_tags, opinion_terms_vocab):
#                 continue
#
#             term = __get_aspect_term_from_matched_pattern(pl, dep_tags, pos_tags, idx)
#             if term is None:
#                 term = __get_aspect_term_from_matched_pattern(pr, dep_tags, pos_tags, j)
#             if term is None or term in filter_terms_vocab:
#                 # print(p, 'term not found')
#                 continue
#
#             terms.append(term)
#     return terms


def __filter_l1_patterns_through_matching(patterns, dep_tags_list, pos_tags_list,
                                          aspect_terms_list, opinion_terms_vocab, filter_terms_vocab):
    patterns_keep = list()
    for p in patterns:
        hit_cnt, cnt = 0, 0
        for dep_tags, pos_tags, aspect_terms in zip(dep_tags_list, pos_tags_list, aspect_terms_list):
            aspect_terms = set(aspect_terms)
            terms = rulescommon.find_terms_by_l1_pattern(
                p, dep_tags, pos_tags, opinion_terms_vocab, filter_terms_vocab)

            for term in terms:
                if term in filter_terms_vocab:
                    continue
                cnt += 1
                if term in aspect_terms:
                    hit_cnt += 1

        if hit_cnt / (cnt + 1e-5) > 0.6:
            # print(p, hit_cnt, cnt, hit_cnt / (cnt + 1e-5))
            patterns_keep.append(p)
    return patterns_keep


def __filter_l2_patterns_through_matching(patterns, dep_tags_list, pos_tags_list, aspect_terms_list,
                                          opinion_terms_vocab, filter_terms_vocab):
    patterns_keep = list()
    for p in patterns:
        hit_cnt, cnt = 0, 0
        for dep_tags, pos_tags, aspect_terms in zip(dep_tags_list, pos_tags_list, aspect_terms_list):
            aspect_terms = set(aspect_terms)
            terms = rulescommon.find_terms_by_l2_pattern(p, dep_tags, pos_tags, opinion_terms_vocab, filter_terms_vocab)
            for term in terms:
                cnt += 1
                if term in aspect_terms:
                    hit_cnt += 1
                    # print(p)
                    # print(dep_tag_l, dep_tag_r)
                    # print(term)

        if hit_cnt / (cnt + 1e-5) > 0.6:
            # print(p, hit_cnt, cnt, hit_cnt / (cnt + 1e-5))
            patterns_keep.append(p)
    return patterns_keep


def __load_data(dep_tags_file, pos_tags_file, sents_file, train_valid_split_file):
    tvs_line = utils.read_lines(train_valid_split_file)[0]
    tvs_arr = [int(v) for v in tvs_line.split()]

    dep_tags_list = utils.load_dep_tags_list(dep_tags_file)
    pos_tags_list = utils.load_pos_tags(pos_tags_file)
    sents = utils.load_json_objs(sents_file)

    assert len(tvs_arr) == len(dep_tags_list)

    dep_tags_list_train, dep_tags_list_valid = list(), list()
    pos_tags_list_train, pos_tags_list_valid = list(), list()
    sents_train, sents_valid = list(), list()
    for tvs_label, dep_tags, pos_tags, sent in zip(tvs_arr, dep_tags_list, pos_tags_list, sents):
        if tvs_label == 0:
            dep_tags_list_train.append(dep_tags)
            pos_tags_list_train.append(pos_tags)
            sents_train.append(sent)
        else:
            dep_tags_list_valid.append(dep_tags)
            pos_tags_list_valid.append(pos_tags)
            sents_valid.append(sent)

    data_train = RuleMineData(dep_tags_list_train, pos_tags_list_train, sents_train)
    data_valid = RuleMineData(dep_tags_list_valid, pos_tags_list_valid, sents_valid)
    return data_train, data_valid


def __get_term_filter_dict(dep_tag_seqs, pos_tag_seqs, terms_list, filter_rate):
    term_cnts_dict = dict()
    for dep_tag_seq, pos_tag_seq, terms in zip(dep_tag_seqs, pos_tag_seqs, terms_list):
        words = [tup[2][1] for tup in dep_tag_seq]
        noun_phrases = rules.rec_rule1(words, pos_tag_seq, None)

        verbs = list()
        for w, pos_tag in zip(words, pos_tag_seq):
            if pos_tag in rulescommon.VB_POS_TAGS:
                verbs.append(w)

        term_cands = noun_phrases + verbs
        # term_cands = verbs
        for t in term_cands:
            cnts = term_cnts_dict.get(t, (0, 0))
            hit_cnt = cnts[0] + 1 if t in terms else cnts[0]
            term_cnts_dict[t] = (hit_cnt, cnts[1] + 1)

    filter_terms = set()
    for t, (hit_cnt, cnt) in term_cnts_dict.items():
        if hit_cnt / cnt < filter_rate:
            filter_terms.add(t)
    return filter_terms


def __gen_aspect_patterns(dep_tags_file, pos_tags_file, sents_file,
                          train_valid_split_file, opinion_terms_file, word_cnts_file, dst_file):
    opinion_terms_vocab = set(utils.read_lines(opinion_terms_file))
    data_train, data_valid = __load_data(dep_tags_file, pos_tags_file, sents_file, train_valid_split_file)

    aspect_terms_list_train = utils.aspect_terms_list_from_sents(data_train.sents)
    filter_terms_vocab = __get_term_filter_dict(
        data_train.dep_tag_seqs, data_train.pos_tag_seqs, aspect_terms_list_train, term_filter_rate)

    patterns_l1, patterns_l2 = __find_rule_candidates(
        data_train.dep_tag_seqs, data_train.pos_tag_seqs, aspect_terms_list_train,
        opinion_terms_vocab, word_cnts_file)
    print(len(patterns_l1), 'l1 patterns', len(patterns_l2), 'l2 patterns')

    aspect_terms_list_valid = utils.aspect_terms_list_from_sents(data_valid.sents)

    patterns_l1 = __filter_l1_patterns_through_matching(
        patterns_l1, data_valid.dep_tag_seqs, data_valid.pos_tag_seqs, aspect_terms_list_valid,
        opinion_terms_vocab, filter_terms_vocab)

    patterns_l2 = __filter_l2_patterns_through_matching(
        patterns_l2, data_valid.dep_tag_seqs, data_valid.pos_tag_seqs, aspect_terms_list_valid,
        opinion_terms_vocab, filter_terms_vocab)

    fout = open(dst_file, 'w', encoding='utf-8', newline='\n')
    for p in patterns_l1:
        fout.write('{}\n'.format(' '.join(p)))
    for p in patterns_l2:
        (pl, ipl), (pr, ipr) = p
        fout.write('{} {} {} {}\n'.format(' '.join(pl), ipl, ' '.join(pr), ipr))
    fout.close()


def __gen_filter_terms_vocab_file(dep_tags_file, pos_tags_file, sents_file, dst_file):
    dep_tags_list = utils.load_dep_tags_list(dep_tags_file)
    pos_tags_list = utils.load_pos_tags(pos_tags_file)
    sents = utils.load_json_objs(sents_file)
    aspect_terms_list = utils.aspect_terms_list_from_sents(sents)
    filter_terms_vocab = __get_term_filter_dict(dep_tags_list, pos_tags_list, aspect_terms_list, term_filter_rate)
    with open(dst_file, 'w', encoding='utf-8', newline='\n') as fout:
        for t in filter_terms_vocab:
            fout.write('{}\n'.format(t))


term_filter_rate = 0.1
dep_tags_file = 'd:/data/aspect/semeval14/laptops/laptops-train-rule-dep.txt'
pos_tags_file = 'd:/data/aspect/semeval14/laptops/laptops-train-rule-pos.txt'
sent_texts_file = 'd:/data/aspect/semeval14/laptops/laptops_train_texts.txt'
train_valid_split_file = config.SE14_LAPTOP_TRAIN_VALID_SPLIT_FILE
opinion_terms_file = 'd:/data/aspect/semeval14/opinion-terms-full.txt'
word_cnts_file = 'd:/data/aspect/semeval14/laptops/word_cnts.txt'
# train_sents_file = config.SE14_LAPTOP_TRAIN_SENTS_FILE
sents_file = config.SE14_LAPTOP_TRAIN_SENTS_FILE
patterns_file = 'd:/data/aspect/semeval14/laptops/mined_rule_patterns.txt'

full_train_term_filter_file = 'd:/data/aspect/semeval14/laptops/aspect_filter_vocab_full.txt'

# __gen_aspect_patterns(dep_tags_file, pos_tags_file, sents_file, train_valid_split_file,
#                       opinion_terms_file, word_cnts_file, patterns_file)

__gen_filter_terms_vocab_file(dep_tags_file, pos_tags_file, sents_file, full_train_term_filter_file)
