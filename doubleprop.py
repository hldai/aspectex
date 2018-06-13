import json
from collections import Counter
import utils
import config


SET_MR = {'amod', 'prep', 'nsubj', 'csubj', 'xsubj'}
SET_JJ = {'JJ', 'JJR', 'JJS'}
SET_NN = {'NN', 'NNS'}


def __is_word(w):
    for ch in w:
        if ch.isalpha():
            return True
    return False


def __parse_indexed_word(w):
    p = w.rfind('-')
    s = w[:p]
    idx = int(w[p + 1:]) - 1
    return s, idx


def __get_phrase(w_idx, pos_list, sent_words):
    wind = 2
    widx_list = [w_idx]
    for i in range(w_idx - 1, w_idx - wind - 1, -1):
        if i < 0:
            break
        if pos_list[i] == 'NN':
            widx_list.insert(0, i)
        else:
            break

    for i in range(w_idx + 1, w_idx + wind + 1):
        if i >= len(pos_list):
            break
        if pos_list[i] == 'NN':
            widx_list.append(i)
        else:
            break

    # if len(widx_list) == 1 and w_idx - 1 > -1 and pos_list[w_idx - 1] == 'JJ':
    #     widx_list.insert(0, w_idx - 1)

    return ' '.join([sent_words[i] for i in widx_list])


def __proc_sent_based_on_opinion(sent_words, dep_tags, pos_tags, opinions, aspects):
    # dep_list = utils.next_sent_dependency(f_dep)
    # sent_words = sent['text'].split(' ')
    assert len(sent_words) == len(dep_tags)
    ao_pairs, ao_idx_pairs = list(), list()

    for gov, dep, reln in dep_tags:
        w_gov, w_gov_idx = __parse_indexed_word(gov)
        w_dep, w_dep_idx = __parse_indexed_word(dep)
        if reln in SET_MR:
            # print(w)
            # print(w_gov_idx)
            if w_dep in opinions and __is_word(w_gov) and pos_tags[w_gov_idx] in SET_NN:
                aspect_phrase = __get_phrase(w_gov_idx, pos_tags, sent_words)
                ao_pairs.append((aspect_phrase, w_dep))
                ao_idx_pairs.append((w_gov_idx, w_dep_idx))
                # if w_gov != aspect_phrase:
                #     print(w_gov, aspect_phrase)
                # print(gov, dep, reln)

        if reln == 'conj':
            if w_gov in opinions and pos_tags[w_dep_idx] in SET_JJ and w_dep not in opinions:
                # print(w_dep)
                opinions.add(w_dep)
            if w_dep in opinions and pos_tags[w_gov_idx] in SET_JJ and w_gov not in opinions:
                opinions.add(w_gov)
    # for a, o in ao_pairs:
    #     if a == 'month':
    #         print(ao_pairs)
    #         print(sent)
    return ao_pairs, ao_idx_pairs


def __proc_sent_based_on_aspect(sent, dep_tags, pos_tags, opinions, aspects):
    # dep_list = utils.next_sent_dependency(f_dep)
    sent_words = sent['text'].split(' ')
    assert len(sent_words) == len(dep_tags)
    results = set()

    for gov, dep, reln in dep_tags:
        w_gov, w_gov_idx = __parse_indexed_word(gov)
        w_dep, w_dep_idx = __parse_indexed_word(dep)

        if reln == 'conj':
            if w_gov in aspects and pos_tags[w_dep_idx] in SET_NN and w_dep not in aspects:
                aspects.add(w_dep)
                aspect_phrase = __get_phrase(w_dep_idx, pos_tags, sent_words)
                results.add((aspect_phrase, w_dep))
            if w_dep in aspects and pos_tags[w_gov_idx] in SET_NN and w_gov not in aspects:
                aspects.add(w_gov)
                aspect_phrase = __get_phrase(w_gov_idx, pos_tags, sent_words)
                results.add((aspect_phrase, w_dep))

        if reln in SET_MR:
            # if w_gov == 'looks':
            #     print(gov, dep, reln)
            if w_gov in aspects and __is_word(w_dep) and pos_tags[w_dep_idx] in SET_JJ:
                opinions.add(w_dep)
                # results.add((w_gov, w_dep))
                aspect_phrase = __get_phrase(w_gov_idx, pos_tags, sent_words)
                if pos_tags[w_gov_idx] == 'NNS':
                    if aspect_phrase.endswith('s'):
                        aspect_phrase = aspect_phrase[:-1]
                results.add((aspect_phrase, w_dep))
    return results


def __prune_aspects(ao_pairs_dict, ao_idx_pairs_dict, sents, pos_tags_list):
    aspects = list()
    for ao_pairs in ao_pairs_dict.values():
        aspects += [a for a, o in ao_pairs]
    aspect_cnts = Counter(aspects)
    # print(aspect_cnts)

    ao_pairs_dict_pruned, ao_idx_pairs_dict_pruned = dict(), dict()
    for sent_idx, ao_idx_pairs in ao_idx_pairs_dict.items():
        ao_pairs = ao_pairs_dict[sent_idx]

        if len(ao_idx_pairs) < 2:
            ao_pairs_dict_pruned[sent_idx] = ao_pairs
            ao_idx_pairs_dict_pruned[sent_idx] = ao_idx_pairs
            continue
        # sent = sents[sent_idx]

        pos_tags = pos_tags_list[sent_idx]

        list_keep = [True for _ in range(len(ao_idx_pairs))]
        for j0, (a_idx0, o_idx0) in enumerate(ao_idx_pairs):
            for j1 in range(j0 + 1, len(ao_idx_pairs)):
                a_idx1, o_idx1 = ao_idx_pairs[j1]

                idx_left, idx_right = min(a_idx0, a_idx1), max(a_idx0, a_idx1)
                has_conj = False
                for k in range(idx_left + 1, idx_right):
                    if pos_tags[k] == ',' or pos_tags[k] == 'CC':
                        has_conj = True
                        break

                if not has_conj:
                    if aspect_cnts[ao_pairs[j0][0]] < aspect_cnts[ao_pairs[j1][0]]:
                        list_keep[j0] = False
                    else:
                        list_keep[j1] = False

        ao_pairs_tmp, ao_idx_pairs_tmp = list(), list()
        ao_pairs_dict_pruned[sent_idx], ao_idx_pairs_dict_pruned[sent_idx] = ao_pairs_tmp, ao_idx_pairs_tmp
        for ao_pair, ao_idx_pair, keep in zip(ao_pairs, ao_idx_pairs, list_keep):
            if keep:
                ao_pairs_tmp.append(ao_pair)
                ao_idx_pairs_tmp.append(ao_idx_pair)

    return ao_pairs_dict_pruned, ao_idx_pairs_dict_pruned


def __get_true_aspect_word_set(sents):
    aspect_set_true = set()
    for sent in sents:
        sent_aspects = sent.get('aspects', list())
        for x in sent_aspects:
            aspect_set_true.add(x['target'])
    return aspect_set_true


def __get_word_extraction_perf(words_true, words_sys):
    hit_cnt = 0
    for w in words_true:
        if w in words_sys:
            hit_cnt += 1
    print('word set', hit_cnt / len(words_sys), hit_cnt / len(words_true))
    return 0


def __dp(sents, dep_tags_list, pos_tags_list, seed_opinions):
    aspect_set_true = __get_true_aspect_word_set(sents)

    opinions = set(seed_opinions)
    aspects = set()
    ao_pairs = list()
    n_hit, true_ao_pairs_cnt = 0, 0
    prev_opinion_size, prev_aspect_size = 0, 0
    while len(opinions) > prev_opinion_size or len(aspects) > prev_aspect_size:
        n_hit, n_sys = 0, 0
        prev_opinion_size, prev_aspect_size = len(opinions), len(aspects)
        ao_pairs_dict, ao_idx_pairs_dict = dict(), dict()
        for i, sent in enumerate(sents):
            sent_words = sent['text'].split(' ')
            ao_pairs_tmp, ao_idx_pairs_tmp = __proc_sent_based_on_opinion(
                sent_words, dep_tags_list[i], pos_tags_list[i], opinions, aspects)
            # if sent['text'].startswith('this camera also has a'):
            #     print('foooooooo', ao_pairs_tmp)
            #     print(dep_tags_list[i])
            #     print(pos_tags_list[i])
            ao_pairs_dict[i] = [(a, o) for a, o in ao_pairs_tmp]
            ao_idx_pairs_dict[i] = [(a_idx, o_idx) for a_idx, o_idx in ao_idx_pairs_tmp]
        # update aspects & opinions

        ao_pairs_dict_pruned, ao_idx_pairs_dict_pruned = __prune_aspects(
            ao_pairs_dict, ao_idx_pairs_dict, sents, pos_tags_list)
        # print(ao_pairs_dict[664])
        # print(ao_pairs_dict_pruned[664])

        for sent_idx, ao_idx_pairs in ao_idx_pairs_dict_pruned.items():
            sent_words = sents[sent_idx]['text'].split(' ')
            for ao_idx_pair in ao_idx_pairs:
                aspects.add(sent_words[ao_idx_pair[0]])

        ao_pairs = list()
        for i, sent in enumerate(sents):
            opinion_tups_sent = __proc_sent_based_on_aspect(
                sent, dep_tags_list[i], pos_tags_list[i], opinions, aspects)
            for aspect, opinion in opinion_tups_sent:
                ao_pairs.append((aspect, opinion, i))
            if 'better than' in sent['text']:
                print(sent)
                print(opinion_tups_sent)
                print()

        aspect_cnt = Counter([x[0] for x in ao_pairs])
        # ao_pairs_pruned = list()
        # for a, o, sent_idx in ao_pairs:
        #     if ' ' in a and
        ao_pairs = [x for x in ao_pairs if ' ' not in x[0] or aspect_cnt[x[0]] > 1]
        # ao_pairs = [x for x in ao_pairs if aspect_cnt[x[0]] > 1]
        sent_ao_pairs_dict = dict()
        for x in ao_pairs:
            aspect, opinion, sent_idx = x
            sent_ao_pairs = sent_ao_pairs_dict.get(sent_idx, list())
            if not sent_ao_pairs:
                sent_ao_pairs_dict[sent_idx] = sent_ao_pairs
            sent_ao_pairs.append((aspect, opinion))

        aspect_words_sys = {x[0] for x in ao_pairs}
        __get_word_extraction_perf(aspect_set_true, aspect_words_sys)

        # evaluation
        true_ao_pairs_cnt = 0
        for i, sent in enumerate(sents):
            opinions_true = sent.get('aspects', None)
            sent_ao_pairs = sent_ao_pairs_dict.get(i, None)

            if opinions_true is not None:
                true_ao_pairs_cnt += len(opinions_true)

            hit = False
            if opinions_true is not None and sent_ao_pairs is not None:
                targets_true = {x['target'] for x in opinions_true}
                for target, opinion in sent_ao_pairs:
                    if target in targets_true:
                        n_hit += 1
                        hit = True
            # if sent_ao_pairs is not None and not hit:
            # if opinions_true is not None and not hit:
            #     print(i, sent['text'])
            #     print(sent.get('aspects', None))
            #     print(sent_ao_pairs)
            #     print(dep_tags_list[i])
            #     print(pos_tags_list[i])
            #     print()
        prec = n_hit / len(ao_pairs)
        recall = n_hit / true_ao_pairs_cnt
        print('tuples', prec, recall, 2 * prec * recall / (prec + recall))
        # break
    print()
    return n_hit, len(ao_pairs), true_ao_pairs_cnt


def __read_seed_opinions():
    def __read_file(filename):
        words = list()
        with open(filename, encoding='utf-8') as f:
            for _ in range(35):
                next(f)
            for line in f:
                words.append(line.strip())
        return words

    pos_words = __read_file('d:/data/aspect/huliu04/negative-words.txt')
    neg_words = __read_file('d:/data/aspect/huliu04/negative-words.txt')
    return pos_words + neg_words


def __dp_hl04():
    reviews = utils.load_json_objs(config.REVIEWS_FILE_HL04)
    sents = utils.load_json_objs(config.SENTS_FILE_HL04)
    review_prod_dict = {r['review_id']: r['file'] for r in reviews}
    prod_set = {v for v in review_prod_dict.values()}
    prod_sents_dict = {v: list() for v in prod_set}
    for i, sent in enumerate(sents):
        prod_sents_dict[review_prod_dict[sent['review_id']]].append(i)

    # seed_opinions = utils.read_lines(config.SEED_OPINIONS_FILE_HL04)
    seed_opinions = __read_seed_opinions()
    pos_tags_list = utils.load_pos_tags(config.SENT_POS_FILE_HL04)
    dep_tags_list = utils.load_dep_tags_list(config.SENT_DEPENDENCY_FILE_HL04)
    assert len(pos_tags_list) == len(sents)
    assert len(dep_tags_list) == len(sents)

    cnt_hit, cnt_sys, cnt_true = 0, 0, 0
    for prod, sent_idxs in prod_sents_dict.items():
        # if prod != 'Apex AD2600 Progressive-scan DVD player.txt':
        #     continue

        prod_sents = [sents[i] for i in sent_idxs]
        prod_pos_tags_list = [pos_tags_list[i] for i in sent_idxs]
        prod_dep_tags_list = [dep_tags_list[i] for i in sent_idxs]
        n_hit, n_sys, n_true = __dp(prod_sents, prod_dep_tags_list, prod_pos_tags_list, seed_opinions)
        cnt_hit += n_hit
        cnt_sys += n_sys
        cnt_true += n_true
        # break

    prec = cnt_hit / cnt_sys
    recall = cnt_hit / cnt_true
    print(prec, recall, 2 * prec * recall / (prec + recall), cnt_hit, cnt_sys, cnt_true)

    # __dp(sents, dep_tags_list, pos_tags_list, seed_opinions)

    # for i, sent in enumerate(sents):
    #     sent_opinions = sent_opinions_dict.get(i, None)
    #
    #     if sent_opinions is not None:
    #         print(sent)
    #         print(sent_opinions)


if __name__ == '__main__':
    __dp_hl04()
