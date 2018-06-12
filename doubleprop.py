import json
from collections import Counter
import utils
import config


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

    return ' '.join([sent_words[i] for i in widx_list])


def __proc_sent_based_on_opinion(sent, f_dep, f_pos, opinions, aspects):
    dep_list = utils.next_sent_dependency(f_dep)
    pos_list = utils.next_sent_pos(f_pos)
    sent_words = sent['text'].split(' ')
    assert len(sent_words) == len(dep_list)
    ao_pairs, ao_idx_pairs = list(), list()

    for gov, dep, reln in dep_list:
        w_gov, w_gov_idx = __parse_indexed_word(gov)
        w_dep, w_dep_idx = __parse_indexed_word(dep)
        if reln == 'amod':
            # print(w)
            # print(w_gov_idx)
            if w_dep in opinions and __is_word(w_gov) and pos_list[w_gov_idx] == 'NN':
                aspect_phrase = __get_phrase(w_gov_idx, pos_list, sent_words)
                ao_pairs.append((aspect_phrase, w_dep))
                ao_idx_pairs.append((w_gov_idx, w_dep_idx))
                # if w_gov != aspect_phrase:
                #     print(w_gov, aspect_phrase)
                # print(gov, dep, reln)

        if reln == 'conj':
            if w_gov in opinions and pos_list[w_dep_idx] == 'JJ' and w_dep not in opinions:
                # print(w_dep)
                opinions.add(w_dep)
            if w_dep in opinions and pos_list[w_gov_idx] == 'JJ' and w_gov not in opinions:
                opinions.add(w_gov)
    return ao_pairs, ao_idx_pairs


def __proc_sent_based_on_aspect(sent, f_dep, f_pos, opinions, aspects):
    dep_list = utils.next_sent_dependency(f_dep)
    pos_list = utils.next_sent_pos(f_pos)
    sent_words = sent['text'].split(' ')
    assert len(sent_words) == len(dep_list)
    results = set()

    for gov, dep, reln in dep_list:
        w_gov, w_gov_idx = __parse_indexed_word(gov)
        w_dep, w_dep_idx = __parse_indexed_word(dep)

        if reln == 'conj':
            if w_gov in aspects and pos_list[w_dep_idx] == 'NN' and w_dep not in aspects:
                aspects.add(w_dep)
            if w_dep in aspects and pos_list[w_gov_idx] == 'NN' and w_gov not in aspects:
                aspects.add(w_gov)

        if reln == 'amod':
            if w_gov in aspects and __is_word(w_dep) and pos_list[w_dep_idx] == 'JJ':
                opinions.add(w_dep)
                # results.add((w_gov, w_dep))
                aspect_phrase = __get_phrase(w_gov_idx, pos_list, sent_words)
                results.add((aspect_phrase, w_dep))
    return results


def __prune_aspects(ao_pairs_dict, ao_idx_pairs_dict, sents):
    aspects = list()
    for ao_pairs in ao_pairs_dict.values():
        aspects += [a for a, o in ao_pairs]
    aspect_cnts = Counter(aspects)
    print(aspect_cnts)

    for i, ao_idx_pairs in ao_idx_pairs_dict.items():
        sent = sents[i]


def __dp():
    seed_opinions = utils.read_lines(config.SEED_OPINIONS_FILE_HL04)
    sents = utils.load_json_objs(config.SENTS_FILE_HL04)

    opinions = set(seed_opinions)
    aspects = set()
    prev_opinion_size, prev_aspect_size = 0, 0
    sent_opinions_dict = None
    while len(opinions) > prev_opinion_size or len(aspects) > prev_aspect_size:
        n_hit, n_sys = 0, 0
        prev_opinion_size, prev_aspect_size = len(opinions), len(aspects)
        f_dep = open(config.SENT_DEPENDENCY_FILE_HL04, encoding='utf-8')
        f_pos = open(config.SENT_POS_FILE_HL04, encoding='utf-8')
        ao_pairs_dict, ao_idx_pairs_dict = dict(), dict()
        for i, sent in enumerate(sents):
            ao_pairs_tmp, ao_idx_pairs_tmp = __proc_sent_based_on_opinion(sent, f_dep, f_pos, opinions, aspects)
            ao_pairs_dict[i] = [(a, o) for a, o in ao_pairs_tmp]
            ao_idx_pairs_dict[i] = [(a_idx, o_idx) for a_idx, o_idx in ao_idx_pairs_tmp]
        # update aspects & opinions

        __prune_aspects(ao_pairs_dict, ao_idx_pairs_dict, sents)

        f_dep.seek(0)
        f_pos.seek(0)
        ao_pairs = list()
        for i, sent in enumerate(sents):
            opinion_tups_sent = __proc_sent_based_on_aspect(sent, f_dep, f_pos, opinions, aspects)
            for aspect, opinion in opinion_tups_sent:
                ao_pairs.append((aspect, opinion, i))

        f_dep.close()
        f_pos.close()

        aspect_cnt = Counter([x[0] for x in ao_pairs])
        ao_pairs = [x for x in ao_pairs if aspect_cnt[x[0]] > 1]
        sent_opinions_dict = dict()
        for x in ao_pairs:
            aspect, opinion, sent_idx = x
            sent_opinions = sent_opinions_dict.get(sent_idx, list())
            if not sent_opinions:
                sent_opinions_dict[sent_idx] = sent_opinions
            sent_opinions.append((aspect, opinion))

        for i, sent in enumerate(sents):
            opinions_true = sent.get('aspects', None)
            sent_opinions = sent_opinions_dict.get(i, None)
            if opinions_true is not None and sent_opinions is not None:
                targets_true = {x['target'] for x in opinions_true}
                for target, opinion in sent_opinions:
                    if target in targets_true:
                        # print(sent)
                        # print(opinion_tups)
                        n_hit += 1
        print(n_hit / len(ao_pairs))

    # for i, sent in enumerate(sents):
    #     sent_opinions = sent_opinions_dict.get(i, None)
    #
    #     if sent_opinions is not None:
    #         print(sent)
    #         print(sent_opinions)


if __name__ == '__main__':
    __dp()
