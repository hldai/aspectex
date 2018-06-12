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


def __proc_sent_based_on_opinion(sent, f_dep, pos_tags, opinions, aspects):
    dep_list = utils.next_sent_dependency(f_dep)
    sent_words = sent['text'].split(' ')
    assert len(sent_words) == len(dep_list)
    ao_pairs, ao_idx_pairs = list(), list()

    for gov, dep, reln in dep_list:
        w_gov, w_gov_idx = __parse_indexed_word(gov)
        w_dep, w_dep_idx = __parse_indexed_word(dep)
        if reln == 'amod':
            # print(w)
            # print(w_gov_idx)
            if w_dep in opinions and __is_word(w_gov) and pos_tags[w_gov_idx] == 'NN':
                aspect_phrase = __get_phrase(w_gov_idx, pos_tags, sent_words)
                ao_pairs.append((aspect_phrase, w_dep))
                ao_idx_pairs.append((w_gov_idx, w_dep_idx))
                # if w_gov != aspect_phrase:
                #     print(w_gov, aspect_phrase)
                # print(gov, dep, reln)

        if reln == 'conj':
            if w_gov in opinions and pos_tags[w_dep_idx] == 'JJ' and w_dep not in opinions:
                # print(w_dep)
                opinions.add(w_dep)
            if w_dep in opinions and pos_tags[w_gov_idx] == 'JJ' and w_gov not in opinions:
                opinions.add(w_gov)
    return ao_pairs, ao_idx_pairs


def __proc_sent_based_on_aspect(sent, f_dep, pos_tags, opinions, aspects):
    dep_list = utils.next_sent_dependency(f_dep)
    sent_words = sent['text'].split(' ')
    assert len(sent_words) == len(dep_list)
    results = set()

    for gov, dep, reln in dep_list:
        w_gov, w_gov_idx = __parse_indexed_word(gov)
        w_dep, w_dep_idx = __parse_indexed_word(dep)

        if reln == 'conj':
            if w_gov in aspects and pos_tags[w_dep_idx] == 'NN' and w_dep not in aspects:
                aspects.add(w_dep)
            if w_dep in aspects and pos_tags[w_gov_idx] == 'NN' and w_gov not in aspects:
                aspects.add(w_gov)

        if reln == 'amod':
            if w_gov in aspects and __is_word(w_dep) and pos_tags[w_dep_idx] == 'JJ':
                opinions.add(w_dep)
                # results.add((w_gov, w_dep))
                aspect_phrase = __get_phrase(w_gov_idx, pos_tags, sent_words)
                results.add((aspect_phrase, w_dep))
    return results


def __prune_aspects(ao_pairs_dict, ao_idx_pairs_dict, sents, pos_tags_list):
    aspects = list()
    for ao_pairs in ao_pairs_dict.values():
        aspects += [a for a, o in ao_pairs]
    aspect_cnts = Counter(aspects)
    print(aspect_cnts)

    ao_pairs_dict_pruned, ao_idx_pairs_dict_pruned = dict(), dict()
    for sent_idx, ao_idx_pairs in ao_idx_pairs_dict.items():
        if len(ao_idx_pairs) < 2:
            continue
        # sent = sents[sent_idx]
        pos_tags = pos_tags_list[sent_idx]
        ao_pairs = ao_pairs_dict[sent_idx]

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


def __dp():
    seed_opinions = utils.read_lines(config.SEED_OPINIONS_FILE_HL04)
    sents = utils.load_json_objs(config.SENTS_FILE_HL04)
    pos_tags_list = utils.load_pos_tags(config.SENT_POS_FILE_HL04)
    assert len(pos_tags_list) == len(sents)

    opinions = set(seed_opinions)
    aspects = set()
    prev_opinion_size, prev_aspect_size = 0, 0
    sent_opinions_dict = None
    while len(opinions) > prev_opinion_size or len(aspects) > prev_aspect_size:
        n_hit, n_sys = 0, 0
        prev_opinion_size, prev_aspect_size = len(opinions), len(aspects)
        f_dep = open(config.SENT_DEPENDENCY_FILE_HL04, encoding='utf-8')
        ao_pairs_dict, ao_idx_pairs_dict = dict(), dict()
        for i, sent in enumerate(sents):
            ao_pairs_tmp, ao_idx_pairs_tmp = __proc_sent_based_on_opinion(
                sent, f_dep, pos_tags_list[i], opinions, aspects)
            ao_pairs_dict[i] = [(a, o) for a, o in ao_pairs_tmp]
            ao_idx_pairs_dict[i] = [(a_idx, o_idx) for a_idx, o_idx in ao_idx_pairs_tmp]
        # update aspects & opinions

        ao_pairs_dict_pruned, ao_idx_pairs_dict_pruned = __prune_aspects(
            ao_pairs_dict, ao_idx_pairs_dict, sents, pos_tags_list)
        for ao_pairs in ao_pairs_dict_pruned.values():
            for ao_pair in ao_pairs:
                aspects.add(ao_pair[0])

        # print(aspects)

        f_dep.seek(0)
        ao_pairs = list()
        for i, sent in enumerate(sents):
            opinion_tups_sent = __proc_sent_based_on_aspect(sent, f_dep, pos_tags_list[i], opinions, aspects)
            for aspect, opinion in opinion_tups_sent:
                ao_pairs.append((aspect, opinion, i))

        f_dep.close()

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
