import json
import config
from utils import utils
from models import rules


# word-idx to (idx, word)
def __get_word_tup(dep_word):
    p = dep_word.rfind('-')
    s = dep_word[:p]
    # idx = int(dep_word[p + 1:]) - 1
    idx = int(dep_word[p + 1:])
    return s, idx


def __read_sent_dep_tups(fin):
    tups = list()
    for line in fin:
        line = line.strip()
        if not line:
            return tups
        line = line[:-1]
        line = line.replace('(', ' ')
        line = line.replace(', ', ' ')
        rel, gov, dep = line.split(' ')
        w_gov, idx_gov = __get_word_tup(gov)
        w_dep, idx_dep = __get_word_tup(dep)
        tups.append((rel, (idx_gov, w_gov), (idx_dep, w_dep)))
        # tups.append(line.split(' '))
    return tups


def __match_terms(sent_text: str, terms_vocab):
    terms = list()
    for t in terms_vocab:
        pbeg = sent_text.find(t)
        if pbeg < 0:
            continue
        pend = pbeg + len(t)
        if pbeg > 0 and sent_text[pbeg - 1].isalpha():
            continue
        if pend < len(sent_text) and sent_text[pend].isalpha():
            continue
        terms.append(t)
    return terms


def __load_terms_in_train(train_sents_file):
    sents_train = utils.load_json_objs(train_sents_file)
    terms_train = set()
    for sent in sents_train:
        terms = sent.get('terms', None)
        if terms is None:
            continue
        for t in terms:
            terms_train.add(t['term'].lower())
    terms_train = list(terms_train)
    terms_train.sort(key=lambda x: -len(x))
    return terms_train


def __load_opinion_terms_in_train(train_sents_file):
    sents_train = utils.load_json_objs(train_sents_file)
    terms_train = set()
    for sent in sents_train:
        terms_train.update([t.lower() for t in sent.get('opinions', list())])
    # print(terms_train)
    terms_cnt_dict = {t: [0, 0] for t in terms_train}
    for sent in sents_train:
        sent_text = sent['text']
        terms_true = [t.lower() for t in sent.get('opinions', list())]
        terms_matched = __match_terms(sent_text, terms_train)
        for t in terms_matched:
            cnt_tup = terms_cnt_dict[t]
            cnt_tup[0] += 1
            if t in terms_true:
                cnt_tup[1] += 1

    # print(terms_cnt_dict)
    terms_vocab = set()
    for t, v in terms_cnt_dict.items():
        if v[0] > 1 and v[1] / v[0] > 0.5:
            terms_vocab.add(t)
    return terms_vocab


def __get_phrase_span_by_compound(dep_tags, base_word_idx):
    dl = 1
    while base_word_idx - dl > -1:
        if dep_tags[base_word_idx - dl][0] != 'compound':
            break
        dl += 1
    dr = 1
    while base_word_idx + dr < len(dep_tags):
        if dep_tags[base_word_idx + dr][0] != 'compound':
            break
        dr += 1
    return base_word_idx - dl + 1, base_word_idx + dr


def __count_hit(y_true, y_sys):
    hit_cnt = 0
    for yi in y_true:
        if yi in y_sys:
            hit_cnt += 1
    return hit_cnt


def __save_never_hit_terms(sents, terms_sys_list, dst_file):
    all_terms_true = set()
    for sent in sents:
        term_objs = sent.get('terms', list())
        all_terms_true.update([t['term'].lower() for t in term_objs])

    all_terms_sys = set()
    for terms_sys in terms_sys_list:
        all_terms_sys.update(terms_sys)
    with open(dst_file, 'w', encoding='utf-8', newline='\n') as fout:
        for t in all_terms_sys:
            if t not in all_terms_true:
                fout.write('{}\n'.format(t))


def __evaluate(terms_sys_list, terms_true_list, dep_tags_list, pos_tags_list, sent_texts):
    correct_sent_idxs = list()
    hit_cnt, true_cnt, sys_cnt = 0, 0, 0
    for sent_idx, (terms_sys, terms_true, dep_tags, pos_tags) in enumerate(
            zip(terms_sys_list, terms_true_list, dep_tags_list, pos_tags_list)):
        true_cnt += len(terms_true)
        sys_cnt += len(terms_sys)
        # new_hit_cnt = __count_hit(terms_true, aspect_terms)
        new_hit_cnt = __count_hit(terms_true, terms_sys)
        if new_hit_cnt == len(terms_true) and new_hit_cnt == len(terms_sys):
            correct_sent_idxs.append(sent_idx)
        hit_cnt += new_hit_cnt
        if len(terms_true) and new_hit_cnt < len(terms_true):
            print(terms_true)
            print(terms_sys)
            print(sent_texts[sent_idx])
            print(pos_tags)
            print(dep_tags)
            print()

    # __save_never_hit_terms(sents, terms_sys_list, 'd:/data/aspect/semeval14/tmp.txt')

    print('hit={}, true={}, sys={}'.format(hit_cnt, true_cnt, sys_cnt))
    p = hit_cnt / (sys_cnt + 1e-8)
    r = hit_cnt / (true_cnt + 1e-8)
    print(p, r, 2 * p * r / (p + r + 1e-8))
    return correct_sent_idxs


def __opinion_rule1(dep_tags, pos_tags):
    words = [dep_tag[2][1] for dep_tag in dep_tags]
    assert len(words) == len(pos_tags)
    # print(words)
    # print(pos_tags)

    opinion_terms = list()
    for i, w in enumerate(words):
        pos = pos_tags[i]
        if pos in {'JJ', 'RB'} and w not in {"n't", 'not', 'so', 'also'}:
            opinion_terms.append(w)
    # print(' '.join(words))
    # print(noun_phrases)
    return opinion_terms


def __opinion_rule2(dep_tags, pos_tags):
    words = [dep_tag[2][1] for dep_tag in dep_tags]
    assert len(words) == len(pos_tags)
    # print(words)
    # print(pos_tags)

    opinion_terms = list()
    for dep_tup in dep_tags:
        rel, gov, dep = dep_tup
        if rel not in {'nsubj'}:
            continue

        igov, wgov = gov
        idep, wdep = dep

        # if wdep == 'i':
        #     opinion_terms.append(wgov)
        if wdep == 'it' and pos_tags[igov] == 'JJ':
            opinion_terms.append(wgov)

    # print(' '.join(words))
    # print(noun_phrases)
    # print(opinion_terms)
    return opinion_terms


def __opinion_rule3(sent_text: str, terms_dict):
    terms = list()
    for t in terms_dict:
        pbeg = sent_text.find(t)
        pend = pbeg + len(t)
        if pbeg > 0 and sent_text[pbeg - 1].isalpha():
            continue
        if pend < len(sent_text) and sent_text[pend].isalpha():
            continue
        terms.append(t)
    return terms


def __opinion_rule4(dep_tags, pos_tags):
    words = [dep_tag[2][1] for dep_tag in dep_tags]
    assert len(words) == len(pos_tags)
    # print(words)
    # print(pos_tags)

    opinion_terms = list()
    for dep_tup in dep_tags:
        rel, gov, dep = dep_tup
        if rel not in {'amod'}:
            continue

        igov, wgov = gov
        idep, wdep = dep
        print(wgov, wdep)
        opinion_terms.append(wdep)

    # print(' '.join(words))
    # print(noun_phrases)
    # print(opinion_terms)
    return opinion_terms


def __filter_sub_terms(terms):
    terms_sys_tmp = list(terms)
    terms_sys_tmp.sort(key=lambda x: len(x))
    terms_new = list()
    for i, t in enumerate(terms_sys_tmp):
        sub_term = False
        for j in range(i + 1, len(terms_sys_tmp)):
            if t in terms_sys_tmp[j]:
                sub_term = True
                break
        if not sub_term:
            terms_new.append(t)
    return terms_new


def __write_rule_results(terms_list, sent_texts, dst_file):
    if dst_file is not None:
        fout = open(dst_file, 'w', encoding='utf-8', newline='\n')
        for terms_sys, sent_text in zip(terms_list, sent_texts):
            # sent_obj = {'text': sent_text}
            # if terms_sys:
            #     sent_obj['terms'] = terms_sys
            fout.write('{}\n'.format(json.dumps(list(terms_sys), ensure_ascii=False)))
        fout.close()


def __opinion_rule_insight(dep_tags_file, pos_tags_file, sent_text_file, terms_vocab,
                           dst_result_file=None, sents_file=None):
    print('loading data ...')
    dep_tags_list = utils.load_dep_tags_list(dep_tags_file)
    pos_tags_list = utils.load_pos_tags(pos_tags_file)
    sent_texts = utils.read_lines(sent_text_file)
    assert len(dep_tags_list) == len(sent_texts)
    assert len(pos_tags_list) == len(dep_tags_list)
    print('done.')
    opinions_sys_list = list()
    for sent_idx, sent_text in enumerate(sent_texts):
        dep_tags = dep_tags_list[sent_idx]
        pos_tags = pos_tags_list[sent_idx]
        assert len(dep_tags) == len(pos_tags)

        opinion_terms = set()
        # terms_new = __opinion_rule1(dep_tags, pos_tags)
        # opinion_terms.update(terms_new)
        terms_new = __opinion_rule2(dep_tags, pos_tags)
        opinion_terms.update(terms_new)
        # terms_new = __opinion_rule4(dep_tags, pos_tags)
        # opinion_terms.update(terms_new)
        terms_new = __match_terms(sent_text, terms_vocab)
        opinion_terms.update(terms_new)
        opinions_sys_list.append(opinion_terms)

        if sent_idx % 10000 == 0:
            print(sent_idx)

    if dst_result_file is not None:
        __write_rule_results(opinions_sys_list, sent_texts, dst_result_file)

    if sents_file is not None:
        sents = utils.load_json_objs(sents_file)
        opinions_true_list = list()
        for sent in sents:
            opinions_true_list.append([t.lower() for t in sent.get('opinions', list())])
        correct_sent_idxs = __evaluate(opinions_sys_list, opinions_true_list, dep_tags_list, pos_tags_list, sent_texts)


def __rule_insight(opinion_term_dict_file, filter_nouns_file, dep_tags_file, pos_tags_file,
                   sent_text_file, train_sents_file, dst_result_file=None, sents_file=None):
    # print(terms_train)
    print('loading data ...')
    aspect_terms_train = __load_terms_in_train(train_sents_file)
    # print(aspect_terms_train)
    opinion_terms = utils.read_lines(opinion_term_dict_file)
    opinion_terms = set(opinion_terms)
    nouns_filter = utils.read_lines(filter_nouns_file)
    nouns_filter = set(nouns_filter)

    dep_tags_list = utils.load_dep_tags_list(dep_tags_file)
    pos_tags_list = utils.load_pos_tags(pos_tags_file)
    sent_texts = utils.read_lines(sent_text_file)
    print('done.')

    assert len(dep_tags_list) == len(sent_texts)
    assert len(pos_tags_list) == len(dep_tags_list)

    sents = None if sents_file is None else utils.load_json_objs(sents_file)

    aspects_sys_list = list()
    for sent_idx, sent_text in enumerate(sent_texts):
        dep_tags = dep_tags_list[sent_idx]
        pos_tags = pos_tags_list[sent_idx]
        assert len(dep_tags) == len(pos_tags)

        # terms_true = set([t['term'].lower() for t in sents[sent_idx].get('terms', list())])
        terms_true = None if sents is None else set(
            [t['term'].lower() for t in sents[sent_idx].get('terms', list())])

        aspect_terms = set()
        # aspect_terms_new = rules.rule1(dep_tags, pos_tags, opinion_terms, nouns_filter)
        # aspect_terms.update(aspect_terms_new)
        # aspect_terms_new = rules.rule2(dep_tags, pos_tags, opinion_terms, nouns_filter)
        # aspect_terms.update(aspect_terms_new)
        # aspect_terms_new = rules.rule3(dep_tags, pos_tags, opinion_terms, nouns_filter, terms_true)
        # aspect_terms.update(aspect_terms_new)
        aspect_terms_new = rules.rule4(dep_tags, pos_tags, sent_text, opinion_terms, nouns_filter,
                                       aspect_terms_train)
        aspect_terms.update(aspect_terms_new)
        # aspect_terms_new = rules.rule5(dep_tags, pos_tags, opinion_terms, nouns_filter)
        # aspect_terms.update(aspect_terms_new)
        aspect_terms_new = rules.conj_rule(dep_tags, pos_tags, opinion_terms, nouns_filter, aspect_terms)
        aspect_terms.update(aspect_terms_new)

        words = [dep_tag[2][1] for dep_tag in dep_tags]
        aspect_terms_new = rules.rec_rule1(words, pos_tags, nouns_filter)
        aspect_terms.update(aspect_terms_new)

        # if sent_idx > 10:
        #     exit()

        terms_sys = __filter_sub_terms(aspect_terms)
        aspects_sys_list.append(terms_sys)

        if sent_idx % 10000 == 0:
            print(sent_idx)
        # if sent_idx >= 100:
        #     break

    if dst_result_file is not None:
        fout = open(dst_result_file, 'w', encoding='utf-8', newline='\n')
        for terms_sys, sent_text in zip(aspects_sys_list, sent_texts):
            sent_obj = {'text': sent_text}
            if terms_sys:
                sent_obj['terms'] = terms_sys
            fout.write('{}\n'.format(json.dumps(terms_sys, ensure_ascii=False)))
        fout.close()

    if sents_file is not None:
        sents = utils.load_json_objs(sents_file)
        aspect_terms_true = list()
        for sent in sents:
            aspect_terms_true.append([t['term'].lower() for t in sent.get('terms', list())])
        sent_texts = [sent['text'] for sent in sents]
        correct_sent_idxs = __evaluate(aspects_sys_list, aspect_terms_true, dep_tags_list, pos_tags_list, sent_texts)
        with open('d:/data/aspect/semeval14/rules-correct.txt', 'w', encoding='utf-8') as fout:
            fout.write('\n'.join([str(i) for i in correct_sent_idxs]))


def __differ():
    idxs_rule = utils.read_lines('d:/data/aspect/semeval14/rules-correct.txt')
    idxs_neu = utils.read_lines('d:/data/aspect/semeval14/lstmcrf-correct.txt')
    idxs_rule = [int(idx) for idx in idxs_rule]
    idxs_neu = [int(idx) for idx in idxs_neu]
    print(idxs_rule)
    print(idxs_neu)
    idxs_rule_only = list()
    for i in idxs_rule:
        if i not in idxs_neu:
            idxs_rule_only.append(i)
    idxs_neu_only = list()
    for i in idxs_neu:
        if i not in idxs_rule:
            idxs_neu_only.append(i)
    print(idxs_rule_only)
    print(len(idxs_rule_only))
    print(idxs_neu_only)
    print(len(idxs_neu_only))


# sents = utils.load_json_objs(config.SE14_LAPTOP_TEST_SENTS_FILE)

# opinion_terms_file = 'd:/data/aspect/semeval14/opinion-terms.txt'
opinion_terms_file = 'd:/data/aspect/semeval14/opinion-terms-full.txt'
filter_nouns_file = 'd:/data/aspect/semeval14/nouns-filter.txt'
rest_filter_nouns_file = 'd:/data/aspect/semeval14/restaurant/aspect-nouns-filter.txt'

# dep_tags_file = 'd:/data/aspect/semeval14/laptops/laptops-test-rule-dep.txt'
# pos_tags_file = 'd:/data/aspect/semeval14/laptops/laptops-test-rule-pos.txt'
# result_file = 'd:/data/aspect/semeval14/laptops/laptops-test-aspect-rule-result.txt'
# sent_texts_file = 'd:/data/aspect/semeval14/laptops/laptops_test_texts.txt'
# train_sents_file = config.SE14_LAPTOP_TRAIN_SENTS_FILE
# sents_file = config.SE14_LAPTOP_TEST_SENTS_FILE

# dep_tags_file = 'd:/data/aspect/semeval14/laptops/laptops-train-rule-dep.txt'
# pos_tags_file = 'd:/data/aspect/semeval14/laptops/laptops-train-rule-pos.txt'
# result_file = 'd:/data/aspect/semeval14/laptops/laptops-train-aspect-rule-result.txt'
# sent_texts_file = 'd:/data/aspect/semeval14/laptops/laptops_train_texts.txt'
# train_sents_file = config.SE14_LAPTOP_TRAIN_SENTS_FILE
# sents_file = config.SE14_LAPTOP_TRAIN_SENTS_FILE

# dep_tags_file = 'd:/data/amazon/laptops-rule-dep.txt'
# pos_tags_file = 'd:/data/amazon/laptops-rule-pos.txt'
# result_file = 'd:/data/amazon/laptops-opinion-rule-result.txt'
# sent_texts_file = 'd:/data/amazon/laptops-reivews-sent-text.txt'
# sents_file = None

# __rule_insight(opinion_terms_file, filter_nouns_file, dep_tags_file, pos_tags_file, sent_texts_file,
#                train_sents_file, dst_result_file=aspect_result_file, sents_file=sents_file)


# dep_tags_file = 'd:/data/aspect/semeval14/restaurant/restaurants-train-rule-dep.txt'
# pos_tags_file = 'd:/data/aspect/semeval14/restaurant/restaurants-train-rule-pos.txt'
# aspect_result_file = 'd:/data/aspect/semeval14/restaurant/restaurants-train-aspect-rule-result.txt'
# sent_texts_file = 'd:/data/aspect/semeval14/restaurant/restaurants_train_texts.txt'
# train_sents_file = config.SE14_REST_TRAIN_SENTS_FILE
# sents_file = config.SE14_REST_TRAIN_SENTS_FILE

# dep_tags_file = 'd:/data/aspect/semeval14/restaurant/restaurants-test-rule-dep.txt'
# pos_tags_file = 'd:/data/aspect/semeval14/restaurant/restaurants-test-rule-pos.txt'
# aspect_result_file = 'd:/data/aspect/semeval14/restaurant/restaurants-test-aspect-rule-result-p.txt'
# # aspect_result_file = 'd:/data/aspect/semeval14/restaurant/restaurants-test-aspect-rule-result-r.txt'
# opinion_result_file = 'd:/data/aspect/semeval14/restaurant/restaurants-test-opinion-rule-result.txt'
# sent_texts_file = 'd:/data/aspect/semeval14/restaurant/restaurants_test_texts.txt'
# train_sents_file = config.SE14_REST_TRAIN_SENTS_FILE
# sents_file = config.SE14_REST_TEST_SENTS_FILE

dep_tags_file = 'd:/data/res/yelp-review-round-9-dep.txt'
pos_tags_file = 'd:/data/res/yelp-review-round-9-pos.txt'
# aspect_result_file = 'd:/data/aspect/semeval14/restaurant/yelp-aspect-rule-result-p.txt'
aspect_result_file = 'd:/data/aspect/semeval14/restaurant/yelp-aspect-rule-result-r.txt'
opinion_result_file = 'd:/data/aspect/semeval14/restaurant/yelp-opinion-rule-result.txt'
sent_texts_file = 'd:/data/res/yelp-review-eng-tok-sents-round-9.txt'
train_sents_file = config.SE14_REST_TRAIN_SENTS_FILE
sents_file = None

__rule_insight(opinion_terms_file, rest_filter_nouns_file, dep_tags_file, pos_tags_file, sent_texts_file,
               train_sents_file, dst_result_file=aspect_result_file, sents_file=sents_file)

# terms_vocab = __load_opinion_terms_in_train(train_sents_file)
# __opinion_rule_insight(dep_tags_file, pos_tags_file, sent_texts_file, terms_vocab,
#                        dst_result_file=opinion_result_file, sents_file=sents_file)

# __differ()
