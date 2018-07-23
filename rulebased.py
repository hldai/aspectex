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


def __load_terms_in_train():
    sents_train = utils.load_json_objs(config.SE14_LAPTOP_TRAIN_SENTS_FILE)
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


def __get_phrase_new(dep_tags, pos_tags, base_word_idxs, opinion_terms):
    words = [tup[2][1] for tup in dep_tags]
    phrase_word_idxs = set(base_word_idxs)
    # find_compound = True
    # while find_compound:
    #     find_compound = False
    #     for rel, (igov, wgov), (idep, wdep) in dep_tags:
    #         if rel != 'compound':
    #             continue
    #         if igov in phrase_word_idxs and idep not in phrase_word_idxs:
    #             find_compound = True
    #             phrase_word_idxs.add(idep)
    #         if idep in phrase_word_idxs and igov not in phrase_word_idxs:
    #             find_compound = True
    #             phrase_word_idxs.add(igov)

    ileft = min(phrase_word_idxs)
    iright = max(phrase_word_idxs)
    ileft_new, iright_new = ileft, iright
    while ileft_new > 0:
        if pos_tags[ileft_new - 1] in {'NN', 'NNP', 'NNS'}:
            ileft_new -= 1
        else:
            break
    while iright_new < len(pos_tags) - 1:
        if pos_tags[iright_new + 1] in {'NN', 'NNP', 'NNS', 'CD'}:
            iright_new += 1
        else:
            break

    # print(ileft, iright, ileft_new, iright_new)
    # print(pos_tags)
    phrase = ' '.join([words[widx] for widx in range(ileft_new, iright_new + 1)])
    return phrase


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


def __evaluate(terms_sys_list, sents, dep_tags_list, pos_tags_list):
    correct_sent_idxs = list()
    hit_cnt, true_cnt, sys_cnt = 0, 0, 0
    for sent_idx, (terms_sys, sent, dep_tags, pos_tags) in enumerate(
            zip(terms_sys_list, sents, dep_tags_list, pos_tags_list)):
        term_objs = sent.get('terms', list())
        sent_text = sent['text']
        terms_true = [t['term'].lower() for t in term_objs]
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
            print(sent_text)
            print(pos_tags)
            print(dep_tags)
            print()

    # __save_never_hit_terms(sents, terms_sys_list, 'd:/data/aspect/semeval14/tmp.txt')

    print(hit_cnt, true_cnt, sys_cnt)
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


def __rule_insight(opinion_terms_file, filter_nouns_file, dep_tags_file, pos_tags_file,
                   sent_text_file, dst_result_file=None, sents_file=None):
    # print(terms_train)
    print('loading data ...')
    terms_train = __load_terms_in_train()
    opinion_terms = utils.read_lines(opinion_terms_file)
    opinion_terms = set(opinion_terms)
    nouns_filter = utils.read_lines(filter_nouns_file)
    nouns_filter = set(nouns_filter)

    dep_tags_list = utils.load_dep_tags_list(dep_tags_file)
    pos_tags_list = utils.load_pos_tags(pos_tags_file)

    sent_texts = utils.read_lines(sent_text_file)
    print('done.')

    assert len(dep_tags_list) == len(sent_texts)
    assert len(pos_tags_list) == len(dep_tags_list)

    terms_sys_list = list()
    for sent_idx, sent_text in enumerate(sent_texts):
        dep_tags = dep_tags_list[sent_idx]
        pos_tags = pos_tags_list[sent_idx]
        assert len(dep_tags) == len(pos_tags)

        aspect_terms = set()
        # aspect_terms_new = rules.rule1(dep_tags, pos_tags, opinion_terms, nouns_filter)
        # aspect_terms.update(aspect_terms_new)
        # aspect_terms_new = rules.rule2(dep_tags, pos_tags, opinion_terms, nouns_filter)
        # aspect_terms.update(aspect_terms_new)
        # aspect_terms_new = rules.rule3(dep_tags, pos_tags, opinion_terms, nouns_filter)
        # aspect_terms.update(aspect_terms_new)
        # aspect_terms_new = rules.rule4(dep_tags, pos_tags, sent_text, opinion_terms, nouns_filter, terms_train)
        # aspect_terms.update(aspect_terms_new)
        # aspect_terms_new = rules.rule5(dep_tags, pos_tags, opinion_terms, nouns_filter)
        # aspect_terms.update(aspect_terms_new)
        # aspect_terms_new = rules.conj_rule(dep_tags, pos_tags, opinion_terms, nouns_filter, aspect_terms)
        # aspect_terms.update(aspect_terms_new)
        aspect_terms_new = rules.rec_rule1(dep_tags, pos_tags, nouns_filter, opinion_terms)
        aspect_terms.update(aspect_terms_new)

        # aspect_terms_new = __opinion_rule1(dep_tags, pos_tags)
        # aspect_terms.update(aspect_terms_new)
        # if sent_idx > 10:
        #     exit()

        terms_sys_tmp = list(aspect_terms)
        terms_sys_tmp.sort(key=lambda x: len(x))
        terms_sys = list()
        for i, t in enumerate(terms_sys_tmp):
            sub_term = False
            for j in range(i + 1, len(terms_sys_tmp)):
                if t in terms_sys_tmp[j]:
                    sub_term = True
                    break
            if not sub_term:
                terms_sys.append(t)
        terms_sys_list.append(terms_sys)

        if sent_idx % 10000 == 0:
            print(sent_idx)
        # if sent_idx >= 100:
        #     break

    if dst_result_file is not None:
        fout = open(dst_result_file, 'w', encoding='utf-8', newline='\n')
        for terms_sys, sent_text in zip(terms_sys_list, sent_texts):
            sent_obj = {'text': sent_text}
            if terms_sys:
                sent_obj['terms'] = terms_sys
            fout.write('{}\n'.format(json.dumps(terms_sys, ensure_ascii=False)))
        fout.close()

    if sents_file is not None:
        sents = utils.load_json_objs(sents_file)
        correct_sent_idxs = __evaluate(terms_sys_list, sents, dep_tags_list, pos_tags_list)
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

opinion_terms_file = 'd:/data/aspect/semeval14/opinion-terms.txt'
filter_nouns_file = 'd:/data/aspect/semeval14/nouns-filter.txt'

dep_tags_file = 'd:/data/aspect/semeval14/laptops-test-rule-dep.txt'
pos_tags_file = 'd:/data/aspect/semeval14/laptops-test-rule-pos.txt'
result_file = 'd:/data/aspect/semeval14/laptops-test-rule-result.txt'
sent_texts_file = 'd:/data/aspect/semeval14/laptops_test_texts.txt'
sents_file = config.SE14_LAPTOP_TEST_SENTS_FILE

# dep_tags_file = 'd:/data/aspect/semeval14/laptops-train-rule-dep.txt'
# pos_tags_file = 'd:/data/aspect/semeval14/laptops-train-rule-pos.txt'
# result_file = 'd:/data/aspect/semeval14/laptops-train-rule-result.txt'
# sent_texts_file = 'd:/data/aspect/semeval14/laptops_train_texts.txt'
# sents_file = config.SE14_LAPTOP_TRAIN_SENTS_FILE

# dep_tags_file = 'd:/data/amazon/laptops-rule-dep.txt'
# pos_tags_file = 'd:/data/amazon/laptops-rule-pos.txt'
# result_file = 'd:/data/amazon/laptops-rule-result4.txt'
# sent_texts_file = 'd:/data/amazon/laptops-reivews-sent-text.txt'
# sents_file = None

__rule_insight(opinion_terms_file, filter_nouns_file, dep_tags_file,
               pos_tags_file, sent_texts_file, dst_result_file=result_file, sents_file=sents_file)

# __differ()
