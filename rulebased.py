import json
import config
from utils import utils


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


def __get_phrase(dep_tags, pos_tags, base_word_idx):
    words = [tup[2][1] for tup in dep_tags]
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

    iend = base_word_idx + dr
    if pos_tags[iend - 1] in {'VBN', 'VB'} and iend < len(words) and pos_tags[iend] == 'RP':
        iend += 1
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    return ' '.join(words[base_word_idx - dl + 1: iend])


def __get_phrase_for_rule1(dep_tags, pos_tags, words, idep, igov):
    assert len(words) == len(pos_tags)
    ibeg, iend = __get_phrase_span_by_compound(dep_tags, idep)
    print(ibeg, iend, idep, igov)
    for i in range(idep, igov):
        if words[i] not in {'is', 'was', 'are'}:
            continue
        for j in range(iend, i):
            if pos_tags[j] in {'VBN', 'JJ', 'NN', 'NNS'}:
                iend = j
            else:
                break
        break
    for i in range(ibeg, max(ibeg - 5, 0), -1):
        if pos_tags[i] in {'VBN', 'JJ', 'NN', 'NNS'}:
            ibeg = i
        else:
            break
    return ' '.join(words[ibeg: iend])


def __rule1(dep_tags, pos_tags, opinion_terms, nouns_filter):
    aspect_terms = set()
    sent_words = [dep_tup[2][1] for dep_tup in dep_tags]
    for dep_tup in dep_tags:
        rel, gov, dep = dep_tup
        if rel not in {'nsubj', 'csubj', 'nmod'}:
            continue

        igov, wgov = gov
        idep, wdep = dep
        if wdep in nouns_filter:
            continue
        # if wgov not in opinion_terms:
        #     continue

        # print(rel, wgov, wdep)
        # phrase = __get_phrase(dep_tags, pos_tags, idep)
        phrase = __get_phrase_new(dep_tags, pos_tags, [idep], opinion_terms)
        # phrase1 = __get_phrase_for_rule1(dep_tags, pos_tags, sent_words, idep, igov)
        # if phrase in terms_true and wgov not in opinion_terms:
        #     print(dep_tup)
        aspect_terms.add(phrase)
        # if phrase != phrase1:
        #     print(sent_text)
        #     print(pos_tags)
        #     print(terms_true)
        #     print(phrase)
        #     print(phrase1)
        #     print()
        # print(rel, wgov, pos_tags[igov], wdep, pos_tags[idep], phrase)
    return aspect_terms


def __rule2(dep_tags, pos_tags, opinion_terms, nouns_filter):
    aspect_terms = set()
    for dep_tup in dep_tags:
        rel, gov, dep = dep_tup
        if rel != 'amod':
            continue

        igov, wgov = gov
        idep, wdep = dep
        if wgov in nouns_filter:
            continue
        if wdep not in opinion_terms:
            continue

        # print(dep_tup)
        # phrase = __get_phrase(dep_tags, pos_tags, igov)
        phrase = __get_phrase_new(dep_tags, pos_tags, [igov], opinion_terms)
        aspect_terms.add(phrase)
    return aspect_terms


def __rule6(dep_tags, pos_tags, opinion_terms, nouns_filter):
    aspect_terms = set()
    for dep_tup in dep_tags:
        rel, gov, dep = dep_tup
        if rel != 'advmod':
            continue

        igov, wgov = gov
        idep, wdep = dep
        if pos_tags[igov] in {'JJ'}:
            continue
        if wgov in nouns_filter:
            continue
        if wdep not in opinion_terms:
            continue

        # print(dep_tup)
        # phrase = __get_phrase(dep_tags, pos_tags, igov)
        aspect_terms.add(wgov)
        # print(wgov, wdep)
    return aspect_terms


# tmp_tup_list = list()
def __rule3(dep_tags, pos_tags, opinion_terms, nouns_filter):
    aspect_terms = set()
    for dep_tup in dep_tags:
        rel, gov, dep = dep_tup
        if rel not in {'dobj', 'xcomp'}:
            continue

        igov, wgov = gov
        idep, wdep = dep
        if wdep in nouns_filter:
            continue
        # if wdep not in opinion_terms:
        #     continue

        phrase = __get_phrase_new(dep_tags, pos_tags, [idep], opinion_terms)
        # phrase = __get_phrase(dep_tags, pos_tags, idep)
        hit = False
        for j in range(len(dep_tags)):
            if dep_tags[j][1][0] == igov and dep_tags[j][0] == 'nsubj':
                hit = True
                break

        aspect_terms.add(phrase)
        # if hit and wgov in {'has', 'have', 'had', 'got', 'offers', 'enjoy', 'like', 'love'}:
        #     aspect_terms.add(phrase)
        # else:
        #     print(dep_tup)
            # tmp_tup_list.append(dep_tup)
    return aspect_terms


def __remove_embeded(matched_tups):
    matched_tups_new = list()
    for i, t0 in enumerate(matched_tups):
        exist = False
        for j, t1 in enumerate(matched_tups):
            if i != j and t1[0] <= t0[0] and t1[1] >= t0[1]:
                exist = True
                break
        if not exist:
            matched_tups_new.append(t0)
    return matched_tups_new


def __find_word_spans(text_lower, words):
    p = 0
    word_spans = list()
    for w in words:
        wp = text_lower[p:].find(w)
        if wp < 0:
            word_spans.append((-1, -1))
            continue
        word_spans.append((p + wp, p + wp + len(w)))
        p += wp + len(w)
    return word_spans


def __pharse_for_span(span, sent_text_lower, words, pos_tags, dep_tags, opinion_terms):
    word_spans = __find_word_spans(sent_text_lower, words)
    widxs = list()
    for i, wspan in enumerate(word_spans):
        if (wspan[0] <= span[0] < wspan[1]) or (wspan[0] < span[1] <= wspan[1]):
            widxs.append(i)

    if not widxs:
        # print(span)
        # print(sent_text_lower[span[0]: span[1]])
        # print(sent_text_lower)
        # print(words)
        # print(word_spans)
        # exit()
        return None

    phrase = __get_phrase_new(dep_tags, pos_tags, widxs, opinion_terms)
    return phrase


def __rule4(dep_tags, pos_tags, sent_text, opinion_terms, nouns_filter, terms_train):
    sent_text_lower = sent_text.lower()
    matched_tups = list()
    for t in terms_train:
        pbeg = sent_text_lower.find(t)
        if pbeg < 0:
            continue
        if pbeg != 0 and sent_text_lower[pbeg - 1].isalpha():
            continue
        pend = pbeg + len(t)
        if pend != len(sent_text_lower) and sent_text_lower[pend].isalpha():
            continue
        matched_tups.append((pbeg, pend))
        # break

    matched_tups = __remove_embeded(matched_tups)
    sent_words = [tup[2][1] for tup in dep_tags]
    aspect_terms = set()
    for matched_span in matched_tups:
        phrase = __pharse_for_span(matched_span, sent_text_lower, sent_words, pos_tags, dep_tags, opinion_terms)
        if phrase is not None:
            aspect_terms.add(phrase)

    return aspect_terms


def __word_in_terms(word, terms):
    for t in terms:
        if word in t:
            return True
    return False


def __rule5(dep_tags, pos_tags, opinion_terms, nouns_filter, terms_extracted):
    aspect_terms = set()
    for dep_tup in dep_tags:
        rel, gov, dep = dep_tup
        if rel != 'conj':
            continue

        igov, wgov = gov
        idep, wdep = dep

        if __word_in_terms(wgov, terms_extracted) and not __word_in_terms(
                wdep, terms_extracted) and wdep not in nouns_filter:
            phrase = __get_phrase_new(dep_tags, pos_tags, [idep], opinion_terms)
            # phrase = __get_phrase(dep_tags, pos_tags, idep)
            aspect_terms.add(phrase)
        elif __word_in_terms(wdep, terms_extracted) and not __word_in_terms(
                wgov, terms_extracted) and wgov not in nouns_filter:
            phrase = __get_phrase_new(dep_tags, pos_tags, [idep], opinion_terms)
            # phrase = __get_phrase(dep_tags, pos_tags, idep)
            aspect_terms.add(phrase)

    return aspect_terms


def __rec_rule1(dep_tags, pos_tags, nouns_filter, opinion_terms):
    words = [dep_tag[2][1] for dep_tag in dep_tags]
    assert len(words) == len(pos_tags)

    noun_phrases = list()
    pleft = 0
    while pleft < len(words):
        if pos_tags[pleft] not in {'NN', 'NNS', 'NNP'}:
            pleft += 1
            continue
        pright = pleft + 1
        while pright < len(words) and pos_tags[pright] in {'NN', 'NNS', 'NNP', 'CD'}:
            pright += 1

        # if pleft > 0 and pos_tags[pleft - 1] == 'JJ' and words[pleft - 1] not in opinion_terms:
        #     pleft -= 1

        phrase = ' '.join(words[pleft: pright])
        if phrase not in nouns_filter:
            noun_phrases.append(phrase)
        pleft = pright
    # print(' '.join(words))
    # print(noun_phrases)
    return noun_phrases


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
        # aspect_terms_new = __rule1(dep_tags, pos_tags, opinion_terms, nouns_filter)
        # aspect_terms.update(aspect_terms_new)
        # aspect_terms_new = __rule2(dep_tags, pos_tags, opinion_terms, nouns_filter)
        # aspect_terms.update(aspect_terms_new)
        # aspect_terms_new = __rule3(dep_tags, pos_tags, opinion_terms, nouns_filter)
        # aspect_terms.update(aspect_terms_new)
        # aspect_terms_new = __rule4(dep_tags, pos_tags, sent_text, opinion_terms, nouns_filter, terms_train)
        # aspect_terms.update(aspect_terms_new)
        # aspect_terms_new = __rule6(dep_tags, pos_tags, opinion_terms, nouns_filter)
        # aspect_terms.update(aspect_terms_new)
        # aspect_terms_new = __rule5(dep_tags, pos_tags, opinion_terms, nouns_filter, aspect_terms)
        # aspect_terms.update(aspect_terms_new)
        aspect_terms_new = __rec_rule1(dep_tags, pos_tags, nouns_filter, opinion_terms)
        aspect_terms.update(aspect_terms_new)

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
# result_file = 'd:/data/amazon/laptops-rule-result1.txt'
# sent_texts_file = 'd:/data/amazon/laptops-reivews-sent-text.txt'
# sents_file = None

__rule_insight(opinion_terms_file, filter_nouns_file, dep_tags_file,
               pos_tags_file, sent_texts_file, dst_result_file=result_file, sents_file=sents_file)

# __differ()
